"""llama-server 프로세스 실행/중지/상태 감시 (QProcess 기반, macOS/Windows 공통)."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional

import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

from . import server_log
from .netutil import client_host
from .settings import config_dir


def pid_file() -> "Path":
    return config_dir() / "server.pid"


def find_orphan_server() -> Optional[psutil.Process]:
    """앱이 비정상 종료되어 남겨진 llama-server가 있으면 돌려준다."""
    try:
        pid = int(pid_file().read_text().strip())
        proc = psutil.Process(pid)
        if "llama-server" in proc.name().lower() or any("llama-server" in a for a in proc.cmdline()[:1]):
            return proc
    except (OSError, ValueError, psutil.Error):
        pass
    clear_pid_file()
    return None


def clear_pid_file() -> None:
    try:
        pid_file().unlink()
    except OSError:
        pass


class ServerState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

    @property
    def label(self) -> str:
        return {"stopped": "중지됨", "starting": "모델 불러오는 중…", "running": "실행 중",
                "stopping": "중지하는 중…", "error": "오류"}[self.value]


class ServerController(QObject):
    state_changed = Signal(object)        # ServerState
    log_line = Signal(str, str)           # (줄, 분류)
    speed_changed = Signal(str, float)    # ("prompt"|"generate", tokens/s)
    memory_changed = Signal(object)       # 바이트 (int는 2GB에서 넘침)
    error_hint = Signal(str)
    _prepared = Signal(object)            # 준비 작업 결과 (None이면 성공, 문자열이면 오류) — 작업 스레드 → UI 스레드

    HEALTH_INTERVAL_MS = 1000
    STOP_GRACE_MS = 5000

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = ServerState.STOPPED
        self.proc: Optional[QProcess] = None
        self.host = "127.0.0.1"
        self.port = 8080
        self.started_at = 0.0
        self.health_path = "/health"
        self.program_name = "llama-server"
        self._prepare = None
        self._preparing = False
        self._fail_hint: Optional[str] = None
        self._prepared.connect(self._on_prepared)
        self._buf = b""
        self._recent: deque[str] = deque(maxlen=80)
        self._net = QNetworkAccessManager(self)
        self._health_pending = False
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._sample_memory)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)

    # ---- 공개 API ------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{client_host(self.host)}:{self.port}"

    @property
    def is_active(self) -> bool:
        return self.state in (ServerState.STARTING, ServerState.RUNNING, ServerState.STOPPING)

    @property
    def pid(self) -> Optional[int]:
        if self.proc and self.proc.state() != QProcess.NotRunning:
            return int(self.proc.processId())
        return None

    def start(self, program: str, args: list[str], host: str, port: int, health_path: str = "/health",
              env: Optional[dict] = None, prepare=None) -> None:
        """prepare(log): 서버가 응답하기 시작한 뒤, '실행 중'으로 바꾸기 전에 할 일 (예: Ollama 모델 등록)."""
        if self.is_active:
            return
        self.host, self.port = host, port
        self.health_path = health_path
        self.program_name = Path(program).name
        self._prepare = prepare
        self._preparing = False
        self._fail_hint = None
        self._buf = b""
        self._recent.clear()
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        if env:
            penv = QProcessEnvironment.systemEnvironment()
            for k, v in env.items():
                penv.insert(k, str(v))
            self.proc.setProcessEnvironment(penv)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)
        self.started_at = time.monotonic()
        self._set_state(ServerState.STARTING)
        self.proc.started.connect(self._write_pid)
        self.proc.start(program, args)
        self._health_timer.start(self.HEALTH_INTERVAL_MS)
        self._mem_timer.start(2000)

    def stop(self) -> None:
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            self._set_state(ServerState.STOPPED)
            return
        self._set_state(ServerState.STOPPING)
        if sys.platform == "win32":
            # 콘솔 프로그램은 terminate(WM_CLOSE)를 무시하므로 바로 종료
            self.proc.kill()
        else:
            self.proc.terminate()  # SIGTERM → llama-server가 정리 후 종료
            self._kill_timer.start(self.STOP_GRACE_MS)

    def stop_blocking(self, timeout_ms: int = 5000) -> None:
        """앱 종료 시 사용: 프로세스가 끝날 때까지 기다린다."""
        if not self.proc or self.proc.state() == QProcess.NotRunning:
            return
        self.stop()
        if not self.proc.waitForFinished(timeout_ms):
            self.proc.kill()
            self.proc.waitForFinished(2000)

    # ---- 내부 ----------------------------------------------------------
    def _set_state(self, state: ServerState) -> None:
        if state == self.state:
            return
        self.state = state
        if state in (ServerState.STOPPED, ServerState.ERROR):
            self._health_timer.stop()
            self._mem_timer.stop()
            self._kill_timer.stop()
        self.state_changed.emit(state)

    def _on_output(self) -> None:
        if not self.proc:
            return
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for raw in lines:
            line = raw.decode("utf-8", errors="replace").rstrip("\r")
            if not line.strip():
                continue
            self._recent.append(line)
            self.log_line.emit(line, server_log.classify(line))
            sample = server_log.parse_speed(line)
            if sample:
                self.speed_changed.emit(sample.kind, sample.tokens_per_sec)

    def _write_pid(self) -> None:
        try:
            pid_file().parent.mkdir(parents=True, exist_ok=True)
            pid_file().write_text(str(self.pid or ""))
        except OSError:
            pass

    def _on_finished(self, exit_code: int, _status) -> None:
        clear_pid_file()
        self._on_output()
        if self._buf.strip():
            self._recent.append(self._buf.decode("utf-8", errors="replace"))
            self.log_line.emit(self._recent[-1], server_log.classify(self._recent[-1]))
            self._buf = b""
        if self._fail_hint:
            self.error_hint.emit(self._fail_hint)
            self._fail_hint = None
            self._set_state(ServerState.ERROR)
            return
        if self.state == ServerState.STOPPING:
            self._set_state(ServerState.STOPPED)
            return
        hint = server_log.explain_error(list(self._recent))
        if hint is None and exit_code != 0:
            hint = f"서버가 예기치 않게 종료됐어요 (종료 코드 {exit_code}). 아래 로그를 확인해 주세요."
        elif hint is None:
            hint = "서버가 종료됐어요."
        self.error_hint.emit(hint)
        self._set_state(ServerState.ERROR)

    def _on_proc_error(self, err) -> None:
        if err == QProcess.FailedToStart:
            self.error_hint.emit(f"{self.program_name}을(를) 실행하지 못했어요. 실행 파일 경로와 권한을 확인해 주세요.")
            self._set_state(ServerState.ERROR)

    def _force_kill(self) -> None:
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def _check_health(self) -> None:
        if self._health_pending or self.state not in (ServerState.STARTING, ServerState.RUNNING):
            return
        self._health_pending = True
        req = QNetworkRequest(QUrl(self.base_url + self.health_path))
        req.setTransferTimeout(1500)
        reply = self._net.get(req)
        reply.finished.connect(lambda r=reply: self._on_health(r))

    def _on_health(self, reply: QNetworkReply) -> None:
        self._health_pending = False
        status = reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)
        reply.deleteLater()
        if self.state not in (ServerState.STARTING, ServerState.RUNNING):
            return
        if status == 200 and self.state == ServerState.STARTING:
            if self._prepare is None:
                self._set_state(ServerState.RUNNING)
            elif not self._preparing:
                self._preparing = True
                prep = self._prepare
                threading.Thread(target=self._run_prepare, args=(prep,), daemon=True).start()
        elif self.state == ServerState.RUNNING and status is None:
            pass  # 일시적 연결 실패는 무시 (프로세스 종료는 finished로 감지)

    def _run_prepare(self, prep) -> None:
        try:
            prep(lambda line: self.log_line.emit(line, "info"))
        except Exception as e:  # noqa: BLE001
            self._prepared.emit(str(e) or "모델 준비에 실패했어요.")
        else:
            self._prepared.emit(None)

    def _on_prepared(self, err) -> None:
        self._preparing = False
        if self.state != ServerState.STARTING:
            return
        if err:
            self._fail_hint = err   # 프로세스가 끝나면 '중지'가 아니라 '오류'로 보여준다
            self.stop()
        else:
            self._prepare = None
            self._set_state(ServerState.RUNNING)

    def _sample_memory(self) -> None:
        pid = self.pid
        if not pid:
            return
        try:
            p = psutil.Process(pid)
            total = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
        except (psutil.Error, OSError):
            return
        self.memory_changed.emit(total)
