"""첫 실행 마법사: ① 사양 확인 → ② 실행 엔진 확인/설치 → ③ 추천 모델 받기 → ④ 자동 설정으로 실행 → ⑤ 채팅 테스트.

각 단계는 메인 창의 기존 기능(대시보드·런처·모델 찾기·다운로드·서버·채팅)을 그대로 이용한다.
"""

from __future__ import annotations

import html
import re
import threading
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from ..core.chat_client import ChatRequest, stream_chat
from ..core.downloader import Status
from ..core.hardware import format_bytes
from ..core.hf_client import HFModel, QuantFile, RepoFiles, recommend_quant
from ..core.installer import Installer, InstallError
from ..core.launchers import by_id
from ..core.server import ServerState
from .workers import run_in_background

if TYPE_CHECKING:
    from .main_window import MainWindow

STEPS = ["사양 확인", "실행 엔진", "추천 모델 받기", "자동 설정으로 실행", "채팅 테스트"]


class _Bridge(QObject):
    log = Signal(str)
    done = Signal(object)
    chat = Signal(object)


STARTER_MAX = 10 * 1024 ** 3
DERIVED = re.compile(r"distill|merge|roleplay|[-_]rp[-_]|uncensor|abliter|heretic|finetune|frankenmerge", re.I)
LOW_BIT = re.compile(r"(^|[-_])(I?Q1|I?Q2|TQ[12])", re.I)


def pick_starter(models: list[HFModel], files_for, system) -> Optional[tuple[HFModel, QuantFile]]:
    """추천 목록 위에서부터, 이 컴퓨터에서 쾌적하게 도는 추천 파일이 있는 첫 모델.

    첫 실행이므로 너무 큰 파일은 피한다: 10GB와 쓸 수 있는 메모리의 절반 중 작은 값까지.
    """
    cap = min(STARTER_MAX, int(system.fast_budget * 0.5))
    for m in models:
        if m.is_embedding or m.gated or DERIVED.search(m.id):
            continue
        try:
            files = files_for(m.id)
        except Exception:  # noqa: BLE001
            continue
        params = m.params_b * 1e9 if m.params_b else None
        ok = [q for q in files.quants if q.size <= cap and not LOW_BIT.search(q.quant)
              and (q.bits_per_weight(params) or 8) >= 3.5]  # 1~2비트급은 품질이 낮아 첫 모델로 부적합
        small = RepoFiles(files.repo, ok, files.mmproj, files.other_gguf)
        q = recommend_quant(small, system, m.params_b * 1e9 if m.params_b else None, n_ctx_train=m.context_length)
        if q:
            return m, q
    return None


class SetupWizard(QDialog):
    def __init__(self, win: "MainWindow"):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("라마런처 시작하기")
        self.resize(760, 540)
        self.step = 0
        self.choice: Optional[tuple[HFModel, QuantFile]] = None
        self.model_path: Optional[str] = None
        self.download_item = None
        self._bridge = _Bridge()
        self._bridge.log.connect(lambda s: self.log.appendPlainText(s))
        self._bridge.done.connect(self._on_install_done)
        self._bridge.chat.connect(self._on_chat_event)
        self._timer = QTimer(self, interval=300)
        self._timer.timeout.connect(self._poll)

        root = QHBoxLayout(self)
        self.nav = QListWidget()
        self.nav.setFixedWidth(170)
        self.nav.setEnabled(False)
        for i, s in enumerate(STEPS, 1):
            self.nav.addItem(QListWidgetItem(f"{i}. {s}"))
        root.addWidget(self.nav)
        right = QVBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("pageTitle")
        right.addWidget(self.title)
        self.body = QLabel()
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.RichText)
        self.body.setOpenExternalLinks(True)
        self.body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        right.addWidget(self.body)
        self.progress = QProgressBar()
        self.progress.hide()
        right.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.hide()
        right.addWidget(self.log, 1)
        right.addStretch(0)
        btns = QHBoxLayout()
        self.skip_all = QPushButton("나중에 할게요")
        self.skip_all.setToolTip("마법사를 닫아요. 메뉴 [도움말 → 시작 마법사]에서 다시 열 수 있어요.")
        self.skip_all.clicked.connect(self.reject)
        btns.addWidget(self.skip_all)
        btns.addStretch(1)
        self.action_btn = QPushButton()
        self.action_btn.clicked.connect(self._action)
        self.skip_btn = QPushButton("이 단계 건너뛰기")
        self.skip_btn.clicked.connect(self._next)
        self.next_btn = QPushButton("다음 ▶")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self._next)
        for b in (self.action_btn, self.skip_btn, self.next_btn):
            btns.addWidget(b)
        right.addLayout(btns)
        root.addLayout(right, 1)
        self._show_step()

    # ------------------------------------------------------------------ 단계 전환
    def _next(self) -> None:
        if self.step >= len(STEPS) - 1:
            self.accept()
            return
        self.step += 1
        self._show_step()

    def _show_step(self) -> None:
        self.nav.setCurrentRow(self.step)
        self.title.setText(f"{self.step + 1}. {STEPS[self.step]}")
        self.progress.hide()
        self.log.hide()
        self.log.clear()
        self.action_btn.show()
        self.action_btn.setEnabled(True)
        self.skip_btn.show()
        self.next_btn.setEnabled(True)
        self.next_btn.setText("다음 ▶" if self.step < len(STEPS) - 1 else "완료 🎉")
        [self._step_specs, self._step_engine, self._step_model, self._step_run, self._step_chat][self.step]()

    def _action(self) -> None:
        [lambda: None, self._install_engine, self._download_model, self._start_server, self._send_chat][self.step]()

    # ------------------------------------------------------------------ ① 사양
    def _step_specs(self) -> None:
        self.action_btn.hide()
        self.skip_btn.hide()
        sysinfo = self.win.dashboard.system
        if not sysinfo:
            self.body.setText("이 컴퓨터의 사양을 확인하는 중이에요…")
            self.next_btn.setEnabled(False)
            QTimer.singleShot(500, lambda: self.step == 0 and self._step_specs())
            return
        from ..core.recommender import summary_sentence
        gpu = sysinfo.primary_gpu
        self.body.setText(
            "<p>안녕하세요! 라마런처는 내 컴퓨터에서 AI 모델을 실행하도록 도와줘요. 다섯 단계면 첫 대화까지 할 수 있어요.</p>"
            f"<p><b>CPU</b> {html.escape(sysinfo.cpu_name)} · <b>메모리</b> {format_bytes(sysinfo.total_ram)}"
            f"{' · <b>GPU</b> ' + html.escape(gpu.name) if gpu else ''}</p>"
            f"<p style='font-size:15px'>💡 {html.escape(summary_sentence(sysinfo))}</p>")

    # ------------------------------------------------------------------ ② 엔진
    def _engine_status(self):
        return self.win.launchers.statuses.get("llamacpp")

    def _step_engine(self) -> None:
        st = self._engine_status()
        if st is None:
            self.body.setText("llama.cpp가 설치되어 있는지 확인하는 중이에요…")
            self.action_btn.hide()
            QTimer.singleShot(500, lambda: self.step == 1 and self._step_engine())
            return
        if st.installed:
            self.body.setText(f"<p>✅ 실행 엔진 <b>llama.cpp</b>가 준비되어 있어요.</p><p style='color:gray'>버전 "
                              f"{html.escape(st.version)} · {html.escape(str(st.path))}</p>"
                              "<p>Ollama·LM Studio 같은 다른 엔진은 나중에 [🚀 런처] 탭에서 설치할 수 있어요.</p>")
            self.action_btn.hide()
            return
        plans = [p for p in by_id("llamacpp").install_plans(self.win.dashboard.system) if not p.manual]
        self._plan = plans[0] if plans else None
        self.body.setText("<p>AI 모델을 실행하려면 <b>llama.cpp</b>라는 가벼운 실행 엔진이 필요해요. "
                          "버튼 하나로 설치할 수 있어요.</p>" +
                          (f"<p style='color:gray'>설치 방법: {html.escape(self._plan.label)}</p>" if self._plan else ""))
        self.action_btn.setText("⬇️ llama.cpp 설치")
        self.action_btn.setEnabled(self._plan is not None)
        self.next_btn.setEnabled(False)

    def _install_engine(self) -> None:
        if not getattr(self, "_plan", None):
            return
        self.action_btn.setEnabled(False)
        self.log.show()
        plan = self._plan
        inst = Installer(log=self._bridge.log.emit)

        def work():
            try:
                inst.run(plan)
                self._bridge.done.emit(None)
            except InstallError as e:
                self._bridge.done.emit(str(e))
            except Exception as e:  # noqa: BLE001
                self._bridge.done.emit(f"예상하지 못한 오류: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _on_install_done(self, err) -> None:
        if err:
            self.log.appendPlainText(f"❌ {err}")
            self.action_btn.setEnabled(True)
            return
        self.log.appendPlainText("✅ 설치했어요!")
        self.win.launchers.detect("llamacpp")
        self.win.server.detect_launcher()
        self.next_btn.setEnabled(True)

    # ------------------------------------------------------------------ ③ 모델
    def _step_model(self) -> None:
        existing = [m for m in self.win.models.models if not m.error and not m.is_embedding and m.fmt == "gguf"]
        self.action_btn.setText("⬇️ 받기")
        self.action_btn.setEnabled(False)
        self.next_btn.setEnabled(bool(existing) or bool(self.model_path))
        if existing:
            best = max(existing, key=lambda m: m.size if (fit := self.win.models._fit_of(m)) and fit.fit.value == "comfortable" else 0)
            self.model_path = str(best.path)
            self.body.setText(f"<p>✅ 이미 받아 둔 모델이 있어요: <b>{html.escape(best.display_name)}</b> "
                              f"({format_bytes(best.size)})</p><p>이 모델로 바로 실행해 볼게요. 새 모델을 받으려면 아래 추천 모델을 받아도 돼요.</p>")
        else:
            self.body.setText("<p>이 컴퓨터에 맞는 추천 모델을 허깅페이스에서 찾는 중이에요…</p>")
        system = self.win.dashboard.system
        hf = self.win.hf

        def work():
            models = hf.search(category="recommended", system=system, limit=30)
            return pick_starter(models, hf.list_files, system)

        run_in_background(work, self._on_starter, lambda e: self.body.setText(
            self.body.text() + f"<p style='color:#c62828'>추천 모델을 찾지 못했어요: {html.escape(str(e))}<br>"
                               "[🔍 모델 찾기] 탭에서 직접 골라 주세요.</p>"))

    def _on_starter(self, choice) -> None:
        if self.step != 2:
            return
        self.choice = choice
        if not choice:
            self.body.setText(self.body.text() + "<p>쾌적하게 돌아가는 추천 모델을 찾지 못했어요. [🔍 모델 찾기] 탭을 이용해 주세요.</p>")
            return
        m, q = choice
        self.body.setText(self.body.text() + (
            f"<p>⭐ 추천: <b>{html.escape(m.name)}</b> · {html.escape(q.quant)} · {format_bytes(q.size)}<br>"
            f"<span style='color:gray'>{html.escape(m.id)} · 다운로드 {m.downloads:,}회</span></p>"))
        self.action_btn.setEnabled(True)

    def _download_model(self) -> None:
        if not self.choice:
            return
        m, q = self.choice
        item = self.win.downloads.enqueue(m, q, None)
        if not item:
            return
        self.download_item = item
        self.action_btn.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 1000)
        self._timer.start()

    def _poll(self) -> None:
        it = self.download_item
        if not it:
            self._timer.stop()
            return
        self.progress.setValue(int(1000 * it.downloaded / max(1, it.total)))
        self.progress.setFormat(f"{format_bytes(it.downloaded)} / {format_bytes(it.total)}  {it.status.label}")
        if it.status == Status.DONE:
            self._timer.stop()
            self.model_path = str(it.main_path)
            self.body.setText(self.body.text() + "<p>✅ 다 받았어요!</p>")
            self.next_btn.setEnabled(True)
        elif it.status in (Status.ERROR, Status.CANCELED):
            self._timer.stop()
            self.body.setText(self.body.text() + f"<p style='color:#c62828'>❌ {html.escape(it.error or it.status.label)}</p>")
            self.action_btn.setEnabled(True)

    # ------------------------------------------------------------------ ④ 실행
    def _step_run(self) -> None:
        self.action_btn.setText("▶ 자동 설정으로 실행")
        if not self.model_path:
            self.body.setText("모델이 없어요. 이전 단계에서 모델을 받거나 [📁 내 모델]에서 추가해 주세요.")
            self.action_btn.setEnabled(False)
            return
        if self.win.server.controller.state == ServerState.RUNNING:
            self.body.setText("✅ 서버가 이미 실행 중이에요.")
            self.action_btn.hide()
            return
        self.body.setText("<p>[🪄 자동 설정]으로 이 컴퓨터에 맞는 옵션(문맥 길이, GPU 사용, 스레드 수 등)을 정하고 "
                          "모델을 실행해요. 큰 모델은 메모리에 올리는 데 1분 가까이 걸릴 수 있어요.</p>")
        self.next_btn.setEnabled(False)

    def _start_server(self) -> None:
        srv = self.win.server
        if srv.launcher.id != "llamacpp" and not srv.controller.is_active:
            srv.set_engine("llamacpp")
        srv.select_model_path(self.model_path)
        srv.auto_configure()
        srv.start_server()
        self.action_btn.setEnabled(False)
        self.body.setText(self.body.text() + "<p>⏳ 모델을 불러오는 중…</p>")
        srv.controller.state_changed.connect(self._on_server_state)

    def _on_server_state(self, state) -> None:
        if self.step != 3:
            return
        if state == ServerState.RUNNING:
            self.body.setText(self.body.text() + f"<p>✅ 실행됐어요! 주소: <code>{self.win.server.controller.base_url}</code></p>")
            self.next_btn.setEnabled(True)
        elif state == ServerState.ERROR:
            self.body.setText(self.body.text() + "<p style='color:#c62828'>❌ 실행하지 못했어요. [▶️ 서버] 탭의 안내와 기록을 확인해 주세요.</p>")
            self.action_btn.setEnabled(True)

    # ------------------------------------------------------------------ ⑤ 채팅
    def _step_chat(self) -> None:
        self.skip_btn.hide()
        running = self.win.server.controller.state == ServerState.RUNNING
        self.action_btn.setText("💬 인사해 보기")
        self.action_btn.setEnabled(running)
        self.body.setText("<p>마지막으로 AI에게 말을 걸어 볼게요. 답이 글자 단위로 나오면 성공이에요! 🎉</p>" if running else
                          "<p>서버가 실행 중이 아니에요. 나중에 [💬 채팅 테스트] 탭에서 대화해 보세요.</p>")
        self.log.show()

    def _send_chat(self) -> None:
        srv = self.win.server
        info = srv.running
        req = ChatRequest(base_url=srv.controller.base_url, model=info.get("alias") or "local-model",
                          api_key=info.get("api_key") or "",
                          messages=[{"role": "user", "content": "안녕! 너는 누구인지 한두 문장으로 소개해 줘."}])
        self.action_btn.setEnabled(False)
        self.log.setPlainText("🙋 안녕! 너는 누구인지 한두 문장으로 소개해 줘.\n🤖 ")
        threading.Thread(target=stream_chat, args=(req, self._bridge.chat.emit), daemon=True).start()

    def _on_chat_event(self, ev) -> None:
        self.log.moveCursor(QTextCursor.End)
        if ev.kind == "content":
            self.log.insertPlainText(ev.text)
        elif ev.kind == "reasoning" and "💭" not in self.log.toPlainText():
            self.log.insertPlainText("💭(생각하는 중…) ")
        elif ev.kind == "done":
            tps = ev.stats.get("tps")
            self.log.appendPlainText(f"\n✅ 성공! {f'{tps:.1f} 토큰/초' if tps else ''}\n"
                                     "이제 [💬 채팅 테스트]에서 대화하거나 [🔌 연동] 탭에서 코딩 도구에 연결해 보세요.")
            self.action_btn.setEnabled(True)
        elif ev.kind == "error":
            self.log.appendPlainText(f"\n❌ {ev.text}")
            self.action_btn.setEnabled(True)

    def done(self, r: int) -> None:  # noqa: D401 — QDialog.done 재정의
        self._timer.stop()
        self.win.settings.first_run_done = True
        self.win.save_settings()
        super().done(r)
