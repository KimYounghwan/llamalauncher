"""서버 탭: 모델 선택 → 옵션(자동 설정) → llama-server 실행/모니터링."""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from ..core import gguf
from ..core.hardware import SystemInfo, format_bytes
from ..core.launchers.base import LauncherStatus
from ..core.launchers.llamacpp import LlamaCppLauncher
from ..core.netutil import find_free_port, generate_api_key, is_port_free, lan_ip
from ..core.recommender import (
    CTX_STEPS, Fit, ModelSpec, Preset, ServerOptions, assess_fit, build_server_args, format_command, recommend,
)
from ..core.server import ServerController, ServerState, clear_pid_file, find_orphan_server
from ..core.settings import Settings, logs_dir
from .workers import run_in_background

STATE_COLORS = {
    ServerState.STOPPED: "#9aa0a6", ServerState.STARTING: "#f4b400", ServerState.RUNNING: "#0f9d58",
    ServerState.STOPPING: "#f4b400", ServerState.ERROR: "#db4437",
}
LOG_COLORS = {"error": "#e06c75", "warn": "#d19a66", "ok": "#4caf50", "info": None}

TIPS = {
    "ctx": "AI가 한 번에 기억할 수 있는 대화 길이(토큰 수)예요. 클수록 긴 대화·문서를 다룰 수 있지만 메모리를 더 써요. "
           "한글은 대략 1글자 ≈ 1~2토큰이에요.",
    "ngl": "모델의 몇 개 층을 GPU에 올릴지 정해요. 많을수록 빨라요. ‘999’는 전부 올린다는 뜻이에요.",
    "threads": "계산에 쓸 CPU 코어 수예요. 보통 성능 코어 수가 가장 좋아요.",
    "batch": "입력(프롬프트)을 한 번에 처리하는 양이에요. 크면 긴 입력을 빨리 읽지만 메모리를 조금 더 써요.",
    "ubatch": "실제로 GPU에 한 번에 넘기는 양이에요. 보통 기본값(512)을 그대로 쓰면 돼요.",
    "fa": "Flash Attention: 메모리를 아끼고 속도를 높이는 계산 방식이에요. Apple/NVIDIA GPU에서는 켜는 게 좋아요.",
    "cache": "대화 기억 공간(KV 캐시)의 정밀도예요. q8_0은 메모리를 절반 가까이 아끼면서 품질 차이가 거의 없어요.",
    "mlock": "모델을 메모리에 고정해서 디스크로 밀려나지 않게 해요. 메모리가 넉넉할 때만 켜세요.",
    "no_mmap": "모델 파일을 전부 메모리로 읽어 들여요. 보통은 끈 상태(mmap 사용)가 빠르게 시작돼요.",
    "host": "‘이 컴퓨터만’은 내 컴퓨터에서만 접속할 수 있어요. ‘같은 네트워크에 공유’는 같은 와이파이의 다른 기기도 접속할 수 있어요.",
    "port": "서버 주소의 번호예요. 다른 프로그램과 겹치면 바꿔 주세요.",
    "api_key": "접속할 때 필요한 비밀번호예요. 네트워크에 공유할 때는 꼭 설정하세요.",
    "parallel": "동시에 처리할 요청 수예요. 늘리면 문맥 길이가 요청 수만큼 나뉘어요. 혼자 쓰면 1이 좋아요.",
    "alias": "외부 프로그램에서 부를 모델 이름이에요. 모델을 바꿔도 같은 이름을 쓰면 연동 설정을 다시 안 해도 돼요.",
    "temp": "답변의 창의성이에요. 낮으면(0.2) 정확하고 일관되게, 높으면(1.0) 다양하고 창의적으로 답해요.",
    "top_p": "확률이 높은 단어들만 골라 쓰는 범위예요. 보통 0.9~0.95.",
    "top_k": "후보 단어 개수 제한이에요. 보통 20~40.",
    "min_p": "너무 확률이 낮은 단어를 걸러요. 보통 0.05.",
    "repeat": "같은 말을 반복하지 않게 해요. 1.0은 끔, 1.1 정도가 무난해요.",
    "jinja": "모델에 들어 있는 채팅 형식(템플릿)을 그대로 사용해요. 도구 호출(코딩 에이전트 연동)에 필요해요.",
    "extra": "llama-server에 직접 넘길 추가 인자예요. 잘 모르면 비워 두세요. 예: --seed 42",
    "mmproj": "비전 모델이 이미지를 이해하는 데 필요한 파일(mmproj)이에요. 모델과 같은 폴더에 있으면 자동으로 찾아요.",
}


def _tip_label(text: str, tip: str) -> QLabel:
    lbl = QLabel(f"{text} <span style='color:gray'>ⓘ</span>")
    lbl.setToolTip(tip)
    return lbl


class ServerTab(QWidget):
    server_state_changed = Signal(object)

    def __init__(self, settings: Settings, save_settings):
        super().__init__()
        self.settings = settings
        self.save_settings = save_settings
        self.system: Optional[SystemInfo] = None
        self.launcher = LlamaCppLauncher()
        self.launcher_status: Optional[LauncherStatus] = None
        self.model_info: Optional[gguf.GGUFInfo] = None
        self.controller = ServerController(self)
        self._loading_ui = False
        self._log_file = None
        self._uptime_timer = QTimer(self)
        self._uptime_timer.timeout.connect(self._update_uptime)

        self.controller.state_changed.connect(self._on_state)
        self.controller.log_line.connect(self._append_log)
        self.controller.speed_changed.connect(self._on_speed)
        self.controller.memory_changed.connect(lambda b: self.mem_lbl.setText(f"메모리 사용: {format_bytes(b)}"))
        self.controller.error_hint.connect(self._on_error_hint)

        root = QVBoxLayout(self)
        root.addWidget(self._build_header())
        split = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(self._build_options())
        top.addWidget(self._build_control())
        top.setSizes([560, 480])
        split.addWidget(top)
        split.addWidget(self._build_log())
        split.setSizes([520, 240])
        root.addWidget(split, 1)

        self._load_options(ServerOptions.from_dict(settings.server_options))
        self._refresh_model_combo()
        self.detect_launcher()
        if settings.last_model and Path(settings.last_model).exists():
            self._select_model(settings.last_model, auto_configure=not settings.server_options)
        self._on_state(ServerState.STOPPED)
        QTimer.singleShot(500, self._check_orphan)

    def _check_orphan(self) -> None:
        proc = find_orphan_server()
        if proc is None:
            return
        answer = QMessageBox.question(
            self, "이전 서버가 남아 있어요",
            f"지난번에 프로그램이 비정상 종료되면서 llama-server(PID {proc.pid})가 계속 실행 중이에요.\n"
            "메모리를 차지하고 있으니 종료할까요?")
        if answer == QMessageBox.Yes:
            try:
                proc.terminate()
                proc.wait(5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            clear_pid_file()

    # ------------------------------------------------------------------ UI 구성
    def _build_header(self) -> QWidget:
        box = QFrame()
        box.setObjectName("card")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("<b>모델</b>"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(420)
        self.model_combo.setPlaceholderText("모델 파일을 선택해 주세요 (.gguf)")
        self.model_combo.activated.connect(lambda i: self._select_model(self.model_combo.itemData(i)))
        row.addWidget(self.model_combo, 1)
        pick = QPushButton("📂 파일 선택…")
        pick.clicked.connect(self._pick_model)
        row.addWidget(pick)
        lay.addLayout(row)

        self.model_lbl = QLabel("선택한 모델이 없어요.")
        self.model_lbl.setObjectName("cardDetail")
        self.model_lbl.setWordWrap(True)
        self.model_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(self.model_lbl)

        row2 = QHBoxLayout()
        self.launcher_lbl = QLabel("llama.cpp 확인 중…")
        self.launcher_lbl.setObjectName("cardDetail")
        self.launcher_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row2.addWidget(self.launcher_lbl, 1)
        path_btn = QPushButton("실행 파일 지정…")
        path_btn.clicked.connect(self._pick_launcher)
        row2.addWidget(path_btn)
        lay.addLayout(row2)
        return box

    def _build_options(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        lay = QVBoxLayout(body)

        # 프리셋 + 자동 설정
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for p in Preset:
            self.preset_combo.addItem(p.label, p.value)
        self.preset_combo.setToolTip("빠르게: 짧은 대화, 빠른 응답 / 균형: 일반적인 사용 / 긴 대화용: 긴 문서·코딩")
        idx = self.preset_combo.findData(self.settings.server_preset)
        self.preset_combo.setCurrentIndex(max(0, idx))
        self.auto_btn = QPushButton("🪄 이 컴퓨터에 맞게 자동 설정")
        self.auto_btn.setObjectName("primary")
        self.auto_btn.clicked.connect(lambda: self.auto_configure(show_notes=True))
        preset_row.addWidget(QLabel("용도"))
        preset_row.addWidget(self.preset_combo)
        preset_row.addWidget(self.auto_btn, 1)
        lay.addLayout(preset_row)

        user_row = QHBoxLayout()
        self.user_preset_combo = QComboBox()
        self.user_preset_combo.setPlaceholderText("내 프리셋")
        self.user_preset_combo.activated.connect(self._apply_user_preset)
        save_p = QPushButton("현재 설정 저장")
        save_p.clicked.connect(self._save_user_preset)
        del_p = QPushButton("삭제")
        del_p.clicked.connect(self._delete_user_preset)
        user_row.addWidget(self.user_preset_combo, 1)
        user_row.addWidget(save_p)
        user_row.addWidget(del_p)
        lay.addLayout(user_row)
        self._refresh_user_presets()

        self.notes_lbl = QLabel("")
        self.notes_lbl.setObjectName("hint")
        self.notes_lbl.setWordWrap(True)
        lay.addWidget(self.notes_lbl)

        # 기본
        g = QGroupBox("기본")
        f = QFormLayout(g)
        self.ctx = QComboBox()
        self.ctx.setEditable(True)
        for c in CTX_STEPS + [262144]:
            self.ctx.addItem(f"{c:,}", c)
        f.addRow(_tip_label("문맥 길이 (Context)", TIPS["ctx"]), self.ctx)
        self.ngl = QSpinBox()
        self.ngl.setRange(0, 999)
        self.ngl.setSpecialValueText("0 (CPU 전용)")
        f.addRow(_tip_label("GPU 레이어 수", TIPS["ngl"]), self.ngl)
        self.threads = QSpinBox()
        self.threads.setRange(1, 256)
        f.addRow(_tip_label("CPU 스레드", TIPS["threads"]), self.threads)
        lay.addWidget(g)

        # 성능
        g = QGroupBox("성능")
        f = QFormLayout(g)
        self.batch = QSpinBox()
        self.batch.setRange(32, 16384)
        self.batch.setSingleStep(256)
        f.addRow(_tip_label("배치 크기", TIPS["batch"]), self.batch)
        self.ubatch = QSpinBox()
        self.ubatch.setRange(32, 8192)
        self.ubatch.setSingleStep(128)
        f.addRow(_tip_label("마이크로 배치", TIPS["ubatch"]), self.ubatch)
        self.fa = QComboBox()
        for label, v in [("자동", "auto"), ("켜기", "on"), ("끄기", "off")]:
            self.fa.addItem(label, v)
        f.addRow(_tip_label("Flash Attention", TIPS["fa"]), self.fa)
        self.cache_k, self.cache_v = QComboBox(), QComboBox()
        for cb in (self.cache_k, self.cache_v):
            for label, v in [("f16 (기본, 최고 품질)", "f16"), ("q8_0 (메모리 절약)", "q8_0"), ("q4_0 (최대 절약)", "q4_0")]:
                cb.addItem(label, v)
        f.addRow(_tip_label("KV 캐시 (K)", TIPS["cache"]), self.cache_k)
        f.addRow(_tip_label("KV 캐시 (V)", TIPS["cache"]), self.cache_v)
        self.mlock = QCheckBox("메모리에 고정 (mlock)")
        self.mlock.setToolTip(TIPS["mlock"])
        self.no_mmap = QCheckBox("파일 매핑 사용 안 함 (no-mmap)")
        self.no_mmap.setToolTip(TIPS["no_mmap"])
        f.addRow(self.mlock)
        f.addRow(self.no_mmap)
        lay.addWidget(g)

        # 네트워크
        g = QGroupBox("네트워크 / 접속")
        f = QFormLayout(g)
        self.host = QComboBox()
        self.host.addItem("이 컴퓨터만 (127.0.0.1)", "127.0.0.1")
        self.host.addItem("같은 네트워크에 공유 (0.0.0.0)", "0.0.0.0")
        f.addRow(_tip_label("접속 허용", TIPS["host"]), self.host)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        f.addRow(_tip_label("포트", TIPS["port"]), self.port)
        key_row = QHBoxLayout()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        self.api_key.setPlaceholderText("비워 두면 비밀번호 없이 접속")
        gen = QPushButton("생성")
        gen.clicked.connect(lambda: self.api_key.setText(generate_api_key()))
        show = QPushButton("보기")
        show.setCheckable(True)
        show.toggled.connect(lambda on: self.api_key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        key_row.addWidget(self.api_key, 1)
        key_row.addWidget(gen)
        key_row.addWidget(show)
        f.addRow(_tip_label("API 키", TIPS["api_key"]), key_row)
        self.parallel = QSpinBox()
        self.parallel.setRange(1, 64)
        f.addRow(_tip_label("동시 요청 수", TIPS["parallel"]), self.parallel)
        self.alias = QLineEdit()
        self.alias.setPlaceholderText("예: local-model")
        f.addRow(_tip_label("모델 별칭 (alias)", TIPS["alias"]), self.alias)
        self.net_warn = QLabel("")
        self.net_warn.setWordWrap(True)
        self.net_warn.setStyleSheet("color:#d19a66")
        f.addRow(self.net_warn)
        lay.addWidget(g)

        # 샘플링
        g = QGroupBox("답변 스타일 (샘플링 기본값)")
        f = QFormLayout(g)

        def dspin(lo, hi, step, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(dec)
            return s

        self.temp = dspin(0, 2, 0.05)
        self.top_p = dspin(0, 1, 0.01)
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 200)
        self.min_p = dspin(0, 1, 0.01)
        self.repeat = dspin(0.8, 2, 0.01)
        f.addRow(_tip_label("온도 (temperature)", TIPS["temp"]), self.temp)
        f.addRow(_tip_label("top-p", TIPS["top_p"]), self.top_p)
        f.addRow(_tip_label("top-k", TIPS["top_k"]), self.top_k)
        f.addRow(_tip_label("min-p", TIPS["min_p"]), self.min_p)
        f.addRow(_tip_label("반복 억제", TIPS["repeat"]), self.repeat)
        lay.addWidget(g)

        # 고급
        g = QGroupBox("고급")
        f = QFormLayout(g)
        self.jinja = QCheckBox("모델 내장 채팅 템플릿 사용 (--jinja)")
        self.jinja.setToolTip(TIPS["jinja"])
        f.addRow(self.jinja)
        mm_row = QHBoxLayout()
        self.mmproj = QLineEdit()
        self.mmproj.setPlaceholderText("비전 모델이 아니면 비워 두세요")
        mm_pick = QPushButton("선택…")
        mm_pick.clicked.connect(self._pick_mmproj)
        mm_row.addWidget(self.mmproj, 1)
        mm_row.addWidget(mm_pick)
        f.addRow(_tip_label("비전 파일 (mmproj)", TIPS["mmproj"]), mm_row)
        self.extra = QLineEdit()
        self.extra.setPlaceholderText("예: --seed 42")
        f.addRow(_tip_label("추가 인자", TIPS["extra"]), self.extra)
        lay.addWidget(g)
        lay.addStretch()

        for w in (self.ngl, self.threads, self.batch, self.ubatch, self.port, self.parallel, self.top_k):
            w.valueChanged.connect(self._on_option_changed)
        for w in (self.temp, self.top_p, self.min_p, self.repeat):
            w.valueChanged.connect(self._on_option_changed)
        for w in (self.fa, self.cache_k, self.cache_v, self.host):
            w.currentIndexChanged.connect(self._on_option_changed)
        self.ctx.currentTextChanged.connect(self._on_option_changed)
        for w in (self.mlock, self.no_mmap, self.jinja):
            w.toggled.connect(self._on_option_changed)
        for w in (self.api_key, self.alias, self.extra, self.mmproj):
            w.textChanged.connect(self._on_option_changed)
        return scroll

    def _build_control(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        status_row = QHBoxLayout()
        self.light = QLabel()
        self.light.setFixedSize(16, 16)
        self.state_lbl = QLabel("중지됨")
        self.state_lbl.setObjectName("sectionTitle")
        status_row.addWidget(self.light)
        status_row.addWidget(self.state_lbl)
        status_row.addStretch()
        lay.addLayout(status_row)

        self.hint_lbl = QLabel("")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setObjectName("errorBanner")
        self.hint_lbl.hide()
        lay.addWidget(self.hint_lbl)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶  서버 시작")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn = QPushButton("■  중지")
        self.stop_btn.clicked.connect(self.controller.stop)
        self.restart_btn = QPushButton("⟳  재시작")
        self.restart_btn.clicked.connect(self.restart_server)
        for b in (self.start_btn, self.stop_btn, self.restart_btn):
            b.setMinimumHeight(36)
            btns.addWidget(b)
        lay.addLayout(btns)

        conn = QGroupBox("접속 정보")
        cf = QVBoxLayout(conn)
        self.url_lbl = QLabel("—")
        self.url_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.url_lbl.setOpenExternalLinks(True)
        self.url_lbl.setTextFormat(Qt.RichText)
        self.url_lbl.setWordWrap(True)
        cf.addWidget(self.url_lbl)
        url_btns = QHBoxLayout()
        self.open_btn = QPushButton("🌐 브라우저에서 웹 UI 열기")
        self.open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.controller.base_url)))
        self.copy_url_btn = QPushButton("API 주소 복사")
        self.copy_url_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.controller.base_url + "/v1"))
        url_btns.addWidget(self.open_btn)
        url_btns.addWidget(self.copy_url_btn)
        cf.addLayout(url_btns)
        lay.addWidget(conn)

        stats = QGroupBox("상태")
        sf = QVBoxLayout(stats)
        self.gen_lbl = QLabel("답변 생성 속도: —")
        self.prompt_lbl = QLabel("입력 처리 속도: —")
        self.mem_lbl = QLabel("메모리 사용: —")
        self.uptime_lbl = QLabel("실행 시간: —")
        for l in (self.gen_lbl, self.prompt_lbl, self.mem_lbl, self.uptime_lbl):
            sf.addWidget(l)
        lay.addWidget(stats)

        cmd = QGroupBox("실행될 명령어")
        cmf = QVBoxLayout(cmd)
        self.cmd_view = QPlainTextEdit()
        self.cmd_view.setReadOnly(True)
        self.cmd_view.setFont(QFont("Menlo, Consolas, monospace", 11))
        self.cmd_view.setMaximumHeight(110)
        copy_cmd = QPushButton("명령어 복사")
        copy_cmd.clicked.connect(lambda: QGuiApplication.clipboard().setText(self._command_text(mask=False)))
        cmf.addWidget(self.cmd_view)
        cmf.addWidget(copy_cmd, 0, Qt.AlignRight)
        lay.addWidget(cmd)
        lay.addStretch()
        return w

    def _build_log(self) -> QWidget:
        box = QGroupBox("서버 로그")
        lay = QVBoxLayout(box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Menlo, Consolas, monospace", 11))
        lay.addWidget(self.log)
        row = QHBoxLayout()
        row.addStretch()
        clear = QPushButton("로그 지우기")
        clear.clicked.connect(self.log.clear)
        open_dir = QPushButton("로그 폴더 열기")
        open_dir.clicked.connect(self._open_logs_dir)
        row.addWidget(clear)
        row.addWidget(open_dir)
        lay.addLayout(row)
        return box

    # ------------------------------------------------------------------ 옵션 ↔ UI
    def current_options(self) -> ServerOptions:
        try:
            ctx = int(self.ctx.currentText().replace(",", "").strip())
        except ValueError:
            ctx = 4096
        return ServerOptions(
            ctx_size=max(256, ctx), n_gpu_layers=self.ngl.value(), threads=self.threads.value(),
            batch_size=self.batch.value(), ubatch_size=self.ubatch.value(), flash_attn=self.fa.currentData(),
            cache_type_k=self.cache_k.currentData(), cache_type_v=self.cache_v.currentData(),
            mlock=self.mlock.isChecked(), no_mmap=self.no_mmap.isChecked(), parallel=self.parallel.value(),
            host=self.host.currentData(), port=self.port.value(), temperature=self.temp.value(),
            top_p=self.top_p.value(), top_k=self.top_k.value(), min_p=self.min_p.value(),
            repeat_penalty=self.repeat.value(), jinja=self.jinja.isChecked(), alias=self.alias.text().strip(),
            api_key=self.api_key.text().strip(), extra_args=self.extra.text(), mmproj=self.mmproj.text().strip(),
        )

    def _load_options(self, o: ServerOptions) -> None:
        self._loading_ui = True
        try:
            self.ctx.setCurrentText(f"{o.ctx_size:,}")
            self.ngl.setValue(o.n_gpu_layers)
            self.threads.setValue(o.threads)
            self.batch.setValue(o.batch_size)
            self.ubatch.setValue(o.ubatch_size)
            self.fa.setCurrentIndex(max(0, self.fa.findData(o.flash_attn)))
            self.cache_k.setCurrentIndex(max(0, self.cache_k.findData(o.cache_type_k)))
            self.cache_v.setCurrentIndex(max(0, self.cache_v.findData(o.cache_type_v)))
            self.mlock.setChecked(o.mlock)
            self.no_mmap.setChecked(o.no_mmap)
            self.host.setCurrentIndex(max(0, self.host.findData(o.host)))
            self.port.setValue(o.port)
            self.api_key.setText(o.api_key)
            self.parallel.setValue(o.parallel)
            self.alias.setText(o.alias)
            self.temp.setValue(o.temperature)
            self.top_p.setValue(o.top_p)
            self.top_k.setValue(o.top_k)
            self.min_p.setValue(o.min_p)
            self.repeat.setValue(o.repeat_penalty)
            self.jinja.setChecked(o.jinja)
            self.extra.setText(o.extra_args)
            self.mmproj.setText(o.mmproj)
        finally:
            self._loading_ui = False
        self._on_option_changed()

    def _on_option_changed(self, *_):
        if self._loading_ui:
            return
        o = self.current_options()
        self.cmd_view.setPlainText(self._command_text(mask=True))
        warn = []
        if o.host == "0.0.0.0":
            ip = lan_ip()
            warn.append(f"같은 네트워크의 기기가 http://{ip or '내-IP'}:{o.port} 로 접속할 수 있어요.")
            if not o.api_key:
                warn.append("⚠️ API 키가 없으면 누구나 접속할 수 있어요. [생성]을 눌러 키를 설정하세요.")
        if o.parallel > 1:
            warn.append(f"동시 요청 {o.parallel}개 → 요청 하나당 문맥 길이는 약 {o.ctx_size // o.parallel:,} 토큰이에요.")
        self.net_warn.setText("\n".join(warn))
        self._update_fit_label()

    def _command_text(self, mask: bool) -> str:
        exe = str(self.launcher_status.path) if self.launcher_status and self.launcher_status.path else "llama-server"
        model = str(self.model_info.path) if self.model_info else "<모델 파일>"
        caps = self.launcher_status.capabilities if self.launcher_status and self.launcher_status.capabilities else None
        o = self.current_options()
        args = build_server_args(o, model, caps=caps) if caps else build_server_args(o, model)
        return format_command([exe, *args], api_key=o.api_key if mask else "")

    # ------------------------------------------------------------------ 시스템/런처/모델
    def set_system(self, info: SystemInfo) -> None:
        first = self.system is None
        self.system = info
        self.threads.setMaximum(max(1, info.logical_cores))
        self.auto_btn.setEnabled(True)
        if first and self.model_info and not self.settings.server_options:
            self.auto_configure()
        self._update_fit_label()

    def detect_launcher(self) -> None:
        self.launcher_lbl.setText("llama.cpp 확인 중…")
        custom = self.settings.launcher_paths.get("llamacpp")
        run_in_background(lambda: self.launcher.detect(custom), self._on_launcher)

    def _on_launcher(self, st: LauncherStatus) -> None:
        self.launcher_status = st
        if st.installed:
            self.launcher_lbl.setText(f"✅ llama.cpp {st.version} · {st.path}")
        else:
            self.launcher_lbl.setText(f"❌ {st.message}")
        self._on_option_changed()
        self._on_state(self.controller.state)

    def _pick_launcher(self) -> None:
        flt = "llama-server (llama-server.exe)" if QGuiApplication.platformName() == "windows" else ""
        path, _ = QFileDialog.getOpenFileName(self, "llama-server 실행 파일 선택", "", flt)
        if path:
            self.settings.launcher_paths["llamacpp"] = path
            self.save_settings()
            self.detect_launcher()

    def _pick_model(self) -> None:
        start = self.settings.last_model or self.settings.models_dir
        path, _ = QFileDialog.getOpenFileName(self, "모델 파일 선택", str(Path(start).parent if Path(start).is_file()
                                                                         else start), "GGUF 모델 (*.gguf)")
        if path:
            self._select_model(path)

    def _pick_mmproj(self) -> None:
        start = str(self.model_info.path.parent) if self.model_info else self.settings.models_dir
        path, _ = QFileDialog.getOpenFileName(self, "비전 파일(mmproj) 선택", start, "GGUF (*.gguf)")
        if path:
            self.mmproj.setText(path)

    def select_model_path(self, path: str) -> None:
        """다른 탭(다운로드 등)에서 모델을 넘겨받을 때 사용."""
        self._select_model(path)

    def _refresh_model_combo(self) -> None:
        self.model_combo.clear()
        for p in self.settings.recent_models:
            if Path(p).exists():
                self.model_combo.addItem(Path(p).name, p)
                self.model_combo.setItemData(self.model_combo.count() - 1, p, Qt.ToolTipRole)

    def _select_model(self, path: Optional[str], auto_configure: bool = True) -> None:
        if not path:
            return
        try:
            info = gguf.inspect(path)
        except (OSError, gguf.GGUFError) as e:
            QMessageBox.warning(self, "모델을 열 수 없어요", f"{path}\n\n{e}")
            return
        if info.missing_parts:
            QMessageBox.warning(self, "분할 파일 누락", "이 모델은 여러 조각으로 나뉘어 있는데 일부가 없어요:\n" +
                                "\n".join(p.name for p in info.missing_parts))
        self.model_info = info
        mm = gguf.find_mmproj(info.path)
        self.mmproj.setText(str(mm) if mm else "")
        self.settings.add_recent_model(str(info.path))
        self.save_settings()
        self._refresh_model_combo()
        self.model_combo.setCurrentIndex(self.model_combo.findData(str(info.path)))
        if auto_configure and self.system:
            self.auto_configure()
        self._on_option_changed()
        self._on_state(self.controller.state)

    def _update_fit_label(self) -> None:
        i = self.model_info
        if not i:
            self.model_lbl.setText("선택한 모델이 없어요. [📂 파일 선택…]으로 .gguf 파일을 골라 주세요.")
            return
        parts = [f"<b>{html.escape(i.display_name)}</b>"]
        if i.size_label:
            parts.append(i.size_label)
        if i.architecture:
            parts.append(html.escape(i.architecture))
        parts.append(format_bytes(i.file_size))
        if i.n_ctx_train:
            parts.append(f"최대 문맥 {i.n_ctx_train:,}")
        if i.supports_tools:
            parts.append("🛠 도구 호출 지원")
        text = " · ".join(parts)
        if self.system:
            o = self.current_options()
            spec = ModelSpec.from_gguf(i)
            fit = assess_fit(self.system, spec, o.ctx_size)
            need = spec.memory_needed(o.ctx_size, o.cache_type_k)
            text += f"<br>{fit.fit.badge} — {html.escape(fit.reason)} (현재 설정 예상 메모리 {format_bytes(need)})"
        self.model_lbl.setText(text)

    def auto_configure(self, show_notes: bool = False) -> None:
        if not self.system:
            self.notes_lbl.setText("사양을 확인하는 중이에요. 잠시 후 다시 눌러 주세요.")
            return
        if not self.model_info:
            self.notes_lbl.setText("먼저 모델을 선택해 주세요.")
            return
        preset = Preset(self.preset_combo.currentData())
        self.settings.server_preset = preset.value
        rec = recommend(self.system, ModelSpec.from_gguf(self.model_info), preset)
        # 네트워크/답변 스타일 등 사용자가 정한 값은 유지하고, 하드웨어 관련 값만 바꾼다
        merged = self.current_options()
        for name in ServerOptions.HARDWARE_FIELDS:
            setattr(merged, name, getattr(rec.options, name))
        if not merged.alias:
            merged.alias = "local-model"
        self._load_options(merged)
        self.notes_lbl.setText(f"🪄 ‘{preset.label}’ 기준으로 설정했어요.\n• " + "\n• ".join(rec.notes))

    # ------------------------------------------------------------------ 사용자 프리셋
    def _refresh_user_presets(self) -> None:
        self.user_preset_combo.clear()
        for name in sorted(self.settings.user_presets):
            self.user_preset_combo.addItem(name)
        self.user_preset_combo.setCurrentIndex(-1)

    def _save_user_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "프리셋 저장", "프리셋 이름:")
        if ok and name.strip():
            data = self.current_options().to_dict()
            data.pop("api_key", None)  # 비밀번호는 프리셋에 저장하지 않음
            self.settings.user_presets[name.strip()] = data
            self.save_settings()
            self._refresh_user_presets()

    def _apply_user_preset(self, idx: int) -> None:
        name = self.user_preset_combo.itemText(idx)
        data = dict(self.settings.user_presets.get(name, {}))
        data["api_key"] = self.api_key.text()
        self._load_options(ServerOptions.from_dict(data))
        self.notes_lbl.setText(f"‘{name}’ 프리셋을 불러왔어요.")

    def _delete_user_preset(self) -> None:
        name = self.user_preset_combo.currentText()
        if name and name in self.settings.user_presets:
            del self.settings.user_presets[name]
            self.save_settings()
            self._refresh_user_presets()

    # ------------------------------------------------------------------ 실행
    def start_server(self) -> None:
        st = self.launcher_status
        if not st or not st.installed or not st.path:
            QMessageBox.warning(self, "llama.cpp 없음", "llama-server를 찾지 못했어요. [실행 파일 지정…]으로 위치를 알려 주세요.")
            return
        if not self.model_info or not self.model_info.path.exists():
            QMessageBox.information(self, "모델 선택", "먼저 실행할 모델 파일을 선택해 주세요.")
            return
        o = self.current_options()
        if not is_port_free(o.port, o.host):
            free = find_free_port(o.port + 1, o.host)
            if free and QMessageBox.question(
                    self, "포트 사용 중",
                    f"포트 {o.port}은(는) 이미 다른 프로그램이 쓰고 있어요.\n빈 포트 {free}(으)로 바꿔서 시작할까요?"
            ) == QMessageBox.Yes:
                self.port.setValue(free)
                o = self.current_options()
            else:
                return
        if self.system:
            spec = ModelSpec.from_gguf(self.model_info)
            fit = assess_fit(self.system, spec, o.ctx_size)
            if fit.fit == Fit.INSUFFICIENT and QMessageBox.warning(
                    self, "메모리 부족 예상", fit.reason + "\n\n그래도 시작할까요?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return

        self.settings.server_options = {k: v for k, v in o.to_dict().items() if k not in ("api_key", "mmproj")}
        self.save_settings()
        args = build_server_args(o, str(self.model_info.path), caps=st.capabilities)
        self.hint_lbl.hide()
        self.gen_lbl.setText("답변 생성 속도: —")
        self.prompt_lbl.setText("입력 처리 속도: —")
        self._open_log_file()
        self._append_log("$ " + format_command([str(st.path), *args], api_key=o.api_key), "info")
        self.controller.start(str(st.path), args, o.host, o.port)

    def restart_server(self) -> None:
        if not self.controller.is_active:
            self.start_server()
            return
        self.controller.state_changed.connect(self._restart_when_stopped)
        self.controller.stop()

    def _restart_when_stopped(self, state) -> None:
        if state in (ServerState.STOPPED, ServerState.ERROR):
            self.controller.state_changed.disconnect(self._restart_when_stopped)
            QTimer.singleShot(300, self.start_server)

    def shutdown(self) -> None:
        self.controller.stop_blocking()
        self._close_log_file()

    # ------------------------------------------------------------------ 상태 표시
    def _on_state(self, state: ServerState) -> None:
        color = STATE_COLORS[state]
        self.light.setStyleSheet(f"background:{color}; border-radius:8px;")
        self.state_lbl.setText(state.label)
        active = self.controller.is_active
        ready = bool(self.launcher_status and self.launcher_status.installed and self.model_info)
        self.start_btn.setEnabled(not active and ready)
        self.stop_btn.setEnabled(state in (ServerState.STARTING, ServerState.RUNNING))
        self.restart_btn.setEnabled(state in (ServerState.RUNNING, ServerState.ERROR) and ready)
        running = state == ServerState.RUNNING
        self.open_btn.setEnabled(running)
        self.copy_url_btn.setEnabled(running)
        if running:
            url = self.controller.base_url
            lines = [f"웹 UI: <a href='{url}'>{url}</a>",
                     f"OpenAI 호환 API: <code>{url}/v1</code> (모델 이름: <code>{html.escape(self.alias.text() or '아무거나')}</code>)"]
            if self.controller.host == "0.0.0.0":
                ip = lan_ip()
                if ip:
                    lines.append(f"다른 기기에서: <code>http://{ip}:{self.controller.port}/v1</code>")
            self.url_lbl.setText("<br>".join(lines))
            self._uptime_timer.start(1000)
        elif state == ServerState.STARTING:
            self.url_lbl.setText("모델을 메모리에 올리는 중이에요. 큰 모델은 1분 이상 걸릴 수 있어요…")
            self._uptime_timer.start(1000)
        else:
            self.url_lbl.setText("서버가 꺼져 있어요.")
            self._uptime_timer.stop()
            self.mem_lbl.setText("메모리 사용: —")
            if state == ServerState.STOPPED:
                self._close_log_file()
        self.server_state_changed.emit(state)

    def _update_uptime(self) -> None:
        secs = int(time.monotonic() - self.controller.started_at)
        self.uptime_lbl.setText(f"실행 시간: {secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}")

    def _on_speed(self, kind: str, tps: float) -> None:
        if kind == "generate":
            self.gen_lbl.setText(f"답변 생성 속도: <b>{tps:.1f}</b> 토큰/초")
        else:
            self.prompt_lbl.setText(f"입력 처리 속도: {tps:.1f} 토큰/초")

    def _on_error_hint(self, hint: str) -> None:
        self.hint_lbl.setText("💡 " + hint)
        self.hint_lbl.show()

    def _append_log(self, line: str, level: str) -> None:
        color = LOG_COLORS.get(level)
        esc = html.escape(line)
        self.log.appendHtml(f"<span style='color:{color}'>{esc}</span>" if color else esc)
        if self._log_file:
            try:
                self._log_file.write(line + "\n")
            except OSError:
                self._log_file = None

    def _open_log_file(self) -> None:
        self._close_log_file()
        try:
            d = logs_dir()
            d.mkdir(parents=True, exist_ok=True)
            self._log_file = open(d / f"server-{datetime.now():%Y%m%d}.log", "a", encoding="utf-8", buffering=1)
            self._log_file.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 서버 시작 =====\n")
        except OSError:
            self._log_file = None

    def _close_log_file(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _open_logs_dir(self) -> None:
        d = logs_dir()
        d.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))
