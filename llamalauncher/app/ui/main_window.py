"""메인 창: 탭 구성. 아직 만들지 않은 탭은 '준비 중'으로 표시한다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from ..core.hf_client import HFClient, HFModel, QuantFile
from ..core.recommender import Fit, ModelSpec, assess_fit
from ..core.hardware import format_bytes
from ..core.secrets_store import get_hf_token
from ..core.server import ServerState
from ..core.settings import Settings
from .browse_tab import BrowseTab
from .chat_tab import ChatTab
from .dashboard import DashboardTab
from .downloads_tab import DownloadsTab
from .launchers_tab import LaunchersTab
from .models_tab import ModelsTab
from .server_tab import ServerTab

UPCOMING_TABS = [("🔌 연동", 6)]

STYLE = """
QLabel#pageTitle { font-size: 22px; font-weight: 700; }
QLabel#sectionTitle { font-size: 16px; font-weight: 600; }
QLabel#hint { color: palette(placeholder-text); }
QLabel#tierBanner {
    font-size: 15px; padding: 14px 16px; border-radius: 10px;
    background: rgba(80, 140, 255, 0.12); border: 1px solid rgba(80, 140, 255, 0.35);
}
QLabel#errorBanner {
    padding: 10px 12px; border-radius: 8px;
    background: rgba(219, 68, 55, 0.10); border: 1px solid rgba(219, 68, 55, 0.45);
}
QFrame#card {
    border-radius: 10px; padding: 6px;
    background: palette(base); border: 1px solid palette(mid);
}
QLabel#cardTitle { font-size: 13px; color: palette(placeholder-text); }
QLabel#cardValue { font-size: 16px; font-weight: 600; }
QLabel#cardDetail { font-size: 12px; color: palette(placeholder-text); }
QPushButton#primary {
    background: #1a73e8; color: white; font-weight: 600;
    border: none; border-radius: 6px; padding: 6px 14px;
}
QPushButton#primary:hover { background: #1765cc; }
QPushButton#primary:disabled { background: rgba(26, 115, 232, 0.35); }
QGroupBox { font-weight: 600; }
"""


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("라마런처 (LlamaLauncher)")
        self.resize(1200, 860)
        self.setStyleSheet(STYLE)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.hf = HFClient(token=get_hf_token())
        self.dashboard = DashboardTab(Path(settings.models_dir))
        self.browse = BrowseTab(settings, self.hf, self.save_settings)
        self.downloads = DownloadsTab(settings, self.save_settings)
        self.models = ModelsTab(settings, self.save_settings)
        self.server = ServerTab(settings, self.save_settings)
        self.chat = ChatTab(settings, self.save_settings)
        self.launchers = LaunchersTab(settings, self.save_settings)
        self.dashboard.system_detected.connect(self.launchers.set_system)
        self.launchers.statuses_changed.connect(self._on_launcher_statuses)
        self.launchers.use_engine.connect(self._use_engine)
        self.launchers.find_models.connect(self._find_models_for)
        self.server.open_launchers.connect(lambda: self.tabs.setCurrentIndex(self.launchers_index))
        self.browse.ollama_pull_requested.connect(self._ollama_pull)
        self.server.name_for = self.models.display_name_for
        self.server.mmproj_for = self.models.mmproj_for
        self.models.in_use = lambda p: self.server.controller.is_active and self.server.running.get("model_path") == p
        self.dashboard.system_detected.connect(self.server.set_system)
        self.dashboard.system_detected.connect(self.browse.set_system)
        self.dashboard.system_detected.connect(self.models.set_system)
        self.server.server_state_changed.connect(self._on_server_state)
        self.server.model_file_picked.connect(self.models.add_file)
        self.models.run_model.connect(self._run_model)
        self.models.models_changed.connect(self.server.refresh_model_names)
        self.chat.go_to_server.connect(lambda: self.tabs.setCurrentIndex(self.server_index))
        self.browse.download_requested.connect(self._on_download_requested)
        self.downloads.run_model.connect(self._run_model)
        self.downloads.active_count_changed.connect(self._on_downloads_count)
        self.downloads.active_count_changed.connect(lambda _n: self.models.refresh_later())
        self.downloads.token_changed.connect(lambda tok: setattr(self.hf, "token", tok))

        self.tabs.addTab(self.dashboard, "📊 대시보드")
        self.tabs.addTab(self.browse, "🔍 모델 찾기")
        self.downloads_index = self.tabs.addTab(self.downloads, "⬇️ 다운로드")
        self.models_index = self.tabs.addTab(self.models, "📁 내 모델")
        self.server_index = self.tabs.addTab(self.server, "▶️ 서버")
        self.chat_index = self.tabs.addTab(self.chat, "💬 채팅 테스트")
        self.launchers_index = self.tabs.addTab(self.launchers, "🚀 런처")
        for name, stage in UPCOMING_TABS:
            placeholder = QLabel(f"{name.split(' ', 1)[1]} 기능은 {stage}단계에서 추가될 예정이에요.")
            placeholder.setAlignment(Qt.AlignCenter)
            idx = self.tabs.addTab(placeholder, name)
            self.tabs.setTabEnabled(idx, False)
        self.setCentralWidget(self.tabs)

    def save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as e:
            self.statusBar().showMessage(f"설정을 저장하지 못했어요: {e}", 5000)

    def _on_server_state(self, state: ServerState) -> None:
        mark = {ServerState.RUNNING: " 🟢", ServerState.STARTING: " 🟡", ServerState.ERROR: " 🔴"}.get(state, "")
        self.tabs.setTabText(self.server_index, "▶️ 서버" + mark)
        self.chat.set_server(state, self.server.controller.base_url, self.server.running)

    def _on_download_requested(self, model: HFModel, quant: QuantFile, mmproj) -> None:
        system = self.dashboard.system
        if system:
            fit = assess_fit(system, ModelSpec(file_size=quant.size, n_ctx_train=model.context_length), 4096)
            if fit.fit == Fit.INSUFFICIENT and QMessageBox.warning(
                    self, "이 컴퓨터에서는 실행하기 어려워요",
                    f"{model.name} {quant.quant} ({format_bytes(quant.size)})\n\n{fit.reason}\n\n그래도 받을까요?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        if model.is_embedding:
            QMessageBox.information(self, "임베딩 모델", "이 모델은 문서 검색(RAG)용 임베딩 모델이라 대화에는 쓸 수 없어요.")
        if self.downloads.enqueue(model, quant, mmproj):
            self.statusBar().showMessage(f"⬇️ {model.name} {quant.quant} 다운로드를 시작했어요. [다운로드] 탭에서 진행 상황을 볼 수 있어요.", 6000)

    def _on_downloads_count(self, n: int) -> None:
        self.tabs.setTabText(self.downloads_index, "⬇️ 다운로드" + (f" ({n})" if n else ""))

    # ------------------------------------------------------------------ 런처 연동
    def _on_launcher_statuses(self, statuses: dict) -> None:
        import sys as _sys
        from ..core.launchers import by_id
        opts = [("gguf", "GGUF — llama.cpp·Ollama·LM Studio 등")]
        apple = _sys.platform == "darwin" and (self.dashboard.system is None or self.dashboard.system.arch == "arm64")
        if apple:
            opts.append(("mlx", "MLX — MLX-LM·LM Studio (Apple Silicon)"))
        if by_id("vllm").availability(self.dashboard.system)[0] or getattr(statuses.get("vllm"), "installed", False):
            opts.append(("safetensors", "safetensors — vLLM (Linux+NVIDIA)"))
        self.browse.set_format_options(opts)
        self.browse.ollama_available = bool(getattr(statuses.get("ollama"), "installed", False))
        self.browse._update_download_btn()
        st = statuses.get(self.server.launcher.id)
        cur = self.server.launcher_status
        if st is not None and (cur is None or cur.installed != st.installed or cur.path != st.path):
            self.server.detect_launcher()
        self.models.refresh_later()

    def _use_engine(self, launcher_id: str) -> None:
        self.server.set_engine(launcher_id)
        self.tabs.setCurrentIndex(self.server_index)

    def _find_models_for(self, fmt: str) -> None:
        self.browse.set_format("gguf" if fmt == "ollama" else fmt)
        self.tabs.setCurrentIndex(self.tabs.indexOf(self.browse))

    def _ollama_pull(self, repo: str, quant: str) -> None:
        from ..core.launchers import by_id
        from ..core.launchers.base import CommandStep, InstallPlan
        st = self.launchers.statuses.get("ollama")
        if not (st and st.installed and st.path):
            QMessageBox.information(self, "Ollama 없음", "Ollama가 설치되어 있지 않아요. [런처] 탭에서 설치해 주세요.")
            return
        ref = f"hf.co/{repo}:{quant}"
        plan = InstallPlan(f"Ollama로 받기: {ref}", [CommandStep([str(st.path), "pull", ref], f"ollama pull {ref}")],
                           note="Ollama 앱(또는 ollama serve)이 켜져 있어야 해요. 받은 모델은 Ollama의 저장소에 들어가요.")
        self.tabs.setCurrentIndex(self.launchers_index)
        self.launchers.run_plan(by_id("ollama"), plan, "받기")

    def _run_model(self, path: str, mmproj=None) -> None:
        if self.server.controller.is_active and self.server.running.get("model_path") != path:
            if QMessageBox.question(self, "다른 모델이 실행 중이에요",
                                    "지금 실행 중인 서버를 멈추고 이 모델로 바꿀까요?\n"
                                    "(바꾼 뒤 [서버] 탭에서 ▶ 서버 시작을 눌러 주세요)") != QMessageBox.Yes:
                return
            self.server.controller.stop()
        # 모델 형식에 맞는 엔진으로 바꾼다
        from ..core.launchers import by_id
        from ..core.model_library import dir_format
        if Path(path).is_dir():
            want = "mlx" if dir_format(Path(path)) == "mlx" else "vllm"
            ok, why = by_id(want).availability(self.dashboard.system)
            if not ok:
                QMessageBox.information(self, "이 컴퓨터에서 실행할 수 없는 형식이에요", f"{by_id(want).name}: {why}")
                return
            if self.server.launcher.id != want:
                self.server.set_engine(want)
        elif self.server.launcher.model_is_dir:
            self.server.set_engine("llamacpp")
        self.server.select_model_path(path, mmproj)
        self.tabs.setCurrentIndex(self.server_index)

    def closeEvent(self, event):
        busy = []
        if self.server.controller.is_active:
            busy.append("AI 서버가 실행 중이에요 (종료하면 꺼져요).")
        if self.downloads.has_active():
            busy.append("다운로드 중인 모델이 있어요 (다음에 이어받을 수 있어요).")
        if busy:
            answer = QMessageBox.question(self, "종료", "\n".join(busy) + "\n\n종료할까요?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.chat.shutdown()
        self.launchers.shutdown()
        self.downloads.shutdown()
        self.server.shutdown()
        self.save_settings()
        event.accept()
