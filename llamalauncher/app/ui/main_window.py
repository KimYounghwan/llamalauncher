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
from .dashboard import DashboardTab
from .downloads_tab import DownloadsTab
from .server_tab import ServerTab

UPCOMING_TABS = [("🚀 런처", 5), ("📁 내 모델", 4), ("💬 채팅 테스트", 4), ("🔌 연동", 6)]

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
        self.server = ServerTab(settings, self.save_settings)
        self.dashboard.system_detected.connect(self.server.set_system)
        self.dashboard.system_detected.connect(self.browse.set_system)
        self.server.server_state_changed.connect(self._on_server_state)
        self.browse.download_requested.connect(self._on_download_requested)
        self.downloads.run_model.connect(self._run_model)
        self.downloads.active_count_changed.connect(self._on_downloads_count)
        self.downloads.token_changed.connect(lambda tok: setattr(self.hf, "token", tok))

        self.tabs.addTab(self.dashboard, "📊 대시보드")
        self.tabs.addTab(self.browse, "🔍 모델 찾기")
        self.downloads_index = self.tabs.addTab(self.downloads, "⬇️ 다운로드")
        self.server_index = self.tabs.addTab(self.server, "▶️ 서버")
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

    def _run_model(self, path: str) -> None:
        self.server.select_model_path(path)
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
        self.downloads.shutdown()
        self.server.shutdown()
        self.save_settings()
        event.accept()
