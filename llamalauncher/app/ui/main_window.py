"""메인 창: 탭 구성. 아직 만들지 않은 탭은 '준비 중'으로 표시한다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QTabWidget

from ..core.server import ServerState
from ..core.settings import Settings
from .dashboard import DashboardTab
from .server_tab import ServerTab

UPCOMING_TABS = [
    ("🚀 런처", 5), ("🔍 모델 찾기", 3), ("⬇️ 다운로드", 3), ("📁 내 모델", 4),
    ("💬 채팅 테스트", 4), ("🔌 연동", 6),
]

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
        self.dashboard = DashboardTab(Path(settings.models_dir))
        self.server = ServerTab(settings, self.save_settings)
        self.dashboard.system_detected.connect(self.server.set_system)
        self.server.server_state_changed.connect(self._on_server_state)

        self.tabs.addTab(self.dashboard, "📊 대시보드")
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

    def closeEvent(self, event):
        if self.server.controller.is_active:
            answer = QMessageBox.question(self, "종료", "AI 서버가 실행 중이에요. 서버를 끄고 종료할까요?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.server.shutdown()
        self.save_settings()
        event.accept()
