"""메인 창: 탭 구성. 아직 만들지 않은 탭은 '준비 중'으로 표시한다."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from ..core.settings import Settings
from .dashboard import DashboardTab

UPCOMING_TABS = [
    ("🚀 런처", 5), ("🔍 모델 찾기", 3), ("⬇️ 다운로드", 3), ("📁 내 모델", 4),
    ("▶️ 서버", 2), ("💬 채팅 테스트", 4), ("🔌 연동", 6),
]

STYLE = """
QLabel#pageTitle { font-size: 22px; font-weight: 700; }
QLabel#sectionTitle { font-size: 16px; font-weight: 600; margin-top: 8px; }
QLabel#hint { color: palette(placeholder-text); }
QLabel#tierBanner {
    font-size: 15px; padding: 14px 16px; border-radius: 10px;
    background: rgba(80, 140, 255, 0.12); border: 1px solid rgba(80, 140, 255, 0.35);
}
QFrame#card {
    border-radius: 10px; padding: 6px;
    background: palette(base); border: 1px solid palette(mid);
}
QLabel#cardTitle { font-size: 13px; color: palette(placeholder-text); }
QLabel#cardValue { font-size: 16px; font-weight: 600; }
QLabel#cardDetail { font-size: 12px; color: palette(placeholder-text); }
"""


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("라마런처 (LlamaLauncher)")
        self.resize(1100, 760)
        self.setStyleSheet(STYLE)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        self.dashboard = DashboardTab(Path(settings.models_dir))
        tabs.addTab(self.dashboard, "📊 대시보드")
        for name, stage in UPCOMING_TABS:
            placeholder = QLabel(f"{name.split(' ', 1)[1]} 기능은 {stage}단계에서 추가될 예정이에요.")
            placeholder.setAlignment(Qt.AlignCenter)
            idx = tabs.addTab(placeholder, name)
            tabs.setTabEnabled(idx, False)
        self.setCentralWidget(tabs)
