"""라마런처 진입점: `python -m llamalauncher.main`"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from llamalauncher.app.core.settings import Settings
from llamalauncher.app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LlamaLauncher")
    app.setStyle("Fusion")
    window = MainWindow(Settings.load())
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
