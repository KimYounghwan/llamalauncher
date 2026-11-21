"""라마런처 진입점: `python -m llamalauncher.main`

옵션
  --smoke-test   창을 띄웠다가 몇 초 뒤 스스로 종료 (빌드 확인용)
  --no-wizard    첫 실행 마법사를 띄우지 않음
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from llamalauncher.app import __version__
from llamalauncher.app.core.settings import Settings
from llamalauncher.app.ui.main_window import MainWindow
from llamalauncher.app.ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("LlamaLauncher")
    app.setApplicationDisplayName("라마런처")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    settings = Settings.load()
    apply_theme(app, settings.theme)
    window = MainWindow(settings)
    window.show()
    smoke = "--smoke-test" in sys.argv
    if smoke:
        def finish():
            ok = window.dashboard.system is not None
            print(f"SMOKE-TEST {'OK' if ok else 'NO-SYSTEM'}: tabs={window.tabs.count()} "
                  f"launchers={len(window.launchers.statuses)} integrations={len(window.integrations.items)}", flush=True)
            window.close()
            app.exit(0 if ok else 1)
        QTimer.singleShot(8000, finish)
    elif not settings.first_run_done and "--no-wizard" not in sys.argv:
        QTimer.singleShot(400, window.show_wizard)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
