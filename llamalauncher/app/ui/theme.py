"""테마: 시스템 설정 따르기 / 밝게 / 어둡게.

Qt 6.8+는 색 구성표를 바꾸면 Fusion 스타일이 따라간다 (macOS·Windows). 플랫폼이 지원하지 않으면
(일부 Linux 데스크톱 등) 직접 만든 팔레트를 쓴다.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

THEMES = [("system", "시스템 설정 따르기"), ("light", "밝게"), ("dark", "어둡게")]


def dark_palette() -> QPalette:
    p = QPalette()
    c = {
        QPalette.Window: "#2b2b2b", QPalette.WindowText: "#e8e8e8", QPalette.Base: "#1f1f1f",
        QPalette.AlternateBase: "#2b2b2b", QPalette.ToolTipBase: "#3a3a3a", QPalette.ToolTipText: "#e8e8e8",
        QPalette.Text: "#e8e8e8", QPalette.Button: "#353535", QPalette.ButtonText: "#e8e8e8",
        QPalette.BrightText: "#ff6b6b", QPalette.Link: "#6ea8ff", QPalette.Highlight: "#2f6fe4",
        QPalette.HighlightedText: "#ffffff", QPalette.PlaceholderText: "#9a9a9a", QPalette.Mid: "#4a4a4a",
    }
    for role, color in c.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor("#7a7a7a"))
    return p


def is_dark(app: QApplication) -> bool:
    return app.palette().color(QPalette.Window).lightness() < 128


def apply_theme(app: QApplication, mode: str) -> None:
    want = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}.get(mode, Qt.ColorScheme.Unknown)
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        hints.setColorScheme(want)
    supported = want == Qt.ColorScheme.Unknown or hints.colorScheme() == want
    if mode == "dark" and (not supported or not is_dark(app)):
        app.setPalette(dark_palette())
    elif mode == "light" and (not supported or is_dark(app)):
        app.setPalette(app.style().standardPalette())
    elif mode == "system":
        app.setPalette(app.style().standardPalette() if not supported else QPalette())
