"""앱 아이콘 만들기: assets/icon.png (1024), icon.ico (Windows), icon.icns (macOS, iconutil 필요)."""

import shutil
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath

OUT = Path(__file__).resolve().parent.parent / "assets"


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    m = size * 0.06
    rect = QRectF(m, m, size - 2 * m, size - 2 * m)
    path = QPainterPath()
    path.addRoundedRect(rect, size * 0.2, size * 0.2)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0, QColor("#4f8cff"))
    g.setColorAt(1, QColor("#1a4fd1"))
    p.fillPath(path, g)
    # 이모지 글꼴(저작권이 있는 그림)은 쓰지 않고 글자로만 만든다
    font = QFont("Apple SD Gothic Neo" if sys.platform == "darwin" else "Malgun Gothic")
    font.setWeight(QFont.Black)
    font.setPixelSize(int(size * 0.34))
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(rect, Qt.AlignCenter, "라마")
    p.end()
    return img


def main() -> None:
    app = QGuiApplication(sys.argv)  # noqa: F841 — 글꼴 사용에 필요
    OUT.mkdir(exist_ok=True)
    render(1024).save(str(OUT / "icon.png"))
    render(256).save(str(OUT / "icon.ico"))
    if sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = OUT / "icon.iconset"
        iconset.mkdir(exist_ok=True)
        for s in (16, 32, 128, 256, 512):
            render(s).save(str(iconset / f"icon_{s}x{s}.png"))
            render(s * 2).save(str(iconset / f"icon_{s}x{s}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT / "icon.icns")], check=True)
        shutil.rmtree(iconset)
    print("아이콘을 만들었어요:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
