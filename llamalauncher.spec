# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 빌드 설정: `pyinstaller --noconfirm llamalauncher.spec`
# macOS → dist/LlamaLauncher.app, Windows → dist/LlamaLauncher/LlamaLauncher.exe
import re
import sys
from pathlib import Path

# 스펙 파일은 프로젝트 폴더를 import 경로에 넣지 않으므로 버전은 파일에서 직접 읽는다
__version__ = re.search(r'__version__ = "([^"]+)"', Path("llamalauncher/app/__init__.py").read_text()).group(1)

datas = [("llamalauncher/app/data", "llamalauncher/app/data")]  # 카테고리·연동 템플릿 JSON
hidden = ["keyring.backends.null"]
if sys.platform == "darwin":
    hidden.append("keyring.backends.macOS")
elif sys.platform == "win32":
    hidden.append("keyring.backends.Windows")
else:
    hidden.append("keyring.backends.SecretService")

a = Analysis(
    ["llamalauncher/main.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
              "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtPdf"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = "assets/icon.icns" if sys.platform == "darwin" else "assets/icon.ico"
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LlamaLauncher", console=False, icon=icon,
          upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="LlamaLauncher", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="LlamaLauncher.app",
        icon=icon,
        bundle_identifier="io.github.llamalauncher",
        info_plist={
            "CFBundleDisplayName": "라마런처",
            "CFBundleName": "LlamaLauncher",
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
