#!/usr/bin/env bash
# macOS 빌드: dist/LlamaLauncher.app 과 dist/LlamaLauncher-<버전>-mac.dmg 를 만든다.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt pyinstaller

[ -f assets/icon.icns ] || python tools/make_icons.py

rm -rf build dist
pyinstaller --noconfirm llamalauncher.spec

echo "▶ 빌드 결과 확인 (창을 띄웠다가 닫아요)"
QT_QPA_PLATFORM=offscreen ./dist/LlamaLauncher.app/Contents/MacOS/LlamaLauncher --smoke-test

VERSION=$(python -c "from llamalauncher.app import __version__; print(__version__)")
DMG="dist/LlamaLauncher-${VERSION}-mac.dmg"
hdiutil create -volname "라마런처" -srcfolder dist/LlamaLauncher.app -ov -format UDZO "$DMG" >/dev/null
echo "✅ 완료: dist/LlamaLauncher.app, $DMG"
echo "ℹ️ 서명하지 않은 앱이라 처음 열 때 [우클릭 → 열기]를 해야 할 수 있어요."
