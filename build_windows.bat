@echo off
rem Windows 빌드: dist\LlamaLauncher\LlamaLauncher.exe 와 dist\LlamaLauncher-<버전>-windows.zip 을 만든다.
setlocal
cd /d "%~dp0"

if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt pyinstaller || goto :error

if not exist assets\icon.ico python tools\make_icons.py

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller --noconfirm llamalauncher.spec || goto :error

echo 빌드 결과 확인 (창을 띄웠다가 닫아요)
set QT_QPA_PLATFORM=offscreen
dist\LlamaLauncher\LlamaLauncher.exe --smoke-test
set QT_QPA_PLATFORM=

for /f %%v in ('python -c "from llamalauncher.app import __version__; print(__version__)"') do set VERSION=%%v
powershell -NoProfile -Command "Compress-Archive -Path dist\LlamaLauncher -DestinationPath dist\LlamaLauncher-%VERSION%-windows.zip -Force" || goto :error
echo 완료: dist\LlamaLauncher\LlamaLauncher.exe, dist\LlamaLauncher-%VERSION%-windows.zip
exit /b 0

:error
echo 빌드에 실패했어요. 위의 오류 메시지를 확인해 주세요.
exit /b 1
