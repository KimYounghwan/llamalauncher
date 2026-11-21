# 라마런처 (LlamaLauncher)

초보자도 내 컴퓨터에서 로컬 AI 모델을 쉽게 찾고, 받고, 실행하고, 코딩 도구에 연결할 수 있게 해 주는 파이썬 GUI 프로그램입니다.
llama.cpp를 기본 엔진으로 사용하며 macOS와 Windows에서 동작합니다. 전체 기획은 [PROMPT.md](PROMPT.md)를 참고하세요.

## 할 수 있는 일

| 탭 | 기능 |
|---|---|
| 📊 대시보드 | CPU·메모리·GPU·디스크를 자동으로 확인하고 "이 컴퓨터는 어느 크기 모델까지 원활한지" 알려줘요 |
| 🔍 모델 찾기 | 허깅페이스에서 카테고리별(추천·대화·한국어·코딩·추론·비전·임베딩·경량) 모델을 이 컴퓨터 적합도와 함께 보여줘요. 형식(GGUF·MLX·safetensors) 선택, ⭐ 추천 파일 |
| ⬇️ 다운로드 | 이어받기·일시정지·SHA-256 검증·동시 다운로드 제한, 비공개 모델용 토큰(OS 키체인 저장) |
| 📁 내 모델 | 받은 모델, 직접 고른 파일, 가져온 폴더, LM Studio·Ollama·허깅페이스 캐시 모델을 한곳에서 보고 실행·삭제 |
| ▶️ 서버 | 실행 엔진(llama.cpp·Ollama·KoboldCpp·MLX-LM·vLLM) 선택, 🪄 이 컴퓨터에 맞게 자동 설정, 프리셋(빠르게·균형·긴 대화·코딩 에이전트용), 명령어 미리보기, 실시간 기록·속도·메모리 |
| 💬 채팅 테스트 | 스트리밍 답변, 시스템 프롬프트, 생각 과정 보기, 속도 표시 |
| 🚀 런처 | 실행 엔진 8종의 설치 상태 확인과 자동 설치·업데이트·제거 (Homebrew·winget·GitHub 배포·전용 파이썬 환경) |
| 🔌 연동 | Claude Code·OpenCode·Codex·Continue·Cline·Copilot·Aider·Zed·JetBrains·Cursor·Open WebUI 설정 자동 생성·적용·되돌리기, 연결 테스트 |

처음 실행하면 **시작 마법사**가 사양 확인 → 엔진 설치 → 추천 모델 받기 → 자동 설정으로 실행 → 첫 대화까지 안내해요
(메뉴 **도움말 → 시작 마법사**로 다시 열 수 있어요). 테마는 **보기 → 테마**에서 바꿀 수 있어요.

## 진행 상황

- [x] 1단계: 시스템 사양 감지 + 추천 옵션 계산 + 테스트 + 대시보드 화면
- [x] 2단계: llama.cpp 런처 감지/실행 + 서버 탭 + 옵션/자동 설정 (+ GGUF 메타데이터 읽기)
- [x] 3단계: 허깅페이스 모델 검색(카테고리·적합도·추천 파일) + 다운로드(이어받기·검증)
- [x] 4단계: 내 모델(LM Studio·Ollama 모델 자동 탐지, 폴더 가져오기) + 파일 직접 선택 + 채팅 테스트(스트리밍)
- [x] 5단계: 런처 8종(llama.cpp·Ollama·LM Studio·KoboldCpp·MLX-LM·vLLM·Jan·text-generation-webui) 감지·자동 설치, 엔진별 서버 실행, 형식별(GGUF·MLX·safetensors) 모델 찾기
- [x] 6단계: 개발 도구 연동 12종 (Claude Code·OpenCode·Codex는 실제 CLI로 동작 확인), 설정 병합·백업·되돌리기, 연결 테스트, Claude Code 호환 주소
- [x] 7단계: 첫 실행 마법사, 테마(시스템·밝게·어둡게), PyInstaller 빌드 (macOS .app/.dmg, Windows .exe/.zip)

## 실행 방법 (소스에서)

### macOS
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m llamalauncher.main
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m llamalauncher.main
```

옵션: `--no-wizard` (시작 마법사 끄기), `--smoke-test` (창을 띄웠다가 스스로 종료 — 빌드 확인용)

## 빌드 (설치용 앱 만들기)

| OS | 명령 | 결과 |
|---|---|---|
| macOS | `./build_mac.sh` | `dist/LlamaLauncher.app`, `dist/LlamaLauncher-<버전>-mac.dmg` |
| Windows | `build_windows.bat` | `dist\LlamaLauncher\LlamaLauncher.exe`, `dist\LlamaLauncher-<버전>-windows.zip` |

- 각 OS에서 직접 빌드해야 해요 (PyInstaller는 다른 OS용으로 빌드하지 못해요).
- 서명하지 않은 앱이라 macOS에서는 처음 열 때 **우클릭 → 열기**, Windows에서는 SmartScreen에서 **추가 정보 → 실행**을 눌러야 할 수 있어요.
- 아이콘은 `tools/make_icons.py`로 다시 만들 수 있어요.
- 패키징된 앱에서 MLX-LM·vLLM을 설치하려면 컴퓨터에 파이썬 3.10 이상이 있어야 해요 (전용 가상환경을 만들 때 사용).

## 테스트
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```
Windows/Linux 전용 코드(레지스트리, nvidia-smi, winget 등)는 가짜 출력(mock)으로 검증해요.
테스트는 실제 OS 키체인이나 개발 도구 설정 파일을 건드리지 않아요.

## 데이터가 저장되는 곳

| 내용 | 위치 |
|---|---|
| 설정·다운로드 목록·연동 기록·로그 | macOS `~/Library/Application Support/LlamaLauncher/`, Windows `%LOCALAPPDATA%\LlamaLauncher\` |
| 모델 (기본값, 다운로드 탭에서 변경 가능) | `~/LlamaLauncher/models/` |
| 라마런처가 설치한 엔진 | `~/LlamaLauncher/bin/<엔진>/`, `~/LlamaLauncher/venvs/<엔진>/` |
| 허깅페이스 토큰, 서버 API 키 | OS 키체인 (macOS 키체인 / Windows 자격 증명 관리자) |
| 개발 도구 설정 백업 | 원래 설정 파일 옆의 `*.bak-날짜시간` |

## 구조
```
llamalauncher/
  main.py                    # 진입점 (테마 적용, 첫 실행 마법사)
  app/core/hardware.py       # 사양 감지 (macOS: sysctl/system_profiler, Windows: 레지스트리/nvidia-smi, Linux: nvidia-smi/rocm-smi)
  app/core/recommender.py    # 메모리 계산, 적합도(쾌적/느림/부족), llama-server 추천 옵션, 프리셋
  app/core/settings.py       # 설정 저장 (OS별 사용자 설정 폴더)
  app/core/gguf.py           # GGUF 헤더(층 수, 최대 문맥, 채팅 템플릿, 양자화) 읽기
  app/core/launchers/        # 런처 어댑터: 감지·설치 계획·서버 실행 방법 (llamacpp, ollama, others)
  app/core/installer.py      # 설치 실행기 (brew/winget 명령, GitHub 배포 받기·압축 풀기, 전용 파이썬 환경 pip)
  app/core/server.py         # 서버 실행/중지/상태 감시 (QProcess, 엔진별 상태 확인 주소, 준비 단계)
  app/core/server_log.py     # 로그 분류, 속도 추출, 오류를 쉬운 말로 설명
  app/core/netutil.py        # 포트 확인, 내 IP, API 키 생성
  app/core/hf_client.py      # 허깅페이스 검색, 카테고리, 파일(양자화) 분석, 추천 파일, 캐시
  app/core/downloader.py     # 다운로드 관리 (이어받기, 일시정지, SHA-256 검증, 폴더형 모델)
  app/core/model_library.py  # 내 모델 목록 (받은 모델·직접 추가·폴더·LM Studio·Ollama·HF 캐시), 삭제
  app/core/chat_client.py    # OpenAI 호환 API 스트리밍 채팅 (생각 과정 분리, 쉬운 오류 설명)
  app/core/integrations/     # 개발 도구 연동: 템플릿 렌더링·호환성 점검(base), 병합·백업·복구(config_merge), 연결 테스트(tester)
  app/core/compat_proxy.py   # Claude Code 호환 주소 (대화 중간 system 메시지 변환)
  app/core/secrets_store.py  # 허깅페이스 토큰·서버 API 키를 OS 키체인에 저장
  app/data/catalog.json      # 카테고리 정의, 검증된 제작자, 숨길 키워드 (앱 업데이트 없이 수정 가능)
  app/data/integrations/     # 도구별 설정 템플릿 (공식 문서 URL·확인 날짜·실제 확인한 버전)
  app/ui/                    # PySide6 화면 (탭별 위젯, 시작 마법사, 테마)
tests/                       # 단계별 테스트
llamalauncher.spec           # PyInstaller 빌드 설정
build_mac.sh, build_windows.bat
tools/make_icons.py          # 앱 아이콘 생성
```
