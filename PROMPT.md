# 라마런처 (LlamaLauncher) 개발 프롬프트

아래 내용을 그대로 AI 코딩 도구(Claude Code 등)에 붙여넣어 사용하세요.

---

## 역할과 목표

너는 숙련된 파이썬 데스크톱 앱 개발자다. **AI 모델을 잘 모르는 초보자**도 내 컴퓨터에서 로컬 LLM을 쉽게 다운로드하고 실행(서비스)할 수 있는 **파이썬 GUI 프로그램 "라마런처"**를 만들어라.

- 기본 실행 엔진: **llama.cpp** (`llama-server`)
- 대상 OS: **macOS(Apple Silicon/Intel)와 Windows 10/11** 모두에서 동작해야 한다. (Linux는 가능하면 지원)
- 현재 개발 환경: macOS, Apple M4 Max, RAM 48GB, Python 3.12, llama.cpp는 Homebrew로 설치됨(`/opt/homebrew/bin/llama-server`)
- UI 언어: **한국어** (용어는 쉬운 말 + 괄호 안에 원어, 예: "문맥 길이(Context Size)")

## 기술 스택

- GUI: **PySide6 (Qt6)** — 크로스플랫폼, 다크/라이트 테마, 스레드 지원
- 하드웨어 감지: `psutil`, `platform`, `subprocess` (macOS: `sysctl`, `system_profiler`; Windows: `nvidia-smi`, WMI/`wmic` 또는 PowerShell `Get-CimInstance`)
- 허깅페이스: `huggingface_hub` (`HfApi.list_models`, `list_repo_tree`/`model_info`, `hf_hub_download`)
- HTTP: `requests` / `httpx`
- 설정 저장: JSON 파일 (`platformdirs`로 OS별 사용자 설정 경로 사용)
- 패키징: `pyinstaller`로 macOS `.app`, Windows `.exe` 빌드 스크립트 제공
- 모든 경로 처리는 `pathlib` 사용, 쉘 명령은 OS 분기 처리. 하드코딩된 `/` 경로 금지.

## 기능 요구사항

### 1. 시스템 사양 자동 인식 (대시보드 탭)
프로그램 시작 시 자동으로 아래를 감지해 카드 형태로 보여준다.
- OS / 버전, CPU 모델·코어 수, 전체/가용 RAM
- GPU 종류와 VRAM
  - macOS Apple Silicon: 통합 메모리(Unified Memory) → Metal 사용 가능, 사용 가능 VRAM ≈ RAM의 약 65~75%
  - Windows/Linux: NVIDIA(`nvidia-smi`) → CUDA, AMD → Vulkan/ROCm, Intel → Vulkan/SYCL, 없으면 CPU 전용
- 남은 디스크 공간 (모델 저장 폴더 기준)
- 결과를 바탕으로 **"이 컴퓨터 등급"**(예: 입문 / 보통 / 고성능 / 워크스테이션)과 **"실행 가능한 최대 모델 크기"**를 쉬운 문장으로 표시
  예) "이 컴퓨터는 약 30GB 크기의 모델까지 원활하게 실행할 수 있어요 (예: 32B 모델 Q4)."

### 2. 런처(실행 엔진) 관리 탭
llama.cpp 외에 유명 런처를 목록으로 보여주고 설치 상태를 표시한다.
- 대상: **llama.cpp, Ollama, LM Studio, KoboldCpp, vLLM(Linux/NVIDIA 전용 안내), MLX-LM(Apple Silicon 전용), Jan, text-generation-webui** 정도
- 각 항목: 설명(초보자용 한 줄), 지원 OS, 설치 여부, 설치 경로, 버전, [설치] [업데이트] [제거] [폴더 열기] 버튼
- **자동 설치**
  - macOS: Homebrew가 있으면 `brew install llama.cpp` / `brew install ollama` 등, 없으면 GitHub Releases에서 바이너리 다운로드
  - Windows: `winget` 우선(`winget install ggml.llamacpp`, `Ollama.Ollama` 등), 실패 시 GitHub Releases 최신 zip을 GPU 종류(CUDA/Vulkan/CPU)에 맞게 골라 다운로드·압축 해제
  - pip 기반(MLX-LM, vLLM)은 앱 전용 가상환경에 설치
  - 설치 진행 로그를 실시간으로 보여주고, 관리자 권한이 필요하면 안내
- 사용자가 실행 파일 경로를 직접 지정할 수도 있게 한다
- 런처마다 지원 모델 형식을 정의: llama.cpp/KoboldCpp/LM Studio/Jan → **GGUF**, MLX-LM → **MLX**, vLLM → **safetensors/AWQ/GPTQ**, Ollama → Ollama 라이브러리 + GGUF

### 3. 모델 찾기 탭 (허깅페이스)
- **현재 선택된 런처에 맞는 형식**의 모델만 검색 (예: llama.cpp → `library=gguf` 필터)
- **카테고리별 제공**: 💬 일반 대화, 🇰🇷 한국어 특화, 💻 코딩, 🧠 추론(Reasoning), 👁️ 비전(이미지 이해), 📚 임베딩, 🪶 경량(저사양용)
  - 카테고리는 태그·검색어·큐레이션 목록(JSON, 앱에 내장 + 업데이트 가능)을 조합해 구성
  - 예시 큐레이션: Qwen, Llama, Gemma, Mistral, Phi, DeepSeek, EXAONE(한국어), 코딩용 Qwen-Coder 등 (실제 존재하는 저장소인지 API로 확인할 것)
- 각 모델에 대해 저장소 파일 목록을 조회해 **양자화(Q2_K ~ Q8_0, F16)별 파일 크기**를 표시
- **내 컴퓨터 적합도 배지**: 파일 크기 + 예상 KV 캐시 메모리와 사용 가능 메모리를 비교해
  ✅ 쾌적 / ⚠️ 가능하지만 느림 / ❌ 메모리 부족
- 기본 정렬: "내 컴퓨터에 추천" (적합 + 다운로드 수 + 최신순), 그 외 다운로드수/좋아요/최신순 정렬과 검색창 제공
- 모델 상세 패널: 설명(모델 카드 요약), 라이선스, 파라미터 수, 지원 문맥 길이, 추천 양자화 파일 하이라이트
- 분할 GGUF(`-00001-of-0000N.gguf`)와 비전 모델의 `mmproj` 파일도 함께 처리
- 네트워크 오류·API 속도 제한 시 친절한 메시지, 결과는 로컬 캐시(예: 6시간)

### 4. 다운로드 관리
- 목록에서 [다운로드] 클릭 → 백그라운드 다운로드(QThread), 진행률·속도·남은 시간 표시
- 일시정지/재개(이어받기), 취소, 동시 다운로드 수 제한
- 다운로드 전 디스크 공간 확인, 완료 후 파일 크기/해시 검증
- 비공개(gated) 모델은 허깅페이스 토큰 입력 안내 (설정에 저장, OS 키체인 사용 권장 `keyring`)
- 모델 저장 폴더 설정 가능 (기본: 사용자 홈 아래 `LlamaLauncher/models`)

### 5. 내 모델 탭
- 다운로드한 모델 목록(이름, 양자화, 크기, 다운로드 날짜, 적합도 배지)
- **[파일 직접 선택]** 버튼으로 로컬 `.gguf` 파일 추가, 폴더 스캔으로 기존 모델 가져오기 (LM Studio·Ollama 모델 폴더 자동 탐지도 시도)
- GGUF 메타데이터 읽기(`gguf` 파이썬 패키지): 아키텍처, 파라미터 수, 학습 문맥 길이, 레이어 수, 채팅 템플릿
- 삭제, 폴더 열기, 이름 변경

### 6. 실행 옵션 & 추천 옵션 (서버 탭)
모델을 선택하면 옵션 화면을 보여준다. **[🪄 이 컴퓨터에 맞게 자동 설정]** 버튼이 핵심.
- 옵션 목록 (llama-server 인자와 매핑, 각 옵션 옆에 ⓘ 툴팁으로 초보자용 설명):
  - 문맥 길이 `-c`, GPU 레이어 수 `-ngl`, CPU 스레드 `-t`, 배치 크기 `-b`/`-ub`
  - Flash Attention `-fa`, KV 캐시 양자화 `--cache-type-k/v`, mmap/mlock
  - 병렬 요청 수 `-np`, 호스트 `--host`, 포트 `--port`, API 키 `--api-key`
  - 샘플링 기본값: temperature, top-p, top-k, min-p, repeat penalty
  - 채팅 템플릿 `--jinja`, 비전 모델이면 `--mmproj`
  - 고급: 직접 인자 추가 입력란
- **추천 로직**(별도 모듈 `recommender.py`로 분리, 단위 테스트 작성):
  - GPU 메모리에 모델이 다 들어가면 `-ngl 999`, 아니면 남는 메모리 비율로 레이어 수 계산
  - 스레드 = 물리 코어 수(Apple Silicon은 성능 코어 수)
  - 문맥 길이 = (가용 메모리 − 모델 크기 − 여유분)으로 KV 캐시 크기를 역산해 4K/8K/16K/32K 중 선택, 학습 문맥 길이를 넘지 않게
  - Apple Silicon/최신 NVIDIA면 Flash Attention 켜기
  - 메모리가 빠듯하면 KV 캐시 q8_0 권장
- 프리셋: "빠르게", "균형", "긴 대화용" + 사용자 프리셋 저장/불러오기
- 실행될 **명령어 미리보기**(복사 가능) 표시

### 7. 서버 실행 & 모니터링
- [▶ 서버 시작] / [■ 중지] / [재시작], 상태 표시등
- `subprocess.Popen`으로 실행, 로그를 실시간 표시(색상 구분), 앱 종료 시 프로세스 정리 (Windows는 `CREATE_NEW_PROCESS_GROUP` 등 고려)
- `/health` 엔드포인트로 로딩 완료 감지, 로딩 진행 상태 표시
- 접속 정보 표시: `http://127.0.0.1:8080` (OpenAI 호환 API `/v1/chat/completions`), [브라우저에서 웹 UI 열기] 버튼
- 같은 네트워크 공유 옵션(`--host 0.0.0.0`) 선택 시 내 IP와 보안 경고 표시
- 토큰 생성 속도(tokens/s), 메모리 사용량 표시
- 포트 충돌 감지 후 빈 포트 제안
- 다른 런처(Ollama 등) 선택 시 해당 런처의 실행 방식으로 분기 (런처별 어댑터 클래스)

### 8. 간단한 채팅 테스트 탭
- 실행 중인 서버에 OpenAI 호환 API로 질문하고 스트리밍 응답 표시 → 초보자가 바로 동작 확인 가능
- 시스템 프롬프트, 대화 초기화

### 9. 초보자 배려
- 첫 실행 시 **마법사(Wizard)**: ① 사양 확인 → ② 런처 확인/설치 → ③ 추천 모델 1개 다운로드 → ④ 자동 설정으로 실행 → ⑤ 채팅 테스트
- 모든 오류 메시지는 원인 + 해결 방법을 쉬운 한국어로 (예: "메모리가 부족해요. 문맥 길이를 줄이거나 더 작은 양자화(Q4_K_M)를 선택해 보세요.")
- 용어 사전 도움말 (양자화, GGUF, 문맥 길이, GPU 레이어 등)

### 10. 외부 개발 도구 연동 탭 (🔌 연동)
실행 중인 로컬 모델을 **코딩 도구에서 바로 쓸 수 있도록** 도구별 설정을 자동으로 만들어 주는 탭.

#### 10-1. 탭 동작
- 왼쪽: 지원 도구 목록 (아이콘, 설치 감지 여부, 연동 상태 ✅/⚪)
- 오른쪽: 선택한 도구의
  1. **호환성 점검** — 현재 모델/서버 설정이 이 도구에 적합한지 검사 (아래 10-3 표 기준)
  2. **설정 내용 자동 생성** — 현재 서버의 `{BASE_URL}`, `{MODEL}`, `{API_KEY}`, `{CONTEXT}` 값을 채워 넣은 설정 표시 + [복사] 버튼
  3. **[설정 파일에 자동 적용]** — 설정 파일 위치를 OS별로 찾아서, **기존 파일을 백업**(`*.bak-날짜`)한 뒤 JSON/YAML/TOML을 **병합**(기존 설정을 덮어쓰지 않음). 적용 전에 바뀌는 내용(diff)을 미리 보여주고 확인받기
  4. **[원래대로 되돌리기]** — 백업에서 복구
  5. **[연결 테스트]** — 도구가 쓰는 API(`/v1/chat/completions`, `/v1/messages`, `/v1/responses`)로 실제 요청을 보내고, 도구 호출(tool calling) 응답까지 확인
  6. GUI에서만 설정하는 도구(Cline, Copilot 등)는 **그림이 있는 단계별 안내** + 입력할 값 복사 버튼
- 모델 이름이 바뀌지 않도록 서버 실행 시 **`--alias {MODEL}`** 을 항상 붙인다 (예: `local-coder`)
- 연동용 서버 프리셋 **"코딩 에이전트용"**: `--jinja`, `-c 65536`(메모리가 허용하면), `-np 1`, Flash Attention 켜기, `--api-key` 자동 생성
- 도구별 설정 형식은 자주 바뀌므로 **템플릿을 `app/data/integrations/*.json|yaml|toml` 파일로 분리**하고, 구현 시점에 각 도구의 공식 문서를 확인해 최신 형식으로 맞출 것. 템플릿마다 "확인한 날짜 / 공식 문서 URL"을 기록한다.

#### 10-2. 도구별 설정 방법 (템플릿 초안)

> 아래 `{...}` 는 앱이 자동으로 채운다. 기본값: `{BASE_URL}` = `http://127.0.0.1:8080`, `{MODEL}` = `--alias` 값

**① Claude Code (CLI + VS Code / JetBrains 확장)**
- Anthropic Messages API(`/v1/messages`)가 필요하다. 런처가 이를 지원하는지 먼저 검사한다 (최신 llama-server는 지원; 지원하지 않는 런처면 앱 안에서 **LiteLLM 프록시**를 띄워 변환하는 옵션 제공)
- VS Code 확장도 CLI와 같은 `~/.claude/settings.json`(Windows: `%USERPROFILE%\.claude\settings.json`)을 읽는다
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "{BASE_URL}",
    "ANTHROPIC_AUTH_TOKEN": "{API_KEY}",
    "ANTHROPIC_MODEL": "{MODEL}",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "{MODEL}",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```
- 주의 안내: 시스템 프롬프트와 도구 정의가 길어서 **문맥 길이 64K 이상 권장**. 연동을 해제하면 원래 Claude 계정으로 돌아가도록 되돌리기 버튼 강조

**② OpenCode** — `~/.config/opencode/opencode.json` (프로젝트별: `./opencode.json`)
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamalauncher": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "라마런처 (로컬)",
      "options": { "baseURL": "{BASE_URL}/v1", "apiKey": "{API_KEY}" },
      "models": {
        "{MODEL}": { "name": "{MODEL}", "limit": { "context": {CONTEXT}, "output": 8192 } }
      }
    }
  },
  "model": "llamalauncher/{MODEL}"
}
```

**③ Continue (VS Code / JetBrains)** — `~/.continue/config.yaml`
```yaml
name: LlamaLauncher
version: 1.0.0
schema: v1
models:
  - name: 라마런처 채팅
    provider: openai
    model: {MODEL}
    apiBase: {BASE_URL}/v1
    apiKey: {API_KEY}
    roles: [chat, edit, apply]
  - name: 라마런처 자동완성
    provider: openai
    model: {AUTOCOMPLETE_MODEL}
    apiBase: {AUTOCOMPLETE_BASE_URL}/v1
    roles: [autocomplete]
```
- 자동완성은 작은 FIM 지원 모델(예: Qwen2.5-Coder 1.5B~7B)을 **두 번째 서버(다른 포트)** 로 띄우는 옵션 제공

**④ Cline / Roo Code / Kilo Code (VS Code 확장, GUI 설정)**
- 안내: 확장 설정 → API Provider: **OpenAI Compatible** → Base URL `{BASE_URL}/v1`, API Key `{API_KEY}`, Model ID `{MODEL}`, Context Window `{CONTEXT}`
- Ollama/LM Studio 런처 사용 시에는 해당 전용 Provider 선택 안내

**⑤ GitHub Copilot (VS Code, 모델 직접 연결 BYOK)**
- 안내: Copilot 채팅 → 모델 선택 → **Manage Models** → Ollama 또는 OpenAI Compatible 선택 → 주소 `{BASE_URL}` 입력
- 플랜/버전에 따라 지원 여부가 다르므로 "사용 불가 시" 안내 포함

**⑥ Aider** — `~/.aider.conf.yml` 또는 환경변수
```yaml
openai-api-base: {BASE_URL}/v1
openai-api-key: {API_KEY}
model: openai/{MODEL}
```
- `.aider.model.settings.yml`에 문맥 길이·편집 형식(`diff`/`whole`)도 함께 생성

**⑦ OpenAI Codex CLI** — `~/.codex/config.toml`
```toml
model = "{MODEL}"
model_provider = "llamalauncher"

[model_providers.llamalauncher]
name = "LlamaLauncher"
base_url = "{BASE_URL}/v1"
env_key = "LLAMALAUNCHER_API_KEY"
wire_api = "responses"   # 서버가 /v1/responses 미지원이면 "chat" (Codex 버전에 따라 확인)
```

**⑧ Zed 에디터** — `settings.json`의 `language_models` 에 OpenAI 호환 제공자 추가 (`api_url: {BASE_URL}/v1`, `available_models`에 `{MODEL}`, `max_tokens: {CONTEXT}`, tools 지원 여부)

**⑨ JetBrains AI Assistant** — 설정 → Tools → AI Assistant → Models → 로컬 모델(Ollama/LM Studio/OpenAI 호환) 활성화, 주소 `{BASE_URL}` 안내

**⑩ Cursor** — Settings → Models → OpenAI API Key + **Override Base URL**. 단 Cursor는 서버를 거쳐 요청하므로 `localhost`는 안 됨 → **cloudflared / ngrok 터널** 실행 옵션과 보안 경고(API 키 필수) 제공

**⑪ Open WebUI / 기타 채팅 앱** — 관리자 설정 → Connections → OpenAI API: `{BASE_URL}/v1`, 키 `{API_KEY}`

**⑫ 범용(LangChain, OpenAI SDK 등)** — 파이썬/JS 예제 코드 스니펫 생성
```python
from openai import OpenAI
client = OpenAI(base_url="{BASE_URL}/v1", api_key="{API_KEY}")
print(client.chat.completions.create(model="{MODEL}", messages=[{"role": "user", "content": "안녕"}]).choices[0].message.content)
```

#### 10-3. 모델별 도구 적합성 (호환성 점검 기준)
앱은 아래 기준으로 현재 모델을 검사하고, 맞지 않으면 대안 모델을 "모델 찾기" 탭에서 바로 보여준다.

| 도구 유형 | 해당 도구 | 모델 필수 조건 | 권장 문맥 길이 | 추천 모델 예시 (구현 시 HF API로 최신 확인) |
|---|---|---|---|---|
| 코딩 에이전트 | Claude Code, OpenCode, Cline, Roo, Codex, Copilot 에이전트 | **도구 호출(tool calling)** 지원 + 채팅 템플릿(`--jinja`) | 64K 이상 (최소 32K) | Qwen3-Coder 계열, gpt-oss, Devstral, GLM 계열 |
| 편집/채팅 | Continue 채팅, Aider, Zed, JetBrains | 지시 따르기(Instruct) 모델 | 16K~32K | Qwen-Coder Instruct, Gemma, Llama Instruct |
| 자동완성 | Continue autocomplete | **FIM(중간 채우기)** 지원, 빠른 응답 | 4K~8K | Qwen2.5-Coder 1.5B/7B (Base) |
| 일반 채팅 | Open WebUI 등 | 제한 없음 | 8K~ | 카테고리별 추천 모델 |

- 판별 방법: GGUF 메타데이터의 채팅 템플릿에 tool 관련 구문이 있는지, 모델 카드 태그(`tool-use`, `function-calling`), FIM 토큰(`<|fim_prefix|>` 등) 존재 여부, 그리고 [연결 테스트]의 실제 도구 호출 결과
- **사양별 추천**: 메모리 16GB 이하 → 에이전트 용도는 "느리거나 품질이 낮을 수 있음" 경고 + 7B~14B Q4 추천 / 32GB → 20B~30B급(MoE 우선) / 48GB 이상 → 30B급 Q6~Q8 또는 더 큰 MoE
- 각 도구 화면에 "이 컴퓨터 + 이 도구에 가장 추천하는 모델 1개"를 [다운로드하고 바로 연동] 버튼으로 제공

## 아키텍처

```
llamalauncher/
  main.py                 # 진입점
  app/
    ui/                   # 탭별 위젯 (dashboard, launchers, browse, downloads, my_models, server, chat, wizard)
    core/
      hardware.py         # 사양 감지 (OS별 구현)
      recommender.py      # 추천 옵션/적합도 계산
      hf_client.py        # 허깅페이스 검색·파일목록·캐시
      downloader.py       # 이어받기 다운로드 매니저
      model_registry.py   # 로컬 모델 목록·GGUF 메타데이터
      settings.py         # 설정 저장
      launchers/
        base.py           # Launcher 추상 클래스 (detect/install/start/stop/supported_formats)
        llamacpp.py, ollama.py, lmstudio.py, koboldcpp.py, mlx.py, vllm.py ...
      integrations/
        base.py           # Integration 추상 클래스 (detect/config_path/render/apply/revert/test)
        claude_code.py, opencode.py, continue_dev.py, cline.py, aider.py, codex.py, zed.py, cursor.py ...
        config_merge.py   # JSON/YAML/TOML 병합 + 백업/복구
    data/curated_models.json
    data/integrations/    # 도구별 설정 템플릿 (확인 날짜·공식 문서 URL 포함)
  tests/
  requirements.txt
  build_mac.sh, build_windows.bat
  README.md
```

- UI 스레드를 막지 않도록 네트워크·설치·다운로드·프로세스 I/O는 모두 워커 스레드 + 시그널 사용
- 런처는 플러그인처럼 추가 가능하게 공통 인터페이스로 설계
- 로그는 파일로도 저장 (문제 신고용)

## 개발 진행 방식

1. 먼저 전체 설계와 파일 구조를 제시하고 확인받은 뒤 구현한다.
2. 단계별로 구현하고 각 단계마다 실제 실행해 확인한다:
   - 1단계: 사양 감지 + 추천 로직 + 테스트
   - 2단계: llama.cpp 런처 감지/실행 + 서버 탭 + 옵션/자동 설정
   - 3단계: 허깅페이스 검색(카테고리, 적합도) + 다운로드
   - 4단계: 내 모델 탭 + 파일 직접 선택 + 채팅 테스트
   - 5단계: 다른 런처 어댑터 + 자동 설치
   - 6단계: 외부 개발 도구 연동 탭 (Claude Code·OpenCode·Continue 먼저, 나머지 순차) + 설정 병합/백업 테스트
   - 7단계: 첫 실행 마법사, 다국어/테마 정리, PyInstaller 빌드
3. Windows 전용 코드는 macOS에서 실행할 수 없으므로 분기를 명확히 하고, 테스트 시 모킹(mock)으로 검증한다.
4. README에 설치·실행·빌드 방법을 한국어로 작성한다.
