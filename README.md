# 라마런처 (LlamaLauncher)

초보자도 내 컴퓨터에서 로컬 AI 모델을 쉽게 받아서 실행할 수 있게 해 주는 파이썬 GUI 프로그램입니다.
llama.cpp를 기본 엔진으로 사용하며 macOS와 Windows에서 동작합니다. 전체 기획은 [PROMPT.md](PROMPT.md)를 참고하세요.

## 진행 상황

- [x] 1단계: 시스템 사양 감지 + 추천 옵션 계산 + 테스트 + 대시보드 화면
- [x] 2단계: llama.cpp 런처 감지/실행 + 서버 탭 + 옵션/자동 설정 (+ GGUF 메타데이터 읽기)
- [x] 3단계: 허깅페이스 모델 검색(카테고리·적합도·추천 파일) + 다운로드(이어받기·검증)
- [ ] 4단계: 내 모델 + 파일 직접 선택 + 채팅 테스트
- [ ] 5단계: 다른 런처 어댑터 + 자동 설치
- [ ] 6단계: 외부 개발 도구 연동
- [ ] 7단계: 첫 실행 마법사, 빌드

## 실행 방법

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

## 테스트
```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

## 구조
```
llamalauncher/
  main.py                    # 진입점
  app/core/hardware.py       # 사양 감지 (macOS: sysctl/system_profiler, Windows: 레지스트리/nvidia-smi, Linux: nvidia-smi/rocm-smi)
  app/core/recommender.py    # 메모리 계산, 적합도(쾌적/느림/부족), llama-server 추천 옵션
  app/core/settings.py       # 설정 저장 (OS별 사용자 설정 폴더)
  app/core/gguf.py           # GGUF 헤더(층 수, 최대 문맥, 채팅 템플릿) 읽기
  app/core/launchers/        # 런처 감지 (llamacpp.py: 경로 탐색, 버전, 지원 옵션 확인)
  app/core/server.py         # llama-server 실행/중지/상태 감시 (QProcess)
  app/core/server_log.py     # 로그 분류, 속도 추출, 오류를 쉬운 말로 설명
  app/core/netutil.py        # 포트 확인, 내 IP, API 키 생성
  app/core/hf_client.py      # 허깅페이스 검색, 카테고리, 파일(양자화) 분석, 추천 파일, 캐시
  app/core/downloader.py     # 다운로드 관리 (이어받기, 일시정지, SHA-256 검증, 동시 다운로드 제한)
  app/core/secrets_store.py  # 허깅페이스 토큰을 OS 키체인에 저장
  app/data/catalog.json      # 카테고리 정의, 검증된 제작자, 숨길 키워드 (앱 업데이트 없이 수정 가능)
  app/ui/                    # PySide6 화면
tests/                       # Windows/Linux 출력은 가짜 데이터(mock)로 검증
```
