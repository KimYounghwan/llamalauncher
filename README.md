# 라마런처 (LlamaLauncher)

초보자도 내 컴퓨터에서 로컬 AI 모델을 쉽게 받아서 실행할 수 있게 해 주는 파이썬 GUI 프로그램입니다.
llama.cpp를 기본 엔진으로 사용하며 macOS와 Windows에서 동작합니다. 전체 기획은 [PROMPT.md](PROMPT.md)를 참고하세요.

## 진행 상황

- [x] 1단계: 시스템 사양 감지 + 추천 옵션 계산 + 테스트 + 대시보드 화면
- [ ] 2단계: llama.cpp 런처 감지/실행 + 서버 탭 + 옵션/자동 설정
- [ ] 3단계: 허깅페이스 모델 검색 + 다운로드
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
  app/ui/                    # PySide6 화면
tests/                       # Windows/Linux 출력은 가짜 데이터(mock)로 검증
```
