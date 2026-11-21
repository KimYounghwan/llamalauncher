"""llama-server 로그 해석: 줄 분류, 속도(tokens/s) 추출, 오류를 쉬운 말로 설명."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_TPS_RE = re.compile(r"(prompt eval|eval) time\s*=.*?([\d.]+)\s*tokens per second", re.I)


@dataclass
class SpeedSample:
    kind: str  # "prompt" | "generate"
    tokens_per_sec: float


def classify(line: str) -> str:
    """'error' | 'warn' | 'ok' | 'info'"""
    low = line.lower()
    if any(k in low for k in ("error", "failed", "out of memory", "exception", "abort")):
        return "error"
    if "warn" in low:
        return "warn"
    if "server is listening" in low or "model loaded" in low or "all slots are idle" in low:
        return "ok"
    return "info"


def parse_speed(line: str) -> Optional[SpeedSample]:
    m = _TPS_RE.search(line)
    if not m:
        return None
    kind = "prompt" if m.group(1).lower().startswith("prompt") else "generate"
    try:
        return SpeedSample(kind, float(m.group(2)))
    except ValueError:
        return None


_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("out of memory", "failed to allocate", "unable to allocate", "cudamalloc failed",
      "insufficient memory", "failed to create context", "kiogpucommandbuffercallbackerroroutofmemory"),
     "메모리가 부족해요. 문맥 길이를 줄이거나, GPU 레이어 수를 낮추거나, 더 작은 양자화(Q4_K_M 등) 모델을 선택해 보세요. "
     "[🪄 자동 설정]을 누르면 안전한 값으로 맞춰 드려요."),
    (("couldn't bind", "address already in use", "failed to bind", "bind failed"),
     "포트가 이미 사용 중이에요. 다른 프로그램(또는 이미 실행 중인 서버)이 같은 포트를 쓰고 있어요. 포트 번호를 바꿔 보세요."),
    (("unknown model architecture", "unknown architecture"),
     "설치된 llama.cpp가 이 모델 종류를 아직 지원하지 않아요. llama.cpp를 최신 버전으로 업데이트해 보세요."),
    (("failed to load model", "error loading model", "invalid magic", "gguf_init_from_file", "tensor data is not within"),
     "모델 파일을 열 수 없어요. 다운로드가 덜 됐거나 손상된 파일일 수 있어요. 다시 받아 보세요."),
    (("invalid argument", "unknown argument", "error: invalid", "unrecognized option"),
     "설치된 llama.cpp 버전이 지원하지 않는 옵션이 있어요. 고급 옵션의 직접 입력 인자를 확인하거나 llama.cpp를 업데이트해 보세요."),
    (("no such file", "cannot open", "permission denied"),
     "파일을 찾을 수 없거나 열 권한이 없어요. 모델 파일 경로를 확인해 주세요."),
]


def explain_error(lines: list[str]) -> Optional[str]:
    """최근 로그에서 원인을 찾아 해결 방법을 돌려준다."""
    text = "\n".join(lines).lower()
    for keys, hint in _HINTS:
        if any(k in text for k in keys):
            return hint
    return None
