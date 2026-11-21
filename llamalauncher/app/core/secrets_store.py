"""허깅페이스 토큰 저장: OS 키체인(macOS 키체인, Windows 자격 증명 관리자) 우선."""

from __future__ import annotations

import os
from typing import Optional

SERVICE = "LlamaLauncher"
USER = "huggingface"


def get_hf_token() -> Optional[str]:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    try:
        import keyring

        tok = keyring.get_password(SERVICE, USER)
        return tok.strip() if tok else None
    except Exception:  # noqa: BLE001 — 키체인을 못 쓰는 환경
        return None


def set_hf_token(token: Optional[str]) -> bool:
    """저장 성공 여부를 돌려준다. None/빈 값이면 삭제."""
    try:
        import keyring

        if token:
            keyring.set_password(SERVICE, USER, token.strip())
        else:
            try:
                keyring.delete_password(SERVICE, USER)
            except Exception:  # noqa: BLE001 — 원래 없던 경우
                pass
        return True
    except Exception:  # noqa: BLE001
        return False
