"""비밀 값 저장: OS 키체인(macOS 키체인, Windows 자격 증명 관리자) 우선.

- 허깅페이스 토큰
- 서버 API 키: 외부 도구(Claude Code 등)에 적용한 키가 앱을 다시 켜도 바뀌지 않도록 보관한다
"""

from __future__ import annotations

import os
from typing import Optional

SERVICE = "LlamaLauncher"
USER = "huggingface"
SERVER_KEY_USER = "server-api-key"


def _get(user: str) -> Optional[str]:
    try:
        import keyring

        v = keyring.get_password(SERVICE, user)
        return v.strip() if v else None
    except Exception:  # noqa: BLE001 — 키체인을 못 쓰는 환경
        return None


def _set(user: str, value: Optional[str]) -> bool:
    """저장 성공 여부를 돌려준다. None/빈 값이면 삭제."""
    try:
        import keyring

        if value:
            keyring.set_password(SERVICE, user, value.strip())
        else:
            try:
                keyring.delete_password(SERVICE, user)
            except Exception:  # noqa: BLE001 — 원래 없던 경우
                pass
        return True
    except Exception:  # noqa: BLE001
        return False


def get_hf_token() -> Optional[str]:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    return _get(USER)


def set_hf_token(token: Optional[str]) -> bool:
    return _set(USER, token)


def get_server_api_key() -> Optional[str]:
    return _get(SERVER_KEY_USER)


def set_server_api_key(key: Optional[str]) -> bool:
    return _set(SERVER_KEY_USER, key)
