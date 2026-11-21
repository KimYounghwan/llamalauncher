"""포트/주소 관련 도우미."""

from __future__ import annotations

import secrets
import socket
import sys
from typing import Optional


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    bind_host = "0.0.0.0" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != "win32":
            # llama-server처럼 SO_REUSEADDR를 켜야 방금 닫힌 연결(TIME_WAIT)을 '사용 중'으로 오인하지 않는다.
            # (Windows에서는 이 옵션이 남의 포트를 가로채는 의미라 쓰지 않는다)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_host, port))
            s.listen(1)
        except OSError:
            return False
    return True


def find_free_port(start: int = 8080, host: str = "127.0.0.1", tries: int = 50) -> Optional[int]:
    for p in range(start, min(start + tries, 65536)):
        if is_port_free(p, host):
            return p
    return None


def lan_ip() -> Optional[str]:
    """같은 네트워크의 다른 기기가 접속할 때 쓸 내 IP (실제 패킷은 보내지 않음)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") else ip


def client_host(host: str) -> str:
    """서버가 0.0.0.0에서 듣고 있어도 이 컴퓨터에서는 127.0.0.1로 접속한다."""
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


def generate_api_key() -> str:
    return "sk-local-" + secrets.token_urlsafe(24)
