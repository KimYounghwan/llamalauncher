"""Claude Code 호환 프록시.

최신 Claude Code는 대화 중간에 role이 "system"인 메시지를 보낸다. Qwen 계열 등 많은 모델의 채팅 템플릿은
system 메시지가 맨 앞에만 있어야 해서 "System message must be at the beginning" 오류를 낸다 (실제 확인).
이 프록시는 /v1/messages 요청에서 중간 system 메시지를 <system-reminder> 사용자 메시지로 바꿔 서버에 넘긴다.
그 밖의 요청은 그대로 전달한다 (스트리밍 포함).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import requests

HOP = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "accept-encoding"}


def _blocks(content) -> list:
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def rewrite_messages(body: dict) -> dict:
    """중간 system 메시지를 user 메시지로 바꾸고, 이어지는 같은 역할 메시지를 합친다."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not any(m.get("role") == "system" for m in msgs if isinstance(m, dict)):
        return body
    out: list[dict] = []
    for m in msgs:
        if m.get("role") == "system":
            text = "".join(b.get("text", "") for b in _blocks(m.get("content")) if isinstance(b, dict))
            m = {"role": "user", "content": [{"type": "text", "text": f"<system-reminder>\n{text}\n</system-reminder>"}]}
        if out and out[-1]["role"] == m["role"] == "user":
            out[-1] = {"role": "user", "content": _blocks(out[-1]["content"]) + _blocks(m["content"])}
        else:
            out.append(m)
    return {**body, "messages": out}


class _Handler(BaseHTTPRequestHandler):
    upstream = ""
    session: requests.Session

    def _forward(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if body and method == "POST" and self.path.split("?")[0] == "/v1/messages":
            try:
                body = json.dumps(rewrite_messages(json.loads(body))).encode()
            except (ValueError, AttributeError):
                pass
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        try:
            r = self.session.request(method, self.upstream + self.path, data=body, headers=headers, stream=True,
                                     timeout=(5, 3600))
        except requests.RequestException as e:
            msg = json.dumps({"type": "error", "error": {"type": "api_error",
                                                         "message": f"라마런처 서버에 연결할 수 없어요: {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(r.status_code)
        for k, v in r.headers.items():
            if k.lower() not in HOP and k.lower() != "content-encoding":
                self.send_header(k, v)
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for chunk in r.raw.stream(8192, decode_content=True):
                self.wfile.write(chunk)
                self.wfile.flush()
        except OSError:
            pass
        finally:
            r.close()

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def do_HEAD(self):
        self._forward("HEAD")

    def log_message(self, *a):
        pass


class CompatProxy:
    def __init__(self, upstream: str, host: str = "127.0.0.1", port: int = 9080):
        self.upstream = upstream.rstrip("/")
        self.host, self.port = host, port
        self._srv: Optional[ThreadingHTTPServer] = None

    @property
    def url(self) -> str:
        return f"http://{'127.0.0.1' if self.host == '0.0.0.0' else self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._srv is not None

    def start(self) -> None:
        if self._srv:
            return
        handler = type("H", (_Handler,), {"upstream": self.upstream, "session": requests.Session()})
        self._srv = ThreadingHTTPServer((self.host, self.port), handler)
        self._srv.daemon_threads = True
        threading.Thread(target=self._srv.serve_forever, daemon=True, name="compat-proxy").start()

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
