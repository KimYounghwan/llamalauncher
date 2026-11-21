"""실행 중인 서버에 OpenAI 호환 API(/v1/chat/completions)로 질문하고 스트리밍 응답을 받는다."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests


@dataclass
class ChatEvent:
    kind: str                 # "content" | "reasoning" | "done" | "error"
    text: str = ""
    stats: dict = field(default_factory=dict)


@dataclass
class ChatRequest:
    base_url: str             # http://127.0.0.1:8080 (뒤에 /v1 없이)
    messages: list[dict]
    model: str = "local-model"
    api_key: str = ""
    temperature: Optional[float] = None   # None이면 서버 기본값
    max_tokens: Optional[int] = None

    def payload(self) -> dict:
        body: dict = {"model": self.model or "local-model", "messages": self.messages, "stream": True,
                      "stream_options": {"include_usage": True}}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return body


def parse_sse_line(line: str) -> Optional[dict]:
    """'data: {...}' 한 줄을 dict로. 끝 표시([DONE])는 {'done': True}."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def chunk_events(chunk: dict) -> list[ChatEvent]:
    out: list[ChatEvent] = []
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        r = delta.get("reasoning_content") or delta.get("reasoning")
        if r:
            out.append(ChatEvent("reasoning", r))
        c = delta.get("content")
        if c:
            out.append(ChatEvent("content", c))
    return out


def friendly_error(status: Optional[int], body: str, exc: Optional[Exception] = None) -> str:
    if isinstance(exc, requests.ConnectionError):
        return "서버에 연결할 수 없어요. [서버] 탭에서 서버가 켜져 있는지 확인해 주세요."
    if isinstance(exc, requests.Timeout):
        return "서버가 너무 오래 응답하지 않아요. 모델이 아직 로딩 중이거나 컴퓨터가 바쁠 수 있어요."
    msg = ""
    try:
        err = json.loads(body).get("error")
        msg = err.get("message", "") if isinstance(err, dict) else str(err or "")
    except (ValueError, AttributeError):
        msg = body[:300]
    low = msg.lower()
    if status == 401:
        return "API 키가 맞지 않아요. 서버 탭에 입력한 API 키를 확인해 주세요."
    if "context" in low and ("exceed" in low or "too long" in low or "size" in low):
        return ("대화가 너무 길어져서 문맥 길이를 넘었어요. [🧹 대화 초기화]를 누르거나 "
                "서버 탭에서 문맥 길이를 늘려 주세요.")
    if status == 503:
        return "모델을 아직 불러오는 중이에요. 잠시 후 다시 보내 주세요."
    if status:
        return f"서버 오류({status}): {msg or '알 수 없는 오류'}"
    return f"요청 실패: {exc or msg}"


def stream_chat(req: ChatRequest, on_event: Callable[[ChatEvent], None],
                stop: Optional[threading.Event] = None, session=None, timeout: float = 600.0) -> None:
    """응답이 오는 대로 on_event를 호출한다 (호출한 스레드에서). 마지막에는 반드시 done 또는 error."""
    stop = stop or threading.Event()
    sess = session or requests
    headers = {"Content-Type": "application/json"}
    if req.api_key:
        headers["Authorization"] = f"Bearer {req.api_key}"
    started = time.monotonic()
    first_token_at: Optional[float] = None
    n_chunks = 0
    stats: dict = {}
    try:
        resp = sess.post(req.base_url.rstrip("/") + "/v1/chat/completions", json=req.payload(),
                         headers=headers, stream=True, timeout=(5, timeout))
    except requests.RequestException as e:
        on_event(ChatEvent("error", friendly_error(None, "", e)))
        return
    try:
        if resp.status_code != 200:
            on_event(ChatEvent("error", friendly_error(resp.status_code, resp.text)))
            return
        # 바이트로 줄을 나눈 뒤 UTF-8로 직접 디코딩한다. text/event-stream에 charset이 없으면
        # requests가 ISO-8859-1로 읽어 한글이 깨지기 때문.
        for raw in resp.iter_lines():
            if stop.is_set():
                stats["stopped"] = True
                break
            if not raw:
                continue
            chunk = parse_sse_line(raw.decode("utf-8", errors="replace"))
            if chunk is None:
                continue
            if chunk.get("done"):
                break
            if "error" in chunk:
                on_event(ChatEvent("error", friendly_error(None, json.dumps(chunk))))
                return
            for ev in chunk_events(chunk):
                if first_token_at is None:
                    first_token_at = time.monotonic()
                n_chunks += 1
                on_event(ev)
            if chunk.get("timings"):
                stats["timings"] = chunk["timings"]
            if chunk.get("usage"):
                stats["usage"] = chunk["usage"]
    except requests.RequestException as e:
        if not stop.is_set():
            on_event(ChatEvent("error", friendly_error(None, "", e)))
            return
        stats["stopped"] = True
    finally:
        resp.close()
    elapsed = time.monotonic() - started
    stats["elapsed"] = elapsed
    stats["first_token"] = (first_token_at - started) if first_token_at else None
    t = stats.get("timings") or {}
    if t.get("predicted_per_second"):
        stats["tps"] = float(t["predicted_per_second"])
    elif first_token_at and n_chunks > 1:
        stats["tps"] = n_chunks / max(1e-6, time.monotonic() - first_token_at)  # 대략값
    on_event(ChatEvent("done", stats=stats))
