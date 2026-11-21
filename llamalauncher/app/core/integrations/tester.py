"""연결 테스트: 도구가 쓰는 API로 실제 요청을 보내 동작을 확인한다 (도구 호출 포함)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOL_PROMPT = "What's the weather in Seoul right now? Use the get_weather tool."


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str
    seconds: float = 0.0

    @property
    def icon(self) -> str:
        return "✅" if self.ok else "❌"


def _err(r: requests.Response) -> str:
    try:
        e = r.json().get("error")
        msg = e.get("message") if isinstance(e, dict) else str(e)
    except (ValueError, AttributeError):
        msg = r.text[:200]
    hint = {401: " — API 키가 맞지 않아요.", 404: " — 이 엔진은 이 API를 지원하지 않아요.",
            501: " — 이 엔진은 이 기능을 지원하지 않아요."}.get(r.status_code, "")
    return f"HTTP {r.status_code}: {msg}{hint}"


class Tester:
    def __init__(self, base_url: str, model: str, api_key: str = "", session=None, timeout: float = 180):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = api_key
        self.s = session or requests.Session()
        self.timeout = timeout

    def _headers(self, anthropic: bool = False) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
            if anthropic:
                h["x-api-key"] = self.key
        if anthropic:
            h["anthropic-version"] = "2023-06-01"
        return h

    def _run(self, name: str, fn: Callable[[], tuple[bool, str]]) -> TestResult:
        t = time.monotonic()
        try:
            ok, detail = fn()
        except requests.ConnectionError:
            ok, detail = False, "서버에 연결할 수 없어요. 서버가 켜져 있는지 확인해 주세요."
        except requests.Timeout:
            ok, detail = False, "응답이 너무 오래 걸려요 (모델 로딩 중이거나 너무 큰 모델일 수 있어요)."
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            ok, detail = False, f"오류: {e}"
        return TestResult(name, ok, detail, time.monotonic() - t)

    def _post(self, path: str, body: dict, anthropic: bool = False) -> requests.Response:
        return self.s.post(self.base + path, json=body, headers=self._headers(anthropic), timeout=self.timeout)

    # ---- 개별 테스트 ---------------------------------------------------
    def models(self) -> TestResult:
        def fn():
            r = self.s.get(self.base + "/v1/models", headers=self._headers(), timeout=10)
            if r.status_code != 200:
                return False, _err(r)
            ids = [m.get("id") for m in r.json().get("data", [])]
            return True, "모델 목록: " + (", ".join(map(str, ids[:5])) or "(비어 있음)")
        return self._run("모델 목록 (/v1/models)", fn)

    def chat(self) -> TestResult:
        def fn():
            r = self._post("/v1/chat/completions", {"model": self.model, "max_tokens": 64,
                                                    "messages": [{"role": "user", "content": "Say 'ok' only."}]})
            if r.status_code != 200:
                return False, _err(r)
            msg = r.json()["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            return True, f"답변: {text[:60]!r}"
        return self._run("채팅 (/v1/chat/completions)", fn)

    def chat_tools(self) -> TestResult:
        def fn():
            r = self._post("/v1/chat/completions", {"model": self.model, "max_tokens": 512, "tools": [WEATHER_TOOL],
                                                    "messages": [{"role": "user", "content": TOOL_PROMPT}]})
            if r.status_code != 200:
                return False, _err(r) + (" (서버 탭에서 --jinja를 켜야 할 수 있어요)" if r.status_code in (400, 500) else "")
            calls = r.json()["choices"][0]["message"].get("tool_calls") or []
            if not calls:
                return False, "모델이 도구를 호출하지 않았어요. 코딩 에이전트에서 제대로 동작하지 않을 수 있어요."
            f = calls[0]["function"]
            return True, f"도구 호출 성공: {f['name']}({f.get('arguments', '')})"
        return self._run("도구 호출 (OpenAI 방식)", fn)

    def messages(self, tools: bool = False) -> TestResult:
        def fn():
            # 최신 Claude Code처럼 대화 중간에 system 메시지를 넣어 템플릿 호환성까지 확인한다
            body = {"model": self.model, "max_tokens": 512, "system": "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": TOOL_PROMPT if tools else "Say 'ok' only."},
                                 {"role": "system", "content": "Answer briefly."}]}
            if tools:
                f = WEATHER_TOOL["function"]
                body["tools"] = [{"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}]
            r = self._post("/v1/messages", body, anthropic=True)
            if r.status_code != 200:
                if "System message must be at the beginning" in r.text:
                    return False, ("이 모델의 채팅 템플릿이 대화 중간의 system 메시지를 거부해요. "
                                   "설정의 주소가 라마런처 호환 주소(서버 포트 + 1000)인지 확인해 주세요.")
                return False, _err(r)
            content = r.json().get("content") or []
            if tools:
                use = next((c for c in content if c.get("type") == "tool_use"), None)
                if not use:
                    return False, "모델이 도구를 호출하지 않았어요."
                return True, f"도구 호출 성공: {use['name']}({json.dumps(use.get('input'), ensure_ascii=False)})"
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
            return True, f"답변: {text[:60]!r}"
        return self._run("Anthropic API" + (" 도구 호출" if tools else "") + " (/v1/messages)", fn)

    def responses(self) -> TestResult:
        def fn():
            r = self._post("/v1/responses", {"model": self.model, "input": "Say 'ok' only.", "max_output_tokens": 64})
            if r.status_code != 200:
                return False, _err(r)
            d = r.json()
            text = d.get("output_text") or ""
            if not text:
                for item in d.get("output", []):
                    if item.get("type") not in (None, "message"):
                        continue  # 생각 과정(reasoning) 항목은 건너뛴다
                    for c in item.get("content", []) or []:
                        text += c.get("text", "")
            return True, f"답변: {text.strip()[:60]!r}"
        return self._run("Responses API (/v1/responses)", fn)

    # ---- 도구별 묶음 ---------------------------------------------------
    def for_api(self, api: str, needs_tools: bool, on_result: Optional[Callable[[TestResult], None]] = None
                ) -> list[TestResult]:
        steps: list[Callable[[], TestResult]] = [self.models]
        if api == "messages":
            steps += [self.messages] + ([lambda: self.messages(tools=True)] if needs_tools else [])
        elif api == "responses":
            steps += [self.responses, self.chat] + ([self.chat_tools] if needs_tools else [])
        else:
            steps += [self.chat] + ([self.chat_tools] if needs_tools else [])
        results = []
        for st in steps:
            res = st()
            results.append(res)
            if on_result:
                on_result(res)
            if st == self.models and not res.ok and "연결할 수 없어요" in res.detail:
                break
        return results
