"""외부 개발 도구 연동: 템플릿(app/data/integrations/*.json)을 읽어 설정을 만들고 적용/되돌린다.

템플릿 문법
- 문자열 안의 {BASE_URL} {API_BASE} {MODEL} {MODEL_NAME} {API_KEY} {API_KEY_OR_DUMMY} {CONTEXT} {OUTPUT} 치환
- 값 전체가 "{CONTEXT}"/"{OUTPUT}"이면 숫자, "{TOOLS}"/"{VISION}"이면 true/false로 바뀐다
- {"$if": "api_key"|"tools", ...}: 조건이 거짓이면 그 항목을 뺀다.
  "$value"가 있으면 그 값으로, "$list"가 있으면 그 목록으로 바뀌고, 아니면 "$if"만 지운 객체가 된다
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import config_merge as cm

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "integrations"
DUMMY_KEY = "sk-local"
CATEGORY_LABELS = {"agent": "코딩 에이전트", "edit": "편집·채팅", "autocomplete": "자동완성", "chat": "일반 채팅"}
_DROP = object()


@dataclass
class Context:
    """현재 서버 정보 (설정에 채워 넣을 값)."""
    base_url: str = "http://127.0.0.1:8080"
    model: str = "local-model"
    api_key: str = ""
    context: int = 4096
    running: bool = False
    engine_id: str = "llamacpp"
    engine_name: str = "llama.cpp"
    model_name: str = ""
    supports_tools: Optional[bool] = None   # None: 모름
    vision: bool = False
    jinja: bool = True
    host: str = "127.0.0.1"
    total_ram: int = 0
    anthropic_base_url: str = ""            # Claude Code 호환 주소 (없으면 base_url)

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/") + "/v1"

    @property
    def output(self) -> int:
        return max(1024, min(8192, self.context // 4))

    def values(self) -> dict[str, str]:
        return {"{BASE_URL}": self.base_url.rstrip("/"), "{API_BASE}": self.api_base, "{MODEL}": self.model,
                "{MODEL_NAME}": self.model_name or self.model, "{API_KEY}": self.api_key,
                "{API_KEY_OR_DUMMY}": self.api_key or DUMMY_KEY, "{CONTEXT}": str(self.context),
                "{CONTEXT_STR}": str(self.context),
                "{ANTHROPIC_BASE_URL}": (self.anthropic_base_url or self.base_url).rstrip("/"),
                "{OUTPUT}": str(self.output)}


def _cond(name: str, ctx: Context) -> bool:
    return {"api_key": bool(ctx.api_key), "tools": bool(ctx.supports_tools), "vision": ctx.vision}.get(name, False)


def render(obj: Any, ctx: Context) -> Any:
    if isinstance(obj, dict):
        if "$if" in obj:
            if not _cond(obj["$if"], ctx):
                return _DROP
            if "$value" in obj:
                return render(obj["$value"], ctx)
            if "$list" in obj:
                return render(obj["$list"], ctx)
            obj = {k: v for k, v in obj.items() if k != "$if"}
        out = {}
        for k, v in obj.items():
            rv = render(v, ctx)
            if rv is not _DROP:
                out[render(k, ctx)] = rv
        return out
    if isinstance(obj, list):
        return [x for x in (render(v, ctx) for v in obj) if x is not _DROP]
    if isinstance(obj, str):
        if obj == "{CONTEXT}":
            return ctx.context
        if obj == "{OUTPUT}":
            return ctx.output
        if obj == "{TOOLS}":
            return bool(ctx.supports_tools)
        if obj == "{VISION}":
            return ctx.vision
        for k, v in ctx.values().items():
            obj = obj.replace(k, v)
        return obj
    return obj


@dataclass
class Check:
    level: str   # ok | warn | error
    text: str

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "error": "❌"}[self.level]


def expand_path(p: str) -> Path:
    return Path(os.path.expandvars(p)).expanduser()


def vscode_extension_dirs() -> list[Path]:
    home = Path.home()
    return [home / ".vscode" / "extensions", home / ".vscode-insiders" / "extensions",
            home / ".cursor" / "extensions", home / ".windsurf" / "extensions"]


@dataclass
class Integration:
    id: str
    name: str
    icon: str = "🔌"
    category: str = "chat"
    description: str = ""
    docs_url: str = ""
    checked: str = ""
    api: str = "chat"                 # chat | messages | responses
    needs_tools: bool = False
    min_context: int = 4096
    recommended_context: int = 8192
    detect_rules: dict = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    copy_values: list[list[str]] = field(default_factory=list)
    snippets: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    tunnel: bool = False
    order: int = 99
    verified: str = ""               # 실제 도구로 확인한 버전

    @classmethod
    def from_dict(cls, d: dict) -> "Integration":
        return cls(id=d["id"], name=d["name"], icon=d.get("icon", "🔌"), category=d.get("category", "chat"),
                   description=d.get("description", ""), docs_url=d.get("docs_url", ""), checked=d.get("checked", ""),
                   api=d.get("api", "chat"), needs_tools=d.get("needs_tools", False),
                   min_context=d.get("min_context", 4096), recommended_context=d.get("recommended_context", 8192),
                   detect_rules=d.get("detect", {}), files=d.get("files", []), steps=d.get("steps", []),
                   copy_values=d.get("copy_values", []), snippets=d.get("snippets", []), notes=d.get("notes", []),
                   env=d.get("env", {}), tunnel=d.get("tunnel", False), order=d.get("order", 99),
                   verified=d.get("verified", ""))

    @property
    def kind(self) -> str:
        if self.files:
            return "file"
        if self.snippets:
            return "snippet"
        return "gui"

    # ---- 감지 ----------------------------------------------------------
    def detect(self) -> bool:
        r = self.detect_rules
        if any(shutil.which(c) for c in r.get("commands", [])):
            return True
        if any(expand_path(p).exists() for p in r.get("paths", [])):
            return True
        for ext in r.get("vscode_extensions", []):
            for d in vscode_extension_dirs():
                if d.is_dir() and any(x.name.lower().startswith(ext.lower() + "-") for x in d.iterdir()):
                    return True
        return False

    # ---- 설정 만들기 ---------------------------------------------------
    def file_path(self, f: dict) -> Path:
        p = f.get(f"path_{sys.platform}") or f["path"]
        return expand_path(p)

    def rendered_files(self, ctx: Context) -> list[tuple[Path, str, Any]]:
        return [(self.file_path(f), f["format"], render(f["template"], ctx)) for f in self.files]

    def preview(self, ctx: Context) -> str:
        """화면에 보여줄 설정 내용 (파일별)."""
        parts = []
        for path, fmt, patch in self.rendered_files(ctx):
            parts.append(f"# {path}\n{cm.dump(patch, fmt)}")
        env = self.env_vars(ctx)
        if env:
            parts.append("# 환경 변수 (터미널 설정 파일에 추가)\n" + "\n".join(
                (f"setx {k} \"{v}\"" if sys.platform == "win32" else f"export {k}=\"{v}\"") for k, v in env.items()))
        return "\n".join(parts)

    def env_vars(self, ctx: Context) -> dict[str, str]:
        r = render(self.env, ctx) if self.env else {}
        return {} if r is _DROP else dict(r)

    def gui_values(self, ctx: Context) -> list[tuple[str, str]]:
        return [(label, render(value, ctx)) for label, value in self.copy_values]

    def rendered_snippets(self, ctx: Context) -> list[tuple[str, str]]:
        return [(s["title"], render(s["text"], ctx)) for s in self.snippets]

    def rendered_steps(self, ctx: Context) -> list[str]:
        return [render(s, ctx) for s in self.steps]

    # ---- 적용/되돌리기 -------------------------------------------------
    def plans(self, ctx: Context) -> list[cm.ApplyPlan]:
        return [cm.plan_merge(path, fmt, patch) for path, fmt, patch in self.rendered_files(ctx)]

    def apply(self, plans: list[cm.ApplyPlan], store: cm.StateStore) -> list[dict]:
        records = store.get(self.id)
        for p in plans:
            if p.changed:
                rec = cm.apply_plan(p)
                # 같은 파일을 여러 번 적용하면 '맨 처음' 백업을 유지한다 (진짜 원래 상태로 되돌리기)
                if not any(r["path"] == rec["path"] for r in records):
                    records.append(rec)
        store.set(self.id, records)
        return records

    def revert(self, store: cm.StateStore) -> list[str]:
        msgs = [cm.revert(r) for r in reversed(store.get(self.id))]
        store.set(self.id, [])
        return msgs

    def applied(self, store: cm.StateStore) -> bool:
        return bool(store.get(self.id))

    # ---- 호환성 점검 ---------------------------------------------------
    def check(self, ctx: Context) -> list[Check]:
        out: list[Check] = []
        if not ctx.running:
            out.append(Check("warn", "서버가 꺼져 있어요. 지금 보이는 값은 서버 탭의 설정 기준이에요. 연결 테스트 전에 서버를 켜 주세요."))
        if self.needs_tools:
            if ctx.supports_tools is False:
                out.append(Check("error", "이 모델은 도구 호출(tool calling)을 지원하지 않는 것 같아요. 코딩 에이전트는 파일을 읽고 "
                                          "고칠 때 도구 호출을 쓰므로 제대로 동작하지 않아요. 💻 코딩 카테고리에서 🛠 표시 모델을 고르세요."))
            elif ctx.supports_tools is None:
                out.append(Check("warn", "도구 호출 지원 여부를 알 수 없어요. [연결 테스트]로 확인해 보세요."))
            else:
                out.append(Check("ok", "도구 호출을 지원하는 모델이에요."))
            if ctx.engine_id == "llamacpp" and not ctx.jinja:
                out.append(Check("error", "서버 탭에서 '채팅 템플릿(--jinja)'을 켜야 도구 호출이 동작해요."))
        if ctx.context < self.min_context:
            out.append(Check("error", f"문맥 길이 {ctx.context:,}은(는) 너무 짧아요. 최소 {self.min_context:,} "
                                      f"(권장 {self.recommended_context:,}) — 서버 탭에서 늘리거나 '코딩 에이전트용' 프리셋을 쓰세요."))
        elif ctx.context < self.recommended_context:
            out.append(Check("warn", f"문맥 길이 {ctx.context:,}: 동작은 하지만 {self.recommended_context:,} 이상을 권장해요."))
        else:
            out.append(Check("ok", f"문맥 길이 {ctx.context:,} — 충분해요."))
        if self.api == "messages":
            if ctx.engine_id == "llamacpp":
                out.append(Check("ok", "llama.cpp는 Claude Code가 쓰는 Anthropic API(/v1/messages)를 지원해요."))
            else:
                out.append(Check("warn", f"{self.name}은(는) Anthropic API(/v1/messages)가 필요해요. {ctx.engine_name}이(가) "
                                         "지원하는지 [연결 테스트]로 확인하세요. 안 되면 llama.cpp 엔진을 쓰세요."))
        if self.api == "responses":
            out.append(Check("warn" if ctx.engine_id != "llamacpp" else "ok",
                             "Responses API(/v1/responses)가 필요해요. " +
                             ("llama.cpp 최신 버전은 지원해요." if ctx.engine_id == "llamacpp" else "[연결 테스트]로 확인하세요.")))
        if ctx.host == "0.0.0.0" and not ctx.api_key:
            out.append(Check("warn", "네트워크 공유 중인데 API 키가 없어요. 같은 네트워크의 누구나 쓸 수 있어요."))
        if self.tunnel and not ctx.api_key:
            out.append(Check("error", "터널(인터넷 공개)을 쓰려면 API 키가 꼭 필요해요. 서버 탭에서 [생성]을 누르세요."))
        if self.category == "agent" and ctx.total_ram and ctx.total_ram <= 16 * 1024 ** 3:
            out.append(Check("warn", "메모리 16GB 이하 컴퓨터에서는 코딩 에이전트가 느리거나 품질이 낮을 수 있어요. "
                                     "7B~14B Q4 모델을 권장해요."))
        return out


def load_all(data_dir: Path = DATA_DIR) -> list[Integration]:
    items = []
    for p in sorted(Path(data_dir).glob("*.json")):
        items.append(Integration.from_dict(json.loads(p.read_text(encoding="utf-8"))))
    return sorted(items, key=lambda i: (i.order, i.name))
