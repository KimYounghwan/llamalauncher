"""6단계: 개발 도구 연동 — 템플릿, 설정 병합/백업/복구, 호환성 점검, 연결 테스트, 연동 탭."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import tomlkit
import yaml

from llamalauncher.app.core.integrations import Context, StateStore, Tester, load_all
from llamalauncher.app.core.integrations import config_merge as cm
from llamalauncher.app.core.integrations.base import render


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    return h


def tool(tid):
    return next(i for i in load_all() if i.id == tid)


CTX = Context(base_url="http://127.0.0.1:8080", model="local-coder", api_key="", context=65536, running=True,
              supports_tools=True)


# ---------------------------------------------------------------- 병합 도우미
def test_strip_jsonc_keeps_strings():
    text = '{\n  // 주석\n  "url": "http://a//b", /* 블록 */ "x": [1, 2,],\n}'
    assert json.loads(cm.strip_jsonc(text)) == {"url": "http://a//b", "x": [1, 2]}
    assert cm.has_comments(text) and not cm.has_comments('{"a": "b//c",}')


def test_merge_preserves_existing_and_matches_list_items():
    base = {"permissions": {"allow": ["Bash(ls)"]}, "env": {"KEEP": "1", "ANTHROPIC_MODEL": "old"},
            "models": [{"name": "다른 모델", "x": 1}, {"name": "라마런처 채팅", "model": "old"}]}
    patch = {"env": {"ANTHROPIC_MODEL": "new"}, "models": [{"name": "라마런처 채팅", "model": "new"}]}
    out = cm.merge(base, patch)
    assert out["permissions"] == {"allow": ["Bash(ls)"]}
    assert out["env"] == {"KEEP": "1", "ANTHROPIC_MODEL": "new"}
    assert out["models"] == [{"name": "다른 모델", "x": 1}, {"name": "라마런처 채팅", "model": "new"}]


def test_apply_backup_and_revert_existing_file(tmp_path):
    p = tmp_path / "settings.json"
    original = '{\n  "theme": "dark",\n  "env": {"FOO": "1"}\n}\n'
    p.write_text(original)
    plan = cm.plan_merge(p, "json", {"env": {"BAR": "2"}})
    assert plan.changed and "+    \"BAR\": \"2\"" in plan.diff()
    rec = cm.apply_plan(plan)
    assert json.loads(p.read_text()) == {"theme": "dark", "env": {"FOO": "1", "BAR": "2"}}
    assert Path(rec["backup"]).read_text() == original and ".bak-" in rec["backup"]
    cm.revert(rec)
    assert p.read_text() == original


def test_backup_names_unique_within_same_second(tmp_path):
    from datetime import datetime
    p = tmp_path / "a.json"
    p.write_text("{}")
    now = datetime(2026, 11, 21, 12, 0, 0)
    first = cm.backup_path(p, now)
    first.write_text("x")
    assert cm.backup_path(p, now) != first


def test_apply_new_file_then_revert_deletes(tmp_path):
    p = tmp_path / "sub" / "new.yaml"
    rec = cm.apply_plan(cm.plan_merge(p, "yaml", {"a": 1}))
    assert yaml.safe_load(p.read_text()) == {"a": 1} and rec["backup"] is None
    cm.revert(rec)
    assert not p.exists()


def test_broken_existing_file_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json")
    with pytest.raises(cm.MergeError):
        cm.plan_merge(p, "json", {"a": 1})


def test_toml_merge_keeps_comments(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('# 내 설정\nmodel = "gpt"\n\n[mcp_servers.x]\ncommand = "npx" # 주석 유지\n')
    plan = cm.plan_merge(p, "toml", {"model": "local", "model_providers": {"ll": {"base_url": "u"}}})
    assert "# 내 설정" in plan.new_text and "# 주석 유지" in plan.new_text
    doc = tomlkit.parse(plan.new_text)
    assert doc["model"] == "local" and doc["mcp_servers"]["x"]["command"] == "npx"
    assert doc["model_providers"]["ll"]["base_url"] == "u"


def test_jsonc_comments_warning(tmp_path):
    p = tmp_path / "zed.json"
    p.write_text('{\n  // 폰트\n  "buffer_font_size": 15,\n}\n')
    plan = cm.plan_merge(p, "jsonc", {"language_models": {}})
    assert plan.warnings and json.loads(plan.new_text)["buffer_font_size"] == 15


# ---------------------------------------------------------------- 템플릿
def test_render_conditions_and_types():
    t = {"n": "{CONTEXT}", "o": "{OUTPUT}", "t": "{TOOLS}", "{MODEL}": "x {API_BASE}",
         "h": {"$if": "api_key", "Authorization": "Bearer {API_KEY}"},
         "c": {"$if": "tools", "$list": ["tool_use"]}, "e": {"$if": "api_key", "$value": "ENV"},
         "l": [{"$if": "api_key", "a": 1}, "keep"]}
    r = render(t, CTX)
    assert r == {"n": 65536, "o": 8192, "t": True, "local-coder": "x http://127.0.0.1:8080/v1",
                 "c": ["tool_use"], "l": ["keep"]}
    ctx = Context(api_key="k", supports_tools=False, context=4096)
    r = render(t, ctx)
    assert r["h"] == {"Authorization": "Bearer k"} and r["e"] == "ENV" and "c" not in r and r["o"] == 1024


def test_all_templates_valid():
    items = load_all()
    assert [i.id for i in items][:3] == ["claude_code", "opencode", "continue"]
    assert len(items) == 12
    for it in items:
        assert it.docs_url.startswith("https://") and it.checked, it.id
        assert it.kind in ("file", "gui", "snippet")
        text = it.preview(CTX) if it.kind == "file" else ""
        for path, fmt, patch in it.rendered_files(CTX):
            assert "{" not in json.dumps(patch).replace("{\"", "").replace("{}", ""), (it.id, patch)
            cm.parse(cm.dump(patch, fmt), fmt)  # 다시 읽을 수 있어야 한다
        for _, v in it.gui_values(CTX):
            assert "{" not in str(v)
        for _, s in it.rendered_snippets(CTX):
            assert "{API_BASE}" not in s and "{MODEL}" not in s
        assert "{BASE_URL}" not in text


def test_claude_code_apply_keeps_user_settings_and_reverts(home):
    s = home / ".claude" / "settings.json"
    s.parent.mkdir()
    s.write_text(json.dumps({"permissions": {"allow": ["Read"]}, "env": {"MY_VAR": "1"}}))
    store = StateStore(home / "state.json")
    cc = tool("claude_code")
    ctx = Context(model="local-coder", api_key="secret", context=65536, running=True, supports_tools=True)
    cc.apply(cc.plans(ctx), store)
    data = json.loads(s.read_text())
    assert data["permissions"] == {"allow": ["Read"]} and data["env"]["MY_VAR"] == "1"
    assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8080"
    assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret" and data["env"]["ANTHROPIC_MODEL"] == "local-coder"
    assert cc.applied(store)
    # 다시 적용해도 '맨 처음' 백업을 유지 → 되돌리면 진짜 원래 상태
    ctx.model = "other"
    cc.apply(cc.plans(ctx), store)
    cc.revert(store)
    assert json.loads(s.read_text()) == {"permissions": {"allow": ["Read"]}, "env": {"MY_VAR": "1"}}
    assert not cc.applied(store)


def test_continue_yaml_appends_model(home):
    c = home / ".continue" / "config.yaml"
    c.parent.mkdir()
    c.write_text(yaml.safe_dump({"name": "Mine", "version": "0.0.1", "schema": "v1",
                                 "models": [{"name": "GPT", "provider": "openai", "model": "gpt-4o"}]}))
    plans = tool("continue").plans(Context(api_key="k", context=32768, supports_tools=True))
    data = yaml.safe_load(plans[0].new_text)
    assert [m["name"] for m in data["models"]] == ["GPT", "라마런처 채팅"]
    assert data["name"] == "LlamaLauncher" or data["name"] == "Mine"
    assert data["models"][1]["requestOptions"]["headers"]["Authorization"] == "Bearer k"


def test_aider_writes_two_files(home):
    plans = tool("aider").plans(CTX)
    assert [p.path.name for p in plans] == [".aider.conf.yml", ".aider.model.metadata.json"]
    meta = json.loads(plans[1].new_text)
    assert meta["openai/local-coder"]["max_input_tokens"] == 65536


def test_codex_env_key_only_with_api_key(home):
    cx = tool("codex")
    no_key = tomlkit.parse(cx.plans(CTX)[0].new_text)
    assert "env_key" not in no_key["model_providers"]["llamalauncher"]
    assert no_key["model_providers"]["llamalauncher"]["wire_api"] == "responses"
    ctx = Context(api_key="k", context=65536)
    with_key = tomlkit.parse(cx.plans(ctx)[0].new_text)
    assert with_key["model_providers"]["llamalauncher"]["env_key"] == "LLAMALAUNCHER_API_KEY"
    assert cx.env_vars(ctx) == {"LLAMALAUNCHER_API_KEY": "k"}


# ---------------------------------------------------------------- 호환성 점검
def levels(checks):
    return [c.level for c in checks]


def test_checks():
    cc = tool("claude_code")
    bad = cc.check(Context(context=8192, supports_tools=False, running=False, engine_id="ollama", engine_name="Ollama"))
    texts = " ".join(c.text for c in bad)
    assert levels(bad).count("error") == 2 and "도구 호출" in texts and "/v1/messages" in texts and "꺼져" in texts
    good = cc.check(CTX)
    assert "error" not in levels(good)
    assert any("--jinja" in c.text for c in cc.check(Context(context=65536, supports_tools=True, jinja=False)))
    cur = tool("cursor").check(Context(context=65536, supports_tools=True, running=True))
    assert any(c.level == "error" and "API 키" in c.text for c in cur)
    small = cc.check(Context(context=65536, supports_tools=True, running=True, total_ram=16 * 1024 ** 3))
    assert any("16GB" in c.text for c in small)
    assert "error" not in levels(tool("openwebui").check(Context(context=8192, running=True)))


def test_agent_preset():
    from llamalauncher.app.core.recommender import ModelSpec, Preset, recommend
    from tests.test_stage5 import system
    s = system()
    s.total_ram = 64 << 30
    s.gpus[0].vram_bytes = 64 << 30
    rec = recommend(s, ModelSpec(file_size=5 << 30, n_layers=36, n_ctx_train=131072, n_head_kv=8, head_dim=128),
                    Preset.AGENT)
    assert rec.options.ctx_size == 65536 and rec.options.jinja and rec.options.parallel == 1
    assert Preset.AGENT.label == "코딩 에이전트용"


# ---------------------------------------------------------------- 연결 테스트
class _API(BaseHTTPRequestHandler):
    tool_ok = True

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"data": [{"id": "local-coder"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            msg = {"role": "assistant", "content": "ok"}
            if body.get("tools") and _API.tool_ok:
                msg["tool_calls"] = [{"function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'}}]
            return self._send(200, {"choices": [{"message": msg}]})
        if self.path == "/v1/messages":
            if self.headers.get("anthropic-version") is None:
                return self._send(400, {"error": {"message": "missing version"}})
            if any(m["role"] == "system" for m in body["messages"]):  # Qwen 계열 템플릿 흉내
                return self._send(500, {"error": {"message": "Jinja Exception: System message must be at the beginning."}})
            content = [{"type": "tool_use", "name": "get_weather", "input": {"city": "Seoul"}}] if body.get("tools") \
                else [{"type": "text", "text": "ok"}]
            return self._send(200, {"content": content})
        self._send(404, {"error": {"message": "Not Found"}})

    def log_message(self, *a):
        pass


@pytest.fixture
def api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _API)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _API.tool_ok = True
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_tester_chat_and_messages(api):
    t = Tester(api, "local-coder")
    res = t.for_api("chat", True)
    assert [r.ok for r in res] == [True, True, True] and "get_weather" in res[2].detail
    res = t.for_api("messages", True)
    assert [r.ok for r in res] == [True, False, False] and "호환 주소" in res[1].detail
    from llamalauncher.app.core.compat_proxy import CompatProxy
    from llamalauncher.app.core.netutil import find_free_port
    proxy = CompatProxy(api, port=find_free_port(19500))
    proxy.start()
    try:
        res = Tester(proxy.url, "local-coder").for_api("messages", True)
        assert [r.ok for r in res] == [True, True, True]  # 호환 주소를 거치면 통과
    finally:
        proxy.stop()
    res = t.for_api("responses", False)
    assert res[1].ok is False and "지원하지 않아요" in res[1].detail


def test_tester_no_tool_call_and_connection_refused(api):
    _API.tool_ok = False
    res = Tester(api, "m").for_api("chat", True)
    assert res[2].ok is False and "호출하지 않았어요" in res[2].detail
    from llamalauncher.app.core.netutil import find_free_port
    res = Tester(f"http://127.0.0.1:{find_free_port(39500)}", "m").for_api("chat", True)
    assert len(res) == 1 and not res[0].ok and "연결할 수 없어요" in res[0].detail


# ---------------------------------------------------------------- 연동 탭
def test_integrations_tab_apply_flow(qapp, home, api, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from llamalauncher.app.ui import integrations_tab as it
    monkeypatch.setattr(it.DiffDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    from llamalauncher.app.core.compat_proxy import CompatProxy
    from llamalauncher.app.core.netutil import find_free_port
    proxy = CompatProxy(api, port=find_free_port(19700))
    proxy.start()
    store = StateStore(home / "state.json")
    tab = it.IntegrationsTab(store)
    tab.context_provider = lambda: Context(base_url=api, model="local-coder", context=65536, running=True,
                                           supports_tools=True, anthropic_base_url=proxy.url)
    tab.list.setCurrentRow(0)  # Claude Code
    tab.refresh()
    assert "ANTHROPIC_BASE_URL" in tab.preview.toPlainText() and proxy.url in tab.preview.toPlainText()
    tab._apply()
    s = json.loads((home / ".claude" / "settings.json").read_text())
    assert s["env"]["ANTHROPIC_MODEL"] == "local-coder" and s["env"]["ANTHROPIC_BASE_URL"] == proxy.url
    assert s["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "65536"
    assert "✅" in tab.list.item(0).text() and tab.revert_btn.isEnabled()
    tab._test()
    end = time.time() + 10
    while time.time() < end and tab._testing:
        qapp.processEvents()
        time.sleep(0.02)
    assert "쓸 준비가 됐어요" in tab.test_lbl.text()
    tab._revert()
    assert not (home / ".claude" / "settings.json").exists()  # 원래 없던 파일은 지운다
    tab.list.setCurrentRow(3)  # Cline: GUI 안내
    assert tab.gui_box.isVisibleTo(tab) and not tab.file_box.isVisibleTo(tab)
    assert tab.values_box.count() == 4
    proxy.stop()


def test_rewrite_mid_system_messages():
    from llamalauncher.app.core.compat_proxy import rewrite_messages
    body = {"system": "top", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "system", "content": "be brief"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": [{"type": "text", "text": "x"}]},
        {"role": "user", "content": "b"}]}
    out = rewrite_messages(body)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"] and out["system"] == "top"
    assert out["messages"][0]["content"][1]["text"] == "<system-reminder>\nbe brief\n</system-reminder>"
    assert out["messages"][2]["content"][-1] == {"type": "text", "text": "b"}
    plain = {"messages": [{"role": "user", "content": "hi"}]}
    assert rewrite_messages(plain) is plain


def test_compat_proxy_streams_sse():
    import requests
    from llamalauncher.app.core.compat_proxy import CompatProxy
    from llamalauncher.app.core.netutil import find_free_port

    class SSE(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), SSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    proxy = CompatProxy(f"http://127.0.0.1:{srv.server_address[1]}", port=find_free_port(19600))
    proxy.start()
    try:
        r = requests.post(proxy.url + "/v1/messages", json={"messages": []}, stream=True, timeout=5)
        lines = [l for l in r.iter_lines(decode_unicode=True) if l]
        assert r.headers["Content-Type"] == "text/event-stream" and lines == ["data: 0", "data: 1", "data: 2"]
    finally:
        proxy.stop()
        srv.shutdown()
