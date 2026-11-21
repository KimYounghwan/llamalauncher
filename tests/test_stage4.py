"""4단계: 내 모델 목록(스캔·LM Studio·Ollama·삭제), 채팅 클라이언트, 채팅/내 모델 탭."""

import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from llamalauncher.app.core import model_library as lib
from llamalauncher.app.core.chat_client import (
    ChatRequest, chunk_events, friendly_error, parse_sse_line, stream_chat,
)
from llamalauncher.app.core.downloader import manifest_path
from tests.test_stage2 import _s, basic_kv, write_gguf


def make_model(path: Path, kv=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, kv or basic_kv())
    return path


def embed_kv():
    return [("general.architecture", 8, _s("nomic-bert")), ("nomic-bert.block_count", 4, struct.pack("<I", 12))]


# ---------------------------------------------------------------- 폴더 스캔
def test_scan_folder_skips_mmproj_parts_and_hidden(tmp_path):
    make_model(tmp_path / "a" / "Model-Q4_K_M.gguf")
    make_model(tmp_path / "a" / "mmproj-F16.gguf")
    make_model(tmp_path / "b" / "Big-Q8_0-00001-of-00002.gguf")
    make_model(tmp_path / "b" / "Big-Q8_0-00002-of-00002.gguf")
    make_model(tmp_path / ".hidden" / "x.gguf")
    (tmp_path / "notes.txt").write_text("hi")
    found = [p.name for p in lib.scan_folder(tmp_path)]
    assert found == ["Model-Q4_K_M.gguf", "Big-Q8_0-00001-of-00002.gguf"]


def test_quant_from_name():
    assert lib.quant_from_name("Qwen3-8B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert lib.quant_from_name("Big-Q8_0-00001-of-00002.gguf") == "Q8_0"
    assert lib.quant_from_name("sha256-abcdef") == ""


# ---------------------------------------------------------------- LM Studio / Ollama
def test_lmstudio_dirs_reads_custom_folder(tmp_path):
    home = tmp_path / "home"
    custom = tmp_path / "Models"
    custom.mkdir()
    (home / ".lmstudio" / "models").mkdir(parents=True)
    (home / ".lmstudio" / "settings.json").write_text(json.dumps({"downloadsFolder": str(custom)}))
    assert lib.lmstudio_dirs(home) == [custom, home / ".lmstudio" / "models"]


def _ollama(home: Path, name="gemma3", tag="4b", with_projector=False) -> Path:
    models = home / ".ollama" / "models"
    blob = make_model(models / "blobs" / f"sha256-{name}0000")
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{name}0000",
               "size": blob.stat().st_size},
              {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:tmpl", "size": 10}]
    if with_projector:
        make_model(models / "blobs" / "sha256-proj")
        layers.append({"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:proj", "size": 1})
    mf = models / "manifests" / "registry.ollama.ai" / "library" / name / tag
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps({"layers": layers}))
    return models


def test_scan_ollama(tmp_path):
    models = _ollama(tmp_path, with_projector=True)
    # 모델 파일이 없는 매니페스트는 무시
    broken = models / "manifests" / "registry.ollama.ai" / "library" / "gone" / "latest"
    broken.parent.mkdir(parents=True)
    broken.write_text(json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
                                              "digest": "sha256:missing"}]}))
    assert lib.ollama_dirs(tmp_path, env={}) == [models]
    found = lib.scan_ollama(models)
    assert [m.name for m in found] == ["gemma3:4b"]
    assert found[0].mmproj == models / "blobs" / "sha256-proj"
    assert not found[0].deletable


def test_collect_all_sources_dedup_and_headers(tmp_path):
    home = tmp_path / "home"
    mdir = tmp_path / "models"
    dl = make_model(mdir / "unsloth" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf")
    manifest_path(mdir).write_text(json.dumps({"models": [
        {"path": str(dl), "repo": "unsloth/Qwen3-8B-GGUF", "quant": "Q4_K_M", "size": 1,
         "downloaded_at": "2026-11-20T10:00:00"},
        {"path": str(mdir / "deleted.gguf"), "repo": "x/y"},  # 지워진 파일은 무시
    ]}))
    added = make_model(tmp_path / "Downloads" / "Embed-F16.gguf", embed_kv())
    folder = tmp_path / "old"
    make_model(folder / "Old-Q5_K_M.gguf")
    lm = make_model(home / ".lmstudio" / "models" / "bartowski" / "Gemma-GGUF" / "gemma-Q6_K.gguf")
    _ollama(home)
    (tmp_path / "broken.gguf").write_bytes(b"NOPE")

    src = lib.LibrarySources(models_dir=mdir, added=[str(added), str(dl), str(tmp_path / "broken.gguf")],
                             folders=[str(folder)], include_external=True, home=home)
    models = {m.display_name: m for m in lib.collect(src)}
    assert set(models) == {"Qwen3-8B-Q4_K_M", "Embed-F16", "Old-Q5_K_M", "gemma-Q6_K", "gemma3:4b", "broken"}
    q = models["Qwen3-8B-Q4_K_M"]
    assert q.source == lib.Source.DOWNLOAD and q.repo == "unsloth/Qwen3-8B-GGUF"  # 직접 추가보다 앞선 출처
    assert q.date.year == 2026 and q.size == dl.stat().st_size and q.info.n_layers == 32
    assert models["Embed-F16"].is_embedding and models["Embed-F16"].source == lib.Source.ADDED
    assert models["gemma-Q6_K"].source == lib.Source.LMSTUDIO and models["gemma-Q6_K"].repo == "bartowski/Gemma-GGUF"
    assert models["gemma3:4b"].quant == "Q4_K_M"  # 파일 이름에 없으니 GGUF의 file_type에서
    assert models["broken"].error and models["broken"].info is None
    # 외부 런처 끄기
    src.include_external = False
    assert {m.source for m in lib.collect(src)} == {lib.Source.DOWNLOAD, lib.Source.ADDED, lib.Source.FOLDER}


def test_mmproj_found_next_to_model(tmp_path):
    p = make_model(tmp_path / "m" / "Vision-Q4_K_M.gguf")
    make_model(tmp_path / "m" / "mmproj-BF16.gguf")
    make_model(tmp_path / "m" / "mmproj-F16.gguf")
    m = lib.load_info(lib.LocalModel(path=p, source=lib.Source.ADDED))
    assert m.mmproj.name == "mmproj-F16.gguf"


# ---------------------------------------------------------------- 삭제
def test_delete_split_model_keeps_shared_mmproj_and_cleans_dirs(tmp_path):
    mdir = tmp_path / "models"
    repo = mdir / "org" / "Repo-GGUF"
    a1 = make_model(repo / "M-Q8_0-00001-of-00002.gguf")
    a2 = make_model(repo / "M-Q8_0-00002-of-00002.gguf")
    b = make_model(repo / "M-Q4_K_M.gguf")
    mm = make_model(repo / "mmproj-F16.gguf")
    manifest_path(mdir).write_text(json.dumps({"models": [{"path": str(a1)}, {"path": str(b)}]}))
    ma = lib.load_info(lib.LocalModel(path=a1, source=lib.Source.DOWNLOAD))
    mb = lib.load_info(lib.LocalModel(path=b, source=lib.Source.DOWNLOAD))
    assert ma.mmproj == mm == mb.mmproj

    removed = lib.delete_model(ma, mdir, [ma, mb])
    assert set(removed) == {a1, a2} and mm.exists() and b.exists()  # 비전 파일은 b가 아직 쓴다
    assert [e["path"] for e in json.loads(manifest_path(mdir).read_text())["models"]] == [str(b)]

    lib.delete_model(mb, mdir, [mb])
    assert not repo.exists() and not (mdir / "org").exists() and mdir.exists()


def test_delete_ollama_refused(tmp_path):
    m = lib.scan_ollama(_ollama(tmp_path))[0]
    with pytest.raises(PermissionError):
        lib.delete_model(m, tmp_path / "models")
    assert m.path.exists()


# ---------------------------------------------------------------- 채팅 클라이언트
def test_sse_parsing():
    assert parse_sse_line("data: [DONE]") == {"done": True}
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("data: {bad") is None
    evs = chunk_events({"choices": [{"delta": {"reasoning_content": "hmm", "content": "Hi"}}]})
    assert [(e.kind, e.text) for e in evs] == [("reasoning", "hmm"), ("content", "Hi")]


def test_friendly_errors():
    import requests
    assert "연결할 수 없어요" in friendly_error(None, "", requests.ConnectionError())
    assert "API 키" in friendly_error(401, '{"error":{"message":"Invalid API Key"}}')
    msg = friendly_error(400, '{"error":{"message":"the request exceeds the available context size"}}')
    assert "문맥 길이" in msg


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _ChatHandler.received.append((dict(self.headers), body))
        last = body["messages"][-1]["content"]
        if self.headers.get("Authorization") == "Bearer wrong":
            return self._json(401, {"error": {"message": "Invalid API Key"}})
        if last == "too long":
            return self._json(400, {"error": {"message": "the request exceeds the available context size"}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "생각 중"}}]}]
        n = 200 if last == "slow" else 3
        chunks += [{"choices": [{"delta": {"content": f"말{i} "}}]} for i in range(n)]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                       "timings": {"predicted_per_second": 42.5},
                       "usage": {"prompt_tokens": 10, "completion_tokens": n}})
        try:
            for c in chunks:
                self._chunk(f"data: {json.dumps(c, ensure_ascii=False)}\n\n")
                if last == "slow":
                    time.sleep(0.02)
            self._chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            pass

    def _chunk(self, s: str):
        b = s.encode()
        self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


@pytest.fixture
def chat_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChatHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _ChatHandler.received = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _collect(req, stop=None):
    events = []
    stream_chat(req, events.append, stop)
    return events


def test_stream_chat_ok(chat_server):
    req = ChatRequest(base_url=chat_server, messages=[{"role": "user", "content": "hi"}], model="local-model",
                      api_key="secret")
    events = _collect(req)
    assert "".join(e.text for e in events if e.kind == "content") == "말0 말1 말2 "
    assert [e.text for e in events if e.kind == "reasoning"] == ["생각 중"]
    done = events[-1]
    assert done.kind == "done" and done.stats["tps"] == 42.5 and done.stats["usage"]["completion_tokens"] == 3
    headers, body = _ChatHandler.received[0]
    assert headers["Authorization"] == "Bearer secret" and body["stream"] is True and body["model"] == "local-model"


@pytest.mark.parametrize("key,text,expect", [("wrong", "hi", "API 키"), ("", "too long", "문맥 길이")])
def test_stream_chat_errors(chat_server, key, text, expect):
    events = _collect(ChatRequest(base_url=chat_server, messages=[{"role": "user", "content": text}], api_key=key))
    assert len(events) == 1 and events[0].kind == "error" and expect in events[0].text


def test_stream_chat_connection_refused():
    from llamalauncher.app.core.netutil import find_free_port
    port = find_free_port(39000)
    events = _collect(ChatRequest(base_url=f"http://127.0.0.1:{port}", messages=[]))
    assert events[-1].kind == "error" and "서버" in events[-1].text


def test_stream_chat_stop(chat_server):
    stop = threading.Event()
    events = []

    def on(ev):
        events.append(ev)
        if sum(e.kind == "content" for e in events) == 5:
            stop.set()

    stream_chat(ChatRequest(base_url=chat_server, messages=[{"role": "user", "content": "slow"}]), on, stop)
    assert events[-1].kind == "done" and events[-1].stats.get("stopped")
    assert sum(e.kind == "content" for e in events) < 50


# ---------------------------------------------------------------- 채팅 표시 도우미
def test_split_think_and_markdown():
    from llamalauncher.app.ui.chat_tab import render_markdown_lite, split_think
    assert split_think("<think>음..</think>\n답") == ("음..", "답")
    assert split_think("<think>아직 생각 중") == ("아직 생각 중", "")
    assert split_think("그냥 답") == ("", "그냥 답")
    out = render_markdown_lite("**굵게** `x<y`\n```python\nprint('<a>')\n```\n끝", "#eee")
    assert "<b>굵게</b>" in out and "<code>x&lt;y</code>" in out
    assert "<pre" in out and "print(&#x27;&lt;a&gt;&#x27;)" in out and out.endswith("</pre>끝")


# ---------------------------------------------------------------- UI
def _wait(qapp, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_models_tab_lists_and_renames(qapp, tmp_path):
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui.models_tab import ModelsTab
    make_model(tmp_path / "models" / "org" / "R" / "A-Q4_K_M.gguf")
    extra = make_model(tmp_path / "elsewhere" / "B-Q8_0.gguf")
    s = Settings(models_dir=str(tmp_path / "models"), show_external_models=False, added_models=[str(extra)])
    tab = ModelsTab(s, lambda: None)
    assert _wait(qapp, lambda: tab.table.rowCount() == 2)
    names = sorted(tab.table.item(r, 0).text() for r in range(2))
    assert names[0].startswith("A-Q4_K_M") and names[1].startswith("B-Q8_0")
    assert "🛠" in names[0]  # 테스트 모델의 채팅 템플릿에 tools가 있다
    s.model_names[str(extra)] = "내 모델"
    assert tab.display_name_for(str(extra)) == "내 모델"
    tab.add_file(str(extra))  # 이미 있으면 중복 추가하지 않는다
    assert s.added_models == [str(extra)]
    tab.filter_edit.setText("내 모델")
    tab._fill_table()
    visible = [r for r in range(tab.table.rowCount()) if not tab.table.isRowHidden(r)]
    assert len(visible) == 1


def test_chat_tab_streams_reply(qapp, chat_server):
    from llamalauncher.app.core.server import ServerState
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui.chat_tab import ChatTab
    tab = ChatTab(Settings(), lambda: None)
    assert not tab.send_btn.isEnabled()
    tab.set_server(ServerState.RUNNING, chat_server, {"alias": "my-alias", "api_key": "", "model_name": "테스트"})
    assert tab.send_btn.isEnabled()
    tab.input.setPlainText("안녕")
    tab.send()
    assert tab.input.toPlainText() == ""
    assert _wait(qapp, lambda: not tab.busy and tab.messages[-1].stats)
    assert [m.role for m in tab.messages] == ["user", "assistant"]
    assert tab.messages[-1].content == "말0 말1 말2 " and tab.messages[-1].reasoning == "생각 중"
    assert "42.5 토큰/초" in tab.stats_lbl.text()
    _, body = _ChatHandler.received[-1]
    assert body["model"] == "my-alias" and body["messages"][0]["role"] == "system"
    # 두 번째 질문에는 이전 대화가 함께 간다 (생각 과정은 빼고)
    tab.send("too long")
    assert _wait(qapp, lambda: not tab.busy and tab.messages[-1].role == "error")
    assert [m.role for m in tab.messages] == ["user", "assistant", "error"]
    assert tab.input.toPlainText() == "too long"  # 실패한 질문은 입력칸으로 되돌아온다
    _, body = _ChatHandler.received[-1]
    assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
    assert all("생각" not in m["content"] for m in body["messages"])
    tab.clear()
    assert tab.messages == []


def test_orphan_check_ignores_own_running_server(qapp, monkeypatch):
    """창을 연 직후 바로 서버를 켜도 '이전 서버가 남아 있어요' 창이 뜨면 안 된다."""
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui import server_tab as st
    tab = st.ServerTab(Settings(), lambda: None)
    called = []
    monkeypatch.setattr(st, "find_orphan_server", lambda: called.append(1))
    monkeypatch.setattr(type(tab.controller), "is_active", property(lambda self: True))
    tab._check_orphan()
    assert called == []
