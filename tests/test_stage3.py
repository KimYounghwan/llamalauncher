import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from llamalauncher.app.core import downloader as dl
from llamalauncher.app.core import hf_client as hf
from llamalauncher.app.core.downloader import DownloadError, DownloadManager, FileSpec, Status
from tests.test_hardware import apple, make_system, nvidia

# ================================================================ 가짜 다운로드 서버
FILES = {
    "a.gguf": bytes(range(256)) * 4096,          # 1 MiB
    "b-00001-of-00002.gguf": b"x" * 300_000,
    "b-00002-of-00002.gguf": b"y" * 200_000,
}


class Handler(BaseHTTPRequestHandler):
    slow = False
    ignore_range = False
    forbidden = False
    requests_seen: list = []

    def do_GET(self):
        Handler.requests_seen.append((self.path, self.headers.get("Range"), self.headers.get("Authorization")))
        if Handler.forbidden:
            self.send_response(403)
            self.end_headers()
            return
        name = self.path.rsplit("/", 1)[-1]
        data = FILES.get(name)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and not Handler.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                self.send_response(416)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        step = 16384 if Handler.slow else len(body)
        try:
            for i in range(0, len(body), step):
                self.wfile.write(body[i:i + step])
                if Handler.slow:
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.slow = Handler.ignore_range = Handler.forbidden = False
    Handler.requests_seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def spec(name, with_hash=True, bad_hash=False):
    data = FILES[name]
    sha = hashlib.sha256(data).hexdigest() if with_hash else None
    if bad_hash:
        sha = "0" * 64
    return FileSpec(path=f"sub/{name}", size=len(data), sha256=sha)


def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def manager(tmp_path, base, **kw):
    return DownloadManager(tmp_path / "state.json", base_url=base, **kw)


def test_download_complete_and_manifest(tmp_path, server):
    m = manager(tmp_path, server, token_getter=lambda: "hf_test")
    dest = dl.repo_dir(tmp_path / "models", "org/Model-GGUF")
    item = m.add("org/Model-GGUF", "Q4_K_M", [spec("a.gguf")], dest)
    assert wait_for(lambda: item.status == Status.DONE), item.error
    assert (dest / "a.gguf").read_bytes() == FILES["a.gguf"]
    assert not (dest / "a.gguf.part").exists()
    assert item.downloaded == item.total
    assert Handler.requests_seen[0][2] == "Bearer hf_test"
    entries = dl.read_manifest(tmp_path / "models")
    assert entries[0]["repo"] == "org/Model-GGUF" and entries[0]["path"] == str(dest / "a.gguf")


def test_split_files_main_path(tmp_path, server):
    m = manager(tmp_path, server)
    files = [spec("b-00001-of-00002.gguf"), spec("b-00002-of-00002.gguf")]
    item = m.add("org/Big", "Q8_0", files, tmp_path / "d")
    assert wait_for(lambda: item.status == Status.DONE), item.error
    assert item.main_path.name == "b-00001-of-00002.gguf"
    assert (tmp_path / "d" / "b-00002-of-00002.gguf").stat().st_size == 200_000


def test_resume_from_partial_file(tmp_path, server):
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "a.gguf.part").write_bytes(FILES["a.gguf"][:400_000])
    m = manager(tmp_path, server)
    item = m.add("org/M", "Q4", [spec("a.gguf")], dest)
    assert wait_for(lambda: item.status == Status.DONE), item.error
    assert Handler.requests_seen[-1][1] == "bytes=400000-"
    assert (dest / "a.gguf").read_bytes() == FILES["a.gguf"]


def test_server_ignores_range_restarts(tmp_path, server):
    Handler.ignore_range = True
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "a.gguf.part").write_bytes(FILES["a.gguf"][:400_000])
    m = manager(tmp_path, server)
    item = m.add("org/M", "Q4", [spec("a.gguf")], dest)
    assert wait_for(lambda: item.status == Status.DONE), item.error
    assert (dest / "a.gguf").read_bytes() == FILES["a.gguf"]


def test_pause_resume_and_cancel(tmp_path, server):
    Handler.slow = True
    m = manager(tmp_path, server)
    item = m.add("org/M", "Q4", [spec("a.gguf")], tmp_path / "d")
    assert wait_for(lambda: item.status == Status.DOWNLOADING and item.downloaded > 0)
    m.pause(item.id)
    assert wait_for(lambda: item.status == Status.PAUSED)
    part = tmp_path / "d" / "a.gguf.part"
    assert 0 < part.stat().st_size < len(FILES["a.gguf"])
    # 앱을 다시 켠 것처럼 상태 파일에서 복원
    m2 = manager(tmp_path, server)
    restored = m2.items[item.id]
    assert restored.status == Status.PAUSED and restored.downloaded == part.stat().st_size
    Handler.slow = False
    m2.resume(item.id)
    assert wait_for(lambda: restored.status == Status.DONE), restored.error
    assert (tmp_path / "d" / "a.gguf").read_bytes() == FILES["a.gguf"]

    Handler.slow = True
    item3 = m2.add("org/Other", "Q4", [spec("b-00001-of-00002.gguf")], tmp_path / "e")
    assert wait_for(lambda: item3.status == Status.DOWNLOADING)
    m2.cancel(item3.id)
    assert wait_for(lambda: item3.status == Status.CANCELED)
    assert not (tmp_path / "e" / "b-00001-of-00002.gguf.part").exists()
    assert item3.id not in m2.items


def test_hash_mismatch(tmp_path, server):
    m = manager(tmp_path, server)
    item = m.add("org/M", "Q4", [spec("a.gguf", bad_hash=True)], tmp_path / "d")
    assert wait_for(lambda: item.status == Status.ERROR)
    assert "손상" in item.error
    assert not (tmp_path / "d" / "a.gguf").exists()


def test_gated_model_error(tmp_path, server):
    Handler.forbidden = True
    m = manager(tmp_path, server)
    item = m.add("org/M", "Q4", [spec("a.gguf")], tmp_path / "d")
    assert wait_for(lambda: item.status == Status.ERROR)
    assert "토큰" in item.error


def test_concurrency_limit(tmp_path, server):
    Handler.slow = True
    m = manager(tmp_path, server, max_concurrent=1)
    a = m.add("org/A", "Q4", [spec("a.gguf")], tmp_path / "a")
    b = m.add("org/B", "Q4", [spec("a.gguf")], tmp_path / "b")
    assert wait_for(lambda: a.status == Status.DOWNLOADING)
    assert b.status == Status.QUEUED
    Handler.slow = False
    assert wait_for(lambda: b.status == Status.DONE, 20)


def test_disk_space_check(tmp_path, server, monkeypatch):
    monkeypatch.setattr(dl, "free_space", lambda p: 100)
    m = manager(tmp_path, server)
    with pytest.raises(DownloadError, match="저장 공간"):
        m.add("org/M", "Q4", [spec("a.gguf")], tmp_path / "d")


def test_duplicate_add_returns_same_item(tmp_path, server):
    Handler.slow = True
    m = manager(tmp_path, server)
    a = m.add("org/M", "Q4", [spec("a.gguf")], tmp_path / "d")
    b = m.add("org/M", "Q4", [spec("a.gguf")], tmp_path / "d")
    assert a is b
    m.cancel(a.id)
    assert wait_for(lambda: a.status == Status.CANCELED)


def test_human_eta():
    assert dl.human_eta(None) == "—"
    assert dl.human_eta(42) == "42초"
    assert dl.human_eta(125) == "2분 5초"
    assert dl.human_eta(3725) == "1시간 2분"


# ================================================================ 허깅페이스 응답 해석
def test_quant_parsing():
    q = hf._quant_of
    assert q("Qwen3.8-27B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert q("Qwen3-4B-Q4_K_M.gguf") == "Q4_K_M"
    assert q("model-IQ4_XS.gguf") == "IQ4_XS"
    assert q("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf") == "BF16"
    assert q("gpt-oss-20b-MXFP4.gguf") == "MXFP4"
    assert q("Llama-3.2-3B-Instruct.Q8_0.gguf") == "Q8_0"


TREE = [
    {"type": "directory", "path": "BF16"},
    {"type": "file", "path": "BF16/M-BF16-00001-of-00002.gguf", "size": 100, "lfs": {"size": 50_000_000_000, "oid": "a"}},
    {"type": "file", "path": "BF16/M-BF16-00002-of-00002.gguf", "size": 100, "lfs": {"size": 4_000_000_000, "oid": "b"}},
    {"type": "file", "path": "MTP/mtp-M-Q4_0.gguf", "lfs": {"size": 1_000_000_000}},
    {"type": "file", "path": "M-Q4_K_M.gguf", "lfs": {"size": 16_000_000_000, "oid": "c"}},
    {"type": "file", "path": "M-Q6_K.gguf", "lfs": {"size": 22_000_000_000}},
    {"type": "file", "path": "M-Q8_0.gguf", "lfs": {"size": 29_000_000_000}},
    {"type": "file", "path": "M-UD-IQ2_XXS.gguf", "lfs": {"size": 7_000_000_000}},
    {"type": "file", "path": "imatrix_unsloth.gguf", "lfs": {"size": 13_000_000}},
    {"type": "file", "path": "mmproj-F16.gguf", "lfs": {"size": 900_000_000}},
    {"type": "file", "path": "README.md", "size": 5000},
]


def test_parse_tree():
    f = hf.parse_tree("org/M", TREE)
    assert [q.quant for q in f.quants] == ["UD-IQ2_XXS", "Q4_K_M", "Q6_K", "Q8_0", "BF16"]
    bf16 = f.quants[-1]
    assert bf16.is_split and bf16.size == 54_000_000_000 and bf16.sha256 == ["a", "b"]
    assert [m.quant for m in f.mmproj] == ["F16"]
    assert set(f.other_gguf) == {"MTP/mtp-M-Q4_0.gguf", "imatrix_unsloth.gguf"}


def test_duplicate_quant_labels():
    tree = [{"type": "file", "path": f"LFM2.5-350M-{q}.gguf", "lfs": {"size": s}}
            for q, s in [("Q4_0", 219), ("QAD-Q4_0", 220), ("Q8_0", 379)]]
    f = hf.parse_tree("x/y", tree)
    assert [q.quant for q in f.quants] == ["Q4_0", "QAD-Q4_0", "Q8_0"]


def test_recommend_quant_by_machine():
    f = hf.parse_tree("org/M", TREE)
    assert hf.recommend_quant(f, apple(48), 27e9, n_ctx_train=262144).quant == "Q6_K"   # 8비트는 제외
    assert hf.recommend_quant(f, apple(32), 27e9, n_ctx_train=262144).quant == "Q4_K_M"
    assert hf.recommend_quant(f, apple(24), 27e9, n_ctx_train=262144).quant == "UD-IQ2_XXS"
    assert hf.recommend_quant(f, nvidia(8, 32), 27e9) is None
    small = hf.parse_tree("org/S", [{"type": "file", "path": "S-Q4_K_M.gguf", "lfs": {"size": 2_500_000_000}},
                                    {"type": "file", "path": "S-Q8_0.gguf", "lfs": {"size": 4_300_000_000}}])
    assert hf.recommend_quant(small, apple(48), 4e9).quant == "Q8_0"  # 작은 모델은 8비트 허용


def test_params_from_name():
    assert hf.params_from_name("Qwen3-30B-A3B-Instruct-GGUF") == 30
    assert hf.params_from_name("gemma-3-270m-GGUF") is None
    assert hf.params_from_name("Llama-3.2-1.5B") == 1.5
    assert hf.params_from_name("EXAONE-4.0-1.2B-GGUF") == 1.2


API_SAMPLE = [
    {"id": "unsloth/Good-8B-GGUF", "downloads": 100000, "likes": 50, "pipeline_tag": "text-generation",
     "lastModified": "2026-10-01T00:00:00.000Z", "tags": ["gguf", "license:apache-2.0"],
     "gguf": {"total": 8_000_000_000, "context_length": 131072, "chat_template": "{% if tools %}"},
     "cardData": {"base_model": ["org/Good-8B"]}},
    {"id": "random/Good-8B-Uncensored-GGUF", "downloads": 900000, "pipeline_tag": "text-generation",
     "gguf": {"total": 8_000_000_000}},
    {"id": "random/Huge-400B-GGUF", "downloads": 5000000, "pipeline_tag": "text-generation",
     "gguf": {"total": 400_000_000_000}},
    {"id": "nomic-ai/embed-GGUF", "downloads": 300000, "pipeline_tag": "sentence-similarity",
     "gguf": {"total": 100_000_000}},
    {"id": "someone/Tiny-1B-GGUF", "downloads": 2000, "pipeline_tag": "text-generation"},
    {"id": "voice/TTS-GGUF", "downloads": 3000000, "pipeline_tag": "text-to-speech", "gguf": {"total": 1e9}},
]


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


class FakeSession:
    def __init__(self, data=API_SAMPLE, status=200):
        self.headers, self.calls, self.data, self.status = {}, [], data, status

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        return FakeResp(self.data, self.status)


def client(tmp_path, **kw):
    return hf.HFClient(cache_dir=tmp_path / "cache", session=FakeSession(**kw))


def test_model_from_api():
    m = hf.HFModel.from_api(API_SAMPLE[0])
    assert m.author == "unsloth" and m.params_b == 8 and m.context_length == 131072
    assert m.supports_tools and m.license == "apache-2.0" and m.base_model == "org/Good-8B"
    assert hf.HFModel.from_api(API_SAMPLE[4]).params_b == 1


def test_search_filters_and_ranking(tmp_path):
    c = client(tmp_path)
    res = c.search(category="chat", system=apple(48))
    ids = [m.id for m in res]
    assert "random/Good-8B-Uncensored-GGUF" not in ids      # 위험 키워드 숨김
    assert "nomic-ai/embed-GGUF" not in ids                   # 대화 카테고리에서 임베딩 제외
    assert ids[0] == "unsloth/Good-8B-GGUF"                   # 신뢰 제작자 + 적합 → 최상위
    assert ids.index("random/Huge-400B-GGUF") > ids.index("unsloth/Good-8B-GGUF")
    shown = c.search(category="chat", system=apple(48), hide_risky=False)
    assert "random/Good-8B-Uncensored-GGUF" in [m.id for m in shown]


def test_search_categories(tmp_path):
    c = client(tmp_path)
    assert [m.id for m in c.search(category="recommended", system=apple(48))] == ["unsloth/Good-8B-GGUF"]
    light = [m.id for m in c.search(category="light", system=make_system(8))]
    assert light == ["someone/Tiny-1B-GGUF"] or set(light) == {"someone/Tiny-1B-GGUF", "voice/TTS-GGUF"}
    assert "voice/TTS-GGUF" not in [m.id for m in c.search(category="coding", system=apple(48))]
    assert [m.id for m in c.search(category="embedding")][0] == "nomic-ai/embed-GGUF"


def test_search_query_params_and_cache(tmp_path):
    c = client(tmp_path)
    c.search(text="qwen coder")
    url, params = c.session.calls[0]
    assert url.endswith("/api/models")
    assert ("filter", "gguf") in params and ("search", "qwen coder") in params
    assert ("expand[]", "gguf") in params
    n = len(c.session.calls)
    c.search(text="qwen coder")
    assert len(c.session.calls) == n  # 6시간 캐시


def test_search_rate_limit_message(tmp_path):
    c = client(tmp_path, status=429)
    with pytest.raises(hf.HFError, match="잠시"):
        c.search(text="x", use_cache=False)


def test_offline_uses_stale_cache(tmp_path):
    c = client(tmp_path)
    c.search(text="abc")
    for p in (tmp_path / "cache").glob("*.json"):
        import os
        os.utime(p, (0, 0))  # 캐시를 오래된 것으로

    def boom(*a, **k):
        import requests
        raise requests.ConnectionError()
    c.session.get = boom
    assert c.search(text="abc")  # 오래된 캐시라도 보여준다
    with pytest.raises(hf.HFError, match="인터넷"):
        c.search(text="never-cached")


def test_summarize_readme():
    md = """---
license: mit
---
# Title
![img](x.png)

Try LFM •
Docs •
LEAP •
Discord

| a | b |
|---|---|

This is the **first** real paragraph describing the [model](http://x) in detail for users.

```bash
llama-server -m x
```

Second paragraph that is also long enough to be included in the summary text.
"""
    s = hf.summarize_readme(md)
    assert s.startswith("This is the first real paragraph describing the model")
    assert "llama-server" not in s and "license" not in s


def test_catalog_loads():
    cat = hf.Catalog.load()
    ids = [c.id for c in cat.categories]
    assert {"recommended", "chat", "korean", "coding", "reasoning", "vision", "embedding", "light"} <= set(ids)
    assert cat.is_trusted("Unsloth") and cat.is_hidden("x/Model-Abliterated-GGUF")
