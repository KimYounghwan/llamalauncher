"""5단계: 런처 어댑터, 설치 실행기, 폴더형(MLX/safetensors) 모델, 엔진별 서버 실행."""

import io
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from llamalauncher.app.core import installer as inst
from llamalauncher.app.core import model_library as lib
from llamalauncher.app.core.hardware import GPUInfo, SystemInfo
from llamalauncher.app.core.hf_client import parse_repo_snapshot, quant_hint, repo_variant_label
from llamalauncher.app.core.launchers import all_launchers, base, by_id
from llamalauncher.app.core.launchers.base import (
    CommandStep, GithubReleaseStep, InstallPlan, LauncherStatus, RemovePathStep,
)
from llamalauncher.app.core.launchers.ollama import model_name_for
from llamalauncher.app.core.recommender import ServerOptions
from tests.test_stage4 import make_model

LLAMA_ASSETS = [
    "cudart-llama-b12602-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-13.5-x64.zip", "llama-b12602-bin-macos-arm64.tar.gz",
    "llama-b12602-bin-macos-x64.tar.gz", "llama-b12602-bin-ubuntu-cuda-12.8-x64.tar.gz",
    "llama-b12602-bin-ubuntu-vulkan-x64.tar.gz", "llama-b12602-bin-ubuntu-x64.tar.gz",
    "llama-b12602-bin-ubuntu-arm64.tar.gz", "llama-b12602-bin-win-cpu-arm64.zip", "llama-b12602-bin-win-cpu-x64.zip",
    "llama-b12602-bin-win-cuda-12.4-x64.zip", "llama-b12602-bin-win-cuda-13.5-x64.zip",
    "llama-b12602-bin-win-vulkan-x64.zip", "llama-b12602-xcframework.zip",
]


def system(vendor="apple", arch="arm64", os_name="macOS"):
    g = [GPUInfo("GPU", vendor, 8 << 30, unified=vendor == "apple")] if vendor else []
    return SystemInfo(os_name, "1", arch, "CPU", 8, 8, None, 32 << 30, 16 << 30, gpus=g)


# ---------------------------------------------------------------- 배포 파일 고르기
@pytest.mark.parametrize("platform,machine,gpu,expected", [
    ("darwin", "arm64", "apple", ["llama-b12602-bin-macos-arm64.tar.gz"]),
    ("darwin", "x86_64", "intel", ["llama-b12602-bin-macos-x64.tar.gz"]),
    ("win32", "AMD64", "nvidia", ["llama-b12602-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]),
    ("win32", "AMD64", "amd", ["llama-b12602-bin-win-vulkan-x64.zip"]),
    ("win32", "AMD64", "unknown", ["llama-b12602-bin-win-cpu-x64.zip"]),
    ("win32", "ARM64", "unknown", ["llama-b12602-bin-win-cpu-arm64.zip"]),
    ("linux", "x86_64", "nvidia", ["llama-b12602-bin-ubuntu-cuda-12.8-x64.tar.gz",
                                   "cudart-llama-b12602-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
    ("linux", "x86_64", "intel", ["llama-b12602-bin-ubuntu-vulkan-x64.tar.gz"]),
    ("linux", "aarch64", "unknown", ["llama-b12602-bin-ubuntu-arm64.tar.gz"]),
])
def test_pick_llamacpp_assets(platform, machine, gpu, expected):
    assert inst.pick_llamacpp_assets(LLAMA_ASSETS, platform, machine, gpu) == expected


def test_pick_llamacpp_none_matching():
    assert inst.pick_llamacpp_assets(["nightly-tag.txt"], "darwin", "arm64", "apple") == []


def test_pick_koboldcpp():
    names = ["koboldcpp-linux-x64", "koboldcpp-linux-x64-nocuda", "koboldcpp-mac-arm64", "koboldcpp-nocuda.exe",
             "koboldcpp.exe"]
    assert inst.pick_koboldcpp_asset(names, "darwin", "arm64", "apple") == ["koboldcpp-mac-arm64"]
    assert inst.pick_koboldcpp_asset(names, "darwin", "x86_64", "intel") == []
    assert inst.pick_koboldcpp_asset(names, "win32", "AMD64", "nvidia") == ["koboldcpp.exe"]
    assert inst.pick_koboldcpp_asset(names, "win32", "AMD64", "amd") == ["koboldcpp-nocuda.exe"]
    assert inst.pick_koboldcpp_asset(names, "linux", "x86_64", "intel") == ["koboldcpp-linux-x64-nocuda"]


# ---------------------------------------------------------------- 압축 풀기
def _tar_with_top(path: Path):
    with tarfile.open(path, "w:gz") as t:
        for name, data in [("llama-b1/llama-server", b"#!/bin/sh\necho hi\n"), ("llama-b1/libggml.1.dylib", b"lib")]:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o755
            t.addfile(ti, io.BytesIO(data))
        link = tarfile.TarInfo("llama-b1/libggml.dylib")
        link.type = tarfile.SYMTYPE
        link.linkname = "libggml.1.dylib"
        t.addfile(link)


def test_extract_tar_strips_top_folder_and_keeps_symlink(tmp_path):
    a = tmp_path / "a.tar.gz"
    _tar_with_top(a)
    dest = tmp_path / "out"
    inst.extract_archive(a, dest)
    assert (dest / "llama-server").is_file() and (dest / "llama-server").stat().st_mode & 0o100
    assert (dest / "libggml.dylib").is_symlink()
    inst.extract_archive(a, dest)  # 업데이트(덮어쓰기)도 된다
    assert (dest / "llama-server").is_file()


def test_extract_zip_flat_and_rejects_traversal(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("llama-server.exe", b"x")
        zf.writestr("ggml.dll", b"y")
    inst.extract_archive(z, tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["ggml.dll", "llama-server.exe"]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", b"x")
    with pytest.raises(inst.InstallError):
        inst.extract_archive(bad, tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()


# ---------------------------------------------------------------- 설치 실행기
def test_installer_command_success_and_failure():
    logs = []
    i = inst.Installer(logs.append)
    i.run(InstallPlan("테스트", [CommandStep([sys.executable, "-c", "print('hello'); print('world')"], "인사")]))
    assert "  hello" in logs and logs[-1] == "✅ 완료"
    with pytest.raises(inst.InstallError, match="종료 코드 3"):
        i.run(InstallPlan("실패", [CommandStep([sys.executable, "-c", "import sys; sys.exit(3)"])]))
    with pytest.raises(inst.InstallError, match="실행하지 못했어요"):
        i.run(InstallPlan("없는 명령", [CommandStep(["/no/such/program"])]))


class _Resp:
    def __init__(self, status=200, js=None, content=b""):
        self.status_code, self._js, self.content = status, js, content
        self.headers = {"Content-Length": str(len(content))}

    def json(self):
        return self._js

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _GitHubSession:
    def __init__(self, archive: bytes):
        self.archive = archive

    def get(self, url, params=None, stream=False, timeout=None):
        if url.endswith("/releases"):
            return _Resp(js=[
                {"tag_name": "v1.0.0", "assets": [{"name": "nightly-tag.txt", "browser_download_url": "x"}]},
                {"tag_name": "beta", "prerelease": True, "assets": [
                    {"name": "llama-b99999-bin-macos-arm64.tar.gz", "browser_download_url": "https://dl/beta"}]},
                {"tag_name": "b12602", "assets": [
                    {"name": "llama-b12602-bin-macos-arm64.tar.gz", "browser_download_url": "https://dl/mac",
                     "size": len(self.archive)}]},
            ])
        assert url == "https://dl/mac"
        return _Resp(content=self.archive)


def test_llamacpp_plan_allows_prerelease():
    """llama.cpp는 정식 빌드도 prerelease로 표시하므로 받아야 한다 (실제 배포에서 확인됨)."""
    step = by_id("llamacpp")._github_plan(system()).steps[0]
    assert step.allow_prerelease and step.repo == "ggml-org/llama.cpp"


def test_installer_github_release_skips_empty_release(tmp_path):
    a = tmp_path / "a.tar.gz"
    _tar_with_top(a)
    dest = tmp_path / "bin" / "llamacpp"
    logs = []
    step = GithubReleaseStep("ggml-org/llama.cpp", dest,
                             lambda names: inst.pick_llamacpp_assets(names, "darwin", "arm64", "apple"))
    inst.Installer(logs.append, session=_GitHubSession(a.read_bytes())).run(InstallPlan("gh", [step]))
    assert (dest / "llama-server").is_file() and (dest / ".version").read_text() == "b12602"
    assert any("b12602" in l for l in logs)


def test_remove_step_only_inside_app_home(tmp_path, monkeypatch):
    monkeypatch.setattr(inst, "app_home", lambda: tmp_path / "LlamaLauncher")
    inside = tmp_path / "LlamaLauncher" / "bin" / "x"
    inside.mkdir(parents=True)
    outside = tmp_path / "keep"
    outside.mkdir()
    i = inst.Installer(lambda s: None)
    i.run(InstallPlan("rm", [RemovePathStep(inside)]))
    assert not inside.exists()
    with pytest.raises(inst.InstallError):
        i.run(InstallPlan("rm", [RemovePathStep(outside)]))
    assert outside.exists()


def test_installer_cancel_before_step():
    i = inst.Installer(lambda s: None)
    i.cancel.set()
    with pytest.raises(inst.Cancelled):
        i.run(InstallPlan("x", [CommandStep([sys.executable, "-c", "print(1)"])]))


# ---------------------------------------------------------------- 런처 어댑터
def test_all_launchers_have_metadata():
    ids = [l.id for l in all_launchers()]
    assert ids == ["llamacpp", "ollama", "lmstudio", "koboldcpp", "mlx", "vllm", "jan", "tgwui"]
    for l in all_launchers():
        assert l.name and l.description and l.homepage and l.formats


def test_availability_by_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert by_id("mlx").availability(system())[0]
    assert not by_id("mlx").availability(system("intel", "x86_64"))[0]
    assert not by_id("vllm").availability(system())[0]
    monkeypatch.setattr(sys, "platform", "linux")
    assert by_id("vllm").availability(system("nvidia", "x86_64", "Linux"))[0]
    ok, why = by_id("vllm").availability(system("amd", "x86_64", "Linux"))
    assert not ok and "NVIDIA" in why
    monkeypatch.setattr(sys, "platform", "win32")
    ok, why = by_id("vllm").availability(system("nvidia", "AMD64", "Windows"))
    assert not ok and "WSL" in why
    assert not by_id("mlx").availability(None)[0]


def test_install_plans_prefer_package_managers(monkeypatch):
    from llamalauncher.app.core.launchers import llamacpp, ollama
    monkeypatch.setattr(llamacpp, "has_brew", lambda: Path("/opt/homebrew/bin/brew"))
    monkeypatch.setattr(llamacpp, "has_winget", lambda: None)
    plans = by_id("llamacpp").install_plans(system())
    assert [p.label for p in plans] == ["Homebrew로 설치", "GitHub에서 받아 설치"]
    assert plans[0].steps[0].args == ["/opt/homebrew/bin/brew", "install", "llama.cpp"]
    monkeypatch.setattr(llamacpp, "has_brew", lambda: None)
    monkeypatch.setattr(llamacpp, "has_winget", lambda: Path("C:/winget.exe"))
    plans = by_id("llamacpp").install_plans(system("nvidia", "AMD64", "Windows"))
    assert plans[0].label == "winget으로 설치" and "ggml.llamacpp" in plans[0].steps[0].args
    monkeypatch.setattr(ollama, "has_brew", lambda: None)
    monkeypatch.setattr(ollama, "has_winget", lambda: None)
    assert by_id("ollama").install_plans(system())[-1].manual


def test_update_and_uninstall_follow_install_method(monkeypatch, tmp_path):
    from llamalauncher.app.core.launchers import llamacpp
    monkeypatch.setattr(llamacpp, "has_brew", lambda: Path("/opt/homebrew/bin/brew"))
    l = by_id("llamacpp")
    st = LauncherStatus(True, Path("/opt/homebrew/bin/llama-server"))
    assert l.update_plans(system(), st)[0].steps[0].args[-2:] == ["upgrade", "llama.cpp"]
    assert l.uninstall_plans(st)[0].steps[0].args[-2:] == ["uninstall", "llama.cpp"]
    monkeypatch.setattr(llamacpp, "bin_dir", lambda _id: tmp_path / "bin" / "llamacpp")
    own = tmp_path / "bin" / "llamacpp" / "llama-server"
    own.parent.mkdir(parents=True)
    own.write_text("x")
    st = LauncherStatus(True, own)
    assert isinstance(l.uninstall_plans(st)[0].steps[0], RemovePathStep)
    assert l.uninstall_plans(LauncherStatus(True, Path("/usr/bin/llama-server"))) == []


def test_ollama_detect_parses_version(monkeypatch, tmp_path):
    from llamalauncher.app.core.launchers import ollama
    exe = tmp_path / "ollama"
    exe.write_text("x")
    monkeypatch.setattr(ollama, "is_running", lambda *a, **k: False)
    st = ollama.OllamaLauncher(runner=lambda cmd: "Warning: could not connect\nollama version is 0.31.1\n").detect(str(exe))
    assert st.installed and st.version == "0.31.1"


def test_ollama_model_name():
    assert model_name_for("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf") == "llamalauncher/qwen3.8-27b-ud-q4_k_xl"
    assert model_name_for("/m/Big Model-Q8_0-00001-of-00002.gguf") == "llamalauncher/big-model-q8_0"


def test_serve_specs():
    o = ServerOptions(ctx_size=8192, n_gpu_layers=999, threads=8, flash_attn="on", cache_type_k="q8_0",
                      port=5001, api_key="k", alias="my", mmproj="/m/mmproj.gguf")
    st = LauncherStatus(True, Path("/bin/koboldcpp"))
    spec = by_id("koboldcpp").serve_spec(st, "/m/a.gguf", o, system("nvidia", "x86_64"))
    a = spec.args
    assert a[:2] == ["--model", "/m/a.gguf"] and "--usecublas" in a and "--flashattention" in a
    assert a[a.index("--quantkv") + 1] == "1" and a[a.index("--password") + 1] == "k"
    assert a[a.index("--mmproj") + 1] == "/m/mmproj.gguf" and spec.health_path == "/api/v1/model"

    spec = by_id("mlx").serve_spec(LauncherStatus(True, Path("/v/bin/mlx_lm.server")), "/m/Qwen-4bit", o)
    assert spec.args == ["--model", "/m/Qwen-4bit", "--host", "127.0.0.1", "--port", "5001"]
    assert spec.model_name == "/m/Qwen-4bit"

    spec = by_id("vllm").serve_spec(LauncherStatus(True, Path("/v/bin/vllm")), "/m/Qwen-AWQ", o)
    a = spec.args
    assert a[:2] == ["serve", "/m/Qwen-AWQ"] and a[a.index("--max-model-len") + 1] == "8192"
    assert a[a.index("--served-model-name") + 1] == "my" and a[a.index("--api-key") + 1] == "k"

    o.port = 11435
    spec = by_id("ollama").serve_spec(LauncherStatus(True, Path("/bin/ollama")), "/m/Qwen3-0.6B-Q4_K_M.gguf", o)
    assert spec.args == ["serve"] and spec.health_path == "/api/version"
    assert spec.env["OLLAMA_HOST"] == "127.0.0.1:11435" and spec.env["OLLAMA_CONTEXT_LENGTH"] == "8192"
    assert spec.env["OLLAMA_KV_CACHE_TYPE"] == "q8_0" and spec.env["OLLAMA_FLASH_ATTENTION"] == "1"
    assert spec.model_name == "llamalauncher/qwen3-0.6b-q4_k_m" and spec.prepare is not None
    assert any("API 키" in n for n in spec.notes)

    with pytest.raises(NotImplementedError):
        by_id("lmstudio").serve_spec(st, "/m/a.gguf", o)


def test_app_launcher_detects_mac_app(tmp_path, monkeypatch):
    from llamalauncher.app.core.launchers import others
    app = tmp_path / "LM Studio.app" / "Contents"
    app.mkdir(parents=True)
    import plistlib
    (app / "Info.plist").write_bytes(plistlib.dumps({"CFBundleShortVersionString": "0.4.28"}))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(others, "_port_open", lambda port: True)
    l = others.LMStudioLauncher()
    monkeypatch.setattr(l, "app_paths", lambda: [tmp_path / "LM Studio.app"])
    st = l.detect()
    assert st.installed and st.version == "0.4.28" and st.running and not l.runnable


# ---------------------------------------------------------------- 폴더형 모델 (MLX/safetensors)
def test_repo_snapshot_parsing():
    entries = [
        {"type": "file", "path": "config.json", "size": 100},
        {"type": "file", "path": "model-00001-of-00002.safetensors", "lfs": {"size": 4000, "oid": "a"}},
        {"type": "file", "path": "model-00002-of-00002.safetensors", "lfs": {"size": 2000, "oid": "b"}},
        {"type": "file", "path": "tokenizer.json", "size": 50},
        {"type": "file", "path": "README.md", "size": 5},
        {"type": "file", "path": ".gitattributes", "size": 5},
        {"type": "file", "path": "original/consolidated.safetensors", "lfs": {"size": 9999}},
        {"type": "file", "path": "pytorch_model.bin", "lfs": {"size": 9999}},
        {"type": "directory", "path": "original"},
    ]
    f = parse_repo_snapshot("mlx-community/Qwen3-8B-4bit", entries, "mlx")
    q = f.quants[0]
    assert q.quant == "4bit" and q.kind == "mlx" and q.size == 6150
    assert q.paths == ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                       "tokenizer.json"]
    assert parse_repo_snapshot("x/y", [{"type": "file", "path": "config.json"}], "mlx").quants == []


@pytest.mark.parametrize("repo,fmt,label,hint", [
    ("mlx-community/Qwen3-8B-4bit", "mlx", "4bit", "Q4_K_M"),
    ("lmstudio-community/Qwen3.8-27B-MLX-8bit", "mlx", "8bit", "Q8_0"),
    ("Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", "safetensors", "AWQ", "Q4_K_M"),
    ("Qwen/Qwen3.8-27B-FP8", "safetensors", "FP8", "Q8_0"),
    ("Qwen/Qwen2.5-7B-Instruct", "safetensors", "BF16 (원본)", "F16"),
])
def test_variant_labels(repo, fmt, label, hint):
    assert repo_variant_label(repo, fmt) == label and quant_hint(repo, fmt) == hint
    assert quant_hint(repo, "gguf") == "Q4_K_M"


def _model_dir(d: Path, mlx=True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"model_type": "qwen3", "max_position_embeddings": 40960, "num_hidden_layers": 28}
    if mlx:
        cfg["quantization"] = {"group_size": 64, "bits": 4}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"0" * 1000)
    return d


def test_library_finds_folder_models(tmp_path):
    home = tmp_path / "home"
    mdir = tmp_path / "models"
    _model_dir(mdir / "mlx" / "mlx-community__Qwen3-8B-4bit")
    _model_dir(mdir / "safetensors" / "Qwen__Qwen3-8B-AWQ", mlx=False)
    snap = _model_dir(home / ".cache" / "huggingface" / "hub" / "models--mlx-community--SmolLM2-135M-4bit"
                      / "snapshots" / "abc123")
    _model_dir(home / ".lmstudio" / "models" / "lmstudio-community" / "Gemma-MLX-8bit")
    src = lib.LibrarySources(models_dir=mdir, include_external=True, home=home)
    models = {m.display_name: m for m in lib.collect(src)}
    q = models["Qwen3-8B-4bit"]
    assert q.fmt == "mlx" and q.quant == "4bit" and q.repo == "mlx-community/Qwen3-8B-4bit"
    assert q.dir_meta["ctx"] == 40960 and q.size > 1000 and q.source == lib.Source.DOWNLOAD
    assert models["Qwen3-8B-AWQ"].fmt == "safetensors"
    hf = models["SmolLM2-135M-4bit"]
    assert hf.source == lib.Source.HFCACHE and hf.path == snap and hf.deletable
    assert models["Gemma-MLX-8bit"].source == lib.Source.LMSTUDIO


def test_delete_folder_model(tmp_path):
    mdir = tmp_path / "models"
    d = _model_dir(mdir / "mlx" / "a__b")
    m = lib.load_info(lib.LocalModel(path=d, source=lib.Source.DOWNLOAD, fmt="mlx"))
    lib.delete_model(m, mdir, [m])
    assert not d.exists() and mdir.exists()


def test_download_item_folder_main_path(tmp_path):
    from llamalauncher.app.core.downloader import DownloadItem, FileSpec
    item = DownloadItem("a/b", "4bit", [FileSpec("sub/config.json", 1, keep_path=True),
                                        FileSpec("model.safetensors", 1, keep_path=True)], str(tmp_path), kind="mlx")
    assert item.main_path == tmp_path and item.files[0].filename == "sub/config.json"
    again = DownloadItem.from_json(item.to_json())
    assert again.kind == "mlx" and again.files[0].keep_path


def test_folder_repo_download_keeps_structure(tmp_path):
    """폴더형 모델은 하위 폴더 구조를 유지하며 받고, 기록에는 폴더 경로가 남는다."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from llamalauncher.app.core.downloader import DownloadManager, FileSpec, Status, read_manifest
    files = {"config.json": b'{"model_type":"x"}', "sub/tokenizer.json": b"{}", "model.safetensors": b"9" * 5000}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.split("/resolve/main/", 1)[-1]
            data = files.get(name)
            self.send_response(200 if data is not None else 404)
            self.send_header("Content-Length", str(len(data or b"")))
            self.end_headers()
            self.wfile.write(data or b"")

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        mdir = tmp_path / "models"
        mgr = DownloadManager(tmp_path / "q.json", base_url=f"http://127.0.0.1:{srv.server_address[1]}")
        specs = [FileSpec(p, len(b), keep_path=True) for p, b in files.items()]
        item = mgr.add("a/b", "4bit", specs, mdir / "mlx" / "a__b", kind="mlx")
        end = time.time() + 10
        while time.time() < end and item.status != Status.DONE:
            time.sleep(0.05)
        assert item.status == Status.DONE, item.error
        assert (mdir / "mlx" / "a__b" / "sub" / "tokenizer.json").read_bytes() == b"{}"
        entry = read_manifest(mdir)[0]
        assert entry["path"] == str(mdir / "mlx" / "a__b") and entry["format"] == "mlx"
        mgr.shutdown()
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- 서버 컨트롤러: 준비 단계
FAKE = Path(__file__).parent / "fake_llama_server.py"


def _wait(qapp, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("fail", [False, True])
def test_controller_prepare_step(qapp, fail):
    from llamalauncher.app.core.netutil import find_free_port
    from llamalauncher.app.core.server import ServerController, ServerState
    c = ServerController()
    hints, logs = [], []
    c.error_hint.connect(hints.append)
    c.log_line.connect(lambda l, k: logs.append(l))
    port = find_free_port(18500)

    def prepare(log):
        log("모델 등록 중")
        time.sleep(0.2)
        if fail:
            raise RuntimeError("등록 실패")

    c.start(sys.executable, [str(FAKE), "--port", str(port), "--load-seconds", "0.2"], "127.0.0.1", port,
            health_path="/api/version", env={"X_TEST": "1"}, prepare=prepare)
    if fail:
        assert _wait(qapp, lambda: c.state == ServerState.ERROR, 15)
        assert hints == ["등록 실패"]
    else:
        assert _wait(qapp, lambda: c.state == ServerState.RUNNING, 15)
        assert "모델 등록 중" in logs
        c.stop_blocking()
    from PySide6.QtCore import QProcess
    assert c.proc is None or c.proc.state() == QProcess.NotRunning


def test_server_tab_engine_switch(qapp, monkeypatch):
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui import server_tab
    s = Settings()
    tab = server_tab.ServerTab(s, lambda: None)
    assert _wait(qapp, lambda: tab.launcher_status is not None)
    tab.port.setValue(8080)
    tab.set_engine("koboldcpp")
    assert s.server_engine == "koboldcpp" and tab.port.value() == 5001
    assert "--contextsize" in tab.cmd_view.toPlainText()
    tab.port.setValue(9999)
    tab.set_engine("llamacpp")
    assert tab.port.value() == 9999  # 사용자가 바꾼 포트는 유지
