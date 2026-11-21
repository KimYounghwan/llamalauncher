import struct
import sys
from pathlib import Path

import pytest

from llamalauncher.app.core import gguf, netutil, server_log
from llamalauncher.app.core.launchers import llamacpp
from llamalauncher.app.core.recommender import ModelSpec, ServerOptions, build_server_args, format_command, split_args

FAKE_SERVER = Path(__file__).parent / "fake_llama_server.py"


# ---------------------------------------------------------------- GGUF
def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, kv: list[tuple[str, int, bytes]]) -> None:
    data = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv))
    for key, vtype, payload in kv:
        data += _s(key) + struct.pack("<I", vtype) + payload
    path.write_bytes(data + b"\0" * 1024)


def basic_kv(ctx=32768):
    return [
        ("general.architecture", 8, _s("llama")),
        ("general.name", 8, _s("Test Model")),
        ("general.size_label", 8, _s("8B")),
        ("llama.block_count", 4, struct.pack("<I", 32)),
        ("llama.context_length", 4, struct.pack("<I", ctx)),
        ("llama.embedding_length", 4, struct.pack("<I", 4096)),
        ("llama.attention.head_count", 4, struct.pack("<I", 32)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", 8)),
        # 큰 문자열 배열(어휘)은 건너뛰어야 한다
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1000) + b"".join(_s(f"t{i}") for i in range(1000))),
        ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 1000) + b"\0" * 4000),
        ("tokenizer.chat_template", 8, _s("{% if tools %}{{ tools }}{% endif %}")),
        ("general.file_type", 4, struct.pack("<I", 15)),
    ]


def test_gguf_inspect(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, basic_kv())
    info = gguf.inspect(p)
    assert info.architecture == "llama" and info.name == "Test Model" and info.size_label == "8B"
    assert (info.n_layers, info.n_ctx_train, info.n_head_kv) == (32, 32768, 8)
    assert info.head_dim == 128  # 4096 / 32
    assert info.supports_tools
    spec = ModelSpec.from_gguf(info)
    assert spec.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2


def test_gguf_per_layer_kv_heads_array(tmp_path):
    kv = [k for k in basic_kv() if k[0] != "llama.attention.head_count_kv"]
    kv.append(("llama.attention.head_count_kv", 9, struct.pack("<IQ", 4, 3) + struct.pack("<III", 0, 8, 4)))
    p = tmp_path / "m.gguf"
    write_gguf(p, kv)
    assert gguf.inspect(p).n_head_kv == 8


def test_gguf_rejects_non_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 100)
    with pytest.raises(gguf.GGUFError):
        gguf.inspect(p)
    p.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 5))
    with pytest.raises(gguf.GGUFError):
        gguf.inspect(p)  # 잘린 파일


def test_gguf_split_parts(tmp_path):
    first = tmp_path / "big-00001-of-00003.gguf"
    write_gguf(first, basic_kv())
    (tmp_path / "big-00002-of-00003.gguf").write_bytes(b"x" * 500)
    info = gguf.inspect(first)
    assert [p.name for p in info.missing_parts] == ["big-00003-of-00003.gguf"]
    assert info.file_size == first.stat().st_size + 500
    assert gguf.split_parts(tmp_path / "single.gguf") == [tmp_path / "single.gguf"]


# ---------------------------------------------------------------- llama.cpp 감지
REAL_HELP = """
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
-ctk,  --cache-type-k TYPE              KV cache data type for K
-ctv,  --cache-type-v TYPE              KV cache data type for V
--mlock                                 force system to keep model in RAM
--no-mmap                               do not memory-map model
--min-p N                               min-p sampling
-a,    --alias STRING                   set model name aliases
--api-key KEY                           API key to use for authentication
--jinja, --no-jinja                     whether to use jinja template engine for chat
--mmproj FILE                           path to a multimodal projector file
"""


def test_parse_version():
    assert llamacpp.parse_version("version: 1.0.0 (build 12359, commit b1b770d90)\nbuilt with") == ("1.0.0 (b12359)", 12359)
    assert llamacpp.parse_version("version: 4567 (abc1234)\nbuilt with clang") == ("b4567", 4567)
    assert llamacpp.parse_version("garbage") == ("", None)


def test_parse_capabilities():
    caps = llamacpp.parse_capabilities(REAL_HELP)
    assert {"fa_value", "alias", "api_key", "jinja", "mmproj", "cache_type_v", "min_p", "mlock", "no_mmap"} <= caps
    old = llamacpp.parse_capabilities("-fa,   --flash-attn    enable Flash Attention\n--alias X")
    assert "fa_flag" in old and "fa_value" not in old and "jinja" not in old


def test_detect_with_custom_path(tmp_path):
    exe = tmp_path / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    exe.write_text("")
    outputs = {"--version": "version: 1.0.0 (build 9000, commit x)", "--help": REAL_HELP}
    launcher = llamacpp.LlamaCppLauncher(runner=lambda cmd: outputs[cmd[1]])
    st = launcher.detect(str(tmp_path))  # 폴더를 지정해도 찾는다
    assert st.installed and st.path == exe and st.build == 9000 and "fa_value" in st.capabilities
    missing = launcher.detect(str(tmp_path / "nope"))
    assert not missing.installed and "찾지 못했" in missing.message


def test_detect_not_installed(monkeypatch):
    monkeypatch.setattr(llamacpp.shutil, "which", lambda name: None)
    monkeypatch.setattr(llamacpp, "candidate_paths", lambda: [])
    st = llamacpp.LlamaCppLauncher(runner=lambda cmd: None).detect()
    assert not st.installed


# ---------------------------------------------------------------- 명령어 만들기
def test_build_args_old_llamacpp_caps():
    o = ServerOptions(flash_attn="on", cache_type_v="q8_0", cache_type_k="q8_0", jinja=True, alias="x", min_p=0.1)
    args = build_server_args(o, "m.gguf", caps={"fa_flag"})
    i = args.index("-fa")
    assert args[i + 1] != "on"  # 예전 방식: 값 없이 플래그만
    assert "--cache-type-v" not in args and "--jinja" not in args and "--alias" not in args and "--min-p" not in args
    assert "--cache-type-k" in args


def test_build_args_api_key_and_extra():
    o = ServerOptions(api_key="sk-secret", extra_args='--seed 42 --system-prompt-file "my file.txt"')
    args = build_server_args(o, "m.gguf")
    assert args[args.index("--api-key") + 1] == "sk-secret"
    assert args[-4:] == ["--seed", "42", "--system-prompt-file", "my file.txt"]
    shown = format_command(["llama-server", *args], api_key="sk-secret")
    assert "sk-secret" not in shown and "********" in shown


def test_split_args_unbalanced_quote():
    assert split_args('--foo "bar') == ["--foo", '"bar']


def test_options_roundtrip_ignores_unknown():
    o = ServerOptions(ctx_size=8192, alias="a")
    back = ServerOptions.from_dict({**o.to_dict(), "unknown_key": 1})
    assert back == o


# ---------------------------------------------------------------- 로그 해석 / 네트워크
def test_log_classify_and_speed():
    assert server_log.classify("llama_model_load: error loading model") == "error"
    assert server_log.classify("main: server is listening on http://127.0.0.1:8080") == "ok"
    assert server_log.classify("load_tensors: offloading 32 layers") == "info"
    s = server_log.parse_speed("prompt eval time =  50.00 ms /  100 tokens (  0.50 ms per token,  2000.00 tokens per second)")
    assert s.kind == "prompt" and s.tokens_per_sec == 2000.0
    s = server_log.parse_speed("       eval time = 1000.00 ms /   45 tokens (  22.22 ms per token,    45.00 tokens per second)")
    assert s.kind == "generate" and s.tokens_per_sec == 45.0
    assert server_log.parse_speed("random line") is None


@pytest.mark.parametrize("line,keyword", [
    ("ggml_backend_metal_buffer_from_ptr: error: failed to allocate buffer", "메모리가 부족"),
    ("cudaMalloc failed: out of memory", "메모리가 부족"),
    ("couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080", "포트"),
    ("llama_model_load: error loading model architecture: unknown model architecture: 'foo'", "업데이트"),
    ("gguf_init_from_file: invalid magic characters", "손상"),
    ("error: invalid argument: --bogus", "옵션"),
])
def test_explain_error(line, keyword):
    assert keyword in server_log.explain_error(["some info", line])


def test_ports():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        busy = s.getsockname()[1]
        assert not netutil.is_port_free(busy)
        free = netutil.find_free_port(busy)
        assert free is not None and free != busy
    # 방금 연결이 오갔던 포트(TIME_WAIT)도 비어 있는 것으로 판단해야 한다
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    cli = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.close()  # 서버 쪽이 먼저 닫으면 서버 포트에 TIME_WAIT가 남는다
    cli.close()
    srv.close()
    if sys.platform != "win32":
        assert netutil.is_port_free(port)
    assert netutil.client_host("0.0.0.0") == "127.0.0.1"
    assert netutil.generate_api_key().startswith("sk-local-")


# ---------------------------------------------------------------- 서버 프로세스 제어 (실제 QProcess)
def _wait(qapp, cond, timeout=10.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_controller_start_ready_stop(qapp):
    from llamalauncher.app.core.server import ServerController, ServerState
    port = netutil.find_free_port(18080)
    c = ServerController()
    states, speeds, lines = [], [], []
    c.state_changed.connect(states.append)
    c.speed_changed.connect(lambda k, v: speeds.append((k, v)))
    c.log_line.connect(lambda l, lvl: lines.append((l, lvl)))
    c.start(sys.executable, [str(FAKE_SERVER), "--port", str(port)], "127.0.0.1", port)
    assert c.state == ServerState.STARTING
    assert _wait(qapp, lambda: c.state == ServerState.RUNNING), lines
    assert _wait(qapp, lambda: speeds)
    assert speeds[0] == ("generate", 42.5)
    assert any(lvl == "ok" for _, lvl in lines)
    c.stop()
    assert _wait(qapp, lambda: c.state == ServerState.STOPPED)
    assert states == [ServerState.STARTING, ServerState.RUNNING, ServerState.STOPPING, ServerState.STOPPED]


def test_controller_crash_gives_hint(qapp):
    from llamalauncher.app.core.server import ServerController, ServerState
    c = ServerController()
    hints = []
    c.error_hint.connect(hints.append)
    c.start(sys.executable, [str(FAKE_SERVER), "--fail", "ggml_metal: failed to allocate buffer of size 30 GB"],
            "127.0.0.1", 18999)
    assert _wait(qapp, lambda: c.state == ServerState.ERROR)
    assert hints and "메모리가 부족" in hints[0]


def test_controller_missing_program(qapp):
    from llamalauncher.app.core.server import ServerController, ServerState
    c = ServerController()
    hints = []
    c.error_hint.connect(hints.append)
    c.start("/nonexistent/llama-server", [], "127.0.0.1", 18998)
    assert _wait(qapp, lambda: c.state == ServerState.ERROR)
    assert "실행하지 못했" in hints[0]


def test_pid_file_and_orphan_detection(qapp, tmp_path, monkeypatch):
    from llamalauncher.app.core import server
    from llamalauncher.app.core.server import ServerController, ServerState
    monkeypatch.setattr(server, "pid_file", lambda: tmp_path / "server.pid")
    port = netutil.find_free_port(18180)
    c = ServerController()
    c.start(sys.executable, [str(FAKE_SERVER), "--port", str(port)], "127.0.0.1", port)
    assert _wait(qapp, lambda: (tmp_path / "server.pid").exists())
    assert int((tmp_path / "server.pid").read_text()) == c.pid
    assert server.find_orphan_server() is None  # 이름이 llama-server가 아니면 무시
    c.stop()
    assert _wait(qapp, lambda: c.state == ServerState.STOPPED)
    assert not (tmp_path / "server.pid").exists()
