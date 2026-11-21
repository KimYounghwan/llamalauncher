import pytest

from llamalauncher.app.core.hardware import GB
from llamalauncher.app.core.recommender import (
    Fit, ModelSpec, Preset, ServerOptions, assess_fit, build_server_args, max_comfortable_params,
    recommend, summary_sentence,
)
from tests.test_hardware import apple, make_system, nvidia

LLAMA3_8B = ModelSpec(file_size=int(4.9 * GB), n_layers=32, n_ctx_train=131072, n_head_kv=8, head_dim=128)


def test_estimate_file_size():
    assert ModelSpec.estimate(8, "Q4_K_M").file_size == pytest.approx(4.85e9, rel=0.01)
    assert ModelSpec.estimate(8, "Q8_0").file_size > ModelSpec.estimate(8, "Q4_K_M").file_size


def test_kv_cache_math_llama3_8b():
    # 2 * 32층 * 8헤드 * 128 * 2바이트 = 128KB/토큰 → 8K 토큰이면 1GB
    assert LLAMA3_8B.kv_bytes_per_token() == 131072
    assert LLAMA3_8B.kv_bytes(8192) == GB
    assert LLAMA3_8B.kv_bytes(8192, "q8_0") < LLAMA3_8B.kv_bytes(8192)


def test_fit_levels():
    assert assess_fit(apple(48), LLAMA3_8B).fit == Fit.COMFORTABLE
    assert assess_fit(nvidia(4, 16), LLAMA3_8B).fit == Fit.SLOW
    assert assess_fit(make_system(8), ModelSpec.estimate(70)).fit == Fit.INSUFFICIENT
    assert assess_fit(make_system(16), ModelSpec.estimate(3)).fit == Fit.COMFORTABLE  # 작은 모델은 CPU도 OK
    assert assess_fit(make_system(32), ModelSpec.estimate(14)).fit == Fit.SLOW


def test_recommend_m4_max_8b_full_gpu():
    rec = recommend(apple(48, perf=12), LLAMA3_8B, Preset.BALANCED)
    o = rec.options
    assert o.n_gpu_layers == 999
    assert o.ctx_size == 16384
    assert o.flash_attn == "on"
    assert o.threads == 12
    assert o.cache_type_k == "f16"
    assert rec.fit.fit == Fit.COMFORTABLE


def test_recommend_long_preset_caps_at_train_ctx():
    spec = ModelSpec(file_size=int(4.9 * GB), n_layers=32, n_ctx_train=32768, n_head_kv=8, head_dim=128)
    rec = recommend(apple(48), spec, Preset.LONG)
    assert rec.options.ctx_size == 32768


def test_recommend_fast_preset_small_ctx():
    assert recommend(apple(48), LLAMA3_8B, Preset.FAST).options.ctx_size == 4096


def test_recommend_tight_memory_uses_q8_kv_cache():
    # 32B Q4 (~19GB)는 36GB 맥(GPU 한도 ~27GB)에서 f16 캐시로는 긴 문맥이 어렵다
    spec = ModelSpec.estimate(32, n_layers=64, n_head_kv=8, head_dim=128, n_ctx_train=131072)
    rec = recommend(apple(36), spec, Preset.LONG)
    assert rec.options.n_gpu_layers == 999
    assert rec.options.cache_type_k == "q8_0"
    assert rec.options.flash_attn == "on"
    assert spec.memory_needed(rec.options.ctx_size, "q8_0") <= apple(36).gpu_budget


def test_recommend_partial_offload_on_small_gpu():
    spec = ModelSpec.estimate(14, n_layers=48)
    rec = recommend(nvidia(8, 32), spec, Preset.BALANCED)
    assert 0 < rec.options.n_gpu_layers < 48
    assert rec.fit.fit == Fit.SLOW


def test_recommend_unified_memory_too_big_model():
    rec = recommend(apple(48), ModelSpec.estimate(70))
    assert rec.fit.fit == Fit.INSUFFICIENT
    assert rec.options.n_gpu_layers == 999
    assert not any("층만" in n for n in rec.notes)


def test_recommend_cpu_only():
    rec = recommend(make_system(16, physical=6), ModelSpec.estimate(4), Preset.BALANCED)
    assert rec.options.n_gpu_layers == 0
    assert rec.options.flash_attn == "auto"
    assert rec.options.threads == 6


def test_recommend_never_exceeds_budget():
    for system in [apple(8), apple(16), apple(48), nvidia(6, 16), nvidia(24, 64), make_system(8)]:
        for params in [1, 4, 8, 14, 32]:
            spec = ModelSpec.estimate(params)
            o = recommend(system, spec).options
            needed = spec.memory_needed(o.ctx_size, o.cache_type_k)
            if assess_fit(system, spec, 2048).fit != Fit.INSUFFICIENT:
                assert needed <= system.total_budget, (system.total_ram, params)


def test_build_server_args():
    opts = ServerOptions(ctx_size=8192, n_gpu_layers=999, threads=12, flash_attn="on",
                         cache_type_k="q8_0", cache_type_v="q8_0")
    args = build_server_args(opts, "/m/model.gguf", alias="local-coder")
    assert args[:2] == ["-m", "/m/model.gguf"]
    assert args[args.index("-c") + 1] == "8192"
    assert args[args.index("-ngl") + 1] == "999"
    assert args[args.index("-fa") + 1] == "on"
    assert "--cache-type-k" in args and "--jinja" in args
    assert args[-2:] == ["--alias", "local-coder"]
    assert "--cache-type-k" not in build_server_args(ServerOptions(), "x.gguf")


def test_summary():
    assert max_comfortable_params(apple(48)) >= 32
    assert max_comfortable_params(apple(8)) is not None and max_comfortable_params(apple(8)) <= 8
    assert "원활" in summary_sentence(apple(48))
