"""이 컴퓨터에 맞는 실행 옵션 추천과 모델 적합도 계산.

메모리 계산은 llama.cpp 기준의 근사치다.
- 모델 가중치: GGUF 파일 크기 그대로
- KV 캐시: 2(K,V) x 레이어 수 x KV 헤드 수 x 헤드 크기 x 토큰 수 x 원소 크기
- 계산용 버퍼: 0.5GB + 가중치의 5%
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Optional

from .hardware import GB, Backend, SystemInfo, format_bytes

CTX_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

KV_BYTES = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}

# 양자화별 파라미터 1개당 평균 비트 수 (GGUF 근사치)
QUANT_BITS = {
    "Q2_K": 3.0, "Q3_K_M": 3.9, "Q4_0": 4.5, "Q4_K_M": 4.85, "Q5_K_M": 5.7,
    "Q6_K": 6.6, "Q8_0": 8.5, "F16": 16.0, "BF16": 16.0,
}


class Fit(str, Enum):
    COMFORTABLE = "comfortable"
    SLOW = "slow"
    INSUFFICIENT = "insufficient"

    @property
    def badge(self) -> str:
        return {"comfortable": "✅ 쾌적", "slow": "⚠️ 가능하지만 느림", "insufficient": "❌ 메모리 부족"}[self.value]


class Preset(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    LONG = "long"

    @property
    def label(self) -> str:
        return {"fast": "빠르게", "balanced": "균형", "long": "긴 대화용"}[self.value]

    @property
    def ctx_target(self) -> int:
        return {"fast": 4096, "balanced": 16384, "long": 131072}[self.value]


@dataclass
class ModelSpec:
    """추천 계산에 필요한 모델 정보. GGUF 메타데이터가 없으면 크기로 추정한다."""

    file_size: int
    n_layers: Optional[int] = None
    n_ctx_train: Optional[int] = None
    n_head_kv: Optional[int] = None
    head_dim: Optional[int] = None
    name: str = ""

    @classmethod
    def estimate(cls, params_b: float, quant: str = "Q4_K_M", **kw) -> "ModelSpec":
        """파라미터 수(십억 단위)와 양자화로 파일 크기를 추정한다."""
        bits = QUANT_BITS.get(quant.upper(), 4.85)
        return cls(file_size=int(params_b * 1e9 * bits / 8), **kw)

    @classmethod
    def from_gguf(cls, info) -> "ModelSpec":
        """gguf.GGUFInfo에서 만든다."""
        return cls(file_size=info.file_size, n_layers=info.n_layers, n_ctx_train=info.n_ctx_train,
                   n_head_kv=info.n_head_kv, head_dim=info.head_dim, name=info.display_name)

    @property
    def layers(self) -> int:
        if self.n_layers:
            return self.n_layers
        # 크기 기반 추정: 4.9GB(8B Q4) ≈ 32층, 크기의 제곱근에 비례
        return max(16, min(96, round(32 * math.sqrt(self.file_size / (4.9 * GB)))))

    def kv_bytes_per_token(self, cache_type: str = "f16") -> float:
        elem = KV_BYTES.get(cache_type, 2.0)
        if self.n_layers and self.n_head_kv and self.head_dim:
            return 2 * self.n_layers * self.n_head_kv * self.head_dim * elem
        # 메타데이터가 없으면 GQA 모델(KV 헤드 8개, 헤드 크기 128) 기준으로 추정
        return 2 * self.layers * 8 * 128 * elem

    def kv_bytes(self, ctx: int, cache_type: str = "f16") -> int:
        return int(self.kv_bytes_per_token(cache_type) * ctx)

    @property
    def overhead(self) -> int:
        return GB // 2 + int(self.file_size * 0.05)

    def memory_needed(self, ctx: int, cache_type: str = "f16") -> int:
        return self.file_size + self.kv_bytes(ctx, cache_type) + self.overhead


@dataclass
class FitResult:
    fit: Fit
    needed: int
    reason: str


def assess_fit(system: SystemInfo, model: ModelSpec, ctx: int = 4096) -> FitResult:
    """이 컴퓨터에서 모델이 얼마나 잘 돌아갈지 판단한다."""
    needed = model.memory_needed(ctx)
    gpu, total = system.gpu_budget, system.total_budget
    if gpu and needed <= gpu:
        return FitResult(Fit.COMFORTABLE, needed, "GPU 메모리에 모두 올라가서 빠르게 동작해요.")
    if not gpu:
        if needed <= total:
            if model.file_size <= 5 * GB:
                return FitResult(Fit.COMFORTABLE, needed, "작은 모델이라 CPU만으로도 쓸 만한 속도예요.")
            return FitResult(Fit.SLOW, needed, "GPU가 없어 CPU로만 실행해요. 답변이 느릴 수 있어요.")
        return FitResult(Fit.INSUFFICIENT, needed,
                         f"메모리가 부족해요. 약 {format_bytes(needed)} 필요하지만 {format_bytes(total)}만 쓸 수 있어요.")
    if needed <= total:
        return FitResult(Fit.SLOW, needed, "일부만 GPU에 올라가고 나머지는 CPU가 처리해서 느려요.")
    return FitResult(Fit.INSUFFICIENT, needed,
                     f"메모리가 부족해요. 약 {format_bytes(needed)} 필요하지만 {format_bytes(total)}만 쓸 수 있어요. "
                     "더 작은 양자화(Q4_K_M 등)나 작은 모델을 골라 보세요.")


@dataclass
class ServerOptions:
    ctx_size: int = 4096
    n_gpu_layers: int = 0
    threads: int = 4
    batch_size: int = 2048
    ubatch_size: int = 512
    flash_attn: str = "auto"  # on | off | auto
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    mlock: bool = False
    no_mmap: bool = False
    parallel: int = 1
    host: str = "127.0.0.1"
    port: int = 8080
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    jinja: bool = True
    alias: str = ""
    api_key: str = ""
    extra_args: str = ""  # 고급: 직접 추가하는 인자

    # 실행 환경(하드웨어/모델)에 따라 정해지는 값 — 사용자 프리셋에는 저장하지 않는다
    HARDWARE_FIELDS = ("ctx_size", "n_gpu_layers", "threads", "batch_size", "ubatch_size",
                       "flash_attn", "cache_type_k", "cache_type_v")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ServerOptions":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class Recommendation:
    options: ServerOptions
    fit: FitResult
    notes: list[str] = field(default_factory=list)


def _largest_ctx(model: ModelSpec, budget: int, cache_type: str, cap: int) -> Optional[int]:
    best = None
    for c in CTX_STEPS:
        if c > cap:
            break
        if model.memory_needed(c, cache_type) <= budget:
            best = c
    return best


def recommend(system: SystemInfo, model: ModelSpec, preset: Preset = Preset.BALANCED) -> Recommendation:
    """시스템과 모델을 보고 llama-server 옵션을 추천한다."""
    opts = ServerOptions(threads=system.recommended_threads)
    notes: list[str] = []
    has_gpu = system.gpu_budget > 0

    cap = preset.ctx_target
    if model.n_ctx_train:
        cap = min(cap, model.n_ctx_train)
    cap = max(cap, CTX_STEPS[0])

    # 1) 최소 문맥 길이로라도 전부 GPU에 올라가면 GPU 메모리 안에서만 계산한다
    fits_gpu = has_gpu and model.memory_needed(CTX_STEPS[0], "q8_0") <= system.gpu_budget
    budget = system.gpu_budget if fits_gpu else system.total_budget
    ctx_f16 = _largest_ctx(model, budget, "f16", cap)
    ctx_q8 = _largest_ctx(model, budget, "q8_0", cap)

    cache = "f16"
    ctx = ctx_f16
    if ctx_q8 and (ctx_f16 is None or ctx_q8 > ctx_f16) and (ctx_f16 or 0) < cap:
        cache, ctx = "q8_0", ctx_q8
        notes.append("메모리를 아끼려고 대화 기억 공간(KV 캐시)을 8비트로 압축했어요. 품질 차이는 거의 없어요.")
    if ctx is None:
        ctx = CTX_STEPS[0]
        cache = "q8_0"
        notes.append("메모리가 부족해서 실행이 실패할 가능성이 높아요. 더 작은 모델이나 양자화(Q4_K_M 등)를 권장해요.")
    opts.ctx_size = ctx
    opts.cache_type_k = opts.cache_type_v = cache

    # 2) GPU 레이어 수
    if not has_gpu:
        opts.n_gpu_layers = 0
        notes.append("GPU를 찾지 못해 CPU로 실행해요.")
    elif model.memory_needed(ctx, cache) <= system.gpu_budget:
        opts.n_gpu_layers = 999
        notes.append("모델 전체를 GPU에 올려요 (가장 빠름).")
    elif system.is_unified_memory:
        # 통합 메모리는 CPU/GPU가 같은 메모리를 쓰므로 나눠 올려도 이득이 없다
        opts.n_gpu_layers = 999
    else:
        free = system.gpu_budget - model.kv_bytes(ctx, cache) - model.overhead
        per_layer = model.file_size / model.layers
        opts.n_gpu_layers = max(0, min(model.layers, int(free / per_layer))) if free > 0 else 0
        notes.append(f"GPU에는 {opts.n_gpu_layers}/{model.layers}개 층만 올라가요. 나머지는 CPU가 처리해서 느릴 수 있어요.")

    # 3) Flash Attention: Metal/CUDA는 켜고, 나머지는 llama.cpp 자동 판단
    if system.backend in (Backend.METAL, Backend.CUDA) and opts.n_gpu_layers > 0:
        opts.flash_attn = "on"
    else:
        opts.flash_attn = "auto"
    if cache != "f16" and opts.flash_attn == "auto" and has_gpu:
        opts.flash_attn = "on"  # V 캐시 양자화에는 Flash Attention이 필요

    # 4) 배치 크기: 메모리가 빠듯하면 줄인다
    if system.fast_budget < 8 * GB:
        opts.batch_size, opts.ubatch_size = 512, 256

    notes.insert(0, f"CPU 스레드는 {opts.threads}개 (성능 코어 수 기준)를 사용해요.")
    if model.n_ctx_train and ctx >= model.n_ctx_train:
        notes.append(f"모델이 학습한 최대 문맥 길이({model.n_ctx_train:,})에 맞췄어요.")

    return Recommendation(options=opts, fit=assess_fit(system, model, ctx), notes=notes)


ALL_CAPS = frozenset({"fa_value", "alias", "api_key", "jinja", "cache_type_v", "min_p", "no_mmap", "mlock"})


def build_server_args(opts: ServerOptions, model_path: str, alias: Optional[str] = None,
                      caps: frozenset[str] | set[str] = ALL_CAPS) -> list[str]:
    """ServerOptions를 llama-server 명령줄 인자로 바꾼다 (실행 파일 경로 제외).

    caps: 설치된 llama-server가 지원하는 기능 (launchers.llamacpp.parse_capabilities)
    """
    args = [
        "-m", model_path,
        "-c", str(opts.ctx_size),
        "-ngl", str(opts.n_gpu_layers),
        "-t", str(opts.threads),
        "-b", str(opts.batch_size),
        "-ub", str(opts.ubatch_size),
    ]
    if "fa_value" in caps:
        args += ["-fa", opts.flash_attn]
    elif opts.flash_attn == "on":
        args.append("-fa")  # 예전 버전은 켜기 플래그만 있다
    args += [
        "-np", str(opts.parallel),
        "--host", opts.host,
        "--port", str(opts.port),
        "--temp", f"{opts.temperature:g}",
        "--top-p", f"{opts.top_p:g}",
        "--top-k", str(opts.top_k),
    ]
    if "min_p" in caps:
        args += ["--min-p", f"{opts.min_p:g}"]
    args += ["--repeat-penalty", f"{opts.repeat_penalty:g}"]
    if opts.cache_type_k != "f16":
        args += ["--cache-type-k", opts.cache_type_k]
    if opts.cache_type_v != "f16" and "cache_type_v" in caps:
        args += ["--cache-type-v", opts.cache_type_v]
    if opts.mlock and "mlock" in caps:
        args.append("--mlock")
    if opts.no_mmap and "no_mmap" in caps:
        args.append("--no-mmap")
    if opts.jinja and "jinja" in caps:
        args.append("--jinja")
    name = alias or opts.alias
    if name and "alias" in caps:
        args += ["--alias", name]
    if opts.api_key and "api_key" in caps:
        args += ["--api-key", opts.api_key]
    if opts.extra_args.strip():
        args += split_args(opts.extra_args)
    return args


def split_args(text: str) -> list[str]:
    import shlex
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def format_command(argv: list[str], api_key: str = "") -> str:
    """미리보기/복사용 명령어 문자열 (OS에 맞는 따옴표, API 키는 가림)."""
    import shlex
    import subprocess
    import sys
    shown = ["********" if api_key and a == api_key else a for a in argv]
    if sys.platform == "win32":
        return subprocess.list2cmdline(shown)
    return shlex.join(shown)


# ---------------------------------------------------------------------------
# 대시보드용 요약
# ---------------------------------------------------------------------------

COMMON_SIZES_B = [1, 3, 4, 7, 8, 12, 14, 24, 27, 32, 70, 120, 235]


def max_comfortable_params(system: SystemInfo, quant: str = "Q4_K_M", ctx: int = 8192) -> Optional[int]:
    """이 컴퓨터가 원활하게 돌릴 수 있는 가장 큰 일반적인 모델 크기(B)."""
    best = None
    for p in COMMON_SIZES_B:
        if assess_fit(system, ModelSpec.estimate(p, quant), ctx).fit == Fit.COMFORTABLE:
            best = p
    return best


def summary_sentence(system: SystemInfo) -> str:
    fast = system.fast_budget
    params = max_comfortable_params(system)
    where = "GPU 메모리" if system.gpu_budget else "메모리"
    if params is None:
        return "이 컴퓨터는 메모리가 작아서 아주 작은 모델(1B 이하)을 권장해요."
    size = format_bytes(ModelSpec.estimate(params).file_size)
    return (f"이 컴퓨터는 {where} 약 {format_bytes(fast)}를 쓸 수 있어서, "
            f"{size} 정도 크기의 모델까지 원활하게 실행할 수 있어요 (예: {params}B 모델 Q4_K_M).")
