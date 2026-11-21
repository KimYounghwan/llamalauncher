"""허깅페이스 모델 검색, 파일 목록 분석, 캐시.

공개 REST API만 사용한다 (https://huggingface.co/docs/hub/api).
네트워크 함수는 모두 동기식이므로 UI에서는 백그라운드 스레드로 호출한다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from .hardware import SystemInfo
from .recommender import Fit, ModelSpec, assess_fit
from .settings import config_dir

HF = "https://huggingface.co"
CACHE_TTL = 6 * 3600
USER_AGENT = "LlamaLauncher/0.3"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

# 형식(런처)별 검색 필터 — 5단계에서 런처가 늘어나면 여기에 추가
FORMAT_FILTERS = {"gguf": "gguf", "mlx": "mlx", "safetensors": "safetensors"}


class HFError(Exception):
    """사용자에게 그대로 보여줄 수 있는 쉬운 오류 메시지."""


# ---------------------------------------------------------------- 카탈로그
@dataclass
class Category:
    id: str
    icon: str
    name: str
    description: str
    queries: list[dict]
    trusted_only: bool = False
    fit_only: bool = False
    require_text: bool = False
    max_params_b: Optional[float] = None
    boost_keywords: list[str] = field(default_factory=list)

    def matches_keywords(self, model_id: str) -> bool:
        low = model_id.lower()
        return any(k in low for k in self.boost_keywords)

    @property
    def label(self) -> str:
        return f"{self.icon}  {self.name}"


@dataclass
class Catalog:
    categories: list[Category]
    trusted_authors: set[str]
    hidden_keywords: list[str]
    text_pipelines: set[str]

    @classmethod
    def load(cls, path: Path = CATALOG_PATH) -> "Catalog":
        data = json.loads(path.read_text(encoding="utf-8"))
        names = Category.__dataclass_fields__
        cats = [Category(**{k: v for k, v in c.items() if k in names}) for c in data["categories"]]
        return cls(cats, {a.lower() for a in data.get("trusted_authors", [])},
                   [k.lower() for k in data.get("hidden_keywords", [])], set(data.get("text_pipelines", [])))

    def get(self, cat_id: str) -> Category:
        return next(c for c in self.categories if c.id == cat_id)

    def is_trusted(self, author: str) -> bool:
        return author.lower() in self.trusted_authors

    def is_hidden(self, model_id: str) -> bool:
        low = model_id.lower()
        return any(k in low for k in self.hidden_keywords)


# ---------------------------------------------------------------- 모델 정보
_PARAMS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


def params_from_name(name: str) -> Optional[float]:
    """이름에서 파라미터 수(B) 추정: 'Qwen3-30B-A3B' → 30 (MoE는 전체 크기)."""
    nums = [float(m.group(1)) for m in _PARAMS_RE.finditer(name)]
    return max(nums) if nums else None


@dataclass
class HFModel:
    id: str
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    pipeline: str = ""
    tags: list[str] = field(default_factory=list)
    params: Optional[int] = None          # 전체 파라미터 수
    context_length: Optional[int] = None
    architecture: str = ""
    supports_tools: bool = False
    gated: bool = False
    license: str = ""
    base_model: str = ""

    @property
    def author(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.id.split("/", 1)[-1]

    @property
    def params_b(self) -> Optional[float]:
        if self.params:
            return self.params / 1e9
        return params_from_name(self.name)

    @property
    def is_vision(self) -> bool:
        return self.pipeline == "image-text-to-text" or "vision" in self.tags or "multimodal" in self.tags

    @property
    def is_embedding(self) -> bool:
        return self.pipeline in ("feature-extraction", "sentence-similarity") or "embedding" in self.name.lower()

    @property
    def url(self) -> str:
        return f"{HF}/{self.id}"

    def estimated_spec(self, quant: str = "Q4_K_M") -> Optional[ModelSpec]:
        p = self.params_b
        if not p:
            return None
        return ModelSpec.estimate(p, quant, n_ctx_train=self.context_length)

    def days_since_update(self) -> Optional[int]:
        try:
            dt = datetime.fromisoformat(self.last_modified.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - dt).days

    @classmethod
    def from_api(cls, d: dict) -> "HFModel":
        g = d.get("gguf") or {}
        card = d.get("cardData") or {}
        tags = d.get("tags") or []
        lic = card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), "")
        base = card.get("base_model") or ""
        if isinstance(base, list):
            base = base[0] if base else ""
        template = g.get("chat_template") or ""
        return cls(
            id=d.get("id") or d.get("modelId", ""),
            downloads=int(d.get("downloads") or 0),
            likes=int(d.get("likes") or 0),
            last_modified=d.get("lastModified") or d.get("createdAt") or "",
            pipeline=d.get("pipeline_tag") or card.get("pipeline_tag") or "",
            tags=tags,
            params=int(g["total"]) if isinstance(g.get("total"), (int, float)) and g["total"] > 0 else None,
            context_length=int(g["context_length"]) if g.get("context_length") else None,
            architecture=g.get("architecture") or "",
            supports_tools=("tools" in template or "tool_call" in template),
            gated=bool(d.get("gated")),
            license=str(lic or ""),
            base_model=str(base),
        )


# ---------------------------------------------------------------- 파일 목록 분석
QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:I?Q[1-8](?:_[0-9A-Z]+)*|F16|BF16|F32|MXFP4(?:_MOE)?|TQ[12]_0))(?=[-_.]|$)", re.I)
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


@dataclass
class QuantFile:
    quant: str                 # 예: "Q4_K_M", "UD-Q4_K_XL"
    paths: list[str]           # 저장소 내 경로 (분할이면 여러 개, 순서대로)
    size: int
    sha256: list[Optional[str]] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return Path(self.paths[0]).name

    @property
    def is_split(self) -> bool:
        return len(self.paths) > 1

    def bits_per_weight(self, params: Optional[float]) -> Optional[float]:
        return self.size * 8 / params if params else None


@dataclass
class RepoFiles:
    repo: str
    quants: list[QuantFile]
    mmproj: list[QuantFile]
    other_gguf: list[str]      # imatrix, MTP(초안 모델) 등 일반 사용자는 필요 없는 파일


def _quant_of(path: str) -> str:
    name = Path(path).name
    name = SPLIT_RE.sub(".gguf", name)
    m = None
    for m in QUANT_RE.finditer(name):
        pass  # 마지막 일치 사용 (모델 이름 속의 숫자보다 뒤쪽이 양자화 표기)
    return m.group(1).upper() if m else Path(name).stem


def parse_tree(repo: str, entries: list[dict]) -> RepoFiles:
    groups: dict[tuple[str, str], list[dict]] = {}
    mmproj: list[QuantFile] = []
    other: list[str] = []
    for e in entries:
        path = e.get("path", "")
        if e.get("type") != "file" or not path.lower().endswith(".gguf"):
            continue
        low = Path(path).name.lower()
        size = int((e.get("lfs") or {}).get("size") or e.get("size") or 0)
        sha = (e.get("lfs") or {}).get("oid")
        if "mmproj" in low:
            mmproj.append(QuantFile(_quant_of(path), [path], size, [sha], [size]))
            continue
        if "imatrix" in low or low.startswith("mtp-") or "/mtp/" in path.lower() or path.lower().startswith("mtp/"):
            other.append(path)
            continue
        m = SPLIT_RE.search(path)
        key = (str(Path(path).parent), SPLIT_RE.sub("", path) if m else path)
        groups.setdefault(key, []).append({"path": path, "size": size, "sha": sha})

    quants = []
    for items in groups.values():
        items.sort(key=lambda x: x["path"])
        quants.append(QuantFile(
            quant=_quant_of(items[0]["path"]),
            paths=[i["path"] for i in items],
            size=sum(i["size"] for i in items),
            sha256=[i["sha"] for i in items],
            sizes=[i["size"] for i in items],
        ))
    _disambiguate(quants)
    quants.sort(key=lambda q: q.size)
    mmproj.sort(key=lambda q: q.size)
    return RepoFiles(repo, quants, mmproj, other)


def _disambiguate(quants: list[QuantFile]) -> None:
    """같은 양자화 이름이 여러 개면(예: Q4_0 과 QAD-Q4_0) 파일 이름에서 구분되는 부분을 보여준다."""
    counts: dict[str, int] = {}
    for q in quants:
        counts[q.quant] = counts.get(q.quant, 0) + 1
    dups = [q for q in quants if counts[q.quant] > 1]
    if not dups:
        return
    stems = [SPLIT_RE.sub("", Path(q.paths[0]).name).removesuffix(".gguf") for q in quants]
    prefix = os.path.commonprefix(stems)
    cut = max(prefix.rfind("-"), prefix.rfind("."), prefix.rfind("_")) + 1 if prefix else 0
    for q in dups:
        stem = SPLIT_RE.sub("", Path(q.paths[0]).name).removesuffix(".gguf")
        label = stem[cut:] or stem
        parent = str(Path(q.paths[0]).parent)
        q.quant = label if label.upper() != q.quant or parent == "." else f"{parent}/{label}"


def recommend_quant(files: RepoFiles, system: SystemInfo, params: Optional[float],
                    ctx: int = 16384, n_ctx_train: Optional[int] = None) -> Optional[QuantFile]:
    """초보자용 추천 파일: 이 컴퓨터에서 '쾌적'하면서 품질과 속도의 균형이 좋은 것.

    - 문맥 16K로도 쾌적해야 하고, 파일이 쓸 수 있는 메모리의 80%를 넘지 않게 여유를 둔다
    - 9B 초과 모델은 6비트급까지만 (8비트는 품질 차이는 작고 느려진다), 작은 모델은 8비트까지
    """
    max_bpw = 8.6 if (params and params <= 9e9) else 6.8
    limit = system.fast_budget * 0.8
    best = None
    for q in files.quants:
        bpw = q.bits_per_weight(params)
        if bpw is not None and bpw > max_bpw:
            continue
        if bpw is None and (q.quant in ("F16", "BF16", "F32") or q.quant.endswith("Q8_0")):
            continue
        if q.size > limit:
            continue
        spec = ModelSpec(file_size=q.size, n_ctx_train=n_ctx_train)
        if assess_fit(system, spec, min(ctx, n_ctx_train or ctx)).fit == Fit.COMFORTABLE:
            best = q
    return best


# ---------------------------------------------------------------- 점수/필터
def score(model: HFModel, fit: Optional[Fit], trusted: bool, keyword_match: bool = False) -> float:
    s = math.log10(model.downloads + 10)
    s += {Fit.COMFORTABLE: 3.0, Fit.SLOW: 0.5, Fit.INSUFFICIENT: -6.0, None: -1.0}[fit]
    if trusted:
        s += 2.5
    if keyword_match:
        s += 2.0
    days = model.days_since_update()
    if days is not None:
        s += 0.8 if days < 120 else (0.3 if days < 365 else 0)
    if model.params_b:
        s += min(model.params_b, 40) / 40  # 같은 조건이면 큰(똑똑한) 모델을 조금 우대
    return s


def model_fit(model: HFModel, system: Optional[SystemInfo]) -> Optional[Fit]:
    if system is None:
        return None
    spec = model.estimated_spec()
    return assess_fit(system, spec, 8192).fit if spec else None


# ---------------------------------------------------------------- 클라이언트
class HFClient:
    EXPAND = ["downloads", "likes", "lastModified", "pipeline_tag", "tags", "gguf", "gated", "cardData"]

    def __init__(self, token: Optional[str] = None, cache_dir: Optional[Path] = None,
                 session: Optional[requests.Session] = None, catalog: Optional[Catalog] = None):
        self.token = token
        self.cache_dir = cache_dir or (config_dir() / "cache" / "hf")
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.catalog = catalog or Catalog.load()

    # ---- HTTP + 캐시
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / (hashlib.sha1(key.encode()).hexdigest() + ".json")

    def get_json(self, path: str, params: Optional[list[tuple[str, Any]]] = None, use_cache: bool = True) -> Any:
        key = path + "?" + "&".join(f"{k}={v}" for k, v in (params or []))
        cp = self._cache_path(key)
        if use_cache:
            try:
                if time.time() - cp.stat().st_mtime < CACHE_TTL:
                    return json.loads(cp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        try:
            r = self.session.get(HF + path, params=params, headers=self._headers(), timeout=20)
        except requests.RequestException as e:
            stale = self._read_stale(cp)
            if stale is not None:
                return stale
            raise HFError("허깅페이스에 연결할 수 없어요. 인터넷 연결을 확인해 주세요.") from e
        if r.status_code == 429:
            raise HFError("요청이 너무 많아 허깅페이스가 잠시 막았어요. 1~2분 뒤에 다시 시도해 주세요. "
                          "(허깅페이스 토큰을 설정하면 한도가 늘어나요)")
        if r.status_code in (401, 403):
            raise HFError("이 모델은 로그인(토큰)이 필요하거나 접근 승인이 필요해요.")
        if r.status_code == 404:
            raise HFError("모델을 찾을 수 없어요. 삭제되었거나 이름이 바뀌었을 수 있어요.")
        if r.status_code >= 400:
            raise HFError(f"허깅페이스 오류가 발생했어요 (HTTP {r.status_code}). 잠시 후 다시 시도해 주세요.")
        data = r.json()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return data

    @staticmethod
    def _read_stale(cp: Path) -> Any:
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # ---- 검색
    def _query(self, q: dict, fmt: str, limit: int, sort: str, use_cache: bool) -> list[HFModel]:
        params: list[tuple[str, Any]] = [("filter", FORMAT_FILTERS.get(fmt, fmt))]
        for k, v in q.items():
            params.append((k, v))
        params += [("sort", sort), ("direction", -1), ("limit", limit)]
        params += [("expand[]", e) for e in self.EXPAND]
        data = self.get_json("/api/models", params, use_cache=use_cache)
        return [HFModel.from_api(d) for d in data if isinstance(d, dict)]

    def search(self, *, category: Optional[str] = None, text: str = "", fmt: str = "gguf",
               system: Optional[SystemInfo] = None, hide_risky: bool = True, trusted_only: bool = False,
               sort: str = "recommended", limit: int = 60, use_cache: bool = True) -> list[HFModel]:
        """카테고리 또는 검색어로 모델을 찾고, 필터링/정렬한 결과를 돌려준다."""
        cat = self.catalog.get(category) if category else None
        if text.strip():
            queries = [{"search": text.strip()}]
            if cat:
                queries = [{**q, "search": text.strip()} for q in cat.queries[:2]]
        elif cat:
            queries = cat.queries
        else:
            queries = [{}]
        api_sort = {"likes": "likes", "updated": "lastModified"}.get(sort, "downloads")
        per_query = 100 if (cat and (cat.max_params_b or cat.fit_only)) else limit

        with ThreadPoolExecutor(max_workers=min(4, len(queries))) as ex:
            batches = list(ex.map(lambda q: self._query(q, fmt, per_query, api_sort, use_cache), queries))
        seen: dict[str, HFModel] = {}
        for batch in batches:
            for m in batch:
                seen.setdefault(m.id, m)
        models = list(seen.values())

        cat_trusted = bool(cat and cat.trusted_only)
        out = []
        for m in models:
            if hide_risky and self.catalog.is_hidden(m.id):
                continue
            if (trusted_only or cat_trusted) and not self.catalog.is_trusted(m.author):
                continue
            if cat and cat.require_text and m.pipeline and m.pipeline not in self.catalog.text_pipelines:
                continue
            if cat and cat.id not in ("embedding",) and m.is_embedding and not text:
                continue
            if cat and cat.max_params_b and (m.params_b is None or m.params_b > cat.max_params_b):
                continue
            if cat and cat.fit_only and system and model_fit(m, system) != Fit.COMFORTABLE:
                continue
            out.append(m)

        if sort == "recommended":
            out.sort(key=lambda m: score(m, model_fit(m, system), self.catalog.is_trusted(m.author),
                                         bool(cat and cat.matches_keywords(m.id))), reverse=True)
        elif sort == "likes":
            out.sort(key=lambda m: m.likes, reverse=True)
        elif sort == "updated":
            out.sort(key=lambda m: m.last_modified, reverse=True)
        else:
            out.sort(key=lambda m: m.downloads, reverse=True)
        return out[:limit]

    # ---- 상세
    def list_files(self, repo: str, use_cache: bool = True) -> RepoFiles:
        data = self.get_json(f"/api/models/{repo}/tree/main", [("recursive", "true")], use_cache=use_cache)
        return parse_tree(repo, data if isinstance(data, list) else [])

    def readme_summary(self, repo: str, max_chars: int = 700) -> str:
        cp = self._cache_path("readme:" + repo)
        try:
            if time.time() - cp.stat().st_mtime < CACHE_TTL:
                return cp.read_text(encoding="utf-8")
        except OSError:
            pass
        try:
            r = self.session.get(f"{HF}/{repo}/raw/main/README.md", headers=self._headers(), timeout=15)
        except requests.RequestException:
            return ""
        if r.status_code != 200:
            return ""
        text = summarize_readme(r.text, max_chars)
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text

    def resolve_url(self, repo: str, path: str, revision: str = "main") -> str:
        return f"{HF}/{repo}/resolve/{revision}/{path}"


def summarize_readme(md: str, max_chars: int = 700) -> str:
    """모델 카드에서 앞부분 설명 문단만 뽑는다 (메타데이터, 이미지, 표, 코드 제외)."""
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = re.sub(r"<[^>]+>", "", md)
    paras = []
    for block in re.split(r"\n\s*\n", md):
        b = block.strip()
        if not b or b.startswith(("#", "|", "!", ">")) or b.startswith("[!["):
            continue
        b = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", b)
        b = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", b)
        b = re.sub(r"[*_`]{1,3}", "", b).strip()
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if len(b) < 30 or (lines and sum(len(ln) for ln in lines) / len(lines) < 25):
            continue  # 링크 목록·배지 같은 짧은 줄 묶음은 설명이 아님
        paras.append(b)
        if sum(len(p) for p in paras) >= max_chars:
            break
    text = "\n\n".join(paras)
    return text[:max_chars].rstrip() + ("…" if len(text) > max_chars else "")

