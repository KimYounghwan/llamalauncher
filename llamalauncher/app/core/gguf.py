"""GGUF 파일의 메타데이터(헤더)만 빠르게 읽는 가벼운 리더.

텐서 데이터는 읽지 않는다. 토크나이저 어휘처럼 큰 배열은 건너뛴다.
형식: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Optional

_SCALAR = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
T_STRING, T_ARRAY = 8, 9
_MAX_KEEP_ARRAY = 256  # 이보다 큰 배열은 값을 버린다 (어휘 목록 등)

SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("파일이 중간에 끝났어요 (다운로드가 덜 된 파일일 수 있어요).")
    return struct.unpack(fmt, data)[0]


def _read_str(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    if n > 64 * 1024 * 1024:
        raise GGUFError("문자열 길이가 비정상적이에요.")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALAR:
        return _read(f, _SCALAR[vtype])
    if vtype == T_STRING:
        return _read_str(f)
    if vtype == T_ARRAY:
        itype = _read(f, "<I")
        count = _read(f, "<Q")
        if count > _MAX_KEEP_ARRAY:
            if itype in _SCALAR:
                f.seek(struct.calcsize(_SCALAR[itype]) * count, 1)
            elif itype == T_STRING:
                for _ in range(count):
                    f.seek(_read(f, "<Q"), 1)
            else:
                for _ in range(count):
                    _read_value(f, itype)
            return None
        return [_read_value(f, itype) for _ in range(count)]
    raise GGUFError(f"알 수 없는 값 형식: {vtype}")


def read_metadata(path: Path | str) -> dict[str, Any]:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError("GGUF 파일이 아니에요.")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError("너무 오래된 GGUF 버전이에요 (v1).")
        _tensor_count = _read(f, "<Q")
        kv_count = _read(f, "<Q")
        meta: dict[str, Any] = {"gguf.version": version}
        for _ in range(kv_count):
            key = _read_str(f)
            vtype = _read(f, "<I")
            meta[key] = _read_value(f, vtype)
        return meta


def split_parts(path: Path) -> list[Path]:
    """분할 GGUF(-00001-of-00003.gguf)면 모든 조각 경로를, 아니면 자기 자신만 돌려준다."""
    m = SPLIT_RE.search(path.name)
    if not m:
        return [path]
    total = int(m.group(2))
    stem = path.name[: m.start()]
    return [path.with_name(f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


@dataclass
class GGUFInfo:
    path: Path
    file_size: int
    architecture: str = ""
    name: str = ""
    size_label: str = ""
    n_layers: Optional[int] = None
    n_ctx_train: Optional[int] = None
    n_embd: Optional[int] = None
    n_head: Optional[int] = None
    n_head_kv: Optional[int] = None
    head_dim: Optional[int] = None
    chat_template: str = ""
    missing_parts: list[Path] = field(default_factory=list)

    @property
    def supports_tools(self) -> bool:
        t = self.chat_template
        return bool(t) and ("tools" in t or "tool_call" in t)

    @property
    def display_name(self) -> str:
        return self.name or self.path.stem


def _int(v) -> Optional[int]:
    if isinstance(v, list):  # 층마다 다른 값(배열)이면 최댓값 사용
        v = max((x for x in v if isinstance(x, (int, float))), default=None)
    return int(v) if isinstance(v, (int, float)) and v > 0 else None


def inspect(path: Path | str) -> GGUFInfo:
    path = Path(path)
    parts = split_parts(path)
    first = parts[0]
    missing = [p for p in parts if not p.exists()]
    size = sum(p.stat().st_size for p in parts if p.exists())
    meta = read_metadata(first if first.exists() else path)

    arch = str(meta.get("general.architecture", ""))
    g = lambda k: meta.get(f"{arch}.{k}")  # noqa: E731
    info = GGUFInfo(
        path=path,
        file_size=size,
        architecture=arch,
        name=str(meta.get("general.name", "") or ""),
        size_label=str(meta.get("general.size_label", "") or ""),
        n_layers=_int(g("block_count")),
        n_ctx_train=_int(g("context_length")),
        n_embd=_int(g("embedding_length")),
        n_head=_int(g("attention.head_count")),
        n_head_kv=_int(g("attention.head_count_kv")),
        head_dim=_int(g("attention.key_length")),
        chat_template=str(meta.get("tokenizer.chat_template", "") or ""),
        missing_parts=missing,
    )
    if info.head_dim is None and info.n_embd and info.n_head:
        info.head_dim = info.n_embd // info.n_head
    if info.n_head_kv is None:
        info.n_head_kv = info.n_head
    return info
