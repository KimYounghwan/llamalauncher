"""내 모델 목록: 받은 모델 + 직접 추가한 파일 + 스캔한 폴더 + LM Studio·Ollama 모델.

GGUF 헤더만 읽으므로 수십 GB 파일도 빠르게 정보를 얻는다.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from . import gguf
from .downloader import MANIFEST_NAME, manifest_path, read_manifest
from .hf_client import QUANT_RE, _quant_of

EMBEDDING_ARCHS = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "t5encoder",
                   "modern-bert", "neo-bert", "eurobert"}
MAX_SCAN_DEPTH = 6
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".Trash", "Library", "AppData", ".cache"}


class Source:
    DOWNLOAD = "download"   # 라마런처로 받은 모델
    ADDED = "added"         # [파일 직접 선택]으로 추가
    FOLDER = "folder"       # 스캔해서 가져온 폴더
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"

    LABELS = {DOWNLOAD: "⬇️ 받은 모델", ADDED: "📄 직접 추가", FOLDER: "📁 폴더",
              LMSTUDIO: "🧪 LM Studio", OLLAMA: "🦙 Ollama"}


@dataclass
class LocalModel:
    path: Path
    source: str
    name: str = ""          # 표시 이름 (Ollama는 gemma3:4b 같은 이름)
    repo: str = ""
    quant: str = ""
    size: int = 0
    date: Optional[datetime] = None
    mmproj: Optional[Path] = None
    info: Optional[gguf.GGUFInfo] = None
    error: str = ""

    @property
    def key(self) -> str:
        return str(self.path)

    @property
    def display_name(self) -> str:
        if self.name:
            return self.name
        stem = gguf.SPLIT_RE.sub("", self.path.name)
        return stem[:-5] if stem.lower().endswith(".gguf") else stem

    @property
    def is_embedding(self) -> bool:
        return bool(self.info and self.info.architecture in EMBEDDING_ARCHS)

    @property
    def deletable(self) -> bool:
        """Ollama 모델은 Ollama가 관리하는 파일이라 직접 지우지 않는다."""
        return self.source != Source.OLLAMA


def quant_from_name(name: str) -> str:
    """파일 이름에 양자화 표기가 있으면 돌려주고, 없으면 빈 문자열."""
    return _quant_of(name) if QUANT_RE.search(gguf.SPLIT_RE.sub(".gguf", name)) else ""


# ---------------------------------------------------------------- 파일 찾기
def is_model_file(p: Path) -> bool:
    n = p.name.lower()
    if not n.endswith(".gguf") or "mmproj" in n or n.startswith("."):
        return False
    m = gguf.SPLIT_RE.search(p.name)
    return not m or int(m.group(1)) == 1  # 분할 모델은 첫 조각만


def scan_folder(root: Path, max_depth: int = MAX_SCAN_DEPTH) -> list[Path]:
    """폴더 아래의 .gguf 모델 파일(비전 파일·분할 조각 제외)을 찾는다."""
    found: list[Path] = []
    root = Path(root).expanduser()
    if not root.is_dir():
        return found
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        if len(d.parts) - base_depth >= max_depth:
            dirnames[:] = []
        dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and not x.startswith(".")]
        for fn in filenames:
            p = d / fn
            if is_model_file(p):
                found.append(p)
    return sorted(found)


def _home() -> Path:
    return Path.home()


def lmstudio_dirs(home: Optional[Path] = None) -> list[Path]:
    """LM Studio 모델 폴더 후보. 설정 파일에 바꾼 위치가 있으면 그것도 본다."""
    home = home or _home()
    cands: list[Path] = []
    for settings_file in (home / ".lmstudio" / "settings.json", home / ".cache" / "lm-studio" / "settings.json"):
        try:
            data = json.loads(settings_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        folder = data.get("downloadsFolder") or data.get("modelsFolder")
        if isinstance(folder, str) and folder:
            cands.append(Path(folder).expanduser())
    cands += [home / ".lmstudio" / "models", home / ".cache" / "lm-studio" / "models"]
    seen, out = set(), []
    for c in cands:
        if c.is_dir() and c.resolve() not in seen:
            seen.add(c.resolve())
            out.append(c)
    return out


def ollama_dirs(home: Optional[Path] = None, env: Optional[dict] = None) -> list[Path]:
    home = home or _home()
    env = os.environ if env is None else env
    cands = []
    if env.get("OLLAMA_MODELS"):
        cands.append(Path(env["OLLAMA_MODELS"]).expanduser())
    cands.append(home / ".ollama" / "models")
    if sys.platform.startswith("linux"):
        cands += [Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/.ollama/models")]
    return [c for c in cands if (c / "manifests").is_dir()]


OLLAMA_MODEL = "application/vnd.ollama.image.model"
OLLAMA_PROJECTOR = "application/vnd.ollama.image.projector"


def _blob(models: Path, digest: str) -> Path:
    return models / "blobs" / digest.replace(":", "-")


def scan_ollama(models: Path) -> list[LocalModel]:
    """manifests/<registry>/<namespace>/<model>/<tag> JSON을 읽어 모델 파일(blob)을 찾는다."""
    out: list[LocalModel] = []
    mroot = models / "manifests"
    for mf in sorted(mroot.rglob("*")):
        if not mf.is_file() or mf.name.startswith("."):
            continue
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        layers = data.get("layers") or []
        model_layer = next((l for l in layers if l.get("mediaType") == OLLAMA_MODEL), None)
        if not model_layer or not model_layer.get("digest"):
            continue
        blob = _blob(models, model_layer["digest"])
        if not blob.is_file():
            continue
        rel = mf.relative_to(mroot).parts  # (registry, namespace, model, tag)
        if len(rel) < 2:
            continue
        model_name, tag = rel[-2], rel[-1]
        ns = rel[-3] if len(rel) >= 3 else "library"
        name = f"{model_name}:{tag}" if ns == "library" else f"{ns}/{model_name}:{tag}"
        proj = next((l for l in layers if l.get("mediaType") == OLLAMA_PROJECTOR), None)
        out.append(LocalModel(
            path=blob, source=Source.OLLAMA, name=name, size=int(model_layer.get("size") or 0),
            date=datetime.fromtimestamp(mf.stat().st_mtime),
            mmproj=_blob(models, proj["digest"]) if proj and proj.get("digest") else None))
    return out


# ---------------------------------------------------------------- 정보 채우기
def _size_of(p: Path) -> int:
    return sum(x.stat().st_size for x in gguf.split_parts(p) if x.exists())


def load_info(m: LocalModel) -> LocalModel:
    """GGUF 헤더를 읽어 m.info를 채운다 (실패하면 m.error)."""
    try:
        m.info = gguf.inspect(m.path)
        m.size = m.info.file_size
        if not m.quant or m.source == Source.OLLAMA:
            m.quant = quant_from_name(m.path.name) or m.info.file_type or m.quant
        if m.mmproj is None and m.source != Source.OLLAMA:
            m.mmproj = gguf.find_mmproj(m.path)
    except (OSError, gguf.GGUFError) as e:
        m.error = str(e) or "파일을 읽을 수 없어요."
        if not m.size:
            try:
                m.size = _size_of(m.path)
            except OSError:
                pass
    return m


# ---------------------------------------------------------------- 목록 모으기
@dataclass
class LibrarySources:
    models_dir: Path
    added: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    include_external: bool = True
    home: Optional[Path] = None


def _file_model(p: Path, source: str) -> LocalModel:
    try:
        st = p.stat()
        date = datetime.fromtimestamp(st.st_mtime)
    except OSError:
        date = None
    return LocalModel(path=p, source=source, quant=quant_from_name(p.name), date=date)


def collect(src: LibrarySources, read_headers: bool = True) -> list[LocalModel]:
    """모든 출처의 모델을 중복 없이 모은다. 같은 파일이 여러 출처에 있으면 앞쪽 출처가 이긴다."""
    models: dict[str, LocalModel] = {}

    def add(m: LocalModel) -> None:
        try:
            k = str(m.path.resolve())
        except OSError:
            k = str(m.path)
        models.setdefault(k, m)

    mdir = Path(src.models_dir)
    for e in read_manifest(mdir):
        p = Path(e.get("path", ""))
        if not p.is_file():
            continue
        date = None
        try:
            date = datetime.fromisoformat(e["downloaded_at"]) if e.get("downloaded_at") else None
        except ValueError:
            pass
        add(LocalModel(path=p, source=Source.DOWNLOAD, repo=e.get("repo", ""), quant=e.get("quant", ""),
                       size=int(e.get("size") or 0), date=date,
                       mmproj=Path(e["mmproj"]) if e.get("mmproj") and Path(e["mmproj"]).is_file() else None))
    for p in scan_folder(mdir):  # 모델 폴더에 직접 복사해 둔 파일
        m = _file_model(p, Source.DOWNLOAD)
        rel = p.relative_to(mdir).parts
        if len(rel) >= 3:
            m.repo = f"{rel[0]}/{rel[1]}"
        add(m)
    for a in src.added:
        p = Path(a)
        if p.is_file():
            add(_file_model(p, Source.ADDED))
    for f in src.folders:
        for p in scan_folder(Path(f)):
            add(_file_model(p, Source.FOLDER))
    if src.include_external:
        for d in lmstudio_dirs(src.home):
            for p in scan_folder(d):
                m = _file_model(p, Source.LMSTUDIO)
                rel = p.relative_to(d).parts
                if len(rel) >= 3:
                    m.repo = f"{rel[0]}/{rel[1]}"
                add(m)
        for d in ollama_dirs(src.home):
            for m in scan_ollama(d):
                add(m)

    out = list(models.values())
    if read_headers:
        for m in out:
            load_info(m)
    return out


# ---------------------------------------------------------------- 삭제
def files_of(m: LocalModel, others: Iterable[LocalModel] = ()) -> list[Path]:
    """삭제할 파일: 모델(분할 조각 포함) + 같은 폴더의 다른 모델이 쓰지 않는 비전 파일."""
    files = [p for p in gguf.split_parts(m.path) if p.exists()]
    if m.mmproj and m.mmproj.exists():
        shared = any(o.path != m.path and o.mmproj and o.mmproj == m.mmproj for o in others)
        if not shared:
            files.append(m.mmproj)
    return files


def delete_model(m: LocalModel, models_dir: Path, others: Iterable[LocalModel] = ()) -> list[Path]:
    if not m.deletable:
        raise PermissionError("Ollama 모델은 Ollama에서 지워 주세요 (터미널: ollama rm 이름).")
    removed = []
    for p in files_of(m, others):
        p.unlink(missing_ok=True)
        removed.append(p)
    # 받은 모델 기록에서도 지운다
    mp = manifest_path(models_dir)
    entries = read_manifest(models_dir)
    keep = [e for e in entries if e.get("path") != str(m.path)]
    if len(keep) != len(entries):
        try:
            mp.write_text(json.dumps({"models": keep}, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass
    # 비어 버린 저장소 폴더 정리 (models_dir 안에서만)
    try:
        parent = m.path.parent
        mroot = Path(models_dir).resolve()
        while parent.resolve() != mroot and mroot in parent.resolve().parents:
            leftovers = [x for x in parent.iterdir() if x.name not in (".DS_Store", MANIFEST_NAME)]
            if leftovers:
                break
            for x in parent.iterdir():
                x.unlink()
            parent.rmdir()
            parent = parent.parent
    except OSError:
        pass
    return removed
