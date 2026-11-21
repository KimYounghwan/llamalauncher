"""설정 파일(JSON/JSONC/YAML/TOML) 병합 + 백업/복구.

원칙
- 기존 설정은 지우지 않는다: 딕셔너리는 재귀 병합, 목록은 같은 이름(name/title) 항목만 바꾸고 나머지는 유지
- 쓰기 전에 원본을 `파일명.bak-날짜시간`으로 복사한다
- 무엇을 했는지 상태 파일에 기록해 [원래대로 되돌리기]가 가능하게 한다 (새로 만든 파일이면 지운다)
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tomlkit
import yaml


class MergeError(Exception):
    pass


# ---------------------------------------------------------------- 읽기/쓰기
def strip_jsonc(text: str) -> str:
    """JSONC(주석·끝 쉼표 허용 JSON)를 일반 JSON으로. 문자열 안의 //는 건드리지 않는다."""
    out, i, n = [], 0, len(text)
    in_str = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def has_comments(text: str) -> bool:
    return strip_jsonc(text) != re.sub(r",(\s*[}\]])", r"\1", text)


def parse(text: str, fmt: str) -> Any:
    if not text.strip():
        return {} if fmt != "toml" else tomlkit.document()
    try:
        if fmt in ("json", "jsonc"):
            return json.loads(strip_jsonc(text))
        if fmt == "yaml":
            data = yaml.safe_load(text)
            return data if data is not None else {}
        if fmt == "toml":
            return tomlkit.parse(text)
    except (json.JSONDecodeError, yaml.YAMLError, tomlkit.exceptions.ParseError) as e:
        raise MergeError(f"기존 설정 파일을 읽을 수 없어요 (형식 오류): {e}") from e
    raise MergeError(f"알 수 없는 형식: {fmt}")


def dump(data: Any, fmt: str) -> str:
    if fmt in ("json", "jsonc"):
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if fmt == "yaml":
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    if fmt == "toml":
        return tomlkit.dumps(data if isinstance(data, tomlkit.TOMLDocument) else _to_toml(data))
    raise MergeError(f"알 수 없는 형식: {fmt}")


def _to_toml(data: dict) -> tomlkit.TOMLDocument:
    doc = tomlkit.document()
    for k, v in data.items():
        doc[k] = v
    return doc


# ---------------------------------------------------------------- 병합
_ID_KEYS = ("name", "title", "id", "model")


def _item_id(x: Any) -> Optional[str]:
    if isinstance(x, dict):
        for k in _ID_KEYS:
            if k in x and isinstance(x[k], str):
                return f"{k}={x[k]}"
    return None


def merge(base: Any, patch: Any) -> Any:
    """patch를 base에 덮어쓴다. base에만 있는 값은 그대로 둔다. (base를 직접 바꾸고 돌려준다)"""
    if isinstance(base, dict) and isinstance(patch, dict):
        for k, v in patch.items():
            old = base.get(k) if k in base else None
            if isinstance(old, dict) and isinstance(v, dict) or isinstance(old, list) and isinstance(v, list):
                base[k] = merge(old, v)
            else:
                base[k] = v
        return base
    if isinstance(base, list) and isinstance(patch, list):
        for item in patch:
            iid = _item_id(item)
            idx = next((i for i, x in enumerate(base) if iid and _item_id(x) == iid), None)
            if idx is not None:
                base[idx] = item
            elif item not in base:
                base.append(item)
        return base
    return patch


# ---------------------------------------------------------------- 적용 계획/실행
@dataclass
class ApplyPlan:
    path: Path
    fmt: str
    old_text: str
    new_text: str
    existed: bool
    warnings: list[str]

    @property
    def changed(self) -> bool:
        return self.old_text != self.new_text

    def diff(self) -> str:
        lines = difflib.unified_diff(self.old_text.splitlines(), self.new_text.splitlines(),
                                     fromfile=f"{self.path.name} (지금)", tofile=f"{self.path.name} (적용 후)",
                                     lineterm="")
        return "\n".join(lines) or "(바뀌는 내용 없음)"


def plan_merge(path: Path, fmt: str, patch: Any) -> ApplyPlan:
    path = Path(path).expanduser()
    existed = path.exists()
    old = path.read_text(encoding="utf-8") if existed else ""
    warnings: list[str] = []
    if fmt == "jsonc" and existed and has_comments(old):
        warnings.append("이 파일에는 주석이 있어요. 자동 적용하면 주석이 사라지니, 직접 복사해 붙여 넣는 것을 권장해요.")
    data = parse(old, fmt)
    if not isinstance(data, dict):
        raise MergeError("기존 설정 파일의 최상위가 객체(딕셔너리)가 아니라서 병합할 수 없어요.")
    merged = merge(data, patch)
    return ApplyPlan(path, fmt, old, dump(merged, fmt), existed, warnings)


def backup_path(path: Path, now: Optional[datetime] = None) -> Path:
    """같은 초에 여러 번 적용해도 이전 백업을 덮어쓰지 않게 번호를 붙인다."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    cand = path.with_name(f"{path.name}.bak-{stamp}")
    n = 1
    while cand.exists():
        n += 1
        cand = path.with_name(f"{path.name}.bak-{stamp}-{n}")
    return cand


def apply_plan(plan: ApplyPlan) -> dict:
    """백업 후 쓴다. 되돌리기용 기록(dict)을 돌려준다."""
    record = {"path": str(plan.path), "existed": plan.existed, "backup": None,
              "applied_at": datetime.now().isoformat(timespec="seconds")}
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    if plan.existed:
        bak = backup_path(plan.path)
        shutil.copy2(plan.path, bak)
        record["backup"] = str(bak)
    tmp = plan.path.with_name(plan.path.name + ".llamalauncher-tmp")
    tmp.write_text(plan.new_text, encoding="utf-8")
    tmp.replace(plan.path)
    return record


def revert(record: dict) -> str:
    """apply_plan의 기록대로 되돌린다. 결과 설명을 돌려준다."""
    path = Path(record["path"])
    if record.get("backup"):
        bak = Path(record["backup"])
        if not bak.exists():
            raise MergeError(f"백업 파일이 없어요: {bak}")
        shutil.copy2(bak, path)
        return f"{path}을(를) 적용 전 상태로 되돌렸어요 (백업: {bak.name})."
    if not record.get("existed") and path.exists():
        path.unlink()
        return f"라마런처가 만든 {path}을(를) 지웠어요."
    return "되돌릴 내용이 없어요."


class StateStore:
    """도구별 적용 기록 (설정 폴더의 integrations.json)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def get(self, tool_id: str) -> list[dict]:
        return self.load().get(tool_id, [])

    def set(self, tool_id: str, records: list[dict]) -> None:
        data = self.load()
        if records:
            data[tool_id] = records
        else:
            data.pop(tool_id, None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
