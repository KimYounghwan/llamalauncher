"""모델 다운로드 관리자: 이어받기, 일시정지, 취소, 동시 다운로드 제한, SHA-256 검증.

Qt에 의존하지 않는다. 진행 상황은 on_update 콜백(작업 스레드에서 호출됨)으로 알린다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import requests

from .hf_client import HF, USER_AGENT

CHUNK = 256 * 1024  # 작을수록 진행률·일시정지 반응이 빠르다 (iter_content는 이만큼 찰 때까지 기다림)
MANIFEST_NAME = ".llamalauncher-models.json"


class Status(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"
    CANCELED = "canceled"

    @property
    def label(self) -> str:
        return {"queued": "대기 중", "preparing": "이어받기 준비 중", "downloading": "다운로드 중",
                "verifying": "파일 검증 중", "paused": "일시정지", "done": "완료", "error": "오류",
                "canceled": "취소됨"}[self.value]

    @property
    def active(self) -> bool:
        return self in (Status.PREPARING, Status.DOWNLOADING, Status.VERIFYING)


@dataclass
class FileSpec:
    path: str                   # 저장소 내 경로
    size: int
    sha256: Optional[str] = None

    @property
    def filename(self) -> str:
        return Path(self.path).name


@dataclass
class DownloadItem:
    repo: str
    quant: str
    files: list[FileSpec]
    dest_dir: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: Status = Status.QUEUED
    downloaded: int = 0
    error: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    mmproj: Optional[str] = None  # 함께 받은 비전 파일 이름
    revision: str = "main"
    # 실행 중에만 쓰는 값
    speed: float = 0.0

    @property
    def total(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def main_path(self) -> Path:
        """서버에 넘길 모델 파일 (분할이면 첫 조각)."""
        model_files = [f for f in self.files if f.filename != self.mmproj]
        return Path(self.dest_dir) / (model_files or self.files)[0].filename

    @property
    def display_name(self) -> str:
        return f"{self.repo.split('/')[-1]} · {self.quant}"

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.speed <= 0:
            return None
        return max(0.0, (self.total - self.downloaded) / self.speed)

    def to_json(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d.pop("speed", None)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "DownloadItem":
        d = dict(d)
        d["files"] = [FileSpec(**f) for f in d.get("files", [])]
        d["status"] = Status(d.get("status", "paused"))
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})


class DownloadError(Exception):
    pass


def repo_dir(models_dir: Path, repo: str) -> Path:
    owner, _, name = repo.partition("/")
    return Path(models_dir) / owner / name


def free_space(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def human_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}시간 {s % 3600 // 60}분"
    if s >= 60:
        return f"{s // 60}분 {s % 60}초"
    return f"{s}초"


class DownloadManager:
    def __init__(self, state_path: Path, token_getter: Callable[[], Optional[str]] = lambda: None,
                 max_concurrent: int = 2, on_update: Callable[[DownloadItem], None] = lambda item: None,
                 session_factory: Callable[[], requests.Session] = requests.Session, base_url: str = HF):
        self.state_path = Path(state_path)
        self.token_getter = token_getter
        self.max_concurrent = max_concurrent
        self.on_update = on_update
        self.session_factory = session_factory
        self.base_url = base_url
        self.items: dict[str, DownloadItem] = {}
        self._stop_flags: dict[str, str] = {}   # id → "pause" | "cancel"
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._load()

    # ---------------------------------------------------------------- 상태 저장
    def _load(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for d in data.get("items", []):
            try:
                item = DownloadItem.from_json(d)
            except (TypeError, ValueError):
                continue
            if item.status.active or item.status == Status.QUEUED:
                item.status = Status.PAUSED  # 앱을 다시 켜면 사용자가 이어받기를 누르게 한다
            item.downloaded = self._bytes_on_disk(item)
            self.items[item.id] = item

    def save(self) -> None:
        with self._lock:
            data = {"items": [i.to_json() for i in self.items.values() if i.status != Status.CANCELED]}
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            pass

    @staticmethod
    def _bytes_on_disk(item: DownloadItem) -> int:
        total = 0
        for f in item.files:
            final = Path(item.dest_dir) / f.filename
            part = final.with_name(final.name + ".part")
            if final.exists():
                total += final.stat().st_size
            elif part.exists():
                total += part.stat().st_size
        return total

    # ---------------------------------------------------------------- 공개 API
    def add(self, repo: str, quant: str, files: list[FileSpec], dest_dir: Path,
            mmproj: Optional[str] = None) -> DownloadItem:
        with self._lock:
            for it in self.items.values():
                if it.repo == repo and it.quant == quant and it.status not in (Status.CANCELED, Status.ERROR) \
                        and Path(it.dest_dir) == Path(dest_dir):
                    if it.status == Status.PAUSED:
                        self.resume(it.id)
                    return it
        item = DownloadItem(repo=repo, quant=quant, files=files, dest_dir=str(dest_dir), mmproj=mmproj)
        item.downloaded = self._bytes_on_disk(item)
        need = item.total - item.downloaded
        free = free_space(Path(dest_dir))
        if need + 1024**3 > free:
            raise DownloadError(f"저장 공간이 부족해요. {need / 1024**3:.1f}GB가 필요한데 "
                                f"{free / 1024**3:.1f}GB만 남아 있어요. 모델 저장 위치를 바꾸거나 공간을 확보해 주세요.")
        with self._lock:
            self.items[item.id] = item
        self.save()
        self.on_update(item)
        self._schedule()
        return item

    def pause(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            if not item:
                return
            if item.id in self._threads:
                self._stop_flags[item_id] = "pause"
            elif item.status == Status.QUEUED:
                self._set(item, Status.PAUSED)

    def resume(self, item_id: str) -> None:
        with self._lock:
            item = self.items.get(item_id)
            if item and item.status in (Status.PAUSED, Status.ERROR):
                item.error = ""
                self._set(item, Status.QUEUED)
        self._schedule()

    def cancel(self, item_id: str, delete_files: bool = True) -> None:
        with self._lock:
            item = self.items.get(item_id)
            if not item:
                return
            if item.id in self._threads:
                self._stop_flags[item_id] = "cancel"
                return
        self._finish_cancel(item, delete_files)

    def remove(self, item_id: str) -> None:
        """완료/취소/오류 항목을 목록에서만 지운다 (파일은 그대로)."""
        with self._lock:
            item = self.items.get(item_id)
            if item and not item.status.active:
                del self.items[item_id]
        self.save()

    def active_count(self) -> int:
        return len(self._threads)

    def shutdown(self, timeout: float = 3.0) -> None:
        with self._lock:
            for i in list(self._threads):
                self._stop_flags[i] = "pause"
            threads = list(self._threads.values())
        for t in threads:
            t.join(timeout)
        self.save()

    # ---------------------------------------------------------------- 내부
    def _set(self, item: DownloadItem, status: Status, error: str = "") -> None:
        item.status = status
        if error:
            item.error = error
        self.on_update(item)

    def _schedule(self) -> None:
        with self._lock:
            queued = [i for i in self.items.values() if i.status == Status.QUEUED and i.id not in self._threads]
            for item in queued:
                if len(self._threads) >= self.max_concurrent:
                    break
                t = threading.Thread(target=self._run, args=(item,), daemon=True, name=f"dl-{item.id}")
                self._threads[item.id] = t
                self._stop_flags.pop(item.id, None)
                item.status = Status.PREPARING
                t.start()

    def _finish_cancel(self, item: DownloadItem, delete_files: bool) -> None:
        if delete_files:
            for f in item.files:
                part = Path(item.dest_dir) / (f.filename + ".part")
                try:
                    part.unlink()
                except OSError:
                    pass
        item.downloaded = 0
        item.speed = 0
        self._set(item, Status.CANCELED)
        with self._lock:
            self.items.pop(item.id, None)
        self.save()

    def _run(self, item: DownloadItem) -> None:
        session = self.session_factory()
        session.headers["User-Agent"] = USER_AGENT
        token = self.token_getter()
        if token:
            session.headers["Authorization"] = f"Bearer {token}"
        result: Status = Status.DONE
        error = ""
        try:
            Path(item.dest_dir).mkdir(parents=True, exist_ok=True)
            done_before = 0
            for f in item.files:
                outcome = self._download_file(session, item, f, done_before)
                if outcome != Status.DONE:
                    result = outcome
                    break
                done_before += f.size
        except DownloadError as e:
            result, error = Status.ERROR, str(e)
        except requests.RequestException:
            result, error = Status.ERROR, "인터넷 연결이 끊겼어요. 연결을 확인한 뒤 [이어받기]를 눌러 주세요."
        except OSError as e:
            result, error = Status.ERROR, f"파일을 저장하지 못했어요: {e}"
        finally:
            session.close()
            with self._lock:
                self._threads.pop(item.id, None)
                flag = self._stop_flags.pop(item.id, None)
            item.speed = 0
            if flag == "cancel":
                self._finish_cancel(item, True)
            else:
                if result == Status.DONE:
                    item.downloaded = item.total
                    write_manifest_entry(item)
                self._set(item, result, error)
                self.save()
            self._schedule()

    def _stop_requested(self, item: DownloadItem) -> Optional[Status]:
        flag = self._stop_flags.get(item.id)
        if flag == "pause":
            return Status.PAUSED
        if flag == "cancel":
            return Status.CANCELED
        return None

    def _download_file(self, session: requests.Session, item: DownloadItem, f: FileSpec, done_before: int) -> Status:
        final = Path(item.dest_dir) / f.filename
        part = final.with_name(final.name + ".part")
        if final.exists() and (not f.size or final.stat().st_size == f.size):
            item.downloaded = done_before + final.stat().st_size
            return Status.DONE

        sha = hashlib.sha256() if f.sha256 else None
        offset = part.stat().st_size if part.exists() else 0
        if f.size and offset > f.size:
            part.unlink()
            offset = 0
        if offset and sha:
            self._set(item, Status.PREPARING)
            with open(part, "rb") as fh:
                read = 0
                while chunk := fh.read(CHUNK * 8):
                    sha.update(chunk)
                    read += len(chunk)
                    if (stop := self._stop_requested(item)):
                        return stop

        url = f"{self.base_url}/{item.repo}/resolve/{item.revision}/{f.path}"
        attempts = 0
        while True:
            if offset < f.size or not f.size:
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    resp = session.get(url, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True)
                except requests.RequestException:
                    attempts += 1
                    if attempts > 3:
                        raise
                    time.sleep(2 * attempts)
                    continue
                with resp:
                    if resp.status_code in (401, 403):
                        raise DownloadError("이 모델은 허깅페이스 로그인(토큰)과 사용 동의가 필요해요. "
                                            "모델 페이지에서 동의한 뒤, [허깅페이스 토큰 설정]에 토큰을 넣어 주세요.")
                    if resp.status_code == 404:
                        raise DownloadError("파일을 찾을 수 없어요. 저장소에서 삭제되었을 수 있어요.")
                    if resp.status_code == 416:  # 이미 다 받음
                        pass
                    elif resp.status_code >= 400:
                        raise DownloadError(f"다운로드 서버 오류 (HTTP {resp.status_code}). 잠시 후 이어받기를 눌러 주세요.")
                    else:
                        if offset and resp.status_code == 200:  # 서버가 이어받기를 지원하지 않음 → 처음부터
                            offset = 0
                            sha = hashlib.sha256() if f.sha256 else None
                        self._set(item, Status.DOWNLOADING)
                        try:
                            stop = self._stream(resp, part, offset, sha, item, done_before)
                        except requests.RequestException:
                            attempts += 1
                            offset = part.stat().st_size if part.exists() else 0
                            if attempts > 3:
                                raise
                            if sha and offset:  # 끊긴 지점까지 다시 해시
                                sha = hashlib.sha256()
                                with open(part, "rb") as fh:
                                    while chunk := fh.read(CHUNK * 8):
                                        sha.update(chunk)
                            time.sleep(2 * attempts)
                            continue
                        if stop:
                            return stop
            break

        size = part.stat().st_size if part.exists() else 0
        if f.size and size != f.size:
            raise DownloadError(f"{f.filename}: 받은 크기가 달라요 ({size:,} / {f.size:,} 바이트). 이어받기를 다시 눌러 주세요.")
        if sha and f.sha256:
            self._set(item, Status.VERIFYING)
            if sha.hexdigest() != f.sha256:
                part.unlink(missing_ok=True)
                raise DownloadError(f"{f.filename}: 파일이 손상되었어요 (검증 실패). 다시 받아 주세요.")
        part.replace(final)
        return Status.DONE

    def _stream(self, resp, part: Path, offset: int, sha, item: DownloadItem, done_before: int) -> Optional[Status]:
        mode = "ab" if offset else "wb"
        last_emit = time.monotonic()
        window_start, window_bytes = last_emit, 0
        written = offset
        with open(part, mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                if sha:
                    sha.update(chunk)
                written += len(chunk)
                window_bytes += len(chunk)
                now = time.monotonic()
                if now - last_emit >= 0.3:
                    inst = window_bytes / max(now - window_start, 1e-6)
                    item.speed = inst if item.speed == 0 else item.speed * 0.7 + inst * 0.3
                    window_start, window_bytes = now, 0
                    item.downloaded = done_before + written
                    last_emit = now
                    self.on_update(item)
                if (stop := self._stop_requested(item)):
                    item.downloaded = done_before + written
                    return stop
        item.downloaded = done_before + written
        return None


# ---------------------------------------------------------------- 받은 모델 기록 (4단계 '내 모델'에서 사용)
def manifest_path(models_dir: Path) -> Path:
    return Path(models_dir) / MANIFEST_NAME


def read_manifest(models_dir: Path) -> list[dict]:
    try:
        return json.loads(manifest_path(models_dir).read_text(encoding="utf-8")).get("models", [])
    except (OSError, json.JSONDecodeError):
        return []


def write_manifest_entry(item: DownloadItem) -> None:
    """dest_dir = <models_dir>/<owner>/<repo> 구조를 가정한다."""
    models_dir = Path(item.dest_dir).parent.parent
    entries = [e for e in read_manifest(models_dir) if e.get("path") != str(item.main_path)]
    entries.append({
        "path": str(item.main_path), "repo": item.repo, "quant": item.quant, "size": item.total,
        "files": [f.filename for f in item.files],
        "mmproj": str(Path(item.dest_dir) / item.mmproj) if item.mmproj else None,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
    })
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        manifest_path(models_dir).write_text(json.dumps({"models": entries}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    except OSError:
        pass
