"""런처 설치 실행기: 설치 계획(InstallPlan)의 단계를 차례로 실행하고 진행 로그를 보낸다.

UI와 분리되어 있어 단위 테스트할 수 있다. 실행은 호출한 스레드에서 이루어지므로
UI에서는 백그라운드 스레드로 돌린다.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from .launchers.base import (
    CommandStep, GithubReleaseStep, InstallPlan, OpenUrlStep, RemovePathStep, VenvPipStep, app_home, venv_bin,
)

Log = Callable[[str], None]
GITHUB_API = "https://api.github.com"


class InstallError(Exception):
    pass


class Cancelled(InstallError):
    pass


# ---------------------------------------------------------------- 배포 파일 고르기
def _arch(machine: str) -> str:
    m = machine.lower()
    return "arm64" if m in ("arm64", "aarch64") else "x64"


def pick_llamacpp_assets(names: list[str], platform: str, machine: str, gpu_vendor: str) -> list[str]:
    """llama.cpp 배포 파일 중 이 컴퓨터에 맞는 것. 예: llama-b12602-bin-win-cuda-12.4-x64.zip"""
    arch = _arch(machine)

    def find(pattern: str) -> Optional[str]:
        rx = re.compile(pattern, re.I)
        return next((n for n in names if rx.search(n)), None)

    if platform == "darwin":
        main = find(rf"^llama-b\d+-bin-macos-{arch}\.(zip|tar\.gz)$")
        return [main] if main else []
    if platform == "win32":
        if gpu_vendor == "nvidia" and arch == "x64":
            main = find(r"^llama-b\d+-bin-win-cuda-12\.\d+-x64\.zip$")
            rt = find(r"^cudart-llama-bin-win-cuda-12\.\d+-x64\.zip$")
            if main and rt:
                return [main, rt]
        if gpu_vendor in ("nvidia", "amd", "intel"):
            main = find(rf"^llama-b\d+-bin-win-vulkan-{arch}\.zip$")
            if main:
                return [main]
        main = find(rf"^llama-b\d+-bin-win-cpu-{arch}\.zip$")
        return [main] if main else []
    # Linux
    if gpu_vendor == "nvidia" and arch == "x64":
        main = find(r"^llama-b\d+-bin-ubuntu-cuda-12\.\d+-x64\.tar\.gz$")
        rt = find(r"^cudart-llama-b\d+-bin-ubuntu-cuda-12\.\d+-x64\.tar\.gz$")
        if main:
            return [main] + ([rt] if rt else [])
    if gpu_vendor in ("nvidia", "amd", "intel"):
        main = find(rf"^llama-b\d+-bin-ubuntu-vulkan-{arch}\.tar\.gz$")
        if main:
            return [main]
    main = find(rf"^llama-b\d+-bin-ubuntu-{arch}\.tar\.gz$")
    return [main] if main else []


def pick_koboldcpp_asset(names: list[str], platform: str, machine: str, gpu_vendor: str) -> list[str]:
    arch = _arch(machine)
    if platform == "darwin":
        want = ["koboldcpp-mac-arm64"] if arch == "arm64" else []
    elif platform == "win32":
        want = ["koboldcpp.exe", "koboldcpp-nocuda.exe"] if gpu_vendor == "nvidia" else ["koboldcpp-nocuda.exe"]
    else:
        want = ["koboldcpp-linux-x64", "koboldcpp-linux-x64-nocuda"] if gpu_vendor == "nvidia" \
            else ["koboldcpp-linux-x64-nocuda"]
    for w in want:
        if w in names:
            return [w]
    return []


# ---------------------------------------------------------------- 실행기
class Installer:
    def __init__(self, log: Log, session=None, open_url: Callable[[str], None] = lambda u: None,
                 cancel: Optional[threading.Event] = None, github_api: str = GITHUB_API):
        self.log = log
        self.session = session or requests.Session()
        self.open_url = open_url
        self.cancel = cancel or threading.Event()
        self.github_api = github_api
        self._proc: Optional[subprocess.Popen] = None

    def run(self, plan: InstallPlan) -> None:
        self.log(f"▶ {plan.label}")
        if plan.note:
            self.log(f"ℹ️ {plan.note}")
        for i, step in enumerate(plan.steps, 1):
            if self.cancel.is_set():
                raise Cancelled("사용자가 취소했어요.")
            desc = getattr(step, "desc", "") or type(step).__name__
            self.log(f"[{i}/{len(plan.steps)}] {desc}")
            if isinstance(step, CommandStep):
                self._command(step)
            elif isinstance(step, GithubReleaseStep):
                self._github(step)
            elif isinstance(step, VenvPipStep):
                self._venv_pip(step)
            elif isinstance(step, RemovePathStep):
                self._remove(step)
            elif isinstance(step, OpenUrlStep):
                self.open_url(step.url)
                self.log(f"🌐 브라우저에서 {step.url} 을(를) 열었어요. 안내에 따라 설치한 뒤 [다시 확인]을 눌러 주세요.")
        self.log("✅ 완료")

    def stop(self) -> None:
        self.cancel.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    # ---- 명령 실행 -----------------------------------------------------
    def _command(self, step: CommandStep) -> None:
        env = {**os.environ, **step.env}
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(step.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                          stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                          bufsize=1, **kwargs)
        except OSError as e:
            raise InstallError(f"명령을 실행하지 못했어요: {step.args[0]} ({e})") from e
        assert self._proc.stdout
        for line in self._proc.stdout:
            line = line.rstrip()
            if line:
                self.log("  " + line)
        code = self._proc.wait()
        if self.cancel.is_set():
            raise Cancelled("사용자가 취소했어요.")
        if code != 0:
            hint = ""
            if step.needs_admin:
                hint = " 관리자 권한이 필요할 수 있어요. 터미널에서 직접 실행해 보세요: " + " ".join(step.args)
            raise InstallError(f"명령이 실패했어요 (종료 코드 {code}).{hint}")

    # ---- GitHub 배포 ---------------------------------------------------
    def _github(self, step: GithubReleaseStep) -> None:
        r = self.session.get(f"{self.github_api}/repos/{step.repo}/releases", params={"per_page": 10}, timeout=20)
        if r.status_code != 200:
            raise InstallError(f"GitHub에서 배포 목록을 받지 못했어요 (HTTP {r.status_code}). 잠시 후 다시 시도해 주세요.")
        chosen, release = [], None
        for rel in r.json():
            if rel.get("draft") or (rel.get("prerelease") and not step.allow_prerelease):
                continue
            assets = {a["name"]: a for a in rel.get("assets", [])}
            chosen = step.picker(list(assets))
            if chosen:
                release = rel
                break
        if not chosen or release is None:
            raise InstallError("이 컴퓨터에 맞는 설치 파일을 찾지 못했어요. 홈페이지에서 직접 받아 주세요.")
        self.log(f"  버전 {release.get('tag_name')}: {', '.join(chosen)}")
        assets = {a["name"]: a for a in release["assets"]}
        step.dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="llamalauncher-") as tmp:
            for name in chosen:
                local = Path(tmp) / name
                self._download(assets[name]["browser_download_url"], local, assets[name].get("size", 0))
                if name.endswith((".zip", ".tar.gz", ".tgz")):
                    self.log(f"  압축 푸는 중: {name}")
                    extract_archive(local, step.dest)
                else:
                    target = step.dest / (step.executable_name or name)
                    shutil.move(str(local), target)
                    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (step.dest / ".version").write_text(str(release.get("tag_name", "")), encoding="utf-8")

    def _download(self, url: str, dest: Path, size: int) -> None:
        with self.session.get(url, stream=True, timeout=(10, 60)) as r:
            if r.status_code != 200:
                raise InstallError(f"파일을 받지 못했어요 (HTTP {r.status_code}).")
            total = size or int(r.headers.get("Content-Length") or 0)
            done, last_pct = 0, -10
            with open(dest, "wb") as f:
                for chunk in r.iter_content(256 * 1024):
                    if self.cancel.is_set():
                        raise Cancelled("사용자가 취소했어요.")
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        if pct >= last_pct + 10:
                            last_pct = pct
                            self.log(f"  받는 중… {pct}% ({done >> 20} / {total >> 20} MB)")

    # ---- 파이썬 가상환경 -----------------------------------------------
    def _venv_pip(self, step: VenvPipStep) -> None:
        py = venv_bin(step.venv, "python")
        if not py.exists():
            base = find_system_python()
            if not base:
                raise InstallError("파이썬 3.10 이상을 찾지 못했어요. python.org에서 파이썬을 설치한 뒤 다시 시도해 주세요.")
            self.log(f"  가상환경 만드는 중: {step.venv} (파이썬 {base})")
            self._command(CommandStep([str(base), "-m", "venv", str(step.venv)]))
        self._command(CommandStep([str(py), "-m", "pip", "install", "--upgrade", "pip"]))
        args = [str(py), "-m", "pip", "install"] + (["--upgrade"] if step.upgrade else []) + step.packages
        self._command(CommandStep(args))

    def _remove(self, step: RemovePathStep) -> None:
        p = Path(step.path)
        if app_home().resolve() not in p.resolve().parents:
            raise InstallError(f"라마런처 폴더 밖은 지우지 않아요: {p}")
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def find_system_python() -> Optional[Path]:
    """가상환경을 만들 파이썬. 앱이 패키징(frozen)되어 있으면 sys.executable은 앱 자체라 쓰지 않는다."""
    if not getattr(sys, "frozen", False) and sys.version_info >= (3, 10):
        return Path(sys.executable)
    names = ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"]
    if sys.platform == "win32":
        names = ["py", "python"]
    for n in names:
        p = shutil.which(n)
        if not p:
            continue
        out = subprocess.run([p, "-c", "import sys; print(sys.version_info >= (3, 10))"],
                             capture_output=True, text=True, timeout=10)
        if out.stdout.strip() == "True":
            return Path(p)
    return None


def extract_archive(archive: Path, dest: Path) -> None:
    """zip/tar.gz를 풀되, 최상위 폴더가 하나뿐이면 그 안의 내용을 dest 바로 아래에 둔다."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".extract-") as tmp:
        tmpd = Path(tmp)
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                for m in z.infolist():
                    target = (tmpd / m.filename).resolve()
                    if tmpd.resolve() not in target.parents and target != tmpd.resolve():
                        raise InstallError("압축 파일 안에 이상한 경로가 있어요.")
                z.extractall(tmpd)
                for m in z.infolist():  # zip은 실행 권한을 잃기 쉬워서 복원
                    mode = (m.external_attr >> 16) & 0o777
                    if mode:
                        (tmpd / m.filename).chmod(mode)
        else:
            with tarfile.open(archive) as t:
                t.extractall(tmpd, filter="data")
        entries = [e for e in tmpd.iterdir() if e.name not in ("__MACOSX",)]
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmpd
        for item in src.iterdir():
            target = dest / item.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            shutil.move(str(item), target)
