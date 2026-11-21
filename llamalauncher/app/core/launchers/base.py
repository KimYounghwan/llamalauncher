"""런처(실행 엔진) 공통 인터페이스.

각 런처는 다음을 정의한다.
- 감지(detect): 설치 여부, 경로, 버전
- 사용 가능 여부(availability): OS/GPU 조건 (예: MLX는 Apple Silicon 전용)
- 설치/업데이트/제거 방법(install_plans 등): 명령어·파일 다운로드·가상환경 pip 단계의 목록
- 서버 실행 방법(serve_spec): 모델 경로와 옵션을 받아 실행 명령을 만든다 (서버형 런처만)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ..settings import APP_NAME

if TYPE_CHECKING:
    from ..hardware import SystemInfo
    from ..recommender import ServerOptions


@dataclass
class LauncherStatus:
    installed: bool
    path: Optional[Path] = None
    version: str = ""
    build: Optional[int] = None
    capabilities: set[str] = field(default_factory=set)
    message: str = ""
    running: bool = False   # (Ollama/LM Studio처럼 늘 켜져 있는 런처) 이미 실행 중인지


# ---------------------------------------------------------------- 설치 단계
@dataclass
class CommandStep:
    args: list[str]
    desc: str = ""
    env: dict = field(default_factory=dict)
    needs_admin: bool = False


@dataclass
class GithubReleaseStep:
    """GitHub 최신 배포에서 이 컴퓨터에 맞는 파일을 골라 받아 dest에 풀거나 둔다."""
    repo: str
    dest: Path
    picker: Callable[[list[str]], list[str]]   # 배포 파일 이름 목록 → 받을 파일들
    desc: str = ""
    executable_name: str = ""                   # 단일 실행 파일이면 이 이름으로 저장
    allow_prerelease: bool = False              # llama.cpp는 모든 빌드를 prerelease로 올린다


@dataclass
class VenvPipStep:
    venv: Path
    packages: list[str]
    desc: str = ""
    upgrade: bool = False


@dataclass
class RemovePathStep:
    path: Path
    desc: str = ""


@dataclass
class OpenUrlStep:
    url: str
    desc: str = ""


Step = CommandStep | GithubReleaseStep | VenvPipStep | RemovePathStep | OpenUrlStep


@dataclass
class InstallPlan:
    label: str          # 버튼/로그에 보일 이름 (예: "Homebrew로 설치")
    steps: list[Step]
    note: str = ""      # 사용자에게 미리 알려줄 점

    @property
    def manual(self) -> bool:
        """다운로드 페이지를 여는 것뿐인 계획 (사용자가 직접 설치)."""
        return all(isinstance(s, OpenUrlStep) for s in self.steps)


# ---------------------------------------------------------------- 서버 실행
@dataclass
class ServeSpec:
    program: str
    args: list[str]
    host: str
    port: int
    health_path: str = "/health"
    env: dict = field(default_factory=dict)
    model_name: str = "local-model"      # API 요청의 model 값
    prepare: Optional[Callable[[Callable[[str], None]], None]] = None  # 서버가 뜬 뒤 할 일 (Ollama 모델 등록 등)
    api_prefix: str = "/v1"
    notes: list[str] = field(default_factory=list)


class Launcher(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    homepage: str = ""
    platforms: tuple[str, ...] = ("darwin", "win32", "linux")
    formats: tuple[str, ...] = ()        # gguf | mlx | safetensors | ollama
    kind: str = "server"                 # server: 라마런처가 직접 실행 / app: 따로 여는 프로그램
    default_port: int = 8080
    model_is_dir: bool = False           # 모델이 파일이 아니라 폴더(MLX, safetensors)

    # 하위 호환
    @property
    def supported_formats(self) -> tuple[str, ...]:
        return self.formats

    @abstractmethod
    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus: ...

    def availability(self, system: Optional["SystemInfo"]) -> tuple[bool, str]:
        """(이 컴퓨터에서 쓸 수 있는지, 못 쓰면 이유)."""
        if sys.platform not in self.platforms:
            names = {"darwin": "macOS", "win32": "Windows", "linux": "Linux"}
            return False, "이 운영체제에서는 쓸 수 없어요 (지원: " + ", ".join(names[p] for p in self.platforms) + ")."
        return True, ""

    def install_plans(self, system: Optional["SystemInfo"]) -> list[InstallPlan]:
        return [InstallPlan("다운로드 페이지 열기", [OpenUrlStep(self.homepage)])] if self.homepage else []

    def update_plans(self, system: Optional["SystemInfo"], status: LauncherStatus) -> list[InstallPlan]:
        return []

    def uninstall_plans(self, status: LauncherStatus) -> list[InstallPlan]:
        return []

    def serve_spec(self, status: LauncherStatus, model: str, opts: "ServerOptions",
                   system: Optional["SystemInfo"] = None) -> ServeSpec:
        raise NotImplementedError(f"{self.name}은(는) 라마런처에서 직접 실행하지 않아요. 프로그램을 열어서 사용해 주세요.")

    @property
    def runnable(self) -> bool:
        return self.kind == "server"


# ---------------------------------------------------------------- 도우미
def run_capture(cmd: list[str], timeout: float = 15.0) -> Optional[str]:
    """stdout+stderr를 합쳐서 돌려준다 (llama.cpp는 버전 정보를 stderr로 출력)."""
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                             text=True, encoding="utf-8", errors="replace", **kwargs)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def exe_name(base: str) -> str:
    return base + ".exe" if sys.platform == "win32" else base


def env_path(var: str) -> Optional[Path]:
    v = os.environ.get(var)
    return Path(v) if v else None


def app_home() -> Path:
    return Path.home() / APP_NAME


def bin_dir(launcher_id: str) -> Path:
    """라마런처가 직접 받아 둔 런처 실행 파일 위치."""
    return app_home() / "bin" / launcher_id


def venv_dir(launcher_id: str) -> Path:
    return app_home() / "venvs" / launcher_id


def venv_bin(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if sys.platform == "win32" else "bin") / exe_name(name)


def which(name: str) -> Optional[Path]:
    p = shutil.which(name)
    return Path(p) if p else None


def has_brew() -> Optional[Path]:
    if sys.platform == "win32":
        return None
    for p in (which("brew"), Path("/opt/homebrew/bin/brew"), Path("/usr/local/bin/brew"),
              Path("/home/linuxbrew/.linuxbrew/bin/brew")):
        if p and p.is_file():
            return p
    return None


def has_winget() -> Optional[Path]:
    return which("winget") if sys.platform == "win32" else None


def brew_plan(brew: Path, formula: str, cask: bool = False) -> InstallPlan:
    args = [str(brew), "install"] + (["--cask"] if cask else []) + [formula]
    return InstallPlan("Homebrew로 설치", [CommandStep(args, f"brew install {'--cask ' if cask else ''}{formula}",
                                                      env={"HOMEBREW_NO_AUTO_UPDATE": "1"})])


def winget_plan(winget: Path, pkg_id: str) -> InstallPlan:
    return InstallPlan("winget으로 설치", [CommandStep(
        [str(winget), "install", "-e", "--id", pkg_id, "--accept-source-agreements", "--accept-package-agreements"],
        f"winget install {pkg_id}")], note="설치 중 Windows가 권한을 물어보면 [예]를 눌러 주세요.")


def is_brew_path(p: Optional[Path]) -> bool:
    s = str(p or "")
    return "/opt/homebrew/" in s or "/usr/local/Cellar" in s or "linuxbrew" in s or (
        s.startswith("/usr/local/bin/") and Path(s).is_symlink() and "Cellar" in os.path.realpath(s))


def is_under(p: Optional[Path], root: Path) -> bool:
    try:
        return p is not None and root.resolve() in Path(p).resolve().parents
    except OSError:
        return False
