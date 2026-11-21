"""llama.cpp (llama-server) 감지."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

from .base import (
    CommandStep, GithubReleaseStep, InstallPlan, Launcher, LauncherStatus, RemovePathStep, ServeSpec, bin_dir,
    brew_plan, env_path, exe_name, has_brew, has_winget, is_brew_path, is_under, run_capture, winget_plan,
)

VERSION_RE = re.compile(r"version:\s*(\S+)\s*\(build\s*(\d+)", re.I)
OLD_VERSION_RE = re.compile(r"version:\s*(\d+)\s*\((\w+)\)", re.I)


def candidate_paths() -> list[Path]:
    """PATH에 없을 때 찾아볼 일반적인 설치 위치."""
    name = exe_name("llama-server")
    home = Path.home()
    own = bin_dir("llamacpp")
    paths: list[Path] = [own / name]
    if own.is_dir():  # 압축 파일 구조가 바뀌어도 찾을 수 있게
        paths += sorted(own.rglob(name))
    if sys.platform == "darwin":
        paths += [Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name]
    elif sys.platform == "win32":
        local = env_path("LOCALAPPDATA")
        if local:
            paths.append(local / "Microsoft" / "WinGet" / "Links" / name)
            pkgs = local / "Microsoft" / "WinGet" / "Packages"
            if pkgs.is_dir():
                paths += sorted(pkgs.glob(f"ggml.llamacpp*/**/{name}"))
        for var in ("ProgramFiles", "ProgramFiles(x86)"):
            base = env_path(var)
            if base:
                paths.append(base / "llama.cpp" / name)
    else:
        paths += [Path("/usr/local/bin") / name, Path("/usr/bin") / name, home / ".local" / "bin" / name]
    return paths


def find_server(custom_path: Optional[str] = None, extra: Iterable[Path] = ()) -> Optional[Path]:
    if custom_path:
        p = Path(custom_path).expanduser()
        if p.is_dir():
            p = p / exe_name("llama-server")
        return p if p.is_file() else None
    found = shutil.which("llama-server")
    if found:
        return Path(found)
    for p in [*extra, *candidate_paths()]:
        if p.is_file():
            return p
    return None


def parse_version(text: str) -> tuple[str, Optional[int]]:
    m = VERSION_RE.search(text or "")
    if m:
        return f"{m.group(1)} (b{m.group(2)})", int(m.group(2))
    m = OLD_VERSION_RE.search(text or "")
    if m:
        return f"b{m.group(1)}", int(m.group(1))
    return "", None


def parse_capabilities(help_text: str) -> set[str]:
    """--help 출력에서 버전마다 다른 기능의 지원 여부를 확인한다."""
    caps: set[str] = set()
    t = help_text or ""
    if re.search(r"--flash-attn\s+\[on\|off\|auto\]", t):
        caps.add("fa_value")  # -fa on|off|auto (최신)
    elif "--flash-attn" in t:
        caps.add("fa_flag")  # -fa (예전 방식: 켜기만 가능)
    for flag, cap in [("--alias", "alias"), ("--api-key", "api_key"), ("--jinja", "jinja"),
                      ("--mmproj", "mmproj"), ("--cache-type-v", "cache_type_v"), ("--min-p", "min_p"),
                      ("--no-mmap", "no_mmap"), ("--mlock", "mlock")]:
        if flag in t:
            caps.add(cap)
    return caps


class LlamaCppLauncher(Launcher):
    id = "llamacpp"
    name = "llama.cpp"
    description = "가볍고 빠른 기본 실행 엔진. GGUF 모델을 거의 모든 컴퓨터에서 실행해요."
    homepage = "https://github.com/ggml-org/llama.cpp"
    formats = ("gguf",)
    default_port = 8080

    def __init__(self, runner: Callable[[list[str]], Optional[str]] = run_capture):
        self.runner = runner

    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus:
        path = find_server(custom_path)
        if path is None:
            msg = ("지정한 경로에서 llama-server를 찾지 못했어요." if custom_path
                   else "llama.cpp가 설치되어 있지 않아요. 런처 탭에서 설치하거나 경로를 직접 지정해 주세요.")
            return LauncherStatus(installed=False, message=msg)
        version, build = parse_version(self.runner([str(path), "--version"]) or "")
        caps = parse_capabilities(self.runner([str(path), "--help"]) or "")
        if not caps:
            # --help가 실패해도 최신 빌드라고 가정하고 기본 기능은 쓴다
            caps = {"fa_value", "alias", "api_key", "jinja", "cache_type_v", "min_p", "no_mmap", "mlock", "mmproj"}
        return LauncherStatus(installed=True, path=path, version=version or "버전 확인 불가",
                              build=build, capabilities=caps)

    # ---- 설치 ----------------------------------------------------------
    def _github_plan(self, system) -> InstallPlan:
        from ..installer import pick_llamacpp_assets
        gpu = system.primary_gpu.vendor if system and system.primary_gpu else "unknown"
        machine = system.arch if system else ""
        return InstallPlan("GitHub에서 받아 설치", [GithubReleaseStep(
            repo="ggml-org/llama.cpp", dest=bin_dir(self.id),
            picker=lambda names: pick_llamacpp_assets(names, sys.platform, machine, gpu),
            desc=f"llama.cpp 최신 버전 받기 ({_gpu_label(gpu)})", allow_prerelease=True)],
            note=f"설치 위치: {bin_dir(self.id)} (관리자 권한 필요 없음)")

    def install_plans(self, system) -> list[InstallPlan]:
        plans = []
        if brew := has_brew():
            plans.append(brew_plan(brew, "llama.cpp"))
        if winget := has_winget():
            plans.append(winget_plan(winget, "ggml.llamacpp"))
        plans.append(self._github_plan(system))
        return plans

    def update_plans(self, system, status: LauncherStatus) -> list[InstallPlan]:
        if is_brew_path(status.path) and (brew := has_brew()):
            return [InstallPlan("Homebrew로 업데이트", [CommandStep([str(brew), "upgrade", "llama.cpp"], "brew upgrade llama.cpp")])]
        if is_under(status.path, bin_dir(self.id)):
            return [self._github_plan(system)]
        if "WinGet" in str(status.path) and (winget := has_winget()):
            return [InstallPlan("winget으로 업데이트", [CommandStep([str(winget), "upgrade", "-e", "--id", "ggml.llamacpp"])])]
        return []

    def uninstall_plans(self, status: LauncherStatus) -> list[InstallPlan]:
        if is_brew_path(status.path) and (brew := has_brew()):
            return [InstallPlan("Homebrew로 제거", [CommandStep([str(brew), "uninstall", "llama.cpp"], "brew uninstall llama.cpp")])]
        if is_under(status.path, bin_dir(self.id)):
            return [InstallPlan("라마런처가 받은 파일 지우기", [RemovePathStep(bin_dir(self.id), "llama.cpp 폴더 삭제")])]
        if "WinGet" in str(status.path) and (winget := has_winget()):
            return [InstallPlan("winget으로 제거", [CommandStep([str(winget), "uninstall", "-e", "--id", "ggml.llamacpp"])])]
        return []

    # ---- 실행 ----------------------------------------------------------
    def serve_spec(self, status, model, opts, system=None) -> ServeSpec:
        from ..recommender import build_server_args
        return ServeSpec(program=str(status.path), args=build_server_args(opts, model, caps=status.capabilities),
                         host=opts.host, port=opts.port, health_path="/health",
                         model_name=opts.alias or "local-model")


def _gpu_label(vendor: str) -> str:
    return {"nvidia": "NVIDIA CUDA", "amd": "Vulkan", "intel": "Vulkan", "apple": "Apple Metal"}.get(vendor, "CPU")
