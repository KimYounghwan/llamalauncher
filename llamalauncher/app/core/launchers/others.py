"""그 밖의 런처: KoboldCpp, MLX-LM, vLLM (서버형) / LM Studio, Jan, text-generation-webui (앱형)."""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path
from typing import Callable, Optional

import requests

from .base import (
    CommandStep, GithubReleaseStep, InstallPlan, Launcher, LauncherStatus, OpenUrlStep, RemovePathStep, ServeSpec,
    VenvPipStep, bin_dir, brew_plan, env_path, exe_name, has_brew, has_winget, run_capture, venv_bin, venv_dir,
    which, winget_plan,
)


def _port_open(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.5).status_code < 500
    except requests.RequestException:
        return False


def _read_version_file(d: Path) -> str:
    try:
        return (d / ".version").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


# ---------------------------------------------------------------- KoboldCpp
class KoboldCppLauncher(Launcher):
    id = "koboldcpp"
    name = "KoboldCpp"
    description = "실행 파일 하나로 끝나는 llama.cpp 기반 런처. 소설·롤플레이용 웹 화면이 함께 있어요."
    homepage = "https://github.com/LostRuins/koboldcpp/releases"
    formats = ("gguf",)
    default_port = 5001

    def _exe(self) -> Path:
        return bin_dir(self.id) / exe_name("koboldcpp")

    def availability(self, system):
        ok, why = super().availability(system)
        if ok and sys.platform == "darwin" and system and system.arch not in ("arm64", "aarch64"):
            return False, "Intel Mac용 실행 파일은 제공되지 않아요."
        return ok, why

    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus:
        path = Path(custom_path).expanduser() if custom_path else None
        if not path:
            path = self._exe() if self._exe().is_file() else which("koboldcpp")
        if not path or not Path(path).is_file():
            return LauncherStatus(installed=False, message="KoboldCpp가 설치되어 있지 않아요.")
        return LauncherStatus(installed=True, path=Path(path), version=_read_version_file(Path(path).parent) or "설치됨")

    def _plan(self, system, label: str) -> InstallPlan:
        from ..installer import pick_koboldcpp_asset
        gpu = system.primary_gpu.vendor if system and system.primary_gpu else "unknown"
        machine = system.arch if system else ""
        return InstallPlan(label, [GithubReleaseStep(
            repo="LostRuins/koboldcpp", dest=bin_dir(self.id), executable_name=exe_name("koboldcpp"),
            picker=lambda names: pick_koboldcpp_asset(names, sys.platform, machine, gpu),
            desc="KoboldCpp 최신 버전 받기")], note=f"설치 위치: {bin_dir(self.id)}")

    def install_plans(self, system):
        return [self._plan(system, "GitHub에서 받아 설치")]

    def update_plans(self, system, status):
        return [self._plan(system, "최신 버전으로 업데이트")] if status.path == self._exe() else []

    def uninstall_plans(self, status):
        if status.path == self._exe():
            return [InstallPlan("라마런처가 받은 파일 지우기", [RemovePathStep(bin_dir(self.id), "KoboldCpp 폴더 삭제")])]
        return []

    def serve_spec(self, status, model, opts, system=None) -> ServeSpec:
        args = ["--model", model, "--host", opts.host, "--port", str(opts.port),
                "--contextsize", str(opts.ctx_size), "--gpulayers", str(opts.n_gpu_layers),
                "--threads", str(opts.threads), "--skiplauncher"]
        vendor = system.primary_gpu.vendor if system and system.primary_gpu else ""
        if vendor == "nvidia":
            args.append("--usecublas")
        elif vendor in ("amd", "intel"):
            args.append("--usevulkan")
        if opts.flash_attn == "on":
            args.append("--flashattention")
            if opts.cache_type_k == "q8_0":
                args += ["--quantkv", "1"]
        if opts.mmproj:
            args += ["--mmproj", opts.mmproj]
        if opts.api_key:
            args += ["--password", opts.api_key]
        return ServeSpec(program=str(status.path), args=args, host=opts.host, port=opts.port,
                         health_path="/api/v1/model", model_name=opts.alias or "koboldcpp",
                         notes=["KoboldCpp 웹 화면(소설·대화)은 서버 주소를 브라우저로 열면 나와요."])


# ---------------------------------------------------------------- 파이썬 기반 (MLX-LM, vLLM)
class _VenvLauncher(Launcher):
    package = ""
    server_exe = ""
    model_is_dir = True

    @property
    def venv(self) -> Path:
        return venv_dir(self.id)

    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus:
        path = Path(custom_path).expanduser() if custom_path else None
        if not path:
            own = venv_bin(self.venv, self.server_exe)
            path = own if own.is_file() else which(self.server_exe)
        if not path or not Path(path).is_file():
            return LauncherStatus(installed=False, message=f"{self.name}이(가) 설치되어 있지 않아요.")
        version = ""
        py = next((Path(path).parent / exe_name(n) for n in ("python", "python3")
                   if (Path(path).parent / exe_name(n)).is_file()), None)
        if py:
            out = run_capture([str(py), "-c", f"import importlib.metadata as m; print(m.version('{self.package}'))"])
            version = (out or "").strip().splitlines()[-1] if out and out.strip() else ""
        return LauncherStatus(installed=True, path=Path(path), version=version or "설치됨")

    def install_plans(self, system):
        return [InstallPlan("라마런처 전용 파이썬 환경에 설치", [VenvPipStep(self.venv, [self.package], f"pip install {self.package}")],
                            note=f"설치 위치: {self.venv} (시스템 파이썬은 건드리지 않아요)")]

    def update_plans(self, system, status):
        if status.path and str(self.venv) in str(status.path):
            return [InstallPlan("최신 버전으로 업데이트", [VenvPipStep(self.venv, [self.package], f"pip install -U {self.package}", upgrade=True)])]
        return []

    def uninstall_plans(self, status):
        if status.path and str(self.venv) in str(status.path):
            return [InstallPlan("전용 파이썬 환경 지우기", [RemovePathStep(self.venv, f"{self.venv} 삭제")])]
        return []


class MlxLauncher(_VenvLauncher):
    id = "mlx"
    name = "MLX-LM"
    description = "애플이 만든 Apple Silicon 전용 엔진. 맥에서는 GGUF보다 빠른 경우가 많아요 (MLX 형식 모델)."
    homepage = "https://github.com/ml-explore/mlx-lm"
    platforms = ("darwin",)
    formats = ("mlx",)
    default_port = 8081
    package = "mlx-lm"
    server_exe = "mlx_lm.server"

    def availability(self, system):
        if sys.platform != "darwin":
            return False, "Apple Silicon 맥 전용이에요."
        if system and system.arch not in ("arm64", "aarch64"):
            return False, "Apple Silicon(M1 이후) 맥에서만 동작해요."
        return True, ""

    def serve_spec(self, status, model, opts, system=None) -> ServeSpec:
        args = ["--model", model, "--host", opts.host, "--port", str(opts.port)]
        return ServeSpec(program=str(status.path), args=args, host=opts.host, port=opts.port,
                         health_path="/v1/models", model_name=model,
                         notes=["MLX-LM은 문맥 길이·GPU 설정을 자동으로 정해요. 아래 옵션 중 일부는 쓰이지 않아요."])


class VllmLauncher(_VenvLauncher):
    id = "vllm"
    name = "vLLM"
    description = "여러 사람이 동시에 쓰는 서버용 고성능 엔진. Linux + NVIDIA GPU 전용이에요 (safetensors/AWQ/GPTQ)."
    homepage = "https://docs.vllm.ai/en/latest/getting_started/installation/"
    platforms = ("linux",)
    formats = ("safetensors",)
    default_port = 8000
    package = "vllm"
    server_exe = "vllm"

    def availability(self, system):
        if not sys.platform.startswith("linux"):
            return False, "Linux + NVIDIA GPU 전용이에요. Windows에서는 WSL2(우분투) 안에서 설치할 수 있어요."
        if system and not any(g.vendor == "nvidia" for g in system.gpus):
            return False, "NVIDIA GPU가 필요해요."
        return True, ""

    def serve_spec(self, status, model, opts, system=None) -> ServeSpec:
        name = opts.alias or "local-model"
        args = ["serve", model, "--host", opts.host, "--port", str(opts.port), "--max-model-len", str(opts.ctx_size),
                "--served-model-name", name, "--gpu-memory-utilization", "0.90"]
        if opts.api_key:
            args += ["--api-key", opts.api_key]
        if opts.parallel > 1:
            args += ["--max-num-seqs", str(opts.parallel)]
        return ServeSpec(program=str(status.path), args=args, host=opts.host, port=opts.port,
                         health_path="/health", model_name=name)


# ---------------------------------------------------------------- 앱형 런처
class _AppLauncher(Launcher):
    kind = "app"
    brew_cask = ""
    winget_id = ""

    def app_paths(self) -> list[Path]:
        return []

    def version_of(self, path: Path) -> str:
        if sys.platform == "darwin" and path.suffix == ".app":
            try:
                with open(path / "Contents" / "Info.plist", "rb") as f:
                    return plistlib.load(f).get("CFBundleShortVersionString", "")
            except (OSError, plistlib.InvalidFileException):
                return ""
        return ""

    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus:
        cands = [Path(custom_path).expanduser()] if custom_path else self.app_paths()
        path = next((p for p in cands if p.exists()), None)
        running = _port_open(self.default_port)
        if not path:
            return LauncherStatus(installed=False, message=f"{self.name}이(가) 설치되어 있지 않아요.", running=running)
        return LauncherStatus(installed=True, path=path, version=self.version_of(path) or "설치됨", running=running)

    def install_plans(self, system):
        plans = []
        if self.brew_cask and sys.platform == "darwin" and (brew := has_brew()):
            plans.append(brew_plan(brew, self.brew_cask, cask=True))
        if self.winget_id and (winget := has_winget()):
            plans.append(winget_plan(winget, self.winget_id))
        plans.append(InstallPlan("다운로드 페이지 열기", [OpenUrlStep(self.homepage)]))
        return plans

    def _brew_managed(self) -> Optional[Path]:
        """Homebrew cask로 설치한 경우에만 brew로 업데이트/제거한다 (직접 받은 앱은 앱 자체 업데이트 사용)."""
        if not (self.brew_cask and sys.platform == "darwin"):
            return None
        brew = has_brew()
        if brew and (brew.parent.parent / "Caskroom" / self.brew_cask).is_dir():
            return brew
        return None

    def uninstall_plans(self, status):
        if status.installed and (brew := self._brew_managed()):
            return [InstallPlan("Homebrew로 제거", [CommandStep([str(brew), "uninstall", "--cask", self.brew_cask])])]
        return []

    def update_plans(self, system, status):
        if status.installed and (brew := self._brew_managed()):
            return [InstallPlan("Homebrew로 업데이트", [CommandStep([str(brew), "upgrade", "--cask", self.brew_cask])])]
        return []


def _mac_apps(name: str) -> list[Path]:
    return [Path("/Applications") / f"{name}.app", Path.home() / "Applications" / f"{name}.app"]


def _win_programs(*parts: str) -> list[Path]:
    local = env_path("LOCALAPPDATA")
    return [local / "Programs" / Path(*parts)] if local else []


class LMStudioLauncher(_AppLauncher):
    id = "lmstudio"
    name = "LM Studio"
    description = "예쁜 화면에서 모델 검색·채팅·서버까지 다 되는 앱. 맥에서는 MLX 모델도 실행해요."
    homepage = "https://lmstudio.ai/download"
    formats = ("gguf", "mlx") if sys.platform == "darwin" else ("gguf",)
    default_port = 1234
    brew_cask = "lm-studio"
    winget_id = "ElementLabs.LMStudio"

    def app_paths(self):
        if sys.platform == "darwin":
            return _mac_apps("LM Studio")
        if sys.platform == "win32":
            return _win_programs("LM Studio", "LM Studio.exe") + _win_programs("lm-studio", "LM Studio.exe")
        return [Path.home() / ".lmstudio" / "bin" / "lms"]


class JanLauncher(_AppLauncher):
    id = "jan"
    name = "Jan"
    description = "ChatGPT처럼 쓰는 오픈소스 데스크톱 앱. 인터넷 없이 내 컴퓨터에서만 동작해요."
    homepage = "https://jan.ai/download"
    formats = ("gguf",)
    default_port = 1337
    brew_cask = "jan"

    def app_paths(self):
        if sys.platform == "darwin":
            return _mac_apps("Jan")
        if sys.platform == "win32":
            return _win_programs("Jan", "Jan.exe") + _win_programs("jan", "Jan.exe")
        return [Path.home() / "Applications" / "Jan.AppImage"]


class TextGenWebUILauncher(_AppLauncher):
    id = "tgwui"
    name = "text-generation-webui"
    description = "기능이 아주 많은 고급 사용자용 웹 화면 (oobabooga). GGUF·safetensors 등 여러 형식을 지원해요."
    homepage = "https://github.com/oobabooga/text-generation-webui#installation"
    formats = ("gguf", "safetensors")
    default_port = 5000

    def app_paths(self):
        home = Path.home()
        return [home / "text-generation-webui", home / "Applications" / "text-generation-webui"]
