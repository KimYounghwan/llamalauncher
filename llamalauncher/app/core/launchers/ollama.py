"""Ollama: 모델 관리가 쉬운 인기 런처. GGUF 파일을 등록(ollama create)해서 실행한다."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import requests

from .base import (
    CommandStep, InstallPlan, Launcher, LauncherStatus, OpenUrlStep, ServeSpec, brew_plan, env_path, exe_name,
    has_brew, has_winget, is_brew_path, run_capture, which, winget_plan,
)

VERSION_RE = re.compile(r"version(?: is)?\s+v?(\d[\w.\-]*)", re.I)


def candidate_paths() -> list[Path]:
    paths: list[Path] = []
    if sys.platform == "darwin":
        paths += [Path("/Applications/Ollama.app/Contents/Resources/ollama"),
                  Path("/opt/homebrew/bin/ollama"), Path("/usr/local/bin/ollama")]
    elif sys.platform == "win32":
        local = env_path("LOCALAPPDATA")
        if local:
            paths.append(local / "Programs" / "Ollama" / "ollama.exe")
    else:
        paths += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    return paths


def model_name_for(path: str) -> str:
    """GGUF 파일 이름으로 Ollama 모델 이름을 만든다 (소문자, 허용 문자만)."""
    stem = Path(path).name
    stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", stem, flags=re.I)
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    slug = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-._")[:60] or "model"
    return f"llamalauncher/{slug}"


def is_running(port: int = 11434, host: str = "127.0.0.1") -> bool:
    try:
        return requests.get(f"http://{host}:{port}/api/version", timeout=0.5).status_code == 200
    except requests.RequestException:
        return False


class OllamaLauncher(Launcher):
    id = "ollama"
    name = "Ollama"
    description = "명령 한 줄로 모델을 받고 실행하는 인기 런처. 다른 앱들이 많이 지원해요."
    homepage = "https://ollama.com/download"
    formats = ("gguf", "ollama")
    default_port = 11435  # 이미 켜져 있을 수 있는 Ollama 앱(11434)과 겹치지 않게

    def __init__(self, runner: Callable[[list[str]], Optional[str]] = run_capture):
        self.runner = runner

    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus:
        path = Path(custom_path).expanduser() if custom_path else (which("ollama") or next(
            (p for p in candidate_paths() if p.is_file()), None))
        if not path or not path.is_file():
            return LauncherStatus(installed=False, message="Ollama가 설치되어 있지 않아요.", running=is_running())
        out = self.runner([str(path), "--version"]) or ""
        m = VERSION_RE.search(out)
        version = m.group(1) if m else "버전 확인 불가"
        return LauncherStatus(installed=True, path=path, version=version, running=is_running())

    def install_plans(self, system) -> list[InstallPlan]:
        plans = []
        if brew := has_brew():
            plans.append(brew_plan(brew, "ollama"))
        if winget := has_winget():
            plans.append(winget_plan(winget, "Ollama.Ollama"))
        if sys.platform.startswith("linux"):
            plans.append(InstallPlan("공식 설치 안내 열기", [OpenUrlStep("https://ollama.com/download/linux")],
                                     note="터미널에서 실행: curl -fsSL https://ollama.com/install.sh | sh (관리자 암호 필요)"))
        plans.append(InstallPlan("다운로드 페이지 열기", [OpenUrlStep(self.homepage)]))
        return plans

    def update_plans(self, system, status) -> list[InstallPlan]:
        if is_brew_path(status.path) and (brew := has_brew()):
            return [InstallPlan("Homebrew로 업데이트", [CommandStep([str(brew), "upgrade", "ollama"], "brew upgrade ollama")])]
        if sys.platform == "win32" and (winget := has_winget()):
            return [InstallPlan("winget으로 업데이트", [CommandStep([str(winget), "upgrade", "-e", "--id", "Ollama.Ollama"])])]
        return []

    def uninstall_plans(self, status) -> list[InstallPlan]:
        if is_brew_path(status.path) and (brew := has_brew()):
            return [InstallPlan("Homebrew로 제거", [CommandStep([str(brew), "uninstall", "ollama"], "brew uninstall ollama")])]
        if sys.platform == "win32" and (winget := has_winget()):
            return [InstallPlan("winget으로 제거", [CommandStep([str(winget), "uninstall", "-e", "--id", "Ollama.Ollama"])])]
        return []

    # ---- 실행 ----------------------------------------------------------
    def serve_spec(self, status, model, opts, system=None) -> ServeSpec:
        from ..model_library import ollama_dirs, scan_ollama
        exe = str(status.path)
        host = f"{opts.host}:{opts.port}"
        env = {"OLLAMA_HOST": host, "OLLAMA_CONTEXT_LENGTH": str(opts.ctx_size), "OLLAMA_KEEP_ALIVE": "24h",
               "OLLAMA_NUM_PARALLEL": str(opts.parallel)}
        if opts.flash_attn == "on":
            env["OLLAMA_FLASH_ATTENTION"] = "1"
        if opts.cache_type_k in ("q8_0", "q4_0"):
            env["OLLAMA_KV_CACHE_TYPE"] = opts.cache_type_k
        notes = []
        # 이미 Ollama에 있는 모델(blob)이면 그 이름을 그대로 쓴다
        name = None
        for d in ollama_dirs():
            for m in scan_ollama(d):
                if str(m.path) == str(model):
                    name = m.name
        is_gguf = str(model).lower().endswith(".gguf")
        create = name is None and is_gguf
        if name is None:
            name = model_name_for(model) if is_gguf else model
        if create:
            notes.append("Ollama는 GGUF 파일을 자기 저장소로 복사해서 등록해요. 처음 한 번은 시간이 걸리고 "
                         "모델 크기만큼 디스크 공간이 더 필요해요.")
        if opts.api_key:
            notes.append("Ollama는 API 키를 지원하지 않아요. 같은 네트워크 공유 시 주의하세요.")
        if opts.mmproj:
            notes.append("Ollama로 GGUF를 등록할 때는 비전(mmproj) 파일을 함께 쓰지 않아요.")
        base = f"http://{'127.0.0.1' if opts.host == '0.0.0.0' else opts.host}:{opts.port}"

        def prepare(log: Callable[[str], None]) -> None:
            run_env = {**__import__("os").environ, "OLLAMA_HOST": host}
            exists = subprocess.run([exe, "show", name], capture_output=True, env=run_env, timeout=30).returncode == 0
            if not exists:
                if not create:
                    raise RuntimeError(f"Ollama에 '{name}' 모델이 없어요.")
                log(f"Ollama에 모델 등록 중: {name} (큰 모델은 몇 분 걸려요)")
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    mf = Path(tmp) / "Modelfile"
                    mf.write_text(f'FROM "{Path(model).resolve()}"\nPARAMETER num_ctx {opts.ctx_size}\n', encoding="utf-8")
                    proc = subprocess.run([exe, "create", name, "-f", str(mf)], capture_output=True, text=True,
                                          env=run_env, timeout=3600, encoding="utf-8", errors="replace")
                for line in (proc.stdout + proc.stderr).splitlines()[-5:]:
                    if line.strip():
                        log(line.strip())
                if proc.returncode != 0:
                    raise RuntimeError("Ollama에 모델을 등록하지 못했어요. 이 모델 구조를 Ollama가 아직 지원하지 않을 수 있어요.")
            log("모델을 메모리에 올리는 중…")
            r = requests.post(f"{base}/api/generate", json={"model": name, "prompt": "", "keep_alive": "24h"},
                              timeout=1800)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama가 모델을 불러오지 못했어요: {r.text[:200]}")

        return ServeSpec(program=exe, args=["serve"], host=opts.host, port=opts.port, health_path="/api/version",
                         env=env, model_name=name, prepare=prepare, notes=notes)
