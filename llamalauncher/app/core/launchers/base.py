"""런처(실행 엔진) 공통 인터페이스. 5단계에서 Ollama, LM Studio 등을 추가한다."""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LauncherStatus:
    installed: bool
    path: Optional[Path] = None
    version: str = ""
    build: Optional[int] = None
    capabilities: set[str] = field(default_factory=set)
    message: str = ""


class Launcher(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    supported_formats: tuple[str, ...] = ()

    @abstractmethod
    def detect(self, custom_path: Optional[str] = None) -> LauncherStatus: ...


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
