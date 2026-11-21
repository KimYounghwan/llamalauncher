"""런처 목록."""

from __future__ import annotations

from .base import Launcher, LauncherStatus
from .llamacpp import LlamaCppLauncher
from .ollama import OllamaLauncher
from .others import (
    JanLauncher, KoboldCppLauncher, LMStudioLauncher, MlxLauncher, TextGenWebUILauncher, VllmLauncher,
)

FORMAT_LABELS = {
    "gguf": "GGUF",
    "mlx": "MLX (Apple Silicon)",
    "safetensors": "safetensors (vLLM)",
    "ollama": "Ollama 라이브러리",
}


def all_launchers() -> list[Launcher]:
    return [LlamaCppLauncher(), OllamaLauncher(), LMStudioLauncher(), KoboldCppLauncher(), MlxLauncher(),
            VllmLauncher(), JanLauncher(), TextGenWebUILauncher()]


def by_id(launcher_id: str) -> Launcher:
    for l in all_launchers():
        if l.id == launcher_id:
            return l
    raise KeyError(launcher_id)


__all__ = ["Launcher", "LauncherStatus", "all_launchers", "by_id", "FORMAT_LABELS"]
