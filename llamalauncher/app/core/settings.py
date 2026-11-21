"""앱 설정 저장/불러오기 (OS별 사용자 설정 폴더의 JSON 파일)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_config_dir

APP_NAME = "LlamaLauncher"


def config_dir() -> Path:
    return Path(user_config_dir(APP_NAME, appauthor=False))


def logs_dir() -> Path:
    return config_dir() / "logs"


def default_models_dir() -> Path:
    return Path.home() / APP_NAME / "models"


@dataclass
class Settings:
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    theme: str = "system"  # system | light | dark
    first_run_done: bool = False
    launcher_paths: dict[str, str] = field(default_factory=dict)
    recent_models: list[str] = field(default_factory=list)
    last_model: str = ""
    server_options: dict = field(default_factory=dict)  # 마지막으로 쓴 ServerOptions
    server_preset: str = "balanced"
    user_presets: dict[str, dict] = field(default_factory=dict)

    MAX_RECENT = 15

    def add_recent_model(self, path: str) -> None:
        self.recent_models = [path] + [p for p in self.recent_models if p != path]
        del self.recent_models[self.MAX_RECENT:]
        self.last_model = path

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or config_dir() / "settings.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: Path | None = None) -> None:
        path = path or config_dir() / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
