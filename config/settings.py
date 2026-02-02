from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    text_model: str
    vision_model: str
    temperature: float


@dataclass(frozen=True)
class UIConfig:
    chainlit_port: int


@dataclass(frozen=True)
class SandboxConfig:
    enabled: bool
    root_dir: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    environment: str
    backend: BackendConfig
    ollama: OllamaConfig
    ui: UIConfig
    sandbox: SandboxConfig


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    app = raw["app"]
    backend = raw["backend"]
    ollama = raw["ollama"]
    ui = raw["ui"]
    sandbox = raw["sandbox"]

    return AppConfig(
        name=str(app["name"]),
        environment=str(app["environment"]),
        backend=BackendConfig(host=str(backend["host"]), port=int(backend["port"])),
        ollama=OllamaConfig(
            base_url=str(ollama["base_url"]),
            text_model=str(ollama["text_model"]),
            vision_model=str(ollama["vision_model"]),
            temperature=float(ollama["temperature"]),
        ),
        ui=UIConfig(chainlit_port=int(ui["chainlit_port"])),
        sandbox=SandboxConfig(enabled=bool(sandbox["enabled"]), root_dir=str(sandbox["root_dir"])),
    )
