from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import os

from config.settings import load_config

cfg = load_config()

def _sandbox_root() -> Path:
    root = Path(cfg.sandbox.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()

def _safe_path(user_path: str) -> Path:
    """
    Convierte una ruta relativa (o nombre de archivo) en una ruta ABSOLUTA dentro del sandbox.
    Bloquea intentos de salir del sandbox.
    """
    root = _sandbox_root()
    p = (root / user_path).resolve()

    if not str(p).startswith(str(root)):
        raise ValueError("Ruta fuera del sandbox. Operación bloqueada.")
    return p

def list_files(relative_dir: str = ".") -> List[str]:
    d = _safe_path(relative_dir)
    if not d.exists():
        raise FileNotFoundError(f"No existe: {relative_dir}")
    if not d.is_dir():
        raise ValueError(f"No es un directorio: {relative_dir}")

    items = []
    for name in os.listdir(d):
        items.append(name)
    return sorted(items)

def read_text(relative_path: str, max_chars: int = 8000) -> str:
    p = _safe_path(relative_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {relative_path}")
    if p.is_dir():
        raise ValueError("No puedo leer un directorio.")

    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCADO]"
    return text

def write_text(relative_path: str, content: str, overwrite: bool = True) -> str:
    p = _safe_path(relative_path)
    if p.exists() and not overwrite:
        raise FileExistsError("El archivo ya existe y overwrite=False.")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: escrito {relative_path} ({len(content)} chars)"
