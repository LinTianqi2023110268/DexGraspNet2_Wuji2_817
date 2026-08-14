"""One source of truth for local project and large-data paths."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config/project.json"


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported project schema: {CONFIG_PATH}")
    configured_root = Path(config["project_root"]).expanduser()
    if not configured_root.is_absolute():
        configured_root = PROJECT_ROOT / configured_root
    if configured_root.resolve() != PROJECT_ROOT:
        raise RuntimeError("config/project.json project_root does not match this project")
    return config


def source_path(name: str, *, must_exist: bool = True) -> Path:
    value = Path(load_project_config()["sources"][name]).expanduser()
    path = value if value.is_absolute() else PROJECT_ROOT / value
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Missing configured source {name}: {path}")
    return path


def project_path(value: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a project-root-relative path independently of process cwd."""

    candidate = Path(value).expanduser()
    path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def output_path(name: str) -> Path:
    value = Path(load_project_config()["outputs"][name])
    path = value if value.is_absolute() else PROJECT_ROOT / value
    return path.resolve()
