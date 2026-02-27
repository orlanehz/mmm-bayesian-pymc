"""Gestion de configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    raw_data: Path
    processed_data: Path
    artifacts: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict[str, Any], project_root: str | Path) -> ProjectPaths:
    root = Path(project_root)
    paths = cfg["paths"]
    return ProjectPaths(
        raw_data=root / paths["raw_data"],
        processed_data=root / paths["processed_data"],
        artifacts=root / paths["artifacts"],
    )