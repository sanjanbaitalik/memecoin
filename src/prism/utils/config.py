from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from prism.utils.paths import find_project_root


def load_yaml(relative_path: str) -> dict[str, Any]:
    root = find_project_root()
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
