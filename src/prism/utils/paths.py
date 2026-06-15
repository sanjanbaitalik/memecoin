from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return project root by locating this file's parent structure."""
    current = start or Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "prompt.md").exists() or (candidate / "scripts").exists():
            return candidate
    return Path.cwd()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(*parts: str) -> Path:
    root = find_project_root()
    parts_list = list(parts)
    outdir_override = os.environ.get("PRISM_OUTDIR", "").strip()
    if outdir_override and parts_list and parts_list[0] == "outputs":
        parts_list[0] = outdir_override

    target = root.joinpath(*parts_list)
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target
