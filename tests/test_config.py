from __future__ import annotations

from prism.utils.config import load_yaml


def test_load_yaml() -> None:
    cfg = load_yaml("configs/data/universe.yaml")
    assert "min_rows" in cfg
