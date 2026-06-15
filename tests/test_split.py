from __future__ import annotations

import pandas as pd

from prism.evaluation.split import chronological_split_index


def test_chronological_split_index_no_shuffle() -> None:
    dt = pd.to_datetime([
        "2024-01-03",
        "2024-01-01",
        "2024-01-02",
        "2024-01-04",
        "2024-01-05",
    ])
    split = chronological_split_index(pd.Series(dt), train_ratio=0.6)
    assert split >= 2
