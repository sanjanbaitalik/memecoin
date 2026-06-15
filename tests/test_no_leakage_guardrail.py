from __future__ import annotations

import pandas as pd

from prism.evaluation.split import chronological_split_index


def test_train_test_index_is_chronological() -> None:
    dt = pd.to_datetime(["2024-01-05", "2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"])
    series = pd.Series(dt)
    split = chronological_split_index(series, train_ratio=0.6)
    sorted_idx = series.sort_values(kind="mergesort").index
    train_idx = sorted_idx[: int(len(sorted_idx) * 0.6)]
    test_idx = sorted_idx[int(len(sorted_idx) * 0.6) :]
    assert series.loc[train_idx].max() <= series.loc[test_idx].min()
    assert split >= 0
