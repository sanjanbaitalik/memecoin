from __future__ import annotations

import pandas as pd


def chronological_split_index(datetime_series: pd.Series, train_ratio: float = 0.8) -> int:
    if datetime_series.empty:
        return 0

    sorted_idx = datetime_series.sort_values(kind="mergesort").index
    split_pos = int(len(sorted_idx) * train_ratio)
    train_indices = sorted_idx[:split_pos]
    if len(train_indices) == 0:
        return 0
    return int(train_indices.max()) + 1
