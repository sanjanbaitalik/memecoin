from __future__ import annotations

import pandas as pd

from .common import BaselineRunResult, run_persistence


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], seed: int) -> BaselineRunResult:
    return run_persistence(train, test)
