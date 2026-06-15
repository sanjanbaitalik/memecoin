from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


LOWER_IS_BETTER = {"mae", "rmse", "mape", "smape"}
HIGHER_IS_BETTER = {"directional_accuracy", "roi_proxy"}


def metric_direction(metric: str) -> str:
    m = metric.lower()
    if m in LOWER_IS_BETTER:
        return "lower_better"
    if m in HIGHER_IS_BETTER:
        return "higher_better"
    raise ValueError(f"Unknown metric direction for: {metric}")


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def rank_biserial_effect(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = rankdata(np.abs(nz))
    pos = np.sum(ranks[nz > 0])
    neg = np.sum(ranks[nz < 0])
    denom = pos + neg
    return float((pos - neg) / denom) if denom else 0.0


def paired_wilcoxon_table(
    frame: pd.DataFrame,
    model_col: str,
    token_col: str,
    metric: str,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict] = []
    pvals: list[float] = []

    for left, right in comparisons:
        l = frame.loc[frame[model_col] == left, [token_col, metric]].rename(columns={metric: "left"})
        r = frame.loc[frame[model_col] == right, [token_col, metric]].rename(columns={metric: "right"})
        merged = l.merge(r, on=token_col, how="inner").dropna()

        if merged.empty:
            p = np.nan
            stat = np.nan
            effect = np.nan
            n = 0
            left_mean = np.nan
            right_mean = np.nan
            favored = "insufficient_pairs"
        else:
            stat, p = wilcoxon(merged["left"].to_numpy(), merged["right"].to_numpy(), zero_method="wilcox")
            effect = rank_biserial_effect(merged["left"].to_numpy(), merged["right"].to_numpy())
            n = len(merged)
            left_mean = float(merged["left"].mean())
            right_mean = float(merged["right"].mean())

            direction = metric_direction(metric)
            if direction == "lower_better":
                favored = "left_better" if left_mean < right_mean else "right_better"
            else:
                favored = "left_better" if left_mean > right_mean else "right_better"

        pvals.append(p if np.isfinite(p) else 1.0)
        rows.append(
            {
                "left": left,
                "right": right,
                "metric": metric,
                "n_pairs": n,
                "wilcoxon_stat": stat,
                "p_value": p,
                "effect_rank_biserial": effect,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "metric_direction": metric_direction(metric),
                "favored_model": favored,
            }
        )

    if rows:
        adjusted = holm_correction(pvals)
        for row, adj in zip(rows, adjusted):
            row["p_value_holm"] = adj
            row["significant_0_05"] = bool(adj < 0.05)

    return pd.DataFrame(rows)
