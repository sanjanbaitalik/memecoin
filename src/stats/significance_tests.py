from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * (pvals[idx] if np.isfinite(pvals[idx]) else 1.0)
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


def paired_wilcoxon(frame: pd.DataFrame, group_col: str, token_col: str, metric_col: str, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict] = []
    pvals: list[float] = []
    for left, right in comparisons:
        l = frame.loc[frame[group_col] == left, [token_col, metric_col]].rename(columns={metric_col: "left"})
        r = frame.loc[frame[group_col] == right, [token_col, metric_col]].rename(columns={metric_col: "right"})
        merged = l.merge(r, on=token_col, how="inner").dropna()
        if merged.empty:
            stat = np.nan
            p = np.nan
            effect = np.nan
            n = 0
            left_mean = np.nan
            right_mean = np.nan
        else:
            stat, p = wilcoxon(merged["left"].to_numpy(), merged["right"].to_numpy(), zero_method="wilcox")
            effect = rank_biserial_effect(merged["left"].to_numpy(), merged["right"].to_numpy())
            n = len(merged)
            left_mean = float(merged["left"].mean())
            right_mean = float(merged["right"].mean())
        pvals.append(1.0 if not np.isfinite(p) else float(p))
        rows.append(
            {
                "left": left,
                "right": right,
                "metric": metric_col,
                "n_pairs": n,
                "wilcoxon_stat": stat,
                "p_value": p,
                "effect_rank_biserial": effect,
                "left_mean": left_mean,
                "right_mean": right_mean,
                "favored_model": "left_better" if (np.isfinite(left_mean) and np.isfinite(right_mean) and left_mean < right_mean) else "right_better",
            }
        )
    adjusted = holm_correction(pvals) if rows else []
    for row, adj in zip(rows, adjusted):
        row["p_value_holm"] = adj
        row["significant_0_05"] = bool(adj < 0.05)
    return pd.DataFrame(rows)


def significance_vs_reference(frame: pd.DataFrame, token_col: str, metric_col: str, reference_model: str, model_col: str = "model") -> pd.DataFrame:
    models = sorted([m for m in frame[model_col].astype(str).unique().tolist() if m != reference_model])
    return paired_wilcoxon(frame.rename(columns={model_col: "model"}), group_col="model", token_col=token_col, metric_col=metric_col, comparisons=[(reference_model, m) for m in models])
