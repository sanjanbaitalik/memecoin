from __future__ import annotations

import numpy as np
import pandas as pd

from scie_revision.common import pairwise_correlation_mean


def diversification_summary(full_frame: pd.DataFrame, selected_frame: pd.DataFrame, token_col: str = "sheet_name") -> dict[str, float]:
    full_tokens = sorted(full_frame[token_col].astype(str).unique().tolist())
    selected_tokens = sorted(selected_frame[token_col].astype(str).unique().tolist())
    full_pivot = full_frame.pivot_table(index="datetime", columns=token_col, values="price", aggfunc="last").sort_index()
    selected_pivot = selected_frame.pivot_table(index="datetime", columns=token_col, values="price", aggfunc="last").sort_index()
    full_corr = pairwise_correlation_mean(full_pivot.fillna(method="ffill").fillna(method="bfill"))
    selected_corr = pairwise_correlation_mean(selected_pivot.fillna(method="ffill").fillna(method="bfill"))
    selected_edge_count = int(selected_pivot.shape[1] * (selected_pivot.shape[1] - 1) / 2)
    full_edge_count = int(full_pivot.shape[1] * (full_pivot.shape[1] - 1) / 2)
    selected_avg_degree = float(2 * selected_edge_count / max(len(selected_tokens), 1))
    full_avg_degree = float(2 * full_edge_count / max(len(full_tokens), 1))
    redundancy_reduction = float(1.0 - (selected_corr / full_corr)) if np.isfinite(full_corr) and full_corr not in (0, np.nan) else 0.0
    return {
        "avg_pairwise_corr_full": float(full_corr),
        "avg_pairwise_corr_selected": float(selected_corr),
        "redundancy_reduction_pct": redundancy_reduction * 100.0,
        "selected_token_edge_count": selected_edge_count,
        "selected_token_avg_degree": selected_avg_degree,
        "full_universe_avg_degree": full_avg_degree,
        "selected_token_count": len(selected_tokens),
        "full_token_count": len(full_tokens),
    }
