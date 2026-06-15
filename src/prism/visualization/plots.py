from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from prism.utils.paths import output_path


def save_metric_boxplot(df: pd.DataFrame, model_col: str, metric_col: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df[model_col].dropna().unique().tolist())
    data = [df.loc[df[model_col] == m, metric_col].dropna().to_numpy() for m in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_title(f"{metric_col.upper()} distribution by {model_col}")
    ax.set_xlabel(model_col)
    ax.set_ylabel(metric_col)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_path("outputs", "figures", filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_missingness_heatmap(missingness: pd.DataFrame, filename: str) -> Path:
    pivot = missingness.pivot_table(index="sheet_name", columns="column", values="missing_fraction", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.fillna(1.0).to_numpy(), aspect="auto", cmap="magma")
    ax.set_title("Missingness heatmap by token and column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Token sheet")
    fig.colorbar(im, ax=ax, label="Missing fraction")
    fig.tight_layout()

    path = output_path("outputs", "figures", filename)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
