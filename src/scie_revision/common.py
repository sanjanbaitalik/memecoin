from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from prism.utils.paths import output_path


SEEDS = [11, 17, 23]


def scie_path(*parts: str) -> Path:
    if len(parts) >= 2 and parts[0] == "outputs" and parts[1] == "scie_revision":
        return output_path("outputs", *parts[2:])
    return output_path(*parts)


def ensure_revision_dirs() -> None:
    for part in [
        ("outputs", "scie_revision"),
        ("outputs", "scie_revision", "tables"),
        ("outputs", "scie_revision", "figures"),
        ("outputs", "scie_revision", "reports"),
        ("outputs", "scie_revision", "reports", "manuscript_inserts_round2"),
        ("outputs", "scie_revision", "logs"),
        ("outputs", "scie_revision", "predictions"),
        ("outputs", "scie_revision", "audits"),
        ("outputs", "scie_revision", "configs"),
        ("outputs", "scie_revision", "results"),
    ]:
        output_path(*part)


def _fmt_cell(value: object, float_format: str = ".4f") -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(float(value)):
        return format(float(value), float_format)
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, title: str | None = None, caption: str | None = None, float_format: str = ".4f") -> str:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    if caption:
        lines.append(caption)
        lines.append("")
    if df.empty:
        lines.append("_No rows available._")
        return "\n".join(lines)

    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_cell(row[c], float_format=float_format) for c in cols) + " |")
    return "\n".join(lines)


def dataframe_to_latex(df: pd.DataFrame, caption: str | None = None, label: str | None = None, float_format: str = ".4f") -> str:
    if df.empty:
        return "\\begin{tabular}{l}\\hline\\textit{No rows available}\\\\\\hline\\end{tabular}"
    cols = "l" * len(df.columns)
    header = " & ".join(df.columns)
    row_break = "\\\\"
    rows = [" & ".join(_fmt_cell(row[c], float_format=float_format) for c in df.columns) + f" {row_break}" for _, row in df.iterrows()]
    parts = ["\\begin{table}[htbp]", "\\centering", f"\\begin{{tabular}}{{{cols}}}", "\\hline", f"{header} {row_break}", "\\hline"]
    parts.extend(rows)
    parts.extend(["\\hline", "\\end{tabular}"])
    if caption:
        parts.append(f"\\caption{{{caption}}}")
    if label:
        parts.append(f"\\label{{{label}}}")
    parts.append("\\end{table}")
    return "\n".join(parts)


def save_table_bundle(df: pd.DataFrame, base_path: Path, title: str, caption: str | None = None, float_format: str = ".4f") -> dict[str, Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    md_path = base_path.with_suffix(".md")
    tex_path = base_path.with_suffix(".tex")
    df.to_csv(csv_path, index=False)
    md_path.write_text(dataframe_to_markdown(df, title=title, caption=caption, float_format=float_format), encoding="utf-8")
    tex_path.write_text(dataframe_to_latex(df, caption=caption, label=base_path.stem), encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def chronological_train_val_test_split(frame: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    train_end = max(int(n * train_ratio), 1)
    val_end = max(int(n * (train_ratio + val_ratio)), train_end + 1)
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    if val.empty and not test.empty:
        val = test.iloc[: max(1, len(test) // 2)].copy()
        test = test.iloc[len(val):].copy()
    return train, val, test


def mean_std(series: pd.Series) -> str:
    if series.empty:
        return ""
    mean = float(series.mean())
    std = float(series.std(ddof=0)) if len(series) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stable_mode(values: Iterable[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def pairwise_correlation_mean(frame: pd.DataFrame) -> float:
    if frame.shape[1] < 2:
        return float("nan")
    corr = frame.corr().to_numpy(dtype=float)
    tri = corr[np.triu_indices_from(corr, k=1)]
    tri = tri[np.isfinite(tri)]
    return float(tri.mean()) if tri.size else float("nan")


def make_seeded_order(items: list[str], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    order = list(items)
    rng.shuffle(order)
    return order
