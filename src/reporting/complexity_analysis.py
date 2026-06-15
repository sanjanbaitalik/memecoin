from __future__ import annotations

from pathlib import Path

import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


def generate_complexity_analysis(output_dir: Path | None = None) -> pd.DataFrame:
    rows = [
        ("Feature preparation", "O(NTF)", "Construct lagged and sentiment-aware features for each token over time."),
        ("Graph construction", "O(N^2 T)", "Pairwise token similarity across the candidate universe."),
        ("Pairwise similarity calculation", "O(N^2 T)", "Correlation and trajectory comparisons across all token pairs."),
        ("Greedy MIS selection", "O(N log N + E)", "Sort nodes by degree and suppress adjacent tokens."),
        ("LSTM training", "O(SNTLH)", "Sequence model training across seeds, tokens, lookback length, horizon, and hidden size."),
        ("MAML inner loop", "O(SNTLKH)", "Token-level adaptation across K inner steps."),
        ("MAML outer loop", "O(SNTLBH)", "Meta-update over batches and tokens."),
        ("Full PRISM training/inference", "O(S(N^2T + NTLH + E))", "End-to-end graph, adaptation, and forecasting workflow."),
    ]
    table = pd.DataFrame(rows, columns=["component", "asymptotic_complexity", "notes"])
    save_table_bundle(table, scie_path("outputs", "scie_revision", "tables", "table_complexity_analysis"), "Complexity Analysis", "Asymptotic complexity for the revision pipeline.")
    (scie_path("outputs", "scie_revision", "reports", "complexity_analysis.md")).write_text(
        "\n".join(["# Complexity Analysis", "", *[f"- {r[0]}: {r[1]} ({r[2]})" for r in rows]]),
        encoding="utf-8",
    )
    return table
