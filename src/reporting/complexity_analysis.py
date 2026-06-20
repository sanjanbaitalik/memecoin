from __future__ import annotations

from pathlib import Path

import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


def generate_complexity_analysis(output_dir: Path | None = None) -> pd.DataFrame:
    rows = [
        ("Data preprocessing", "O(NTF)", "Construct lagged and sentiment-aware features for each token over time."),
        ("Graph construction", "O(N^2 T)", "Pairwise token similarity across the candidate universe."),
        ("Pairwise similarity calculation", "O(N^2 T)", "Correlation and trajectory comparisons across all token pairs."),
        ("Greedy MIS selection", "O(N log N + E)", "Sort nodes by degree and suppress adjacent tokens."),
        ("LSTM training (baselines)", "O(SNTLH)", "Sequence model training across seeds, tokens, lookback, horizon, and hidden size."),
        ("First-order MAML meta-training", "O(SNTLKI)", "Token-level meta-learning with K inner steps, I outer epochs, across S seeds, N tokens, T timesteps, L lookback, H hidden size."),
        ("First-order MAML adaptation", "O(NTKI)", "Per-token inner-loop adaptation at test time."),
        ("MAML inference", "O(NTH)", "Adapted model forward pass for each test token."),
        ("Full PRISM training/inference", "O(S(N^2T + NTLKI + E))", "End-to-end graph, meta-learning, and forecasting workflow."),
        ("Statistical testing", "O(NB)", "Paired Wilcoxon tests with Holm correction across B baseline comparisons and N token pairs."),
    ]
    table = pd.DataFrame(rows, columns=["component", "asymptotic_complexity", "notes"])
    save_table_bundle(table, scie_path("outputs", "scie_revision_round3", "tables", "table_complexity_analysis"), "Complexity Analysis", "Asymptotic complexity for the revision pipeline.")
    (scie_path("outputs", "scie_revision_round3", "reports", "complexity_analysis.md")).write_text(
        "\n".join(["# Complexity Analysis", "", *[f"- **{r[0]}**: {r[1]} ({r[2]})" for r in rows]]),
        encoding="utf-8",
    )
    return table
