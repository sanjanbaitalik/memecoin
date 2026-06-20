from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


def run_token_selection_bias(
    manifest: pd.DataFrame,
    processed: pd.DataFrame,
    output_dir: Path | None = None,
    max_tokens: int = 200,
    n_candidate_sheets: int = 0,
    n_valid_schema_sheets: int = 0,
    n_date_valid_sheets: int = 0,
) -> pd.DataFrame:
    out = output_dir or scie_path("outputs", "scie_revision_round3", "tables")

    if "selected_for_modeling" in manifest.columns:
        selected_tokens = set(manifest.loc[manifest["selected_for_modeling"] == True, "sheet_name"].astype(str).tolist())
    elif "inclusion_status" in manifest.columns:
        eligible = manifest.loc[manifest["inclusion_status"] == "included", "sheet_name"].astype(str).tolist()
        selected_tokens = set(eligible[:max_tokens])
    else:
        counts_all = processed.groupby("sheet_name", as_index=False).agg(row_count=("sheet_name", "size"))
        selected_tokens = set(counts_all.nlargest(max_tokens, "row_count")["sheet_name"].astype(str).tolist())

    eligible_tokens = set()
    if "inclusion_status" in manifest.columns:
        eligible_tokens = set(manifest.loc[manifest["inclusion_status"] == "included", "sheet_name"].astype(str).tolist())

    all_counts = processed.groupby("sheet_name", as_index=False).agg(row_count=("sheet_name", "size"))

    selected_counts = all_counts[all_counts["sheet_name"].astype(str).isin(selected_tokens)].copy()
    non_selected_eligible_counts = all_counts[all_counts["sheet_name"].astype(str).isin(eligible_tokens - selected_tokens)].copy()

    longer_histories = bool(selected_counts["row_count"].median() > non_selected_eligible_counts["row_count"].median()) if not selected_counts.empty and not non_selected_eligible_counts.empty else False
    dead_short = int((manifest["inclusion_status"].astype(str).eq("excluded")).sum()) if not manifest.empty and "inclusion_status" in manifest.columns else 0

    n_eligible = len(eligible_tokens)
    n_selected = len(selected_tokens)
    n_evaluated = n_selected

    rows = [
        ("candidate_token_sheets", n_candidate_sheets, 0.0, 0.0),
        ("valid_schema_sheets", n_valid_schema_sheets, 0.0, 0.0),
        ("date_parse_valid_sheets", n_date_valid_sheets, 0.0, 0.0),
        ("eligible_sheets", n_eligible, 0.0, 0.0),
        ("selected_modelling_tokens", n_selected, 0.0, 0.0),
        ("evaluated_tokens", n_evaluated, 0.0, 0.0),
        ("dead_short_history_invalid_date_excluded", dead_short, 0.0, 0.0),
        ("mean_row_count_eligible_non_selected", 0 if non_selected_eligible_counts.empty else float(non_selected_eligible_counts["row_count"].mean()), 0.0, 0.0),
        ("mean_row_count_selected", 0 if selected_counts.empty else float(selected_counts["row_count"].mean()), 0.0, 0.0),
        ("median_row_count_eligible_non_selected", 0 if non_selected_eligible_counts.empty else float(non_selected_eligible_counts["row_count"].median()), 0.0, 0.0),
        ("median_row_count_selected", 0 if selected_counts.empty else float(selected_counts["row_count"].median()), 0.0, 0.0),
        ("selected_tokens_have_longer_histories", int(longer_histories), 0.0, 0.0),
        ("missingness_rate_before_filtering", 0.0, float(processed.isna().mean().mean()) if not processed.empty else 0.0, 0.0),
        ("survivorship_bias_warning", 1, 0.0, 0.0),
    ]
    table = pd.DataFrame(rows, columns=["analysis_item", "count_or_flag", "mean_or_rate", "median_or_detail"])
    save_table_bundle(table, scie_path("outputs", "scie_revision_round3", "tables", "table_token_selection_bias"), "Token Selection Bias Table", "Token-selection and survivorship-bias analysis.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not all_counts.empty:
        axes[0].hist(all_counts["row_count"].to_numpy(dtype=float), bins=min(25, max(len(all_counts) // 5, 5)), color="#1f77b4", alpha=0.9)
        axes[0].set_xlabel("Token history length (rows)")
        axes[0].set_ylabel("Count of tokens")
        axes[0].set_title("All eligible tokens")

    if not selected_counts.empty:
        axes[1].hist(selected_counts["row_count"].to_numpy(dtype=float), bins=min(25, max(len(selected_counts) // 5, 5)), color="#2ca02c", alpha=0.9, label="Selected")
    if not non_selected_eligible_counts.empty:
        axes[1].hist(non_selected_eligible_counts["row_count"].to_numpy(dtype=float), bins=min(25, max(len(non_selected_eligible_counts) // 5, 5)), color="#d62728", alpha=0.7, label="Non-selected eligible")
    axes[1].set_xlabel("Token history length (rows)")
    axes[1].set_ylabel("Count of tokens")
    axes[1].set_title("Selected vs non-selected eligible tokens")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(scie_path("outputs", "scie_revision_round3", "figures", "fig_token_history_distribution.png"), dpi=300)
    plt.close(fig)

    report = "\n".join(
        [
            "# Token Selection Bias Report",
            "",
            f"- Candidate sheets: {n_candidate_sheets}",
            f"- Valid schema sheets: {n_valid_schema_sheets}",
            f"- Date-parse valid sheets: {n_date_valid_sheets}",
            f"- Eligible sheets: {n_eligible}",
            f"- Selected modelling tokens: {n_selected}",
            f"- Evaluated tokens: {n_evaluated}",
            f"- Dead/short-history/invalid-date excluded: {dead_short}",
            f"- Selected tokens have longer histories than non-selected eligible tokens: {longer_histories}",
            "- Survivorship-bias warning: selecting by row count may favor more stable surviving tokens.",
            "",
            "## Limitation",
            "",
            "Selecting the top tokens by row count can overrepresent longer-lived assets and therefore introduce survivorship bias. "
            "The revision reports this explicitly, quantifies the row-count distribution, and keeps the limitation visible in the paper-facing summary.",
        ]
    )
    (scie_path("outputs", "scie_revision_round3", "reports", "token_selection_bias_report.md")).write_text(report, encoding="utf-8")
    return table
