from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


def run_token_selection_bias(manifest: pd.DataFrame, processed: pd.DataFrame, output_dir: Path | None = None, max_tokens: int = 200) -> pd.DataFrame:
    out = output_dir or scie_path("outputs", "scie_revision", "tables")
    counts = processed.groupby("sheet_name", as_index=False).agg(row_count=("sheet_name", "size"))

    if "selected_for_modeling" in manifest.columns:
        selected_tokens = set(manifest.loc[manifest["selected_for_modeling"] == True, "sheet_name"].astype(str).tolist())
    elif "inclusion_status" in manifest.columns:
        eligible = manifest.loc[manifest["inclusion_status"] == "included", "sheet_name"].astype(str).tolist()
        selected_tokens = set(eligible[:max_tokens])
    else:
        selected_tokens = set(counts.nlargest(max_tokens, "row_count")["sheet_name"].astype(str).tolist())

    counts["selected"] = counts["sheet_name"].astype(str).isin(selected_tokens)
    selected = counts[counts["selected"]].copy()
    non_selected = counts[~counts["selected"]].copy()
    longer_histories = bool(selected["row_count"].median() > non_selected["row_count"].median()) if not selected.empty and not non_selected.empty else False
    dead_short = int((manifest["inclusion_status"].astype(str).eq("excluded") & manifest["exclusion_reason"].astype(str).str.contains("too_few_rows|invalid_date", case=False, na=False)).sum()) if not manifest.empty and "exclusion_reason" in manifest.columns else 0

    rows = [
        ("distribution_of_row_counts_across_candidate_tokens", int(counts["row_count"].count()), float(counts["row_count"].mean()), float(counts["row_count"].median())),
        ("row_counts_of_eligible_tokens", int(manifest[manifest["inclusion_status"] == "included"].shape[0]) if not manifest.empty and "inclusion_status" in manifest.columns else 0, float(selected["row_count"].mean()) if not selected.empty else 0.0, float(selected["row_count"].median()) if not selected.empty else 0.0),
        (f"row_counts_of_selected_{max_tokens}_tokens", int(selected.shape[0]), float(selected["row_count"].mean()) if not selected.empty else 0.0, float(selected["row_count"].median()) if not selected.empty else 0.0),
        ("selected_tokens_have_longer_histories_than_non_selected_tokens", int(longer_histories), selected["row_count"].median() if not selected.empty else 0.0, non_selected["row_count"].median() if not non_selected.empty else 0.0),
        ("missingness_rate_by_token_before_filtering", int(counts.shape[0]), float(processed.isna().mean().mean()) if not processed.empty else 0.0, 0.0),
        ("number_of_dead_or_short_history_tokens_excluded", dead_short, 0.0, 0.0),
        ("survivorship_bias_warning", 1, 0.0, 0.0),
    ]
    table = pd.DataFrame(rows, columns=["analysis_item", "count_or_flag", "mean_or_rate", "median_or_detail"])
    save_table_bundle(table, scie_path("outputs", "scie_revision", "tables", "table_token_selection_bias"), "Token Selection Bias Table", "Token-selection and survivorship-bias analysis.")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts["row_count"].to_numpy(dtype=float), bins=min(25, max(len(counts) // 5, 5)), color="#1f77b4", alpha=0.9)
    ax.set_xlabel("Token history length (rows)")
    ax.set_ylabel("Count of tokens")
    ax.set_title("Token history distribution")
    fig.tight_layout()
    fig.savefig(scie_path("outputs", "scie_revision", "figures", "fig_token_history_distribution.png"), dpi=300)
    plt.close(fig)

    report = "\n".join(
        [
            "# Token Selection Bias Report",
            "",
            f"- Selected tokens have longer histories than non-selected tokens: {longer_histories}",
            f"- Dead/short-history tokens excluded: {dead_short}",
            f"- Number of selected tokens: {len(selected_tokens)}",
            "- Survivorship-bias warning: selecting by row count may favor more stable surviving tokens.",
        ]
    )
    (scie_path("outputs", "scie_revision", "reports", "token_selection_bias_report.md")).write_text(report, encoding="utf-8")
    (scie_path("outputs", "scie_revision", "reports", "token_selection_bias_paragraph.md")).write_text(
        "Selecting the top tokens by row count can overrepresent longer-lived assets and therefore introduce survivorship bias. The revision reports this explicitly, quantifies the row-count distribution, and keeps the limitation visible in the paper-facing summary.",
        encoding="utf-8",
    )
    return table
