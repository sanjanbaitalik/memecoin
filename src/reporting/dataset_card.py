from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


MISSING = "Not available in current workbook metadata"


def _field(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def generate_dataset_card(workbook_path: Path, manifest: pd.DataFrame, processed: pd.DataFrame, output_dir: Path | None = None) -> dict[str, object]:
    outdir = output_dir or scie_path("outputs", "scie_revision", "reports")
    token_counts = processed.groupby("sheet_name", as_index=False).agg(rows=("sheet_name", "size"))
    candidate_tokens = int(manifest["sheet_name"].nunique()) if not manifest.empty else 0
    eligible_tokens = int((manifest["inclusion_status"] == "included").sum()) if not manifest.empty else 0
    modeled_tokens = int(processed["sheet_name"].nunique()) if not processed.empty else 0
    rows = [
        ("source_file_name", workbook_path.name),
        ("data_source_name_or_provider", MISSING),
        ("price_source", MISSING),
        ("market_cap_source", MISSING),
        ("volume_source", MISSING),
        ("sentiment_source", MISSING),
        ("social_platform_source", _field("twitter/x, reddit, telegram")),
        ("date_range_start", str(processed["datetime"].min()) if not processed.empty else MISSING),
        ("date_range_end", str(processed["datetime"].max()) if not processed.empty else MISSING),
        ("sampling_frequency", _field("daily")),
        ("number_of_candidate_token_sheets", candidate_tokens),
        ("number_of_eligible_sheets", eligible_tokens),
        ("number_of_selected_tokens", modeled_tokens),
        ("number_of_evaluated_tokens", modeled_tokens),
        ("processed_aligned_observations", int(len(processed))),
        ("mean_token_history_length", float(token_counts["rows"].mean()) if not token_counts.empty else 0.0),
        ("median_token_history_length", float(token_counts["rows"].median()) if not token_counts.empty else 0.0),
        ("missing_value_policy", _field("train-only scaling with zero-fill sentiment imputation and row-wise exclusion of invalid targets")),
        ("target_construction", _field("3-day-ahead closing price target")),
        ("train_validation_test_split_policy", _field("chronological split with validation subset from the training period only")),
        ("token_selection_rule", _field("eligible included tokens ranked by row count and capped at the modeling maximum")),
        ("possible_survivorship_bias_risk", _field("yes")),
        ("limitations_of_using_top_tokens_by_row_count", _field("may overrepresent longer-lived, more liquid tokens and underrepresent short-lived assets")),
    ]
    table = pd.DataFrame(rows, columns=["field", "value"])
    save_table_bundle(table, scie_path("outputs", "scie_revision", "tables", "table_dataset_transparency"), "Dataset Transparency Table", "Dataset transparency and provenance summary.")
    md = "\n".join(["# Dataset Card", "", *[f"- {field}: {value}" for field, value in rows]])
    (scie_path("outputs", "scie_revision", "reports", "dataset_card.md")).write_text(md, encoding="utf-8")
    payload = {"workbook": str(workbook_path), "rows": rows}
    (scie_path("outputs", "scie_revision", "reports", "dataset_card.json")).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_author_action_file(workbook_path)
    return payload


def _write_author_action_file(workbook_path: Path) -> None:
    content = "\n".join([
        "# AUTHOR ACTION REQUIRED: Dataset Metadata",
        "",
        "The following fields are not available in the current workbook metadata and must be filled manually by the author before final manuscript submission.",
        "",
        "## Required Fields",
        "",
        "1. **Price source**: What is the upstream provider for closing price data? (e.g., CoinGecko, CoinMarketCap, Binance API, manual collection)",
        "2. **Volume source**: What is the upstream provider for trading volume data?",
        "3. **Market-cap source**: What is the upstream provider for market capitalization data?",
        "4. **Sentiment source**: What is the upstream provider for sentiment scores? (e.g., LunarCrush, Santiment, custom NLP pipeline)",
        "5. **Social platform source**: Which social platforms were scraped/monitored? (e.g., Twitter/X, Reddit, Telegram, Discord)",
        "6. **Data collection period confirmation**: Confirm the exact start and end dates of data collection.",
        "7. **Sampling frequency confirmation**: Confirm that data is sampled daily at a consistent time (e.g., UTC midnight close).",
        "8. **Missing-value policy**: Describe how missing values were handled beyond the current zero-fill approach.",
        "9. **Survivorship-bias notes**: Describe any steps taken to mitigate survivorship bias beyond the current transparent reporting.",
        "",
        "## Current Workbook",
        "",
        f"- File: {workbook_path.name}",
        f"- Path: {workbook_path.as_posix()}",
        "",
        "## Instructions",
        "",
        "Fill in each field above and update the corresponding entries in `reports/dataset_card.md` and `tables/table_dataset_transparency.csv`.",
        "Do NOT fabricate values. If a field is genuinely unknown, mark it as 'Unknown - author to confirm'.",
    ])
    target = scie_path("outputs", "scie_revision", "reports", "AUTHOR_ACTION_REQUIRED_dataset_metadata.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def dataset_transparency_paragraph() -> str:
    return (
        "The dataset is derived from the workbook input used in this repository, but several provenance fields are not available in the current workbook metadata. "
        "Accordingly, the paper reports the exact workbook file name, token counts, processed observation counts, split policy, target construction, and the token-selection rule, while explicitly marking unavailable source metadata as not available in current workbook metadata. "
        "This framing is transparent about survivorship-bias risk introduced by ranking tokens by row count and about the limitation that the workbook alone does not fully document upstream providers for every variable."
    )
