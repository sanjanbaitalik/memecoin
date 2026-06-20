from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scie_revision.common import save_table_bundle, scie_path


MISSING = "AUTHOR_TO_CONFIRM"


def _field(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def _load_metadata_config(root: Path | None = None) -> dict:
    search_root = root or Path(".")
    filled = search_root / "configs" / "dataset_metadata.yaml"
    template = search_root / "configs" / "dataset_metadata_template.yaml"
    source = filled if filled.exists() else template
    if source.exists():
        with open(source, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def generate_dataset_card(
    workbook_path: Path,
    manifest: pd.DataFrame,
    processed: pd.DataFrame,
    output_dir: Path | None = None,
    max_tokens: int = 200,
) -> dict[str, object]:
    outdir = output_dir or scie_path("outputs", "scie_revision_round3", "reports")

    meta = _load_metadata_config(Path("."))

    token_counts = processed.groupby("sheet_name", as_index=False).agg(rows=("sheet_name", "size"))
    candidate_tokens = int(manifest["sheet_name"].nunique()) if not manifest.empty else 0
    eligible_tokens = int((manifest["inclusion_status"] == "included").sum()) if not manifest.empty and "inclusion_status" in manifest.columns else 0
    selected_tokens = int(processed["sheet_name"].nunique()) if not processed.empty else 0
    evaluated_tokens = selected_tokens

    rows = [
        ("source_file_name", workbook_path.name),
        ("data_source_name_or_provider", _field(meta.get("data_source_name_or_provider", MISSING))),
        ("price_source", _field(meta.get("price_source", MISSING))),
        ("market_cap_source", _field(meta.get("market_cap_source", MISSING))),
        ("volume_source", _field(meta.get("volume_source", MISSING))),
        ("sentiment_source", _field(meta.get("sentiment_source", MISSING))),
        ("social_platform_source", _field(", ".join(meta.get("social_platform_source", ["Twitter/X", "Reddit", "Telegram"])))),
        ("date_range_start", str(processed["datetime"].min()) if not processed.empty else MISSING),
        ("date_range_end", str(processed["datetime"].max()) if not processed.empty else MISSING),
        ("sampling_frequency", _field(meta.get("sampling_frequency", "daily"))),
        ("timezone_or_close_time", _field(meta.get("timezone_or_close_time", MISSING))),
        ("number_of_candidate_token_sheets", candidate_tokens),
        ("number_of_eligible_sheets", eligible_tokens),
        ("number_of_selected_tokens", selected_tokens),
        ("number_of_evaluated_tokens", evaluated_tokens),
        ("max_tokens_configured", max_tokens),
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
    save_table_bundle(table, scie_path("outputs", "scie_revision_round3", "tables", "table_dataset_transparency"), "Dataset Transparency Table", "Dataset transparency and provenance summary.")
    md = "\n".join(["# Dataset Card", "", *[f"- **{field}**: {value}" for field, value in rows]])
    (scie_path("outputs", "scie_revision_round3", "reports", "dataset_card.md")).write_text(md, encoding="utf-8")
    payload = {"workbook": str(workbook_path), "rows": {field: value for field, value in rows}}
    (scie_path("outputs", "scie_revision_round3", "reports", "dataset_card.json")).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_author_action_file(workbook_path)
    return payload


def _write_author_action_file(workbook_path: Path) -> None:
    content = "\n".join([
        "# AUTHOR ACTION REQUIRED: Dataset Metadata",
        "",
        "The following fields are not available in the current workbook metadata and must be filled manually by the author before final manuscript submission.",
        "",
        "## Instructions",
        "",
        "1. Copy `configs/dataset_metadata_template.yaml` to `configs/dataset_metadata.yaml`.",
        "2. Fill in each field marked `AUTHOR_TO_CONFIRM`.",
        "3. Re-run the pipeline. The paper-facing outputs will use the filled values.",
        "",
        "## Required Fields",
        "",
        "1. **data_source_name_or_provider**: The upstream data provider or aggregation service.",
        "2. **price_source**: The upstream provider for closing price data (e.g., CoinGecko, CoinMarketCap, Binance API).",
        "3. **volume_source**: The upstream provider for trading volume data.",
        "4. **market_cap_source**: The upstream provider for market capitalization data.",
        "5. **sentiment_source**: The upstream provider for sentiment scores (e.g., LunarCrush, Santiment, custom NLP pipeline).",
        "6. **timezone_or_close_time**: The timezone or time of day for daily price closes.",
        "",
        "## Current Workbook",
        "",
        f"- File: {workbook_path.name}",
        f"- Path: {workbook_path.as_posix()}",
        "",
        "## Important",
        "",
        "Do NOT fabricate values. If a field is genuinely unknown, mark it as `AUTHOR_TO_CONFIRM`.",
    ])
    target = scie_path("outputs", "scie_revision_round3", "reports", "AUTHOR_ACTION_REQUIRED_dataset_metadata.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def dataset_transparency_paragraph() -> str:
    return (
        "The dataset is derived from the workbook input used in this repository, but several provenance fields are not available in the current workbook metadata. "
        "Accordingly, the paper reports the exact workbook file name, token counts, processed observation counts, split policy, target construction, and the token-selection rule, "
        "while explicitly marking unavailable source metadata as `AUTHOR_TO_CONFIRM`. "
        "This framing is transparent about survivorship-bias risk introduced by ranking tokens by row count and about the limitation that the workbook alone does not fully document upstream providers for every variable."
    )
