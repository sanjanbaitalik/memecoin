from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from prism.data.workbook import (
    OPTIONAL_SOCIAL_COLUMNS,
    REQUIRED_BASE_COLUMNS,
    coerce_datetime_with_diagnostics,
    iter_token_frames,
)
from prism.utils.paths import output_path


@dataclass
class UniverseRule:
    min_rows: int = 30
    min_price_coverage: float = 0.7
    require_any_sentiment: bool = False


def build_token_universe_manifest(workbook_path, rules: UniverseRule) -> pd.DataFrame:
    rows: list[dict] = []
    exclusion_rows: list[dict] = []

    for sheet, frame in iter_token_frames(workbook_path):
        lower_cols = set(frame.columns)
        reasons: list[str] = []
        diagnostics: list[dict] = []

        missing_required = [c for c in REQUIRED_BASE_COLUMNS if c not in lower_cols]
        missing_required_flag = len(missing_required) > 0
        if missing_required_flag:
            reasons.append(f"missing_required_columns:{'|'.join(missing_required)}")

        row_count = len(frame)
        too_few_rows_flag = row_count < rules.min_rows
        if too_few_rows_flag:
            reasons.append(f"too_few_rows<{rules.min_rows}")

        dt = coerce_datetime_with_diagnostics(frame, sheet_name=sheet, diagnostics=diagnostics)
        first_date = dt.min()
        last_date = dt.max()
        valid_date_count = int(dt.notna().sum())
        invalid_date_flag = valid_date_count == 0
        if invalid_date_flag:
            reasons.append("invalid_date")

        price_cov = float(frame["price"].notna().mean()) if "price" in frame.columns and row_count else 0.0
        low_price_coverage_flag = price_cov < rules.min_price_coverage
        if low_price_coverage_flag:
            reasons.append(f"low_price_coverage<{rules.min_price_coverage}")

        sentiment_coverage = {
            col: (float(frame[col].notna().mean()) if col in frame.columns and row_count else 0.0)
            for col in OPTIONAL_SOCIAL_COLUMNS
        }

        no_sentiment_flag = rules.require_any_sentiment and all(v <= 0 for v in sentiment_coverage.values())
        if no_sentiment_flag:
            reasons.append("no_sentiment_coverage")

        token = str(frame["token"].dropna().iloc[0]) if "token" in frame.columns and not frame["token"].dropna().empty else sheet
        ticker = str(frame["ticker"].dropna().iloc[0]) if "ticker" in frame.columns and not frame["ticker"].dropna().empty else ""

        platform = ""
        if "platform" in frame.columns and not frame["platform"].dropna().empty:
            platform = str(frame["platform"].dropna().iloc[0])
        elif "chains" in frame.columns and not frame["chains"].dropna().empty:
            platform = str(frame["chains"].dropna().iloc[0])

        other_reason_flag = False
        include = len(reasons) == 0
        rows.append(
            {
                "sheet_name": sheet,
                "token": token,
                "ticker": ticker,
                "platform_or_chain": platform,
                "row_count": row_count,
                "first_date": first_date.date().isoformat() if pd.notna(first_date) else "",
                "last_date": last_date.date().isoformat() if pd.notna(last_date) else "",
                "valid_date_count": valid_date_count,
                "price_coverage": price_cov,
                "twitter_sentiment_coverage": sentiment_coverage.get("twitter_sentiments", 0.0),
                "telegram_sentiment_coverage": sentiment_coverage.get("telegram_sentiment", 0.0),
                "reddit_sentiment_coverage": sentiment_coverage.get("reddit_sentiment", 0.0),
                "inclusion_status": "included" if include else "excluded",
                "exclusion_reason": "" if include else ";".join(reasons),
            }
        )

        exclusion_rows.append(
            {
                "sheet_name": sheet,
                "too_few_rows": too_few_rows_flag,
                "low_price_coverage": low_price_coverage_flag,
                "invalid_date": invalid_date_flag,
                "missing_required_columns": missing_required_flag,
                "other_reason": other_reason_flag,
                "final_included": include,
                "reason": "" if include else ";".join(reasons),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["inclusion_status", "token", "sheet_name"])
    exclusion_log = pd.DataFrame(exclusion_rows).sort_values(["final_included", "sheet_name"])

    manifest.to_csv(output_path("outputs", "manifests", "token_universe_manifest.csv"), index=False)
    manifest.to_csv(output_path("data", "manifests", "token_universe_manifest.csv"), index=False)
    exclusion_log.to_csv(output_path("outputs", "manifests", "token_universe_exclusion_log.csv"), index=False)
    return manifest
