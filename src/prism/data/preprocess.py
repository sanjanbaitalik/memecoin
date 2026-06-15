from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prism.data.workbook import coerce_datetime_with_diagnostics, iter_token_frames
from prism.utils.paths import output_path


@dataclass
class PreprocessConfig:
    forecast_horizon_days: int = 3
    lookback_days: int = 14
    train_ratio: float = 0.8
    sentiment_mode: str = "raw"


def _aggregate_sentiment(series: pd.Series, mode: str) -> pd.Series:
    if mode == "raw":
        return series
    if mode == "rolling_mean":
        return series.rolling(7, min_periods=1).mean()
    if mode == "rolling_median":
        return series.rolling(7, min_periods=1).median()
    return series


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def preprocess_dataset(
    workbook_path,
    manifest: pd.DataFrame,
    config: PreprocessConfig,
) -> pd.DataFrame:
    included_sheets = manifest.loc[manifest["inclusion_status"] == "included", "sheet_name"].tolist()
    processed_frames: list[pd.DataFrame] = []
    imputation_log: list[dict] = []
    diagnostics: list[dict] = []
    token_row_counts: list[dict] = []

    for sheet, frame in iter_token_frames(workbook_path, sheets=included_sheets):
        frame = frame.copy()
        frame["datetime"] = coerce_datetime_with_diagnostics(frame, sheet_name=sheet, diagnostics=diagnostics)
        before_rows = len(frame)
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        frame = frame.drop_duplicates(subset=["datetime"], keep="last")

        for numeric_col in ["price", "volume", "twitter_sentiments", "reddit_sentiment", "telegram_sentiment"]:
            if numeric_col in frame.columns:
                frame[numeric_col] = _safe_numeric(frame[numeric_col])

        if "price" not in frame.columns:
            continue

        frame = frame[frame["price"] > 0]
        if "volume" in frame.columns:
            frame.loc[frame["volume"] < 0, "volume"] = np.nan

        for sentiment_col in ["twitter_sentiments", "reddit_sentiment", "telegram_sentiment"]:
            if sentiment_col in frame.columns:
                before = int(frame[sentiment_col].isna().sum())
                frame[sentiment_col] = _aggregate_sentiment(frame[sentiment_col], config.sentiment_mode)
                frame[sentiment_col] = frame[sentiment_col].fillna(0.0)
                after = int(frame[sentiment_col].isna().sum())
                imputation_log.append(
                    {
                        "sheet_name": sheet,
                        "column": sentiment_col,
                        "imputed_values": before - after,
                    }
                )

        frame["target_t_plus_h"] = frame["price"].shift(-config.forecast_horizon_days)
        frame["return_1d"] = frame["price"].pct_change().replace([np.inf, -np.inf], np.nan)
        frame["log_return_1d"] = np.log(frame["price"]).diff().replace([np.inf, -np.inf], np.nan)

        for lag in range(1, config.lookback_days + 1):
            frame[f"price_lag_{lag}"] = frame["price"].shift(lag)
            if "volume" in frame.columns:
                frame[f"volume_lag_{lag}"] = frame["volume"].shift(lag)

        frame["sheet_name"] = sheet
        frame["token"] = frame["token"] if "token" in frame.columns else sheet
        frame["ticker"] = frame["ticker"] if "ticker" in frame.columns else ""

        split_idx = int(len(frame) * config.train_ratio)
        frame["split"] = np.where(np.arange(len(frame)) < split_idx, "train", "test")

        token_row_counts.append(
            {
                "sheet_name": sheet,
                "rows_before_filtering": int(before_rows),
                "rows_after_filtering": int(len(frame)),
                "non_null_dates": int(frame["datetime"].notna().sum()),
            }
        )
        processed_frames.append(frame)

    if not processed_frames:
        raise ValueError("No included token sheets available to preprocess.")

    dataset = pd.concat(processed_frames, ignore_index=True)
    dataset = dataset.dropna(subset=["target_t_plus_h"])

    dataset = dataset.sort_values(["sheet_name", "datetime"]).reset_index(drop=True)
    train = dataset[dataset["split"] == "train"]

    # Keep a canonical panel schema to avoid mixed object types from raw columns.
    identifier_cols = ["sheet_name", "token", "ticker", "datetime", "split"]
    base_numeric_cols = [
        "price",
        "volume",
        "target_t_plus_h",
        "return_1d",
        "log_return_1d",
        "twitter_sentiments",
        "reddit_sentiment",
        "telegram_sentiment",
    ]
    feature_cols = [c for c in dataset.columns if c.startswith("price_lag_") or c.startswith("volume_lag_")]
    keep_cols = [c for c in (identifier_cols + base_numeric_cols + feature_cols) if c in dataset.columns]
    dataset = dataset[keep_cols].copy()

    for col in ["sheet_name", "token", "ticker", "split"]:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype(str)

    for col in [c for c in dataset.columns if c not in ["sheet_name", "token", "ticker", "datetime", "split"]]:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    scale_cols = [c for c in dataset.columns if c.startswith("price_lag_") or c.startswith("volume_lag_")]
    scale_cols += [
        c
        for c in ["twitter_sentiments", "reddit_sentiment", "telegram_sentiment", "return_1d", "volume"]
        if c in dataset.columns
    ]
    for col in scale_cols:
        mu = train[col].mean()
        sigma = train[col].std(ddof=0)
        sigma = sigma if sigma and np.isfinite(sigma) and sigma > 0 else 1.0
        dataset[f"{col}_z"] = (dataset[col] - mu) / sigma

    dataset.to_parquet(output_path("data", "processed", "modeling_dataset.parquet"), index=False)
    dataset.to_parquet(output_path("outputs", "processed", "processed_panel.parquet"), index=False)
    dataset.to_csv(output_path("outputs", "processed", "processed_panel.csv"), index=False)

    pd.DataFrame(imputation_log).to_csv(
        output_path("outputs", "audits", "imputation_log.csv"), index=False
    )

    pd.DataFrame(token_row_counts).to_csv(
        output_path("outputs", "processed", "token_row_counts.csv"),
        index=False,
    )

    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(
            output_path("outputs", "audits", "preprocess_date_parse_diagnostics.csv"),
            index=False,
        )

    return dataset
