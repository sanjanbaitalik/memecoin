# AUTHOR ACTION REQUIRED: Dataset Metadata

The following fields are not available in the current workbook metadata and must be filled manually by the author before final manuscript submission.

## Instructions

1. Copy `configs/dataset_metadata_template.yaml` to `configs/dataset_metadata.yaml`.
2. Fill in each field marked `AUTHOR_TO_CONFIRM`.
3. Re-run the pipeline. The paper-facing outputs will use the filled values.

## Required Fields

1. **data_source_name_or_provider**: The upstream data provider or aggregation service.
2. **price_source**: The upstream provider for closing price data (e.g., CoinGecko, CoinMarketCap, Binance API).
3. **volume_source**: The upstream provider for trading volume data.
4. **market_cap_source**: The upstream provider for market capitalization data.
5. **sentiment_source**: The upstream provider for sentiment scores (e.g., LunarCrush, Santiment, custom NLP pipeline).
6. **timezone_or_close_time**: The timezone or time of day for daily price closes.

## Current Workbook

- File: Output_database.xlsx
- Path: C:/Users/91943/Desktop/Sanjan/Research/Memecoin/data/raw/Output_database.xlsx

## Important

Do NOT fabricate values. If a field is genuinely unknown, mark it as `AUTHOR_TO_CONFIRM`.