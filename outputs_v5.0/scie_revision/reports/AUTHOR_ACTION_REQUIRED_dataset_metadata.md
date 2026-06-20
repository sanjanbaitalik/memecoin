# AUTHOR ACTION REQUIRED: Dataset Metadata

The following fields are not available in the current workbook metadata and must be filled manually by the author before final manuscript submission.

## Required Fields

1. **Price source**: What is the upstream provider for closing price data? (e.g., CoinGecko, CoinMarketCap, Binance API, manual collection)
2. **Volume source**: What is the upstream provider for trading volume data?
3. **Market-cap source**: What is the upstream provider for market capitalization data?
4. **Sentiment source**: What is the upstream provider for sentiment scores? (e.g., LunarCrush, Santiment, custom NLP pipeline)
5. **Social platform source**: Which social platforms were scraped/monitored? (e.g., Twitter/X, Reddit, Telegram, Discord)
6. **Data collection period confirmation**: Confirm the exact start and end dates of data collection.
7. **Sampling frequency confirmation**: Confirm that data is sampled daily at a consistent time (e.g., UTC midnight close).
8. **Missing-value policy**: Describe how missing values were handled beyond the current zero-fill approach.
9. **Survivorship-bias notes**: Describe any steps taken to mitigate survivorship bias beyond the current transparent reporting.

## Current Workbook

- File: Output_database.xlsx
- Path: C:/Users/91943/Desktop/Sanjan/Research/Memecoin/data/raw/Output_database.xlsx

## Instructions

Fill in each field above and update the corresponding entries in `reports/dataset_card.md` and `tables/table_dataset_transparency.csv`.
Do NOT fabricate values. If a field is genuinely unknown, mark it as 'Unknown - author to confirm'.