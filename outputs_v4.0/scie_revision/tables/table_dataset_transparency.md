# Dataset Transparency Table

Dataset transparency and provenance summary.

| field | value |
| --- | --- |
| source_file_name | Output_database.xlsx |
| data_source_name_or_provider | Not available in current workbook metadata |
| price_source | Not available in current workbook metadata |
| market_cap_source | Not available in current workbook metadata |
| volume_source | Not available in current workbook metadata |
| sentiment_source | Not available in current workbook metadata |
| social_platform_source | twitter/x, reddit, telegram |
| date_range_start | 2024-11-11 00:00:00 |
| date_range_end | 2025-02-01 00:00:00 |
| sampling_frequency | daily |
| number_of_candidate_token_sheets | 1273 |
| number_of_eligible_sheets | 881 |
| number_of_modeled_tokens | 881 |
| processed_aligned_observations | 60431 |
| mean_token_history_length | 68.5936 |
| median_token_history_length | 76.0000 |
| missing_value_policy | train-only scaling with zero-fill sentiment imputation and row-wise exclusion of invalid targets |
| target_construction | 3-day-ahead closing price target |
| train_validation_test_split_policy | chronological split with validation subset from the training period only |
| token_selection_rule | eligible included tokens ranked by row count and capped at the modeling maximum |
| possible_survivorship_bias_risk | yes |
| limitations_of_using_top_tokens_by_row_count | may overrepresent longer-lived, more liquid tokens and underrepresent short-lived assets |