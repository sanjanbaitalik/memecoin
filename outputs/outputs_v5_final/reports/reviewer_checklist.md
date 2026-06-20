# Reviewer Checklist — prompt_v6 Compliance

- 1. All baselines use the same chronological train/val/test split — CONFIRMED
- 2. All baselines use the same forecast horizon (H=3 days) — CONFIRMED
- 3. All baselines use the same lookback window (L=14 days) — CONFIRMED
- 4. All baselines use the same token subset (200 modelling tokens) — CONFIRMED
- 5. All baselines use the same three random seeds (11, 17, 23) — CONFIRMED
- 6. No ARIMA fallback to persistence — CONFIRMED (ARIMA returns NaN on failure, logged in failure_log.csv)
- 7. No Prophet fallback to persistence — CONFIRMED (Prophet returns NaN on failure, logged in failure_log.csv)
- 8. N-BEATS-lite has its own model class (not _TCNModel) — CONFIRMED
- 9. TCN constructor accepts num_layers parameter — CONFIRMED
- 10. PRISM V3a always applies MAML adaptation — CONFIRMED
- 11. PRISM V3b uses validation-gated adaptation — CONFIRMED
- 12. Graph threshold calibrated on training data — CONFIRMED
- 13. MIS size > 1, density between 0.05 and 0.60, redundancy reduction > 0 — TO BE CONFIRMED WITH 200 TOKENS
- 14. Statistical tests: n_pairs > 0 for every valid comparison, no NaN p-values — CONFIRMED
- 15. Leakage audit: all 10 checks pass — CONFIRMED
- 16. No fabricated metadata or provenance — CONFIRMED (AUTHOR_TO_CONFIRM fields preserved in tables)
- 17. Risk-adjusted metrics clearly labelled as proxies — CONFIRMED
- 18. Output directory: outputs_v5_final/ — CONFIRMED