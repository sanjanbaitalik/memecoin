# Claim Reframing (Honest, Paper-Ready) — prompt_v6

## Recommended Claim
PRISM outperforms PRISM-V2, PRISM-V3a, PRISM-V1, PRISM-V0 on MAE, while none remain strong short-horizon competitors. This indicates that memecoin forecasting is strongly influenced by local price continuity, and that PRISM is better interpreted as a risk-aware, multimodal, and diversified forecasting framework rather than as a universally dominant point forecaster.

## Summary of Evidence
- PRISM-V3b: MAE=0.0033, RMSE=0.0039
- PRISM-V2: MAE=0.0034, RMSE=0.0040
- PRISM-V3a: MAE=0.0034, RMSE=0.0036
- PRISM-V1: MAE=0.0078, RMSE=0.0085
- PRISM-V0: MAE=0.0118, RMSE=0.0120

## Important Caveats
- PRISM is NOT claimed to be the best point forecaster across all metrics.
- Prophet and Persistence are competitive on short-horizon MAE/RMSE.
- PRISM's contribution includes risk-aware graph diversification (MIS) and validation-gated MAML.
- V3b (validation-gated) may or may not beat V3a (always adapted) — both are reported honestly.
- Risk-adjusted metrics (Sharpe, Sortino, VaR, CVaR) are labelled as proxies — not executable P&L.

## Variant Definitions (from prompt_v6)
- V0: price-only LSTM
- V1: V0 + sentiment fusion
- V2: V1 + graph features + MIS selection
- V3a: V2 + first-order MAML (always adapted)
- V3b: V2 + validation-gated MAML (only if improves over V2 on validation MAE)