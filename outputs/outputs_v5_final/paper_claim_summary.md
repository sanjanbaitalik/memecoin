# Paper Claim Summary

- PRISM best MAE: True
- PRISM best RMSE: False
- Baselines better than PRISM on MAE: none
- Baselines better than PRISM on RMSE: PRISM-V3a
- Baselines worse than PRISM on MAE: PRISM-V2, PRISM-V3a, PRISM-V1, PRISM-V0

## Recommended Claim

PRISM outperforms PRISM-V2, PRISM-V3a, PRISM-V1, PRISM-V0 on MAE, while none remain strong short-horizon competitors. This indicates that memecoin forecasting is strongly influenced by local price continuity, and that PRISM is better interpreted as a risk-aware, multimodal, and diversified forecasting framework rather than as a universally dominant point forecaster.

## Limitations Note

Prophet and/or Persistence remain highly competitive under short-horizon MAE/RMSE, indicating strong local continuity in memecoin prices. Baselines better than PRISM on MAE: none.

## Interpretation

The main contribution of PRISM is evaluated not only through point-forecasting error, but also through risk-aware diversification, robustness, ablation behavior, and statistical testing.