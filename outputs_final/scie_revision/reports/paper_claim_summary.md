# Paper Claim Summary

- PRISM best MAE: False
- PRISM best RMSE: False
- Baselines better than PRISM on MAE: prophet, persistence
- Baselines better than PRISM on RMSE: prophet, persistence

PRISM outperforms several classical, tree-based, and recurrent neural baselines; however, Prophet and/or Persistence remain highly competitive under short-horizon MAE/RMSE, indicating strong local continuity in memecoin prices.

The main contribution of PRISM is therefore evaluated not only through point-forecasting error, but also through risk-aware diversification, robustness, ablation behavior, and statistical testing.