# PRISM Algorithm Pseudocode

## Inputs
- Workbook with N token sheets (each containing datetime, price, volume, sentiment columns)
- Forecast horizon H = 3 days, Lookback L = 14 days
- Maximum modelling tokens M = 200

## Outputs
- Per-token forecasts for PRISM and all baselines
- Aggregated metrics (MAE, RMSE, sMAPE, MASE, directional accuracy)
- Risk-adjusted metrics (Sharpe, Sortino, VaR, CVaR) — labelled as proxies
- Statistical significance tests (paired Wilcoxon + Holm correction)

## Preprocessing
1. Load workbook, validate schema, parse dates robustly
2. Filter tokens by minimum history length and price coverage
3. Select top-M tokens by row count
4. Construct lagged features and rolling sentiment/volatility features
5. Z-score normalize features using train-only statistics
6. Construct 3-day-ahead target

## Graph Construction (train-only)
1. For each token pair (i, j), compute multi-factor edge score:
   edge_score(i,j) = mean(|corr(f_i^k, f_j^k)|) for all feature dimensions
2. Calibrate threshold over range [0.50, 0.95] using objective:
   score = MIS_size_norm + redundancy_norm - abs(density - 0.25)
3. Build graph with calibrated threshold
4. Compute greedy maximal independent set (MIS)

## First-Order MAML Training (V3a)
1. Initialize base LSTM parameters theta
2. For each meta-epoch:
   a. Sample batch of token tasks
   b. For each task: clone theta, adapt using support loss (K=3 inner steps, lr=0.01)
   c. Compute query loss on adapted parameters
   d. Update theta using query loss gradients (outer lr=0.001)
3. Early stopping on validation meta-loss
4. Always apply adaptation at test time

## Validation-Gated MAML (V3b)
1. Same meta-training as V3a
2. For each token at test time:
   a. Compute MAML-adapted prediction
   b. Compute V2 (no-MAML) prediction
   c. Compare token-level validation MAE for each
   d. Use adapted prediction only if MAE improves; otherwise use V2 prediction

## Ablation Variants
- V0: Price-only LSTM
- V1: V0 + sentiment fusion
- V2: V1 + graph features + MIS
- V3a: V2 + always-adapted MAML
- V3b: V2 + validation-gated MAML

## Evaluation
1. Compute MAE, RMSE, sMAPE, MASE, directional accuracy per model per token
2. Aggregate across tokens and seeds
3. Compute risk-adjusted proxy metrics (Sharpe, Sortino, VaR, CVaR)
4. Paired Wilcoxon tests with Holm correction
5. Output quality gate validation (10 conditions)

## Leakage Audit Checkpoints (10 checks)
- L1-L10: Cover features, scaling, graph, MIS, MAML, test data separation