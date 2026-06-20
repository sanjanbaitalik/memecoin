# PRISM Algorithm Pseudocode

## Inputs
- Workbook with N token sheets (each containing datetime, price, volume, sentiment columns)
- Forecast horizon H = 3 days, Lookback L = 14 days
- Maximum modelling tokens M = 200

## Outputs
- Per-token forecasts for PRISM and all baselines
- Aggregated metrics (MAE, RMSE, sMAPE, MASE, directional accuracy)
- Risk-adjusted metrics (Sharpe, Sortino, VaR, CVaR)
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
   edge_score(i,j) = mean(|corr(f_i^k, f_j^k)|) for 13 feature dimensions
2. Select edges using quantile thresholding (top 15%)
3. Compute greedy maximal independent set (MIS)

## First-Order MAML Training (V3)
1. Initialize base LSTM parameters theta
2. For each meta-epoch:
   a. Sample batch of token tasks
   b. For each task: clone theta, adapt using support loss (K=3 inner steps, lr=0.01)
   c. Compute query loss on adapted parameters
   d. Update theta using query loss gradients (outer lr=0.001)
3. Early stopping on validation meta-loss

## Ablation Variants
- V0: Price-only LSTM
- V1: V0 + sentiment fusion
- V2: V1 + graph features + MIS
- V3: V2 + first-order MAML

## Evaluation
1. Compute MAE, RMSE, sMAPE, MASE, directional accuracy per model per token
2. Aggregate across tokens and seeds
3. Compute risk-adjusted proxy metrics
4. Paired Wilcoxon tests with Holm correction
5. Output quality gate validation

## Leakage Audit Checkpoints
- Features at time t use only <= t information
- Target t+H never used in feature scaling
- Scaling fit on training data only
- Graph/MIS use training-period data only
- MAML support/query/test are chronological and non-overlapping
- Test data never used for feature selection, scaling, tuning, or meta-training