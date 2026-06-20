# Main Results

PRISM is not included in the main baseline comparison table. The best point forecaster on MAE is Persistence. PRISM's contribution is evaluated through the ablation study (V0-V3) and risk-aware diversification metrics.

All baselines use the same chronological split, target horizon, lookback window, selected token subset, and random seeds (see `tables/table_main_forecasting_comparison.csv`). Neural baselines (LSTM, GRU, BiLSTM, TCN) are implemented as native PyTorch sequence models. Risk-adjusted metrics (Sharpe proxy, Sortino proxy, maximum drawdown, VaR, CVaR) are reported in `tables/table_risk_adjusted_evaluation.csv`. Statistical significance is assessed using paired Wilcoxon signed-rank tests with Holm correction (see `tables/table_statistical_tests.csv`).