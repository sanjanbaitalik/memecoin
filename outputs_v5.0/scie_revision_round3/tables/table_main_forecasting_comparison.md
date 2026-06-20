# Main Forecasting Comparison

Paper-ready comparison table across all models.

| model | mae_mean | mae_std | rmse_mean | rmse_std | smape_mean | smape_std | mase_mean | directional_accuracy_mean | n_tokens | n_seeds | model_display | rank_mae | rank_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence | 0.0057 | 0.0187 | 0.0069 | 0.0223 | 21.5695 | 10.0404 | 0.8478 | 0.0078 | 48 | 3 | Persistence | 1 | 1 |
| XGBoost | 0.0176 | 0.0642 | 0.0188 | 0.0678 | 47.8347 | 30.0332 | 1.8069 | 0.3826 | 48 | 3 | XGBoost | 2 | 2 |
| Random Forest | 0.0202 | 0.0792 | 0.0209 | 0.0811 | 49.9564 | 29.5964 | 1.8200 | 0.3455 | 48 | 3 | Random Forest | 3 | 3 |
| ARIMA | 0.0243 | 0.0897 | 0.0250 | 0.0915 | 54.8384 | 33.4377 | 2.2785 | 0.3290 | 48 | 3 | ARIMA | 4 | 4 |
| Prophet | 0.0243 | 0.0897 | 0.0250 | 0.0915 | 54.8384 | 33.4377 | 2.2785 | 0.3290 | 48 | 3 | Prophet | 4 | 4 |
| BiLSTM | 0.0322 | 0.0881 | 0.0330 | 0.0899 | 132.7870 | 67.6603 | 28052.7655 | 0.4520 | 48 | 3 | BiLSTM | 6 | 6 |
| LSTM | 0.0455 | 0.0924 | 0.0461 | 0.0941 | 131.0026 | 67.3713 | 62824.4663 | 0.4277 | 48 | 3 | LSTM | 7 | 7 |
| GRU | 0.0517 | 0.0989 | 0.0526 | 0.1006 | 140.5175 | 64.4670 | 41391.3702 | 0.4404 | 48 | 3 | GRU | 8 | 8 |