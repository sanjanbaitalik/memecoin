# GitHub Copilot Prompt: PRISM SCIE Revision Error-Fix and Paper-Ready Result Regeneration

You are working on the PRISM memecoin forecasting repository. The current result archive has improved leakage audit, dataset reporting, and risk metrics, but several reviewer-critical issues remain. Your task is to fix the implementation and regenerate a clean, paper-ready result package that can be directly used in the manuscript.

Do not make cosmetic changes only. Implement the corrections, rerun the experiments, validate the outputs, and export paper-ready CSV/Excel tables, figures, and a concise run summary. The goal is not to force PRISM to beat every baseline artificially; the goal is to produce fair, reproducible, defensible, and internally consistent results.

---

## 1. Current Problems to Fix

The latest result set still has the following major issues:

1. PRISM is not included in the main forecasting comparison table.
2. V2 performs better than V3/full PRISM in the ablation table, so the MAML/adaptation stage currently does not support the claimed full framework.
3. Persistence remains the best MAE/RMSE model, so claims must be reframed unless PRISM genuinely improves.
4. Statistical testing is invalid because PRISM comparisons have `n_pairs = 0` and `NaN` p-values.
5. Prophet and ARIMA are using fallback results due to missing dependencies or failed backends.
6. TCN and N-BEATS failed due to implementation/API errors.
7. The graph/MIS module is not meaningful because graph density is 1.0, MIS size is 1, and redundancy reduction is 0%.
8. Dataset metadata still contains `AUTHOR_TO_CONFIRM` fields.
9. Paper-ready result tables must clearly distinguish full PRISM, PRISM variants, and external baselines.
10. The output package must contain a final validation report with PASS/FAIL gates.

Fix these issues systematically.

---

## 2. Required Output Directory

Create a new output directory:

```text
outputs_v5_final/
```

Inside it, create:

```text
outputs_v5_final/
├── tables/
│   ├── table_dataset_summary.csv
│   ├── table_data_quality_audit.csv
│   ├── table_leakage_audit.csv
│   ├── table_hyperparameters.csv
│   ├── table_main_forecasting_comparison.csv
│   ├── table_ablation.csv
│   ├── table_statistical_tests.csv
│   ├── table_risk_adjusted_metrics.csv
│   ├── table_diversification_metrics.csv
│   ├── table_token_selection_bias.csv
│   ├── table_top_tokens.csv
│   └── table_failure_log.csv
│
├── figures/
│   ├── fig_main_forecasting_comparison.png
│   ├── fig_ablation_mae_rmse.png
│   ├── fig_statistical_tests.png
│   ├── fig_risk_adjusted_performance.png
│   ├── fig_graph_diversification.png
│   ├── fig_robustness_heatmap.png
│   └── fig_prediction_examples.png
│
├── reports/
│   ├── run_summary.md
│   ├── reviewer_readiness_checklist.md
│   ├── claim_reframing_notes.md
│   ├── leakage_audit_explanation.md
│   ├── dataset_metadata_to_confirm.md
│   └── manuscript_insert_text.md
│
└── combined_results.xlsx
```

Every CSV must have clear column names. Every figure must have axis labels, readable legends, units where applicable, and a caption-ready title.

---

## 3. Fix Main Forecasting Comparison Table

### Problem
The current main forecasting table does not include full PRISM. This is unacceptable because reviewers need to compare the proposed method against baselines.

### Required Fix
Create `table_main_forecasting_comparison.csv` containing at least these rows:

```text
Persistence
ARIMA
Prophet
Random Forest
XGBoost
LSTM
GRU
BiLSTM
TCN
N-BEATS
PRISM-V0 Price-only LSTM
PRISM-V1 + Sentiment
PRISM-V2 + Graph/MIS
PRISM-V3 Full PRISM
```

Optional rows if implemented successfully:

```text
DeepAR
TFT
Informer
PatchTST
Autoformer
```

### Required Columns

```text
model
model_family
backend_status
n_tokens
n_predictions
MAE_mean
MAE_std
RMSE_mean
RMSE_std
sMAPE_mean
sMAPE_std
MASE_mean
MASE_std
directional_accuracy_mean
directional_accuracy_std
ROI_proxy_mean
ROI_proxy_std
rank_by_MAE
rank_by_RMSE
rank_by_MASE
notes
```

### Backend Rules

`backend_status` must be one of:

```text
native_success
skipped_dependency_unavailable
skipped_runtime_error
fallback_not_used
```

Do not silently use fallback results for major baselines. If a baseline fails, mark it as skipped and put the reason in `table_failure_log.csv`.

Do not include fallback/surrogate baseline values in the main comparison table as if they were real results.

---

## 4. Implement or Fix Native Baselines

### 4.1 Persistence
Implement a causal persistence baseline:

```text
forecast(t+h) = price(t)
```

where `h = 3 days`.

### 4.2 ARIMA
Use `statsmodels` if available. If unavailable, skip with clear failure reason. Do not use a fake fallback.

Suggested implementation:

```python
from statsmodels.tsa.arima.model import ARIMA
```

Use simple order search over:

```text
(1,0,0), (1,1,0), (0,1,1), (1,1,1), (2,1,2)
```

Select by validation MAE or AIC using only training data. Forecast test period causally.

### 4.3 Prophet
Use Prophet only if installed. Try both imports:

```python
from prophet import Prophet
from fbprophet import Prophet
```

If neither is available, skip with reason. Do not invent Prophet results.

### 4.4 Random Forest
Use sklearn `RandomForestRegressor`. Use only lagged market/sentiment features available at time `t` to predict `t+h`.

### 4.5 XGBoost
Use `xgboost.XGBRegressor` if installed. If unavailable, skip with clear failure reason. Do not silently replace it with another model.

### 4.6 LSTM, GRU, BiLSTM
Implement native PyTorch versions for all three:

```text
LSTM
GRU
Bidirectional LSTM
```

Use the same train/test split, lookback, horizon, scaling, and token subset as PRISM.

Do not label them as fallback/surrogate. If training is too slow, reduce epochs but keep implementation real and document settings.

### 4.7 TCN
Fix the TCN constructor error. Do not pass unsupported keyword arguments like `num_layers` if the local class does not accept them.

Implement your own simple PyTorch TCN if needed:

- Causal Conv1D blocks
- dilation rates: 1, 2, 4
- residual connections
- dropout
- final linear layer

### 4.8 N-BEATS
Implement a lightweight N-BEATS-style baseline if external libraries are unavailable:

- Input: lookback sequence flattened
- Several fully connected blocks
- ReLU activations
- Backcast/forecast heads are optional; a simplified feed-forward N-BEATS-like model is acceptable if labelled `N-BEATS-lite native`

Do not allow this baseline to fail due to dependency errors.

---

## 5. Fix PRISM Variant Definitions

Create clear variant definitions:

```text
PRISM-V0: price-only LSTM
PRISM-V1: V0 + sentiment fusion
PRISM-V2: V1 + risk-aware graph construction + MIS diversification
PRISM-V3: V2 + meta-learning adaptation
```

All variants must use:

```text
same selected tokens
same train/test split
same lookback window
same forecast horizon
same target scaling/inverse-scaling policy
same evaluation metrics
same random seeds
```

Export this to `table_ablation.csv`.

Required columns:

```text
variant
description
components_enabled
n_tokens
n_predictions
MAE_mean
MAE_std
RMSE_mean
RMSE_std
sMAPE_mean
MASE_mean
directional_accuracy_mean
ROI_proxy_mean
relative_MAE_change_vs_previous
relative_RMSE_change_vs_previous
relative_MAE_change_vs_V0
relative_RMSE_change_vs_V0
interpretation
```

---

## 6. Handle the V2 Better Than V3 Problem

### Problem
Current ablation reports:

```text
V2 MAE < V3 MAE
```

This means the meta-learning/adaptation stage worsens results. Do not hide this.

### Required Fix Options

Implement both modes and compare:

```text
PRISM-V3a: V2 + MAML-style adaptation
PRISM-V3b: V2 + validation-gated adaptation
```

Validation-gated adaptation logic:

1. Train/adapt using train/support only.
2. Evaluate adaptation candidate on validation/query split inside the training period.
3. If adaptation improves validation MAE, use adapted weights.
4. Otherwise, retain the V2/base weights.
5. Never use test data for this decision.

Then select final PRISM as:

```text
PRISM-V3 Full PRISM = validation-gated adaptation output
```

Add a column:

```text
adaptation_used_rate
```

If MAML/adaptation is not consistently beneficial, write the interpretation honestly:

```text
Meta-learning adaptation improved only a subset of tokens; therefore, the final PRISM uses validation-gated adaptation to avoid harmful token-specific updates.
```

This is acceptable for a paper and defensible.

---

## 7. Fix Statistical Testing

### Problem
Current statistical test table has:

```text
n_pairs = 0
p_value = NaN
```

This means the test is invalid.

### Required Fix
For every model comparison against PRISM, compute paired token-level metrics.

Use a table where each row is:

```text
token_id
model
MAE
RMSE
sMAPE
MASE
directional_accuracy
ROI_proxy
```

Then pivot by token and compare PRISM vs each baseline on the same tokens.

### Required Test
Use Wilcoxon signed-rank test for paired token-level MAE and RMSE:

```python
from scipy.stats import wilcoxon
```

If fewer than 10 paired tokens are available, mark the test as insufficient rather than returning NaN.

Apply Holm correction across comparisons:

```python
from statsmodels.stats.multitest import multipletests
method='holm'
```

Required output columns in `table_statistical_tests.csv`:

```text
comparison
metric
n_pairs
prism_mean
baseline_mean
mean_difference
median_difference
wilcoxon_statistic
p_value
holm_adjusted_p
alpha
significant
favoured_model
interpretation
```

Quality gate:

```text
n_pairs must be > 0 for every comparison where both PRISM and baseline predictions exist.
p_value must not be NaN unless n_pairs is insufficient.
```

---

## 8. Fix Graph/MIS Diversification

### Problem
Current graph statistics:

```text
graph density = 1.0
MIS size = 1
redundancy reduction = 0%
```

This means the graph threshold is too low or the edge score is not discriminative.

### Required Fix
Implement threshold calibration on training data only.

Do not use a fixed threshold blindly. Search over candidate thresholds:

```text
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
```

or use quantile thresholds:

```text
edge_score >= q, q in {0.70, 0.75, 0.80, 0.85, 0.90, 0.95}
```

Select threshold using training-period graph only with the following target:

```text
0.05 <= graph_density <= 0.60
MIS_size >= max(5, 0.05 * number_of_tokens)
redundancy_reduction > 0
```

If no threshold satisfies all constraints, choose the threshold that maximizes:

```text
score = MIS_size_normalized + redundancy_reduction_normalized - abs(graph_density - 0.25)
```

### Edge Score
Ensure edge score combines:

```text
price/return correlation
volatility similarity
sentiment-trend similarity
risk-distance penalty
```

Suggested formula:

```text
edge_score_ij = sigmoid(
    w_corr * abs(corr_returns_ij)
  + w_sent * sentiment_similarity_ij
  + w_vol  * volatility_similarity_ij
  - w_risk * risk_distance_ij
)
```

Use training-period data only.

### Export `table_diversification_metrics.csv`

Required columns:

```text
n_tokens
threshold_type
selected_threshold
graph_edges
graph_density
MIS_size
MIS_fraction
avg_pairwise_corr_before
avg_pairwise_corr_after
median_pairwise_corr_before
median_pairwise_corr_after
redundancy_reduction_percent
avg_risk_before
avg_risk_after
interpretation
```

Quality gate:

```text
graph_density must not be 0.0 or 1.0
MIS_size must be > 1
redundancy_reduction_percent must be computed correctly
```

---

## 9. Fix Dataset Metadata and Token Counts

### Problem
Dataset metadata still has `AUTHOR_TO_CONFIRM`. Token counts are inconsistent.

### Required Fix
Create:

```text
reports/dataset_metadata_to_confirm.md
```

It must contain a clean checklist:

```text
price_source: AUTHOR_TO_CONFIRM
market_cap_source: AUTHOR_TO_CONFIRM
volume_source: AUTHOR_TO_CONFIRM
sentiment_source: AUTHOR_TO_CONFIRM
sentiment_collection_method: AUTHOR_TO_CONFIRM
raw_text_available: yes/no/AUTHOR_TO_CONFIRM
timezone: AUTHOR_TO_CONFIRM
date_range_start: computed from dataset
date_range_end: computed from dataset
sampling_frequency: computed from dataset
```

Also create `table_dataset_summary.csv` with separate rows for:

```text
candidate_token_sheets
eligible_token_sheets
selected_modeling_tokens
evaluated_tokens_after_prediction
processed_observations_full_eligible
processed_observations_selected_subset
train_observations_selected_subset
test_observations_selected_subset
forecast_horizon_days
lookback_window_days
main_train_ratio
robustness_train_ratios
robustness_lookbacks
```

Quality gate:

```text
selected_modeling_tokens must be 200 if max_tokens=200.
processed_observations_selected_subset must be recalculated only for selected 200 tokens.
evaluated_tokens_after_prediction must be <= selected_modeling_tokens.
Do not report 881 as selected tokens if selected subset is 200.
```

---

## 10. Strengthen Leakage Audit

The existing leakage audit is good, but make it stricter.

Create `table_leakage_audit.csv` with rows:

```text
chronological_split
future_target_exclusion
train_only_scaling
rolling_feature_causality
sentiment_alignment_causality
graph_train_period_only
MIS_train_period_only
MAML_support_query_split_train_only
validation_gating_train_only
test_set_never_used_for_tuning
```

Columns:

```text
check_name
status
risk_level
evidence_file_or_function
explanation
```

Every status must be:

```text
PASS
FAIL
WARNING
NOT_APPLICABLE
```

If any `FAIL` exists, stop the run and write a failure summary.

---

## 11. Add Risk-Adjusted and Portfolio Metrics

Since PRISM may not beat Persistence on raw MAE/RMSE, evaluate its risk-aware value properly.

Create `table_risk_adjusted_metrics.csv` with rows for:

```text
Persistence
Prophet
LSTM
GRU
BiLSTM
Random Forest
XGBoost
PRISM-V2
PRISM-V3 Full PRISM
```

Columns:

```text
model
mean_return_proxy
volatility_proxy
sharpe_proxy
sortino_proxy
max_drawdown_proxy
VaR_95_proxy
CVaR_95_proxy
downside_deviation_proxy
hit_rate_mean
directional_accuracy_mean
rank_by_sharpe
rank_by_CVaR
rank_by_max_drawdown
interpretation
```

Use proxy metrics if real trading execution is not available, but clearly label them as proxies.

Also compute transaction-cost-adjusted ROI proxy under simple cost assumptions:

```text
cost = 0.10%, 0.25%, 0.50%
```

Export these to:

```text
table_transaction_cost_sensitivity.csv
```

---

## 12. Claim Reframing Rules

Do not overclaim.

If PRISM does not beat Persistence/Prophet on MAE/RMSE, generate `reports/claim_reframing_notes.md` with recommended manuscript wording.

Use language like:

```text
PRISM outperforms several classical, tree-based, and recurrent baselines, while Persistence and Prophet remain strong short-horizon competitors.
```

```text
The results indicate that memecoin forecasting is strongly affected by local price continuity, and therefore raw point-forecasting accuracy alone is insufficient to evaluate risk-aware multimodal frameworks.
```

```text
PRISM is positioned as an auditable risk-aware multimodal forecasting and diversified token-selection framework rather than as a universally dominant point forecaster.
```

Avoid these claims unless true:

```text
PRISM outperforms all baselines.
PRISM is the best forecasting model.
PRISM achieves superior accuracy over all traditional models.
```

---

## 13. Paper-Ready Manuscript Insert Text

Create:

```text
reports/manuscript_insert_text.md
```

It must contain short paper-ready paragraphs for:

1. Dataset transparency paragraph
2. Leakage audit paragraph
3. Baseline implementation paragraph
4. Graph/MIS threshold calibration paragraph
5. Main comparison interpretation paragraph
6. Ablation interpretation paragraph
7. Statistical testing paragraph
8. Risk-adjusted performance paragraph
9. Claim-limitation paragraph

These paragraphs must be honest and aligned with the actual generated results.

---

## 14. Figure Requirements

Generate the following figures:

### `fig_main_forecasting_comparison.png`
Bar chart of MAE and RMSE for all successful baselines and PRISM variants.

### `fig_ablation_mae_rmse.png`
Line or grouped bar chart for V0, V1, V2, V3a, V3b/Full PRISM.

### `fig_statistical_tests.png`
Heatmap or bar plot of Holm-adjusted p-values and favoured model.

### `fig_risk_adjusted_performance.png`
Bar chart of Sharpe/Sortino/CVaR proxy rankings.

### `fig_graph_diversification.png`
Before/after average pairwise correlation and graph/MIS summary.

### `fig_robustness_heatmap.png`
Heatmap across train ratios and lookback windows.

### `fig_prediction_examples.png`
For 3 representative tokens, plot actual vs predicted for Persistence, PRISM-V2, and Full PRISM.

All plots must be high-resolution PNG, at least 300 DPI.

---

## 15. Final Reviewer Readiness Checklist

Create:

```text
reports/reviewer_readiness_checklist.md
```

It must contain PASS/FAIL for:

```text
[ ] PRISM included in main comparison table
[ ] Full PRISM and variants included in ablation table
[ ] No fallback/surrogate results reported as real baselines
[ ] Failed baselines are logged clearly
[ ] Statistical tests have n_pairs > 0 where applicable
[ ] No NaN p-values for valid comparisons
[ ] Graph density is not 0 or 1
[ ] MIS size is greater than 1
[ ] Redundancy reduction is computed
[ ] Dataset selected token count is consistent
[ ] Leakage audit has no FAIL
[ ] Dataset source metadata fields listed for author confirmation
[ ] Claims are softened if Persistence/Prophet beat PRISM
[ ] All paper-ready tables exported
[ ] All paper-ready figures exported
```

---

## 16. Quality Gates Before Finishing

The script must not print “done” unless all of the following are true:

```text
1. table_main_forecasting_comparison.csv exists and contains PRISM-V3 Full PRISM.
2. table_ablation.csv exists and contains V0, V1, V2, V3a, and final PRISM.
3. table_statistical_tests.csv has n_pairs > 0 for at least PRISM vs all successful baselines.
4. table_diversification_metrics.csv has graph_density not equal to 0 or 1.
5. MIS_size > 1.
6. table_dataset_summary.csv reports selected_modeling_tokens = 200 if max_tokens=200.
7. leakage audit has no FAIL rows.
8. failure log clearly lists any skipped/failed baselines.
9. all required figures exist.
10. combined_results.xlsx exists.
```

If a quality gate fails, write:

```text
outputs_v5_final/reports/FAILED_QUALITY_GATES.md
```

and explain exactly what failed and how to fix it.

---

## 17. Final Terminal Summary

At the end of the run, print:

```text
PRISM SCIE Revision V5 Complete
Output directory: outputs_v5_final/
Successful baselines: ...
Skipped baselines: ...
Full PRISM MAE/RMSE/MASE: ...
Best MAE model: ...
Best risk-adjusted model: ...
Graph density: ...
MIS size: ...
Valid statistical comparisons: ...
Leakage audit: PASS/FAIL
Reviewer readiness: PASS/FAIL
```

If PRISM is not the best MAE model, explicitly print:

```text
NOTE: PRISM is not the best point forecaster by MAE/RMSE. Use the claim-reframing notes and position PRISM as a risk-aware multimodal forecasting and diversified token-selection framework.
```

---

## 18. Important Rule

Do not fabricate improved results. Do not tune on the test set. Do not use future information. Do not hide failed baselines. The final outputs must be honest, reproducible, internally consistent, and directly usable for a revised SCIE manuscript.
