# GitHub Copilot Prompt: PRISM SCIE Revision Round 2 — Fix Remaining Result and Evidence Problems

You are working inside the existing PRISM memecoin forecasting repository. The previous SCIE revision run generated an output folder similar to `outputs/scie_revision/`, but the produced results still contain several reviewer-risk issues. Your task is to modify the code and rerun the pipeline so that the resulting tables, figures, audits, and manuscript inserts can be used directly in the paper without misleading claims.

Do not make cosmetic-only changes. Do not fabricate metadata, performance, or provenance. If a required dataset field is unavailable, report it explicitly and create an author-action checklist. The goal is to generate honest, reviewer-safe, paper-ready outputs.

---

## 1. Current problems detected in the addressed results

The previous output archive shows the following unresolved issues:

1. **PRISM is still not the best point forecaster.**
   - In the revised main comparison table, Prophet and Persistence still have lower MAE/RMSE than PRISM.
   - The final paper claims must therefore avoid saying that PRISM is superior to all baselines.
   - The claim should be reframed around: PRISM outperforms several classical, tree-based, and neural baselines, while Prophet and Persistence remain strong short-horizon competitors.

2. **GRU/LSTM baselines are still fallback/surrogate rows.**
   - Current `actual_backend` values such as `sklearn_mlp_gru_surrogate` and `sklearn_mlp_lstm_surrogate` are unacceptable.
   - `fallback_used=True` must not appear in the final paper-ready comparison table.

3. **Graph and MIS diversification outputs are currently broken or non-informative.**
   - The graph statistics report `number_of_edges = 0`, `graph_density = 0`, and `redundancy_reduction_percentage = 0`.
   - This means MIS selection does nothing and cannot support a graph-diversification claim.
   - Fix graph construction, thresholding, similarity calculation, and selected-token reporting so that graph/MIS metrics are meaningful.

4. **Dataset and token-count reporting is inconsistent.**
   - Some files report `number_of_modeled_tokens = 881`, while the manuscript and hyperparameter summary mention a deterministic modelled subset of 200 tokens.
   - The token-selection bias table also reports invalid values such as `row_counts_of_selected_200_tokens = 0`.
   - The pipeline must clearly distinguish candidate sheets, eligible sheets, selected/modelled tokens, and evaluated tokens.

5. **PRISM metrics are incomplete in the main comparison table.**
   - PRISM currently has missing values for MASE and directional accuracy.
   - Compute all metrics for PRISM and all baselines consistently.

6. **Risk-adjusted metrics currently do not strengthen PRISM enough.**
   - Prophet still appears superior on cumulative return and Sharpe proxy.
   - Either improve the PRISM portfolio/risk evaluation properly or generate an honest interpretation that PRISM is stronger mainly in modularity, diversification framework, and statistically supported improvements over several non-extrapolative baselines.

7. **Leakage audit still contains a warning for MAML.**
   - The previous audit reports: `MAML support/query sets are drawn from the training period during training = WARNING`.
   - Replace proxy MAML with a strict implementation or rename it honestly as an adaptation proxy. If the paper says MAML, the code must implement strict support/query meta-learning without validation/test contamination.

8. **Ablation results are very large and suspicious.**
   - Huge changes such as V0 MAE 4.0539 to V1 MAE 0.6981 may trigger reviewer suspicion.
   - Add a strict ablation audit confirming same token subset, same split, same target transform, same scaler policy, same feature availability timing, and consistent inverse transformation.
   - Add median/IQR and per-seed tables to make the ablation more transparent.

9. **Dataset transparency is still insufficient.**
   - Provider/source fields currently say `Not available in current workbook metadata`.
   - Do not invent values, but create an explicit `AUTHOR_ACTION_REQUIRED_dataset_metadata.md` file listing exactly what the author must fill manually: price source, volume source, market-cap source, sentiment source, social platforms, data collection period confirmation, sampling frequency confirmation, missing-value policy, and survivorship-bias notes.

10. **Some figures are not paper-ready.**
    - The diversification figure is meaningless when the graph has zero edges and mostly plots zero values.
    - Regenerate figures only after fixing metrics. Every figure must have clear axis labels, units where applicable, readable tick labels, and a caption-ready description.

---

## 2. Main implementation goals

Update the repository so that a single command regenerates a complete SCIE-ready revision package:

```bash
python run_scie_revision.py \
  --data data/raw/Output_database.xlsx \
  --output outputs/scie_revision_round2 \
  --seeds 11 17 23 \
  --main_train_ratio 0.8 \
  --horizon 3 \
  --lookback 14 \
  --max_tokens_for_modeling 200 \
  --run_baselines \
  --run_prism \
  --run_ablation \
  --run_graph_sensitivity \
  --run_risk_metrics \
  --run_leakage_audit \
  --run_output_quality_gate
```

The output directory must contain CSV, Markdown, LaTeX, and PNG outputs that can be inserted directly into the manuscript.

---

## 3. Required code corrections

### 3.1 Implement real neural baselines

Replace all fallback/surrogate GRU and LSTM baselines with real trainable sequence models.

Minimum required baselines:

1. Persistence / naive last value
2. ARIMA
3. Prophet
4. Random Forest
5. XGBoost
6. Real LSTM
7. Real GRU
8. Real BiLSTM
9. TCN or N-BEATS

Preferred additional baseline if feasible:

10. PatchTST, Informer, Autoformer, DeepAR, or TFT

Implementation requirements:

- Use PyTorch for LSTM, GRU, BiLSTM, and TCN/N-BEATS.
- Use the same chronological split, target horizon, lookback, selected token subset, scaling policy, and random seeds across all models.
- Store the actual backend name, e.g. `torch_lstm`, `torch_gru`, `torch_bilstm`, `torch_tcn`, `nbeats_torch`.
- The final table must have `fallback_used = False` for every reported model.
- If a model fails, exclude it from the paper-ready table and report the failure separately in `reports/baseline_failure_log.md`.
- Do not include any fallback/surrogate row in the final paper-ready table.

Output:

- `tables/table_main_comparison_round2.csv/.md/.tex`
- `tables/table_neural_baseline_hyperparameters.csv/.md/.tex`
- `reports/baseline_failure_log.md`

Acceptance criteria:

- No row in the paper-ready table contains `fallback`, `surrogate`, or `sklearn_mlp_*_surrogate`.
- All neural baselines have real sequence inputs with lookback windows.

---

### 3.2 Fix graph construction and MIS diversification

The previous graph had zero edges. Fix this.

Tasks:

1. Audit the current graph similarity score and threshold logic.
2. Compute token-token similarity only using training-period data.
3. Build features for graph construction from causal information only:
   - rolling volatility,
   - return trajectories,
   - sentiment trajectories,
   - liquidity/volume trajectory if available,
   - optional drawdown/risk score.
4. Normalize similarity scores into a stable range.
5. Replace fixed threshold-only graphing with one of these reviewer-safe options:
   - quantile thresholding, e.g. top 10% or top 15% similarities become edges, or
   - k-nearest-neighbor graph, e.g. each token connects to top-k most similar tokens, or
   - threshold sweep with selected operating point justified by graph density.
6. Keep graph construction strictly train-only for each split.
7. Implement greedy maximal independent set, and call it correctly as **greedy maximal independent set**, not maximum independent set.
8. Report before/after selected-token redundancy.

Required graph metrics:

- number of nodes
- number of edges
- graph density
- average degree
- connected components
- selected MIS support size
- selected edge count
- average pairwise correlation before MIS
- average pairwise correlation after MIS
- redundancy reduction percentage
- similarity threshold or quantile used
- k value if kNN graph is used

Output:

- `tables/table_graph_statistics_round2.csv/.md/.tex`
- `tables/table_diversification_metrics_round2.csv/.md/.tex`
- `figures/fig_diversification_effect_round2.png`
- `reports/mis_theoretical_explanation_round2.md`
- `reports/graph_threshold_sensitivity.md`

Acceptance criteria:

- Graph edge count must be greater than zero.
- Graph density must be greater than zero but not trivially complete.
- MIS must actually reduce redundancy or selected edge count.
- If no threshold gives meaningful graph structure, write an explicit failure report instead of plotting meaningless zero bars.

---

### 3.3 Fix token-count consistency and token-selection audit

The pipeline must consistently report whether 200 or 881 tokens are used.

Tasks:

1. Track and report the following separately:
   - candidate token sheets,
   - eligible token sheets after filtering,
   - selected/modelled token subset,
   - tokens with successful forecasts,
   - tokens used in each table.
2. If `--max_tokens_for_modeling 200` is used, ensure all model comparison, ablation, robustness, graph, and token-selection tables use the same 200 selected tokens unless explicitly stated otherwise.
3. Fix `table_token_selection_bias` so it does not show impossible values such as selected token count = 0.
4. Compute row-count distributions for selected and non-selected eligible tokens.
5. Report survivorship/token-selection bias honestly.

Output:

- `tables/table_dataset_transparency_round2.csv/.md/.tex`
- `tables/table_token_selection_bias_round2.csv/.md/.tex`
- `reports/token_selection_bias_report_round2.md`
- `reports/dataset_card_round2.md/.json`

Acceptance criteria:

- The number of modelled tokens in all tables must match the configured selected subset.
- The token-selection bias table must contain valid selected-token counts and selected-token row-count statistics.
- If 881 tokens are used instead of 200, update all manuscript inserts and table descriptions accordingly.

---

### 3.4 Compute all metrics consistently for PRISM and all baselines

The previous main table had missing MASE and directional accuracy for PRISM.

Compute the following for every model and every selected token:

Forecasting metrics:

- MAE
- RMSE
- median MAE
- standard deviation of MAE
- sMAPE
- MASE
- directional accuracy
- hit rate if applicable

Risk/trading proxy metrics:

- cumulative return proxy
- mean return proxy
- Sharpe ratio proxy
- Sortino ratio proxy
- maximum drawdown proxy
- VaR 95%
- CVaR 95%
- downside deviation
- turnover proxy

Statistical outputs:

- paired Wilcoxon PRISM-vs-baseline tests on MAE
- Holm correction
- effect size such as rank-biserial correlation
- favored model

Output:

- `tables/table_main_comparison_round2.csv/.md/.tex`
- `tables/table_risk_adjusted_metrics_round2.csv/.md/.tex`
- `tables/table_significance_vs_prism_round2.csv/.md/.tex`

Acceptance criteria:

- No metric column for PRISM is NaN.
- If PRISM is not best, the claim summary must explicitly say so.
- The paper-ready claim must be generated from the actual results, not hardcoded.

---

### 3.5 Implement or honestly rename MAML

The previous leakage audit had a warning that the revision uses an adaptation proxy.

Tasks:

1. If the paper claims MAML, implement real MAML support/query adaptation:
   - support set from training period only,
   - query set from training period only during meta-training,
   - validation set used only for early stopping/tuning,
   - test set used only once for final evaluation.
2. If real MAML is too expensive, rename the method in outputs and manuscript inserts as `meta-learning-inspired adaptation proxy`, not MAML.
3. The leakage audit must not have warnings.

Output:

- `audits/leakage_audit_report_round2.csv/.md/.json/.tex`
- `reports/maml_or_adaptation_audit.md`

Acceptance criteria:

- Every leakage audit item is PASS, or the pipeline exits with a clear failure.
- No warning remains for MAML support/query contamination.

---

### 3.6 Add strict ablation audit and robust ablation outputs

The ablation improvement is large, so make it defensible.

Tasks:

1. Verify and report that V0, V1, V2, and V3 use:
   - identical selected token subset,
   - identical chronological split,
   - identical target horizon,
   - identical target scaling/inverse-scaling policy,
   - identical training/test periods,
   - causal-only features.
2. Add per-seed and per-token result exports for all ablation variants.
3. Add median, IQR, mean, standard deviation, and n-token counts.
4. Add a written leakage-ablation audit explaining why the performance jump is not caused by leakage or scale mismatch.
5. If any check fails, stop the paper-ready export.

Output:

- `tables/table_ablation_strict_round2.csv/.md/.tex`
- `tables/table_ablation_per_seed_round2.csv/.md/.tex`
- `tables/table_ablation_median_iqr_round2.csv/.md/.tex`
- `audits/ablation_strict_audit_round2.md`
- `figures/fig_ablation_strict_round2.png`

Acceptance criteria:

- Ablation table includes mean, std, median, IQR, sMAPE, MASE, directional accuracy, n_tokens, and n_seeds.
- The audit explicitly confirms no leakage and no target-scaling mismatch.

---

### 3.7 Improve dataset transparency without inventing unknowns

Tasks:

1. Parse whatever provenance exists in the workbook and code configuration.
2. If metadata is absent, keep `Not available in workbook metadata`, but also create a manual author-action file.
3. Create a dataset card with:
   - source workbook filename,
   - upstream provider if available,
   - price source if available,
   - market-cap source if available,
   - volume source if available,
   - sentiment source if available,
   - social platform source if available,
   - date range start/end,
   - sampling frequency,
   - candidate sheets,
   - eligible sheets,
   - selected/modelled tokens,
   - processed observations,
   - missing-value policy,
   - target construction,
   - split policy,
   - token-selection rule,
   - survivorship-bias warning.

Output:

- `reports/dataset_card_round2.md/.json`
- `tables/table_dataset_transparency_round2.csv/.md/.tex`
- `reports/AUTHOR_ACTION_REQUIRED_dataset_metadata.md`

Acceptance criteria:

- The paper-ready dataset table must be transparent and not overclaim unknown provenance.
- The author-action file must list missing fields clearly.

---

### 3.8 Reframe claims automatically from results

Generate a claim summary from the actual results.

Rules:

1. If PRISM is not best on MAE/RMSE, do not write `PRISM outperforms all baselines`.
2. If Prophet/Persistence beat PRISM, say:
   `PRISM outperforms several classical, tree-based, and neural baselines, while Prophet and Persistence remain strong short-horizon competitors.`
3. If PRISM wins on risk/diversification metrics, explain that separately and specify which metrics.
4. If PRISM does not win risk metrics either, say that PRISM's contribution is primarily modular, interpretable, statistically tested, and diversification-aware, not raw point-forecast dominance.

Output:

- `reports/paper_claim_summary_round2.md/.json`
- `reports/title_and_abstract_recommendation_round2.md`
- `reports/manuscript_insert_claim_reframing_round2.md`
- `reports/manuscript_insert_limitations_round2.md`
- `reports/manuscript_insert_future_work_round2.md`

Acceptance criteria:

- Generated claims must match the tables.
- No overclaiming is allowed.

---

### 3.9 Add output quality gate

Create a script:

```bash
python scripts/audit_scie_outputs.py --output outputs/scie_revision_round2
```

This script must inspect all final outputs and fail if any of the following is true:

- fallback/surrogate model appears in paper-ready main comparison table,
- graph edge count is zero,
- redundancy reduction is zero while the paper claims diversification benefit,
- PRISM has NaN metrics,
- token counts are inconsistent across dataset, graph, ablation, and main-comparison outputs,
- leakage audit contains WARNING or FAIL,
- paper claim says PRISM is best when Table 5 contradicts it,
- key dataset metadata are missing and no author-action file is created,
- figures are missing,
- required table formats CSV/MD/TEX are missing.

Output:

- `audits/output_quality_gate_round2.md`
- `audits/output_quality_gate_round2.json`

Acceptance criteria:

- The final run should clearly say either `SCIE_OUTPUT_QUALITY_GATE=PASS` or list blocking failures.

---

## 4. Paper-ready output package

After the fixes, regenerate the following final files:

### Tables

- `tables/table_dataset_transparency_round2.csv/.md/.tex`
- `tables/table_token_selection_bias_round2.csv/.md/.tex`
- `tables/table_main_comparison_round2.csv/.md/.tex`
- `tables/table_neural_baseline_hyperparameters.csv/.md/.tex`
- `tables/table_ablation_strict_round2.csv/.md/.tex`
- `tables/table_ablation_per_seed_round2.csv/.md/.tex`
- `tables/table_graph_statistics_round2.csv/.md/.tex`
- `tables/table_diversification_metrics_round2.csv/.md/.tex`
- `tables/table_risk_adjusted_metrics_round2.csv/.md/.tex`
- `tables/table_significance_vs_prism_round2.csv/.md/.tex`
- `tables/table_complexity_analysis_round2.csv/.md/.tex`

### Figures

- `figures/fig_main_comparison_round2.png`
- `figures/fig_ablation_strict_round2.png`
- `figures/fig_graph_diversification_round2.png`
- `figures/fig_risk_adjusted_comparison_round2.png`
- `figures/fig_robustness_heatmap_round2.png`
- `figures/fig_token_history_distribution_round2.png`

### Audits and reports

- `audits/leakage_audit_report_round2.csv/.md/.json/.tex`
- `audits/ablation_strict_audit_round2.md`
- `audits/output_quality_gate_round2.md/.json`
- `reports/dataset_card_round2.md/.json`
- `reports/AUTHOR_ACTION_REQUIRED_dataset_metadata.md`
- `reports/mis_theoretical_explanation_round2.md`
- `reports/complexity_analysis_round2.md`
- `reports/paper_claim_summary_round2.md/.json`
- `reports/title_and_abstract_recommendation_round2.md`
- `reports/code_data_availability_statement_round2.md`

---

## 5. Manuscript-ready text snippets to generate

Generate concise manuscript inserts for:

1. dataset transparency and limitations,
2. leakage audit,
3. graph/MIS diversification explanation,
4. baseline fairness and hyperparameter protocol,
5. ablation interpretation,
6. risk/diversification interpretation,
7. honest claim reframing,
8. limitations and future work.

Do not generate long prose. Each insert should be 1–2 paragraphs and should cite the corresponding table/figure filename.

Output path:

- `reports/manuscript_inserts_round2/`

---

## 6. Final acceptance checklist

Before finishing, print the following checklist:

```text
[ ] Real LSTM baseline implemented and reported without fallback
[ ] Real GRU baseline implemented and reported without fallback
[ ] Real BiLSTM or TCN/N-BEATS baseline implemented and reported
[ ] No fallback/surrogate rows in paper-ready table
[ ] PRISM has MAE, RMSE, sMAPE, MASE, directional accuracy, ROI/risk metrics
[ ] Graph has nonzero edges and meaningful density
[ ] MIS reduces redundancy or a failure is explicitly reported
[ ] Token counts are consistent across all tables
[ ] Leakage audit has no WARNING/FAIL
[ ] Ablation audit confirms same subset/split/scaler/target transformation
[ ] Dataset card reports available metadata and lists unavailable fields honestly
[ ] Claims are consistent with actual table rankings
[ ] Figures are paper-ready and not based on meaningless zero metrics
[ ] Output quality gate passed
```

Only mark an item complete if it is genuinely satisfied by the generated outputs.

