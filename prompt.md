# GitHub Copilot Prompt: Major Revision Implementation for PRISM Memecoin Forecasting Paper

You are working on the PRISM memecoin forecasting codebase and manuscript-support scripts. The goal is to address serious reviewer-style concerns before SCIE submission. Implement the required corrections so that the generated outputs can be used directly in the paper without manual recalculation.

The current paper proposes PRISM: Portfolio Risk-aware Independent Set Meta-learning, combining sentiment fusion, risk-aware token graph construction, maximal independent set diversification, and MAML-enhanced LSTM forecasting. The latest review identifies major weaknesses: PRISM does not beat Prophet/Persistence in the current main result table, GRU/LSTM baselines are marked as fallback/surrogate, dataset provenance is incomplete, the ablation may look suspicious without leakage checks, risk-aware claims are not sufficiently evaluated, and paper claims overstate forecasting superiority.

Your task is to update the repository so that the implementation produces a fair, transparent, reproducible, paper-ready evaluation package.

Do not fabricate results. Do not hardcode metrics. Do not delete unfavorable baselines. If PRISM does not outperform Prophet or Persistence on MAE/RMSE, the generated report must honestly state that short-horizon extrapolative baselines remain competitive. The revised framing should emphasize risk-aware, interpretable, multimodal forecasting and diversified token selection.

---

## 1. First inspect the repository

Before editing, inspect the repository structure and identify:

- dataset loading scripts
- preprocessing scripts
- PRISM model implementation
- graph construction / MIS selection code
- MAML-LSTM code
- baseline model code
- experiment runner scripts
- result table generation scripts
- plotting scripts
- manuscript-support or paper-output scripts

Create or update a short developer note at:

```text
outputs/revision_audit/repo_inspection_summary.md
```

It should list the files modified and the role of each file.

---

## 2. Create a reproducible output structure

All revised outputs must be saved under:

```text
outputs/scie_revision/
```

Create these subfolders:

```text
outputs/scie_revision/tables/
outputs/scie_revision/figures/
outputs/scie_revision/reports/
outputs/scie_revision/logs/
outputs/scie_revision/predictions/
outputs/scie_revision/audits/
outputs/scie_revision/configs/
```

All generated files must be deterministic given the same random seeds.

Use seeds:

```python
SEEDS = [11, 17, 23]
```

Every experiment must save:

- configuration JSON
- predictions CSV
- metrics CSV
- runtime log
- reproducibility metadata

---

## 3. Fix the main claim problem: PRISM must be evaluated honestly

Current issue: Table 5 shows Prophet and Persistence outperforming PRISM on aggregate MAE/RMSE. Therefore the code and generated report must not claim “PRISM is superior to all baselines.”

Implement a result-interpretation module that automatically generates a truthful textual summary based on the actual result table.

Create:

```text
src/reporting/claim_generator.py
```

The generated summary must follow this rule:

### If PRISM beats all baselines on MAE and RMSE:

State:

```text
PRISM achieves the lowest aggregate MAE and RMSE among all evaluated baselines under the chronological protocol.
```

### If Prophet or Persistence beats PRISM:

State:

```text
PRISM outperforms several classical, tree-based, and recurrent neural baselines; however, Prophet and/or Persistence remain highly competitive under short-horizon MAE/RMSE, indicating strong local continuity in memecoin prices.
```

### Always add:

```text
The main contribution of PRISM is therefore evaluated not only through point-forecasting error, but also through risk-aware diversification, robustness, ablation behavior, and statistical testing.
```

Save this generated text to:

```text
outputs/scie_revision/reports/paper_claim_summary.md
```

Also save a JSON version:

```text
outputs/scie_revision/reports/paper_claim_summary.json
```

Fields:

```json
{
  "prism_best_mae": true/false,
  "prism_best_rmse": true/false,
  "baselines_better_than_prism_mae": [],
  "baselines_better_than_prism_rmse": [],
  "recommended_claim": "..."
}
```

---

## 4. Remove fallback/surrogate baseline language by implementing real neural baselines

Current issue: GRU and LSTM are marked as fallback/surrogate. This is unacceptable for journal review.

Implement real trainable baselines using the same chronological split, same feature set rules, and same target construction.

Minimum required neural baselines:

1. LSTM
2. GRU
3. BiLSTM
4. TCN
5. N-BEATS or N-BEATS-lite
6. PatchTST-lite or TransformerEncoder baseline

Classical and machine-learning baselines:

1. Persistence
2. ARIMA
3. Prophet, if available in environment; otherwise use a documented Prophet-compatible fallback only if explicitly unavailable, and mark it separately
4. Random Forest
5. XGBoost, if installed; otherwise HistGradientBoosting or GradientBoosting, but record the fallback clearly

Important rules:

- Do not call real LSTM/GRU/BiLSTM rows fallback.
- Each neural baseline must actually train.
- Use early stopping.
- Use the same train/test split as PRISM.
- Use the same lookback and horizon settings.
- Use the same random seeds.
- Save predictions for each token, seed, and model.
- Save model-level metrics aggregated over tokens and seeds.

Expected files:

```text
src/models/baselines/lstm.py
src/models/baselines/gru.py
src/models/baselines/bilstm.py
src/models/baselines/tcn.py
src/models/baselines/nbeats_lite.py
src/models/baselines/transformer_encoder.py
src/models/baselines/persistence.py
src/models/baselines/arima.py
src/models/baselines/prophet_model.py
src/models/baselines/random_forest.py
src/models/baselines/xgboost_model.py
```

If the repository already has equivalent files, update them instead of duplicating.

---

## 5. Add fair hyperparameter tuning protocol

Implement a common tuning protocol for all models.

Create:

```text
src/experiments/hyperparameter_registry.py
```

Include search spaces:

### PRISM

- lookback: [7, 14, 21]
- horizon: [3]
- graph_threshold: [0.3, 0.5, 0.7]
- volatility_window: [7, 14, 21]
- sentiment_weight: [0.25, 0.50, 0.75, 1.00]
- MAML inner steps: [1, 3, 5]
- inner learning rate: [0.001, 0.005, 0.01]
- hidden size: [32, 64, 128]

### Neural baselines

- hidden size: [32, 64, 128]
- layers: [1, 2]
- dropout: [0.0, 0.2]
- learning rate: [0.001, 0.0005]
- batch size: [32, 64]

### Tree baselines

- n_estimators: [200, 500]
- max_depth: [3, 5, 7, None]
- learning_rate for boosting: [0.01, 0.05, 0.1]

Use a small validation split inside the training period only. Never use the test period for hyperparameter tuning.

Save best hyperparameters to:

```text
outputs/scie_revision/tables/table_hyperparameter_best_by_model.csv
outputs/scie_revision/tables/table_hyperparameter_best_by_model.md
outputs/scie_revision/tables/table_hyperparameter_best_by_model.tex
```

---

## 6. Add leakage audit section and automated leakage checks

Current issue: Ablation improvements look too large, so reviewers may suspect leakage.

Create:

```text
src/audits/leakage_audit.py
```

The audit must verify and report:

1. All features at time t use only information available at or before t.
2. The 3-day-ahead target is never used in feature scaling.
3. Scaling parameters are fitted on training data only.
4. Rolling volatility uses only past/current observations, never future rows.
5. Rolling sentiment aggregation uses only past/current sentiment values.
6. Graph construction is performed using training-period data only for each split.
7. MIS selection is computed using training-period graph only.
8. MAML support/query sets are drawn from the training period during training.
9. Validation data are not used in meta-training updates except for tuning/early stopping.
10. Test data are never used for feature selection, token selection, graph construction, scaling, tuning, or early stopping.
11. All ablation variants use the same token subset, same split, same horizon, and same target transformation.
12. Target normalization, if used, is inverted consistently for all models and variants.

Generate:

```text
outputs/scie_revision/audits/leakage_audit_report.md
outputs/scie_revision/audits/leakage_audit_report.json
```

The markdown report must include a table:

| Audit item | Status | Evidence file/function | Notes |

Statuses:

- PASS
- WARNING
- FAIL

If any item fails, the experiment runner must stop unless a flag `--allow_failed_audit` is passed.

---

## 7. Add dataset transparency report / dataset card

Current issue: “structured memecoin workbook” is not enough for SCIE review.

Create:

```text
src/reporting/dataset_card.py
```

Generate:

```text
outputs/scie_revision/reports/dataset_card.md
outputs/scie_revision/reports/dataset_card.json
outputs/scie_revision/tables/table_dataset_transparency.csv
outputs/scie_revision/tables/table_dataset_transparency.md
outputs/scie_revision/tables/table_dataset_transparency.tex
```

The dataset card must report, if available:

- exact source file name
- data source name or provider
- price source
- market cap source
- volume source
- sentiment source
- social platform source, e.g., Twitter/X, Reddit, Telegram, if available
- date range start
- date range end
- sampling frequency
- number of candidate token sheets
- number of eligible sheets
- number of modelled tokens
- processed aligned observations
- mean token history length
- median token history length
- missing-value policy
- target construction
- train/validation/test split policy
- token-selection rule
- possible survivorship-bias risk
- limitations of using top tokens by row count

If any field is unavailable, do not invent it. Mark it as:

```text
Not available in current workbook metadata
```

Also generate a short paper-ready paragraph:

```text
outputs/scie_revision/reports/dataset_transparency_paragraph.md
```

The paragraph must honestly state any unavailable metadata.

---

## 8. Add survivorship-bias and token-selection-bias analysis

Current issue: selecting the top 200 tokens by row count may favor more stable/surviving tokens.

Implement:

```text
src/audits/token_selection_bias.py
```

Compute and save:

- distribution of row counts across all candidate tokens
- row counts of eligible tokens
- row counts of selected 200 tokens
- whether selected tokens have longer histories than non-selected tokens
- missingness rate by token before filtering
- number of dead/short-history tokens excluded, if identifiable
- warning about survivorship bias if dead/short-lived tokens cannot be identified

Outputs:

```text
outputs/scie_revision/tables/table_token_selection_bias.csv
outputs/scie_revision/tables/table_token_selection_bias.md
outputs/scie_revision/figures/fig_token_history_distribution.png
outputs/scie_revision/reports/token_selection_bias_report.md
```

Paper-ready conclusion should be saved to:

```text
outputs/scie_revision/reports/token_selection_bias_paragraph.md
```

---

## 9. Strengthen novelty: formal graph edge score, MIS explanation, and complexity analysis

Current issue: reviewers may see PRISM as stacking known components.

Implement and document exact novelty in three places:

### 9.1 Graph edge score

Create or update:

```text
src/graph/risk_aware_graph.py
```

Ensure the edge score is explicitly computed as a function of:

- return correlation or trajectory similarity
- sentiment similarity
- rolling volatility similarity or penalty
- liquidity/risk penalty if available
- confidence threshold

Save graph statistics:

```text
outputs/scie_revision/tables/table_graph_statistics.csv
outputs/scie_revision/tables/table_graph_statistics.md
outputs/scie_revision/tables/table_graph_statistics.tex
```

Include:

- number of nodes
- number of edges
- graph density
- average degree
- connected components
- average pairwise correlation before MIS
- average pairwise correlation after MIS
- redundancy reduction percentage

### 9.2 MIS explanation

Create:

```text
src/reporting/mis_explanation.py
```

Generate:

```text
outputs/scie_revision/reports/mis_theoretical_explanation.md
```

Explain formally:

- tokens are nodes
- edges represent high similarity/redundancy/risk-coupling
- an independent set selects tokens with no direct high-redundancy edge between them
- maximal independent set gives a diversified support subset
- this reduces redundant token exposure under the graph definition

Do not overclaim optimal portfolio selection unless maximum independent set is exactly solved. If using greedy MIS, call it “greedy maximal independent set”, not “maximum independent set”.

### 9.3 Complexity analysis

Create:

```text
src/reporting/complexity_analysis.py
```

Generate:

```text
outputs/scie_revision/reports/complexity_analysis.md
outputs/scie_revision/tables/table_complexity_analysis.csv
outputs/scie_revision/tables/table_complexity_analysis.md
outputs/scie_revision/tables/table_complexity_analysis.tex
```

Include asymptotic complexity for:

- feature preparation
- graph construction
- pairwise similarity calculation
- MIS selection
- LSTM training
- MAML inner loop
- MAML outer loop
- full PRISM training/inference

Use variables:

- N = number of tokens
- T = time length
- F = number of features
- E = graph edges
- L = lookback length
- H = hidden size
- S = number of seeds
- K = MAML inner steps
- B = batch size

---

## 10. Add pooled LSTM vs MAML necessity experiment

Current issue: reviewers may ask why MAML is necessary instead of pooled LSTM.

Add experiment variants:

1. Pooled LSTM without MAML
2. Token-specific fine-tuned LSTM
3. MAML-LSTM / PRISM full

Use identical features, splits, tokens, horizon, and seeds.

Outputs:

```text
outputs/scie_revision/tables/table_maml_necessity.csv
outputs/scie_revision/tables/table_maml_necessity.md
outputs/scie_revision/tables/table_maml_necessity.tex
outputs/scie_revision/figures/fig_maml_necessity.png
outputs/scie_revision/reports/maml_necessity_paragraph.md
```

The paragraph must state whether MAML actually helps. If MAML does not outperform pooled LSTM, do not hide it. State that MAML improves adaptability only under certain token heterogeneity settings.

---

## 11. Re-run and verify ablation under strict no-leakage conditions

Current issue: current ablation improvement is too large and may look suspicious.

Recompute ablation under the leakage-audited pipeline.

Ablation variants:

- V0: price-only LSTM
- V1: V0 + sentiment fusion
- V2: V1 + risk-aware graph + MIS support diversification
- V3: V2 + MAML adaptation / full PRISM

All variants must use:

- same selected tokens
- same chronological split
- same target
- same scaling policy
- same seed list
- same train/validation/test separation
- same inverse transformation logic

Save:

```text
outputs/scie_revision/tables/table_ablation_strict.csv
outputs/scie_revision/tables/table_ablation_strict.md
outputs/scie_revision/tables/table_ablation_strict.tex
outputs/scie_revision/figures/fig_ablation_strict.png
outputs/scie_revision/reports/ablation_interpretation_strict.md
```

Table must include:

- variant
- modules included
- MAE mean
- MAE std
- RMSE mean
- RMSE std
- sMAPE mean
- MASE mean
- directional accuracy mean
- number of tokens
- number of seeds
- significance vs previous variant
- Holm-corrected p-value

If the improvement is still very large, the report must include a leakage-audit note and a scale-consistency note.

---

## 12. Add risk-aware and portfolio-diversification evaluation

Current issue: PRISM is “Portfolio Risk-aware” but current results emphasize only raw forecasting error. If Prophet/Persistence beat PRISM on MAE, PRISM must be evaluated on risk-aware and diversification metrics.

Implement:

```text
src/metrics/risk_metrics.py
src/metrics/diversification_metrics.py
src/experiments/portfolio_evaluation.py
```

Compute these metrics:

### Forecasting metrics

- MAE
- RMSE
- sMAPE
- MASE
- directional accuracy

### Trading/economic proxy metrics

- cumulative return proxy
- mean return proxy
- Sharpe ratio proxy
- Sortino ratio proxy
- max drawdown proxy
- hit rate
- turnover proxy if selections change over time

### Risk metrics

- VaR at 95%
- CVaR at 95%
- downside deviation
- volatility of returns

### Diversification metrics

- average pairwise correlation among selected tokens
- average pairwise correlation among full eligible universe
- redundancy reduction percentage
- graph density
- selected-token graph edge count
- selected-token average degree
- rank stability across seeds
- rank stability across train/lookback settings

Outputs:

```text
outputs/scie_revision/tables/table_main_forecasting_metrics.csv
outputs/scie_revision/tables/table_main_forecasting_metrics.md
outputs/scie_revision/tables/table_main_forecasting_metrics.tex

outputs/scie_revision/tables/table_risk_adjusted_metrics.csv
outputs/scie_revision/tables/table_risk_adjusted_metrics.md
outputs/scie_revision/tables/table_risk_adjusted_metrics.tex

outputs/scie_revision/tables/table_diversification_metrics.csv
outputs/scie_revision/tables/table_diversification_metrics.md
outputs/scie_revision/tables/table_diversification_metrics.tex

outputs/scie_revision/figures/fig_forecasting_comparison.png
outputs/scie_revision/figures/fig_risk_adjusted_comparison.png
outputs/scie_revision/figures/fig_diversification_effect.png

outputs/scie_revision/reports/risk_diversification_interpretation.md
```

The report must explicitly answer:

- Does PRISM beat Prophet/Persistence on MAE/RMSE?
- Does PRISM improve risk-adjusted metrics?
- Does MIS reduce selected-token redundancy?
- Does PRISM improve stability across seeds/splits?

---

## 13. Add modern baseline comparison table

Create a final main comparison table that includes:

- Persistence
- ARIMA
- Prophet
- Random Forest
- XGBoost or fallback gradient boosting
- LSTM
- GRU
- BiLSTM
- TCN
- N-BEATS-lite
- TransformerEncoder or PatchTST-lite
- PRISM

The table must include:

- MAE mean ± std
- RMSE mean ± std
- sMAPE mean ± std
- MASE mean ± std
- directional accuracy mean ± std
- rank by MAE
- rank by RMSE
- rank by risk-adjusted score, if available
- Wilcoxon/Holm result vs PRISM
- whether PRISM is significantly better/worse/not different

Outputs:

```text
outputs/scie_revision/tables/table5_revised_main_comparison.csv
outputs/scie_revision/tables/table5_revised_main_comparison.md
outputs/scie_revision/tables/table5_revised_main_comparison.tex
```

---

## 14. Add statistical testing module

Create:

```text
src/stats/significance_tests.py
```

Implement:

- paired Wilcoxon signed-rank test
- Holm correction
- Cliff's delta or rank-biserial effect size
- significance table vs PRISM
- significance table for ablation steps

Outputs:

```text
outputs/scie_revision/tables/table_significance_vs_prism.csv
outputs/scie_revision/tables/table_significance_vs_prism.md
outputs/scie_revision/tables/table_significance_vs_prism.tex

outputs/scie_revision/tables/table_ablation_significance.csv
outputs/scie_revision/tables/table_ablation_significance.md
outputs/scie_revision/tables/table_ablation_significance.tex
```

Do not use paired t-test unless normality is checked and justified. Prefer Wilcoxon for skewed token-level results.

---

## 15. Add robustness and sensitivity analysis

Expand robustness over:

- train ratios: [0.7, 0.8, 0.9]
- lookback windows: [7, 14, 21]
- graph thresholds: [0.3, 0.5, 0.7]
- sentiment weights: [0.25, 0.50, 0.75, 1.00]
- seeds: [11, 17, 23]

Outputs:

```text
outputs/scie_revision/tables/table_robustness_grid.csv
outputs/scie_revision/tables/table_robustness_grid.md
outputs/scie_revision/tables/table_robustness_grid.tex
outputs/scie_revision/figures/fig_robustness_train_lookback_heatmap.png
outputs/scie_revision/figures/fig_sensitivity_graph_threshold.png
outputs/scie_revision/figures/fig_sensitivity_sentiment_weight.png
outputs/scie_revision/reports/robustness_interpretation.md
```

---

## 16. Add code/data availability statement generator

Create:

```text
src/reporting/availability_statement.py
```

Generate:

```text
outputs/scie_revision/reports/code_data_availability_statement.md
```

The generated statement must include:

- whether code can be shared
- whether dataset can be shared
- if dataset cannot be public, state why
- exact reproducibility command
- expected outputs directory
- environment file path

If repository has no license, suggest adding one but do not invent it.

---

## 17. Add manuscript-ready text snippets

Generate paper-ready paragraphs for these sections:

```text
outputs/scie_revision/reports/manuscript_insert_dataset_transparency.md
outputs/scie_revision/reports/manuscript_insert_leakage_audit.md
outputs/scie_revision/reports/manuscript_insert_baseline_protocol.md
outputs/scie_revision/reports/manuscript_insert_ablation_strict.md
outputs/scie_revision/reports/manuscript_insert_risk_diversification.md
outputs/scie_revision/reports/manuscript_insert_claim_reframing.md
outputs/scie_revision/reports/manuscript_insert_limitations.md
outputs/scie_revision/reports/manuscript_insert_future_work.md
```

The claim-reframing paragraph must avoid overclaiming. It should use this style if Prophet/Persistence beat PRISM:

```text
PRISM outperforms several classical, tree-based, and neural baselines, but Prophet and Persistence remain strong short-horizon competitors. This pattern suggests that memecoin prices contain strong local continuity over the evaluated horizon. Therefore, PRISM is interpreted not only as a point-forecasting model, but as a risk-aware and auditable framework that combines multimodal signal fusion, diversified token selection, and adaptive sequence learning.
```

---

## 18. Add grammar/style cleanup script for generated text only

Create:

```text
src/reporting/style_checks.py
```

This should scan generated manuscript snippets and flag phrases such as:

- “has been projected”
- “has been anticipated”
- “has been introduced” when used incorrectly
- “statistical implication has been abridged”
- “existing forecasting research often treat”
- “baseline characteristics are used to analyze are shown”
- “persistent gaps such as existing approaches”

Generate:

```text
outputs/scie_revision/reports/style_check_report.md
```

Do not automatically rewrite the whole manuscript unless a script for manuscript editing already exists. Only generate warnings and suggested replacements.

---

## 19. Master experiment runner

Create or update:

```text
run_scie_revision.py
```

CLI:

```bash
python run_scie_revision.py \
  --data path/to/memecoin_workbook.xlsx \
  --output outputs/scie_revision \
  --seeds 11 17 23 \
  --main_train_ratio 0.8 \
  --horizon 3 \
  --lookback 14 \
  --run_baselines \
  --run_prism \
  --run_ablation \
  --run_robustness \
  --run_risk_metrics \
  --run_audits
```

The runner must:

1. Load dataset.
2. Generate dataset card.
3. Run leakage audit.
4. Run token-selection-bias audit.
5. Train/evaluate baselines.
6. Train/evaluate PRISM.
7. Run strict ablation.
8. Run MAML necessity experiment.
9. Run robustness/sensitivity grids.
10. Run risk/diversification evaluation.
11. Run significance tests.
12. Generate paper-ready tables, figures, and manuscript snippets.
13. Generate final revision summary.

Final summary path:

```text
outputs/scie_revision/reports/final_revision_summary.md
```

This summary must list:

- all generated tables
- all generated figures
- whether PRISM beats Prophet/Persistence
- whether PRISM improves risk/diversification metrics
- whether leakage audit passed
- recommended manuscript claim
- warnings/limitations

---

## 20. Output tables must be paper-ready

For every table, save three versions:

- CSV
- Markdown
- LaTeX

Formatting rules:

- Use mean ± std format for paper tables.
- Include number of tokens and seeds where relevant.
- Do not round internally; round only for display.
- Use 4 decimal places for MAE/RMSE/sMAPE/MASE unless scientific notation is necessary.
- Include clear captions in Markdown and LaTeX outputs.

---

## 21. Output figures must be paper-ready

For every figure:

- save PNG at 300 dpi
- include readable axis labels
- include units where applicable
- include legend
- include title only if useful; otherwise rely on caption
- use consistent font size
- avoid clutter

Save figure captions to:

```text
outputs/scie_revision/reports/figure_captions.md
```

---

## 22. Acceptance criteria

The implementation is complete only if these files exist after running `run_scie_revision.py`:

```text
outputs/scie_revision/reports/dataset_card.md
outputs/scie_revision/audits/leakage_audit_report.md
outputs/scie_revision/reports/token_selection_bias_report.md
outputs/scie_revision/tables/table5_revised_main_comparison.csv
outputs/scie_revision/tables/table_ablation_strict.csv
outputs/scie_revision/tables/table_risk_adjusted_metrics.csv
outputs/scie_revision/tables/table_diversification_metrics.csv
outputs/scie_revision/tables/table_significance_vs_prism.csv
outputs/scie_revision/tables/table_robustness_grid.csv
outputs/scie_revision/reports/paper_claim_summary.md
outputs/scie_revision/reports/final_revision_summary.md
outputs/scie_revision/reports/manuscript_insert_claim_reframing.md
outputs/scie_revision/reports/code_data_availability_statement.md
```

Also ensure there is no occurrence of the phrase:

```text
fallback GRU
fallback LSTM
surrogate GRU
surrogate LSTM
```

inside final paper-ready output tables, unless a dependency genuinely failed and the final summary clearly marks the result as not comparable.

---

## 23. Do not overclaim

The final generated report must avoid these statements unless supported by actual results:

- “PRISM outperforms all baselines.”
- “PRISM is the best forecasting model.”
- “PRISM achieves superior accuracy over Prophet and Persistence.”
- “MAML is always necessary.”
- “MIS provides optimal diversification” if using greedy maximal independent set.

Use safer claims when appropriate:

- “PRISM outperforms several classical, tree-based, and neural baselines.”
- “Prophet and Persistence remain strong short-horizon competitors.”
- “PRISM provides an auditable risk-aware framework for multimodal forecasting and diversified token selection.”
- “MIS-based selection reduces graph-defined redundancy among selected tokens.”
- “MAML improves token-specific adaptation under the evaluated configuration,” only if supported.

---

## 24. Suggested revised title and abstract support

Generate a file:

```text
outputs/scie_revision/reports/title_and_abstract_recommendation.md
```

Include this safer title:

```text
PRISM: A Risk-Aware Graph and Meta-Learning Framework for Multimodal Forecasting and Diversified Selection of Memecoin Assets
```

Generate an honest abstract result sentence automatically based on results.

If Prophet/Persistence beat PRISM:

```text
Experimental results show that PRISM significantly outperforms ARIMA, Random Forest, XGBoost, and recurrent neural baselines, while Prophet and Persistence remain strong short-horizon competitors. These findings suggest that memecoin forecasting requires not only predictive accuracy but also risk-aware diversification, adaptive learning, and transparent evaluation.
```

If PRISM beats all baselines:

```text
Experimental results show that PRISM achieves the strongest aggregate forecasting performance among the evaluated baselines while also improving risk-aware diversification and adaptive token-level modelling.
```

---

## 25. Final deliverable

When done, provide:

1. List of modified files.
2. Command to reproduce everything.
3. Location of main paper-ready tables.
4. Location of main paper-ready figures.
5. Whether leakage audit passed.
6. Whether PRISM beats Prophet and Persistence.
7. Recommended final claim for the manuscript.

Do not stop after partial implementation unless a missing dependency or missing dataset prevents completion. If that happens, clearly document the blocker and create all possible reports from available data.
