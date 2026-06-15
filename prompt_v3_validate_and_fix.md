# prompt.md — Third-pass validation and correction prompt for GitHub Copilot

You are working inside the memecoin forecasting repo for the PRISM manuscript revision.

Your job is **not** to add more reports. Your job is to make the existing outputs **scientifically credible, internally consistent, and manuscript-safe**.

The current pipeline now produces non-empty artifacts, but the outputs contain serious contradictions that would damage a journal submission. Fix those contradictions end to end.

---

## What already exists

The pipeline currently generates:
- `outputs/processed/processed_panel.csv`
- `outputs/tables/table_baseline_comparison.csv`
- `outputs/tables/table_ablation_sequential.csv`
- `outputs/tables/table_robustness.csv`
- `outputs/tables/significance_tests.csv`
- `outputs/tables/table_1_dataset_composition.csv`
- `outputs/tables/table_4_hyperparameters.csv`
- `outputs/reports/*.md`
- `outputs/logs/run_manifest.json`

This means the artifact skeleton exists. Do **not** redesign it. Fix the correctness and validity of the generated results.

---

## Critical problems that must be fixed

### 1. PRISM appears catastrophically worse than all baselines
Current output shows PRISM with enormous errors (for example MAE in the tens of thousands) while persistence/baselines are far lower.
That means one of the following is broken:
- target scaling / inverse scaling for PRISM only
- mismatch between prediction horizon target and reconstructed prediction values
- train/test leakage in baselines or incorrect evaluation for PRISM
- comparing normalized predictions against raw targets
- bad postprocessing for PRISM outputs
- metric computation bug for one branch only

You must trace the entire PRISM evaluation path and fix it.

### 2. Significance direction appears wrong
The significance table currently marks comparisons like `PRISM vs baseline` with `left_better` even though PRISM has much worse MAE.
That means the effect-direction logic is likely inverted or using the wrong comparison operator.
Fix this so:
- for error metrics, lower is better
- for directional accuracy / ROI, higher is better
- the written report and claim guardrail reflect the actual numeric results

### 3. Claim guardrail report is unsafe
The claim guardrail currently says things like “superior performance over traditional models — supported” even though the baseline table contradicts that.
Fix claim validation logic so manuscript claims are only marked supported when:
- the direction is correct,
- the comparison is statistically significant after correction,
- and the aggregate result actually favors PRISM.

### 4. GRU and LSTM outputs are identical
This strongly suggests:
- one model calls the other,
- cached predictions are reused incorrectly,
- model labels differ but underlying predictions are the same,
- or one implementation is stubbed/fallback-cloned.

Fix this so GRU and LSTM are genuinely separate implementations or explicitly remove one if only one recurrent baseline is actually implemented.

### 5. Prophet / XGBoost may be fallback implementations
Current notes indicate fallback behavior such as:
- Prophet fallback to persistence
- XGBoost fallback to gradient boosting

That is acceptable only if:
- it is explicitly declared in the experiment registry,
- the manuscript-facing table labels the actual implementation used,
- and unsupported baselines are either properly installed or renamed honestly.

Do **not** silently present fallback models as canonical baselines.

### 6. Only 200 tokens are modeled while preprocessing reports 881 tokens
The pipeline currently preprocesses a much larger universe but models only 200 tokens.
You must make this explicit and reproducible.
Either:
- model all eligible tokens, or
- keep the 200-token cap but document the exact deterministic selection rule.

Required:
- export `outputs/tables/modeling_subset_manifest.csv`
- include columns such as `sheet_name`, `selection_reason`, `eligible_rank` or equivalent
- update experiment registry and reports to explain the subset rule

### 7. Robustness table looks suspiciously duplicated
The results for lookback 14 and 21 are identical for some train ratios. That suggests one of:
- lookback parameter not actually wired into robustness runs
- cached output reused incorrectly
- reporting aggregation bug

Fix robustness so each config truly changes the underlying features/model input and produces independently computed outputs.

### 8. Stale or contradictory report text still exists
At least one report still references an earlier blocker like “modeling dataset is empty” even though the manifest says success.
Fix report generation so all markdown reports are regenerated from current pipeline state and cannot retain stale text.

### 9. Date parsing still fails for many sheets
There are still many date parse failures.
You must:
- preserve a diagnostic log,
- quantify how many sheets/tokens are excluded because of date issues,
- ensure failed-date sheets never silently leak into modeling,
- and make the dataset composition table clear about what is included vs excluded.

### 10. Manuscript-facing tables need publication-safe semantics
Every exported table intended for the paper must be:
- non-empty,
- numerically valid,
- consistent with reports,
- free of placeholder/supporting text,
- and interpretable without repo knowledge.

---

## Required implementation tasks

### A. Audit and repair metric computation
Inspect and fix all metric computation code.

Requirements:
- Centralize metrics in one module, e.g. `src/evaluation/metrics.py`
- Ensure the same target array and aligned prediction array are passed for every model
- Add assertions for equal length, matching indices, non-empty test sets, no all-NaN outputs
- Add explicit handling for scale/inverse-scale transformations
- For MAPE, guard against zero/near-zero denominators and export both raw MAPE and a safe variant such as sMAPE or epsilon-MAPE

Export:
- `outputs/audits/metric_validation_report.csv`
- `outputs/reports/metric_validation_report.md`

### B. Validate PRISM prediction scale
Instrument PRISM evaluation with per-token diagnostics.

Export a table:
- `outputs/audits/prism_prediction_diagnostics.csv`

Columns must include at minimum:
- `sheet_name`
- `n_test`
- `y_true_min`, `y_true_max`, `y_true_mean`
- `y_pred_min`, `y_pred_max`, `y_pred_mean`
- `used_scaler`
- `inverse_transform_applied`
- `contains_nan`
- `contains_inf`
- `suspected_scale_mismatch` (boolean)

If PRISM predictions are on normalized scale while targets are raw, fix that.

### C. Make baseline implementations honest and reproducible
For each baseline, export metadata describing the actual implementation used.

Create:
- `outputs/tables/baseline_implementation_registry.csv`

Columns:
- `model_label`
- `actual_backend`
- `fallback_used`
- `fallback_reason`
- `library_version`
- `random_seed`

If Prophet or XGBoost are unavailable, either install them or rename the baseline labels in tables to reflect the actual backend.

### D. Separate GRU and LSTM properly
Inspect their training/prediction code paths.
They must not reuse the same architecture or cached predictions unless explicitly intended.

Add a validation check:
- if two model families produce identical prediction vectors for all tokens, fail the pipeline with a clear error unless explicitly whitelisted.

### E. Fix significance and claim logic
Create one shared comparison-spec module that defines metric directionality.

Required metric directions:
- lower is better: MAE, RMSE, MAPE, sMAPE
- higher is better: directional_accuracy, ROI, Sharpe-like metrics

Recompute:
- `outputs/tables/significance_tests.csv`
- `outputs/reports/significance_report.md`
- `outputs/reports/claim_guardrail_report.md`

Guardrail requirements:
- “supported” only if the aggregate result favors PRISM **and** corrected significance agrees
- “partially supported” only if mixed evidence exists
- “requires rewording” if PRISM is not clearly better

### F. Make modeling subset transparent
If `max_tokens_for_modeling` stays at 200, make token selection deterministic and documented.

Create:
- `outputs/tables/modeling_subset_manifest.csv`

Columns:
- `sheet_name`
- `eligible`
- `selected_for_modeling`
- `selection_order`
- `selection_reason`
- `rows_available`
- `date_coverage`
- `sentiment_coverage`

Also update `experiment_registry.md` with the exact selection rule.

### G. Repair robustness experiments
Ensure robustness configs truly vary both preprocessing and downstream modeling as intended.

For each `(lookback, train_ratio, seed, token)` combination:
- rerun feature construction,
- rerun model fit or cached-artefact lookup keyed by full config,
- prevent accidental reuse across configs.

Create an audit:
- `outputs/audits/robustness_config_fingerprint.csv`

Each row should include a hash/fingerprint proving each config was independently materialized.

### H. Regenerate all markdown reports from current outputs only
No stale hardcoded text.
All reports must be generated from present pipeline state.
If a step succeeds, no leftover blocker message may remain.

### I. Add fail-fast manuscript gates
Before declaring pipeline success, check all paper-facing tables.
Fail the pipeline if any of the following happen:
- PRISM is presented as superior but aggregate metrics do not support it
- significance direction disagrees with aggregate means
- a table is empty
- a report contradicts the run manifest
- duplicate model outputs are detected (e.g. GRU == LSTM)
- baseline labels do not match actual backend

Create a summary gate report:
- `outputs/reports/manuscript_integrity_report.md`

---

## Required outputs after the fix

The rerun must produce all of these as non-empty and internally consistent:

### Core data
- `outputs/processed/processed_panel.csv`
- `outputs/tables/modeling_subset_manifest.csv`

### Main manuscript tables
- `outputs/tables/table_1_dataset_composition.csv`
- `outputs/tables/table_2_summary_statistics.csv`
- `outputs/tables/table_4_hyperparameters.csv`
- `outputs/tables/table_baseline_comparison.csv`
- `outputs/tables/table_ablation_sequential.csv`
- `outputs/tables/table_robustness.csv`
- `outputs/tables/significance_tests.csv`

### Diagnostics and audits
- `outputs/audits/date_parse_diagnostics.csv`
- `outputs/audits/metric_validation_report.csv`
- `outputs/audits/prism_prediction_diagnostics.csv`
- `outputs/audits/robustness_config_fingerprint.csv`
- `outputs/tables/baseline_implementation_registry.csv`

### Reports
- `outputs/reports/baseline_experiment_report.md`
- `outputs/reports/ablation_report.md`
- `outputs/reports/robustness_report.md`
- `outputs/reports/significance_report.md`
- `outputs/reports/claim_guardrail_report.md`
- `outputs/reports/manuscript_integrity_report.md`
- `outputs/reports/paper_patch_notes.md`

---

## Acceptance criteria for this coding pass

This pass is successful only if all of the following are true:

1. PRISM metrics are numerically credible and on the same scale as baselines.
2. Significance direction matches aggregate metric direction.
3. Claim guardrail does not mark unsupported superiority claims as supported.
4. GRU and LSTM are either genuinely distinct or one is removed/renamed honestly.
5. Any fallback baseline is clearly labeled as fallback in manuscript-facing artifacts.
6. The 200-token modeling subset is explicitly documented or removed by scaling up modeling.
7. Robustness results differ when configs differ, unless there is a defensible numeric coincidence.
8. No stale blocker text remains in reports.
9. All paper-facing tables are non-empty and consistent.
10. The final `manuscript_integrity_report.md` explicitly states whether the package is safe for manuscript rewriting.

---

## Important coding style requirements

- Do not silently swallow exceptions.
- Do not fabricate results.
- Do not default to “supported” claims when evidence is mixed.
- Prefer explicit warnings/errors over silent fallback.
- Every fallback must be surfaced in exported artifacts.
- Make every random process seeded and logged.

---

## Final deliverable

After code changes, rerun the full pipeline and ensure the output folder is ready for manual manuscript rewriting.
The code should optimize for **scientific validity and reviewer defensibility**, not just for making files appear.
