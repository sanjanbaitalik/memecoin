# Prompt for GitHub Copilot: Fix PRISM revision pipeline after failed first pass

You are working in a research codebase for the PRISM memecoin forecasting paper revision.

## Context
The first implementation pass produced partial artifacts but the main pipeline is still broken.
The generated outputs show:
- workbook audit succeeded
- schema detection succeeded across 1273 sheets
- token universe manifest succeeded
- table_1_dataset_composition.csv exists with 894 included tokens
- table_4_hyperparameters.csv exists
- BUT processed dataset creation failed
- baseline, ablation, robustness, and significance outputs are empty
- run_manifest.json reports dataset_rows=0, baseline_rows=0, ablation_rows=0, robustness_rows=0, significance_rows=0
- pipeline_todo.md reports: `EmptyDataError: No columns to parse from file`
- preprocessing_report.md reports Dataset rows: 0, Tokens: 0
- workbook inventory and table_1 have `first_date` and `last_date` as NaN even though the source Excel workbook contains valid datetime values in the `date` column

This means the issue is NOT the raw Excel file itself. The issue is in the ingestion / preprocessing / downstream file contract logic.

## Primary objective
Make the revision pipeline run end-to-end and generate NON-EMPTY, publication-usable artifacts for:
1. processed dataset
2. baseline comparison table
3. sequential ablation table
4. robustness table
5. significance test table
6. descriptive statistics tables
7. reproducibility manifests/reports

## Required behavior
Implement the fixes below directly in code. Do not leave TODOs unless absolutely unavoidable.

### 1) Fix Excel ingestion and date handling
The raw workbook has many sheets. Each sheet represents one token time series.
Required columns include at least:
- token
- ticker
- date
- price
- volume

Tasks:
- Create a robust workbook loader that iterates through all sheets once and returns standardized per-sheet DataFrames.
- Preserve datetime information in `date` using `pd.to_datetime(..., errors="coerce")`.
- If `date` fails to parse, log the sheet name and the original dtype / sample values.
- If a `time` column exists, safely merge it with `date` only when needed; do not destroy valid parsed dates.
- Ensure `first_date` and `last_date` are populated in audit outputs whenever any valid dates exist.
- Add validation assertions so the pipeline fails fast with a clear message if >90% of sheets lose their date column after parsing.

### 2) Fix data contracts between steps
The current pipeline appears to write empty CSVs that downstream steps then try to read, causing `EmptyDataError`.

Tasks:
- Before reading any intermediate CSV/Parquet, check:
  - file exists
  - file size > 0
  - parsed DataFrame has at least 1 row and 1 column
- Replace silent empty-file propagation with explicit exceptions carrying:
  - step name
  - file path
  - expected schema
  - upstream dependency
- Add helper `safe_read_csv(path, required_columns=None)`.
- Add helper `safe_write_csv(df, path)` that refuses to write completely empty DataFrames unless explicitly allowed.
- Update the run manifest to record step status: `success`, `failed`, `skipped_due_to_upstream_failure`.

### 3) Build a canonical processed dataset
Create one canonical long-format processed dataset at something like:
- outputs/processed/processed_panel.csv or .parquet

Required columns should include at least:
- sheet_name
- token
- ticker
- date
- price
- volume
- log_return or return
- optional sentiment features if available
- split (train/test where relevant)

Rules:
- preserve chronological order within token
- compute train/test split chronologically using configurable `train_ratio`
- support lookback window and forecast horizon settings
- do not leak future information
- include per-token row counts before and after filtering

### 4) Universe filtering must be explicit and reproducible
Current manifest says 894 included and 379 excluded.

Tasks:
- Create a structured exclusion log with one row per sheet and boolean flags for every filter:
  - too_few_rows
  - low_price_coverage
  - invalid_date
  - missing_required_columns
  - other_reason
- Include final inclusion decision and human-readable reason string.
- Ensure included token count in manifest matches actual processed dataset token count.

### 5) Baseline experiments
Implement reproducible baselines on identical chronological splits.
At minimum include:
- persistence / last value
- ARIMA or a simple autoregressive baseline
- RandomForest or XGBoost on lagged features
- GRU
- plain LSTM (no meta-learning)
- full PRISM

Requirements:
- identical forecast horizon across methods
- per-token metrics, not only aggregate metrics
- save outputs to machine-readable tables
- baseline table must not be placeholder text

Suggested output files:
- outputs/results/per_token_baseline_metrics.csv
- outputs/tables/table_baseline_comparison.csv

### 6) Sequential ablation experiments
Implement a sequential ablation matching the reviewer request.
Target variants:
- V0: price-only sequence model
- V1: V0 + sentiment features
- V2: V1 + graph / MIS component
- V3: V2 + meta-learning (full PRISM)

Requirements:
- identical data split and evaluation protocol across all variants
- per-token metrics and aggregate metrics
- save a clean table with mean, std, median where appropriate

Suggested outputs:
- outputs/results/per_token_ablation_metrics.csv
- outputs/tables/table_ablation_sequential.csv

### 7) Robustness evaluation
Implement robustness checks requested by reviewers.
At minimum:
- multiple seeds for neural models
- multiple train ratios (e.g. 0.7, 0.8, 0.9)
- multiple lookback windows (e.g. 7, 14, 21)

Save:
- outputs/results/robustness_metrics.csv
- outputs/tables/table_robustness.csv

### 8) Statistical significance testing
Current significance report is empty.
Implement paired significance testing using per-token metric pairs.

Requirements:
- Wilcoxon signed-rank on paired per-token MAE at minimum
- Holm correction for multiple comparisons
- include effect direction and sample size
- gracefully skip impossible comparisons with explicit reason

Save:
- outputs/tables/significance_tests.csv
- outputs/reports/significance_report.md

### 9) Descriptive statistics for the manuscript
Create real non-empty descriptive tables from actual outputs.

Required tables:
- Table 1: dataset composition and coverage
- Table 2: summary statistics across tokens
- Table 3: top and bottom tokens by ROI or chosen outcome
- Table 4: hyperparameters

Requirements:
- if an aggregate depends on a missing upstream metric, fail that table generation clearly rather than writing an empty CSV
- include N used for each metric
- handle undefined MAPE/ROI carefully and document exclusions or winsorization

### 10) Reporting and manuscript-support exports
Generate markdown reports that are directly usable while rewriting the paper.
At minimum produce:
- data_audit_report.md
- preprocessing_report.md
- descriptive_stats_report.md
- experiment_registry.md
- significance_report.md
- figure_manifest.md

Each report must state:
- timestamp
- input file path(s)
- parameter settings
- record counts
- whether the step succeeded
- any exclusion or failure reasons

### 11) End-to-end validation tests
Add tests or validation scripts that run before the full expensive pipeline.

Create a lightweight validation script that checks:
- at least one sample sheet parses correctly
- dates are non-null after parsing for sample sheets
- processed dataset has >0 rows
- included token count matches manifest
- significance_tests.csv is non-empty when baseline metrics exist
- all output CSVs intended for manuscript tables are readable by pandas without EmptyDataError

### 12) CLI / execution requirements
Provide one reproducible entry point such as:
- `python -m prism_revision.run_revision_pipeline --input data/raw/Output_database.xlsx --outdir outputs/`

The command should:
- run the full pipeline
- exit non-zero on hard failure
- print a concise summary of counts and output file paths

## Important implementation rules
- Do not fake results.
- Do not write placeholder files when inputs are missing.
- Do not silently continue after critical upstream failures.
- Prefer Parquet for intermediate large tables and CSV for manuscript-facing outputs.
- Make logging explicit and human-readable.
- Keep functions modular and testable.

## Deliverables expected from you
1. The actual code changes.
2. A short summary of files added/modified.
3. The exact command to run.
4. A checklist showing which output artifacts are now generated successfully.
5. Any remaining blockers, with exact stack traces and file paths.

## Acceptance criteria
The task is only complete if all of the following are true:
- processed dataset exists and is non-empty
- run manifest shows dataset_rows > 0
- baseline_rows > 0
- ablation_rows > 0
- robustness_rows > 0
- significance_rows > 0
- table_1_dataset_composition.csv is non-empty and contains real dates
- table_2_summary_statistics.csv is non-empty
- table_3_top_bottom_tokens.csv is non-empty
- significance_tests.csv is readable and non-empty
- no manuscript-facing CSV is zero bytes

Work until the pipeline is fixed end-to-end.
