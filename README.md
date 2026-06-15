# PRISM Revision Pipeline

This repository contains a reproducible, manuscript-support experimentation pipeline for PRISM resubmission work.

## Goals

- Audit workbook-level data quality and schema consistency.
- Reconcile token universe inclusion/exclusion with explicit reasons.
- Build leakage-safe chronological modeling datasets.
- Run baseline and PRISM ablation experiments.
- Run robustness and significance analyses.
- Export manuscript-facing tables, figures, and revision reports.

## Expected Input Files

Place raw files in one of these locations:

- `data/raw/Output_database.xlsx` (preferred)
- `Output_database.xlsx` (fallback)

Optional manuscript/comment files can stay in project root.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
```

## Run Step-by-Step

```bash
python scripts/audit_data.py
python scripts/build_universe.py
python scripts/preprocess_data.py
python scripts/train_baselines.py
python scripts/train_prism.py
python scripts/run_ablation.py
python scripts/run_robustness.py
python scripts/run_significance_tests.py
python scripts/export_paper_tables.py
python scripts/generate_paper_figures.py
python scripts/generate_revision_report.py
```

## Run End-to-End

```bash
python run_scie_revision.py \
  --data data/raw/Output_database.xlsx \
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

## Validate Generated Artifacts

```bash
python scripts/validate_pipeline_outputs.py
```

## Key Outputs

- `outputs/scie_revision/audits/workbook_inventory.csv`
- `outputs/scie_revision/audits/schema_summary.json`
- `outputs/scie_revision/audits/leakage_audit_report.md`
- `outputs/scie_revision/manifests/token_universe_manifest.csv`
- `data/processed/modeling_dataset.parquet`
- `outputs/scie_revision/results/*.csv`
- `outputs/scie_revision/tables/*.csv`
- `outputs/scie_revision/figures/*.png`
- `outputs/scie_revision/reports/*.md`
- `outputs/scie_revision/logs/run_manifest.json`

## Reproducibility Notes

- All split logic is chronological.
- Scaling uses training-split statistics only.
- Seeds are configurable under `configs/model/`.
- Fallback model behavior is logged in output tables (`note` columns) rather than hidden.

## Test

```bash
pytest -q
```
