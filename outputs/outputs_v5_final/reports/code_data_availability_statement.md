# Code and Data Availability Statement

- **Code**: The full revision pipeline and manuscript-support scripts are available in this repository.
- **Dataset**: The dataset is derived from a workbook that is available upon reasonable request.
  The upstream data provider is not fully documented in the workbook metadata and requires author confirmation.
  An anonymized processed sample can be provided upon request for reproducibility.
- **Reproducibility**: Run `python run_v5_pipeline.py --data data/raw/Output_database.xlsx --output outputs_v5_final --seeds 11 17 23 --main_train_ratio 0.8 --horizon 3 --lookback 14 --max_tokens_for_modeling 200 --run_baselines --run_ablation --run_leakage_audit --run_output_quality_gate`
- **Environment**: See requirements.txt