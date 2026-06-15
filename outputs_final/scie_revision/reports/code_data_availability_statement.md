# Code and Data Availability Statement

- Code can be shared: yes, this repository contains the full revision pipeline and manuscript-support scripts.
- Dataset can be shared: not fully public from the workbook metadata alone.
- Dataset limitation: the upstream provider details are not fully documented in the workbook metadata, so the repository only asserts the local workbook file used for processing.
- Reproducibility command: python run_scie_revision.py --data C:/Users/91943/Desktop/Sanjan/Research/Memecoin/data/raw/Output_database.xlsx --output outputs/scie_revision --seeds 11 17 23 --main_train_ratio 0.8 --horizon 3 --lookback 14 --run_baselines --run_prism --run_ablation --run_robustness --run_risk_metrics --run_audits
- Expected outputs directory: outputs/scie_revision
- Environment file path: requirements.txt
- License present: no, consider adding one