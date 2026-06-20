# Repository Inspection Summary

- Workbook: C:\Users\91943\Desktop\Sanjan\Research\Memecoin\data\raw\Output_database.xlsx
- Universe sheets: 1273
- Eligible tokens: 881
- Processed tokens: 881
- Max tokens for modeling: 200
- Modified files:
  - run_scie_revision.py: orchestrates the revision pipeline.
  - src/scie_revision/common.py: shared revision output helpers.
  - src/models/baselines/*.py: trainable PyTorch sequence model baselines.
  - src/models/baselines/sequence_models.py: shared PyTorch LSTM/GRU/BiLSTM/TCN implementations.
  - src/metrics/*.py: risk and diversification metrics.
  - src/audits/*.py: leakage, token-selection, and ablation audits.
  - src/reporting/*.py: claim, dataset, MIS, complexity, availability, style reports.
  - src/graph/risk_aware_graph.py: multi-factor graph construction with quantile thresholding.
  - scripts/audit_scie_outputs.py: output quality gate.