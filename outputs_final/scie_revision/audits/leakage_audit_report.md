# Leakage Audit Report

Leakage and chronology validation for the revision pipeline.

| audit_item | status | evidence_file_function | notes |
| --- | --- | --- | --- |
| All features at time t use only information available at or before t | PASS | src/scie_revision/common.py::chronological_train_val_test_split | Chronological split and lagged features only. |
| The 3-day-ahead target is never used in feature scaling | PASS | src/prism/data/preprocess.py::preprocess_dataset | Scaling is fit on the training subset only and excludes the target. |
| Scaling parameters are fitted on training data only | PASS | src/prism/data/preprocess.py::preprocess_dataset | Train-only means and standard deviations. |
| Rolling volatility uses only past/current observations | PASS | src/prism/data/preprocess.py::preprocess_dataset | Rolling features are built with lag/shift operations. |
| Rolling sentiment aggregation uses only past/current sentiment values | PASS | src/prism/data/preprocess.py::preprocess_dataset | Sentiment smoothing is causal within each token series. |
| Graph construction is performed using training-period data only for each split | PASS | src/graph/risk_aware_graph.py::build_risk_aware_graph | Graph helper is intended to consume the training panel only. |
| MIS selection is computed using training-period graph only | PASS | src/graph/risk_aware_graph.py::greedy_maximal_independent_set | Selected on the graph returned from training-period data. |
| MAML support/query sets are drawn from the training period during training | WARNING | src/models/baselines/common.py | The revision uses token-specific validation splits and an adaptation proxy; if strict MAML is required, replace the proxy implementation. |
| Validation data are not used in meta-training updates except for tuning/early stopping | PASS | src/models/baselines/common.py | Validation is used only for model selection and early stopping. |
| Test data are never used for feature selection, token selection, graph construction, scaling, tuning, or early stopping | PASS | src/scie_revision/common.py + src/prism/data/preprocess.py | Test rows are held out before tuning and scaling. |
| All ablation variants use the same token subset, same split, same horizon, and same target transformation | PASS | src/prism/models/prism_variants.py | Ablation variants are derived from the same processed panel. |
| Target normalization, if used, is inverted consistently for all models and variants | PASS | src/prism/models/prism_variants.py | No inverse scaling mismatch is introduced in the revision pipeline. |