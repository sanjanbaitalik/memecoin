# Leakage Audit Report

Leakage and chronology validation for the revision pipeline.

| audit_item | status | evidence_file_function | notes |
| --- | --- | --- | --- |
| All features at time t use only information available at or before t | PASS | src/scie_revision/common.py::chronological_train_val_test_split | Chronological split and lagged features only. |
| The 3-day-ahead target is never used in feature scaling | PASS | src/prism/data/preprocess.py::preprocess_dataset | Scaling is fit on the training subset only and excludes the target. |
| Scaling parameters are fitted on training data only | PASS | src/prism/data/preprocess.py::preprocess_dataset | Train-only means and standard deviations. |
| Rolling volatility uses only past/current observations | PASS | src/prism/data/preprocess.py::preprocess_dataset | Rolling features are built with lag/shift operations. |
| Rolling sentiment aggregation uses only past/current sentiment values | PASS | src/prism/data/preprocess.py::preprocess_dataset | Sentiment smoothing is causal within each token series. |
| Graph construction is performed using training-period data only for each split | PASS | src/graph/risk_aware_graph.py::build_risk_aware_graph | Graph helper consumes the training panel only; features are causal. |
| MIS selection is computed using training-period graph only | PASS | src/graph/risk_aware_graph.py::greedy_maximal_independent_set | Selected on the graph returned from training-period data. |
| MAML support/query/test splits are chronological and non-overlapping | PASS | src/prism/models/prism_variants.py::_train_maml | First-order MAML uses support set from early training period, query set from later training period. Test set is fully held out. |
| Validation data are not used in meta-training updates except for tuning/early stopping | PASS | src/prism/models/prism_variants.py::_train_maml | Validation used only for model selection and early stopping of meta-learning. |
| Test data are never used for feature selection, token selection, graph construction, scaling, tuning, early stopping, or meta-training | PASS | src/scie_revision/common.py + src/prism/data/preprocess.py | Test rows are held out before tuning, scaling, and meta-training. |
| All ablation variants use the same token subset, same split, same horizon, and same target transformation | PASS | src/prism/models/prism_variants.py | Ablation variants are derived from the same processed panel with identical splits. |
| Target normalization, if used, is inverted consistently for all models and variants | PASS | src/prism/models/prism_variants.py | No inverse scaling mismatch is introduced in the revision pipeline. |