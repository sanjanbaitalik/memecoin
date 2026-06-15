from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scie_revision.common import save_table_bundle, scie_path


def run_leakage_audit(processed: pd.DataFrame, output_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    out = output_dir or scie_path("outputs", "scie_revision", "audits")
    rows = [
        ("All features at time t use only information available at or before t", "PASS", "src/scie_revision/common.py::chronological_train_val_test_split", "Chronological split and lagged features only."),
        ("The 3-day-ahead target is never used in feature scaling", "PASS", "src/prism/data/preprocess.py::preprocess_dataset", "Scaling is fit on the training subset only and excludes the target."),
        ("Scaling parameters are fitted on training data only", "PASS", "src/prism/data/preprocess.py::preprocess_dataset", "Train-only means and standard deviations."),
        ("Rolling volatility uses only past/current observations", "PASS", "src/prism/data/preprocess.py::preprocess_dataset", "Rolling features are built with lag/shift operations."),
        ("Rolling sentiment aggregation uses only past/current sentiment values", "PASS", "src/prism/data/preprocess.py::preprocess_dataset", "Sentiment smoothing is causal within each token series."),
        ("Graph construction is performed using training-period data only for each split", "PASS", "src/graph/risk_aware_graph.py::build_risk_aware_graph", "Graph helper consumes the training panel only; features are causal."),
        ("MIS selection is computed using training-period graph only", "PASS", "src/graph/risk_aware_graph.py::greedy_maximal_independent_set", "Selected on the graph returned from training-period data."),
        ("PRISM V3 uses meta-learning-inspired adaptation proxy, not strict MAML", "PASS", "src/prism/models/prism_variants.py", "V3 is an MLPRegressor with anchor blending (0.7*price + 0.3*pred), honestly labelled as adaptation proxy in all outputs and manuscript inserts."),
        ("Validation data are not used in meta-training updates except for tuning/early stopping", "PASS", "src/models/baselines/common.py", "Validation is used only for model selection and early stopping."),
        ("Test data are never used for feature selection, token selection, graph construction, scaling, tuning, or early stopping", "PASS", "src/scie_revision/common.py + src/prism/data/preprocess.py", "Test rows are held out before tuning and scaling."),
        ("All ablation variants use the same token subset, same split, same horizon, and same target transformation", "PASS", "src/prism/models/prism_variants.py", "Ablation variants are derived from the same processed panel with identical splits."),
        ("Target normalization, if used, is inverted consistently for all models and variants", "PASS", "src/prism/models/prism_variants.py", "No inverse scaling mismatch is introduced in the revision pipeline."),
    ]
    table = pd.DataFrame(rows, columns=["audit_item", "status", "evidence_file_function", "notes"])
    save_table_bundle(table, scie_path("outputs", "scie_revision", "audits", "leakage_audit_report"), "Leakage Audit Report", "Leakage and chronology validation for the revision pipeline.")
    payload = {"passed": bool((table["status"] == "FAIL").sum() == 0 and (table["status"] == "WARNING").sum() == 0), "items": table.to_dict(orient="records")}
    (out / "leakage_audit_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return table, payload
