from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scie_revision.common import ensure_parent, scie_path


def generate_claim_summary(main_table: pd.DataFrame, output_dir: Path | None = None) -> dict[str, object]:
    prism_rows = main_table[main_table["model"].astype(str).str.upper().isin(["PRISM", "V3"])]
    if prism_rows.empty:
        prism_rows = main_table[main_table["model"].astype(str).str.contains("PRISM", case=False, na=False)]
    if prism_rows.empty:
        best_row = main_table.loc[main_table["mae_mean"].idxmin()]
        best_model = str(best_row["model"])
        recommended_claim = (
            f"PRISM is not included in the main baseline comparison table. "
            f"The best point forecaster on MAE is {best_model}. "
            "PRISM's contribution is evaluated through the ablation study (V0-V3) and risk-aware diversification metrics."
        )
        limitations_note = f"PRISM row absent from main table. Best MAE model: {best_model}."
        payload = {
            "prism_best_mae": False,
            "prism_best_rmse": False,
            "baselines_better_than_prism_mae": [],
            "baselines_better_than_prism_rmse": [],
            "baselines_worse_than_prism_mae": [],
            "recommended_claim": recommended_claim,
            "limitations_note": limitations_note,
        }
        target_root = output_dir or scie_path("outputs", "scie_revision_round3", "reports")
        ensure_parent(target_root / "paper_claim_summary.md")
        (target_root / "paper_claim_summary.md").write_text(recommended_claim, encoding="utf-8")
        (target_root / "paper_claim_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    prism = prism_rows.iloc[0]
    mae_min = float(main_table["mae_mean"].min())
    rmse_min = float(main_table["rmse_mean"].min())
    prism_best_mae = bool(float(prism["mae_mean"]) <= mae_min + 1e-12)
    prism_best_rmse = bool(float(prism["rmse_mean"]) <= rmse_min + 1e-12)

    baselines_better_mae = main_table.loc[main_table["mae_mean"] < float(prism["mae_mean"]), "model"].astype(str).tolist()
    baselines_better_rmse = main_table.loc[main_table["rmse_mean"] < float(prism["rmse_mean"]), "model"].astype(str).tolist()
    baselines_worse_mae = main_table.loc[main_table["mae_mean"] > float(prism["mae_mean"]), "model"].astype(str).tolist()

    if prism_best_mae and prism_best_rmse:
        recommended_claim = (
            "PRISM achieves the lowest aggregate MAE and RMSE among all evaluated baselines under the chronological protocol."
        )
        limitations_note = "No limitations on point-forecast accuracy."
    else:
        better_list = ", ".join(baselines_better_mae) if baselines_better_mae else "none"
        recommended_claim = (
            f"PRISM outperforms {', '.join(baselines_worse_mae)} on MAE, while {better_list} remain strong short-horizon competitors. "
            "This indicates that memecoin forecasting is strongly influenced by local price continuity, and that PRISM is better interpreted "
            "as a risk-aware, multimodal, and diversified forecasting framework rather than as a universally dominant point forecaster."
        )
        limitations_note = (
            f"Prophet and/or Persistence remain highly competitive under short-horizon MAE/RMSE, "
            f"indicating strong local continuity in memecoin prices. "
            f"Baselines better than PRISM on MAE: {better_list}."
        )

    summary_text = "\n".join(
        [
            "# Paper Claim Summary",
            "",
            f"- PRISM best MAE: {prism_best_mae}",
            f"- PRISM best RMSE: {prism_best_rmse}",
            f"- Baselines better than PRISM on MAE: {', '.join(baselines_better_mae) if baselines_better_mae else 'none'}",
            f"- Baselines better than PRISM on RMSE: {', '.join(baselines_better_rmse) if baselines_better_rmse else 'none'}",
            f"- Baselines worse than PRISM on MAE: {', '.join(baselines_worse_mae) if baselines_worse_mae else 'none'}",
            "",
            "## Recommended Claim",
            "",
            recommended_claim,
            "",
            "## Limitations Note",
            "",
            limitations_note,
            "",
            "## Interpretation",
            "",
            "The main contribution of PRISM is evaluated not only through point-forecasting error, but also through "
            "risk-aware diversification, robustness, ablation behavior, and statistical testing.",
        ]
    )

    payload = {
        "prism_best_mae": prism_best_mae,
        "prism_best_rmse": prism_best_rmse,
        "baselines_better_than_prism_mae": baselines_better_mae,
        "baselines_better_than_prism_rmse": baselines_better_rmse,
        "baselines_worse_than_prism_mae": baselines_worse_mae,
        "recommended_claim": recommended_claim,
        "limitations_note": limitations_note,
    }

    target_root = output_dir or scie_path("outputs", "scie_revision_round3", "reports")
    ensure_parent(target_root / "paper_claim_summary.md")
    (target_root / "paper_claim_summary.md").write_text(summary_text, encoding="utf-8")
    (target_root / "paper_claim_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
