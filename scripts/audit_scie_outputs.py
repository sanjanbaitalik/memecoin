from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_outputs(output_dir: str) -> dict[str, object]:
    root = Path(output_dir)
    failures: list[str] = []
    warnings: list[str] = []

    tables_dir = root / "tables"
    figures_dir = root / "figures"
    audits_dir = root / "audits"
    reports_dir = root / "reports"

    main_comparison_csv = tables_dir / "table5_revised_main_comparison.csv"
    if main_comparison_csv.exists():
        df = pd.read_csv(main_comparison_csv)
        for col in ["fallback", "surrogate"]:
            if col in df.columns:
                bad = df[df[col].astype(str).str.contains("surrogate|sklearn_mlp.*_surrogate", case=False, na=False)]
                if not bad.empty:
                    failures.append(f"Paper-ready table contains fallback/surrogate rows: {bad['model'].tolist()}")
        if "model" in df.columns:
            for pattern in ["sklearn_mlp_gru_surrogate", "sklearn_mlp_lstm_surrogate", "sklearn_mlp_bilstm_surrogate"]:
                if pattern in df["model"].astype(str).values:
                    failures.append(f"Paper-ready table contains surrogate model: {pattern}")

        if "mase_mean" in df.columns:
            prism_row = df[df["model"].astype(str).str.upper().isin(["PRISM", "V3"])]
            if not prism_row.empty and pd.isna(prism_row.iloc[0].get("mase_mean")):
                failures.append("PRISM has NaN MASE in main comparison table")

    graph_csv = tables_dir / "table_graph_statistics_round2.csv"
    if not graph_csv.exists():
        graph_csv = tables_dir / "table_graph_statistics.csv"
    if graph_csv.exists():
        gdf = pd.read_csv(graph_csv)
        edge_row = gdf[gdf["metric"] == "number_of_edges"]
        if not edge_row.empty:
            n_edges = float(edge_row.iloc[0]["value"])
            if n_edges == 0:
                failures.append("Graph edge count is zero")

        redundancy_row = gdf[gdf["metric"] == "redundancy_reduction_percentage"]
        if not redundancy_row.empty:
            red = float(redundancy_row.iloc[0]["value"])
            if red == 0:
                warnings.append("Redundancy reduction is zero while paper may claim diversification benefit")

    leakage_csv = audits_dir / "leakage_audit_report_round2.csv"
    if not leakage_csv.exists():
        leakage_csv = audits_dir / "leakage_audit_report.csv"
    if leakage_csv.exists():
        ldf = pd.read_csv(leakage_csv)
        if "status" in ldf.columns:
            warnings_list = ldf[ldf["status"].isin(["WARNING", "FAIL"])]
            if not warnings_list.empty:
                failures.append(f"Leakage audit contains WARNING/FAIL: {warnings_list['audit_item'].tolist()}")
    else:
        failures.append("Leakage audit report not found")

    token_counts = set()
    for fname in ["table5_revised_main_comparison.csv", "table_ablation_strict.csv", "table_graph_statistics.csv"]:
        fpath = tables_dir / fname
        if fpath.exists():
            tdf = pd.read_csv(fpath)
            if "n_tokens" in tdf.columns:
                token_counts.update(tdf["n_tokens"].dropna().unique().tolist())
    if len(token_counts) > 1:
        warnings.append(f"Token counts are inconsistent across tables: {token_counts}")

    claim_json = reports_dir / "paper_claim_summary_round2.json"
    if not claim_json.exists():
        claim_json = reports_dir / "paper_claim_summary.json"
    if claim_json.exists():
        claim = json.loads(claim_json.read_text(encoding="utf-8"))
        if claim.get("prism_best_mae") and main_comparison_csv.exists():
            mdf = pd.read_csv(main_comparison_csv)
            if not mdf.empty and "mae_mean" in mdf.columns:
                prism_mae = mdf[mdf["model"].astype(str).str.upper().isin(["PRISM", "V3"])]
                if not prism_mae.empty:
                    best_model = mdf.loc[mdf["mae_mean"].idxmin(), "model"]
                    if best_model != prism_mae.iloc[0]["model"]:
                        failures.append("Paper claim says PRISM is best but Table contradicts it")

    if not reports_dir.exists() or not any(reports_dir.glob("AUTHOR_ACTION_REQUIRED*.md")):
        warnings.append("No author-action-required metadata file found")

    required_figures = [
        "fig_forecasting_comparison.png",
        "fig_risk_adjusted_comparison.png",
        "fig_diversification_effect.png",
        "fig_ablation_strict.png",
        "fig_token_history_distribution.png",
    ]
    for fig_name in required_figures:
        if not (figures_dir / fig_name).exists():
            warnings.append(f"Missing figure: {fig_name}")

    required_formats = ["csv", "md", "tex"]
    required_tables = [
        "table5_revised_main_comparison",
        "table_ablation_strict",
        "table_graph_statistics",
        "table_risk_adjusted_metrics",
        "table_significance_vs_prism",
    ]
    for tbl in required_tables:
        for fmt in required_formats:
            fpath = tables_dir / f"{tbl}.{fmt}"
            if not fpath.exists():
                warnings.append(f"Missing table format: {tbl}.{fmt}")

    passed = len(failures) == 0
    gate_status = "SCIE_OUTPUT_QUALITY_GATE=PASS" if passed else "SCIE_OUTPUT_QUALITY_GATE=FAIL"

    result = {
        "gate_status": gate_status,
        "passed": passed,
        "failures": failures,
        "warnings": warnings,
        "n_failures": len(failures),
        "n_warnings": len(warnings),
    }

    audits_dir.mkdir(parents=True, exist_ok=True)
    (audits_dir / "output_quality_gate_round2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_lines = [
        "# Output Quality Gate Report",
        "",
        f"## Status: {gate_status}",
        "",
        f"- Failures: {len(failures)}",
        f"- Warnings: {len(warnings)}",
        "",
    ]
    if failures:
        md_lines.append("## Blocking Failures")
        for f in failures:
            md_lines.append(f"- {f}")
        md_lines.append("")
    if warnings:
        md_lines.append("## Warnings")
        for w in warnings:
            md_lines.append(f"- {w}")
        md_lines.append("")
    md_lines.append("## Acceptance Checklist")
    md_lines.append("")
    md_lines.append("- [ ] Real LSTM baseline implemented and reported without fallback")
    md_lines.append("- [ ] Real GRU baseline implemented and reported without fallback")
    md_lines.append("- [ ] Real BiLSTM or TCN/N-BEATS baseline implemented and reported")
    md_lines.append("- [ ] No fallback/surrogate rows in paper-ready table")
    md_lines.append("- [ ] PRISM has MAE, RMSE, sMAPE, MASE, directional accuracy, ROI/risk metrics")
    md_lines.append("- [ ] Graph has nonzero edges and meaningful density")
    md_lines.append("- [ ] MIS reduces redundancy or a failure is explicitly reported")
    md_lines.append("- [ ] Token counts are consistent across all tables")
    md_lines.append("- [ ] Leakage audit has no WARNING/FAIL")
    md_lines.append("- [ ] Ablation audit confirms same subset/split/scaler/target transformation")
    md_lines.append("- [ ] Dataset card reports available metadata and lists unavailable fields honestly")
    md_lines.append("- [ ] Claims are consistent with actual table rankings")
    md_lines.append("- [ ] Figures are paper-ready and not based on meaningless zero metrics")
    md_lines.append("- [ ] Output quality gate passed")

    (audits_dir / "output_quality_gate_round2.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(gate_status)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SCIE revision outputs")
    parser.add_argument("--output", default="outputs/scie_revision_round2", help="Output directory")
    args = parser.parse_args()
    result = audit_outputs(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
