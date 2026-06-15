from _bootstrap import bootstrap_path

bootstrap_path()

from pathlib import Path

import pandas as pd

from prism.data.workbook import iter_token_frames, locate_workbook
from prism.utils.io_contracts import DataContractError, safe_read_csv, safe_read_parquet
from prism.utils.paths import output_path


def main() -> int:
    workbook = locate_workbook()

    sample_ok = False
    for _, frame in iter_token_frames(workbook):
        if "date" in frame.columns and pd.to_datetime(frame["date"], errors="coerce").notna().sum() > 0:
            sample_ok = True
            break

    if not sample_ok:
        raise RuntimeError("Validation failed: no sample sheet with parseable dates")

    panel = safe_read_parquet(
        output_path("outputs", "processed", "processed_panel.parquet"),
        step="validate",
        required_columns=["sheet_name", "datetime", "price", "target_t_plus_h"],
        upstream_dependency="preprocess",
    )

    if len(panel) <= 0:
        raise RuntimeError("Validation failed: processed panel has zero rows")

    manifest = safe_read_csv(
        output_path("outputs", "manifests", "token_universe_manifest.csv"),
        step="validate",
        required_columns=["sheet_name", "inclusion_status"],
        upstream_dependency="build_universe",
    )
    included = manifest.loc[manifest["inclusion_status"] == "included", "sheet_name"].nunique()
    processed = panel["sheet_name"].nunique()
    if included != processed:
        raise RuntimeError(
            f"Validation failed: included tokens ({included}) != processed panel tokens ({processed})"
        )

    baseline = safe_read_csv(
        output_path("outputs", "results", "per_token_baseline_metrics.csv"),
        step="validate",
        required_columns=["sheet_name", "model", "mae"],
        upstream_dependency="train_baselines",
    )

    sig_path = output_path("outputs", "tables", "significance_tests.csv")
    if len(baseline) > 0:
        sig = safe_read_csv(
            sig_path,
            step="validate",
            required_columns=["left", "right", "p_value_holm"],
            upstream_dependency="significance",
        )
        if sig.empty:
            raise RuntimeError("Validation failed: significance tests are empty while baseline metrics exist")

    manuscript_tables = [
        output_path("outputs", "tables", "table_1_dataset_composition.csv"),
        output_path("outputs", "tables", "table_2_summary_statistics.csv"),
        output_path("outputs", "tables", "table_3_top_bottom_tokens.csv"),
        output_path("outputs", "tables", "table_4_hyperparameters.csv"),
    ]

    for table in manuscript_tables:
        safe_read_csv(table, step="validate", upstream_dependency="export_tables")

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
