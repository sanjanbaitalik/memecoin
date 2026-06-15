from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.baselines.runner import BaselineConfig, run_baselines_with_diagnostics
from prism.data.preprocess import PreprocessConfig, preprocess_dataset
from prism.data.universe import UniverseRule, build_token_universe_manifest
from prism.data.workbook import audit_workbook, locate_workbook, write_audit_outputs
from prism.evaluation.metrics import evaluate_frame
from prism.models.prism_variants import PrismExperimentConfig, run_ablation
from prism.stats.significance import paired_wilcoxon_table
from prism.utils.config import load_yaml
from prism.utils.io_contracts import DataContractError, safe_read_csv, safe_read_parquet, safe_write_csv
from prism.utils.paths import output_path
from prism.utils.registry import write_run_manifest
from prism.utils.seed import set_global_seed
from prism.visualization.plots import save_metric_boxplot, save_missingness_heatmap


RESULTS_DIR = ("outputs", "results")
TABLES_DIR = ("outputs", "tables")
REPORTS_DIR = ("outputs", "reports")
AUDITS_DIR = ("outputs", "audits")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_status(name: str, status: str, detail: str = "") -> dict[str, Any]:
    return {"step": name, "status": status, "detail": detail, "timestamp": _timestamp()}


def _read_csv_contract(path: str, step: str, required_columns: list[str] | None = None, upstream: str | None = None) -> pd.DataFrame:
    return safe_read_csv(
        output_path(*path.split("/")),
        step=step,
        required_columns=required_columns,
        upstream_dependency=upstream,
    )


def _read_parquet_contract(path: str, step: str, required_columns: list[str] | None = None, upstream: str | None = None) -> pd.DataFrame:
    return safe_read_parquet(
        output_path(*path.split("/")),
        step=step,
        required_columns=required_columns,
        upstream_dependency=upstream,
    )


def _write_step_report(filename: str, title: str, lines: list[str]) -> None:
    payload = [f"# {title}", "", f"- Timestamp: {_timestamp()}"] + lines
    output_path(*REPORTS_DIR, filename).write_text("\n".join(payload), encoding="utf-8")


def _remove_stale_reports() -> None:
    stale = output_path(*REPORTS_DIR, "pipeline_todo.md")
    if stale.exists():
        stale.unlink()


def _selected_tokens_from_manifest() -> list[str]:
    subset = _read_csv_contract(
        "outputs/tables/modeling_subset_manifest.csv",
        step="modeling_subset",
        required_columns=["sheet_name", "selected_for_modeling"],
        upstream="preprocess",
    )
    selected = subset.loc[subset["selected_for_modeling"] == True, "sheet_name"].astype(str).tolist()
    if not selected:
        raise DataContractError("[modeling_subset] no selected tokens available for modeling")
    return selected


def _build_modeling_subset_manifest(manifest: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    cfg = load_yaml("configs/model/baselines.yaml")
    max_tokens = int(cfg.get("max_tokens_for_modeling", 200))

    panel_stats = (
        dataset.groupby("sheet_name", as_index=False)
        .agg(
            rows_available=("sheet_name", "size"),
            panel_first_date=("datetime", "min"),
            panel_last_date=("datetime", "max"),
            sentiment_coverage=(
                "twitter_sentiments",
                lambda s: float(s.notna().mean()) if "twitter_sentiments" in dataset.columns else 0.0,
            ),
        )
    )

    merged = manifest.merge(panel_stats, on="sheet_name", how="left")
    merged["rows_available"] = merged["rows_available"].fillna(0).astype(int)
    merged["date_coverage"] = np.where(
        merged["panel_first_date"].notna() & merged["panel_last_date"].notna(),
        merged["panel_first_date"].astype(str) + " -> " + merged["panel_last_date"].astype(str),
        "",
    )

    eligible_mask = merged["inclusion_status"].eq("included") & merged["rows_available"].gt(0)
    merged["eligible"] = eligible_mask

    eligible = (
        merged.loc[eligible_mask, ["sheet_name", "rows_available", "date_coverage", "sentiment_coverage"]]
        .sort_values(["rows_available", "sheet_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    eligible["selection_order"] = np.arange(1, len(eligible) + 1)
    if max_tokens > 0:
        selected = set(eligible.head(max_tokens)["sheet_name"].tolist())
    else:
        selected = set(eligible["sheet_name"].tolist())

    merged = merged.merge(eligible[["sheet_name", "selection_order"]], on="sheet_name", how="left")
    merged["selected_for_modeling"] = merged["sheet_name"].isin(selected)
    merged["selection_reason"] = np.where(
        ~merged["eligible"],
        "not_eligible_after_preprocess",
        np.where(merged["selected_for_modeling"], "top_rows_deterministic_rank", "eligible_but_not_selected_due_to_cap"),
    )

    subset_cols = [
        "sheet_name",
        "eligible",
        "selected_for_modeling",
        "selection_order",
        "selection_reason",
        "rows_available",
        "date_coverage",
        "sentiment_coverage",
    ]
    subset_manifest = merged[subset_cols].copy().sort_values(["selected_for_modeling", "selection_order", "sheet_name"], ascending=[False, True, True])
    safe_write_csv(subset_manifest, output_path(*TABLES_DIR, "modeling_subset_manifest.csv"), step="preprocess")
    return subset_manifest


def run_audit() -> pd.DataFrame:
    workbook = locate_workbook()
    audit = audit_workbook(workbook)
    write_audit_outputs(audit)

    inventory = audit["inventory"]
    missingness = audit["missingness"]
    diagnostics = audit.get("date_parse_diagnostics", pd.DataFrame())

    assert isinstance(inventory, pd.DataFrame)
    assert isinstance(missingness, pd.DataFrame)

    if not missingness.empty:
        save_missingness_heatmap(missingness, "eda_missingness_heatmap.png")

    date_issue_sheets = int((inventory.get("valid_date_count", 0) == 0).sum()) if not inventory.empty else 0
    _write_step_report(
        "data_audit_report.md",
        "Data Audit Report",
        [
            f"- Input workbook: {workbook}",
            f"- Total sheets: {len(inventory)}",
            f"- Valid schema sheets: {int(inventory['valid_schema'].sum()) if not inventory.empty else 0}",
            f"- Sheets with date parse failures: {date_issue_sheets}",
            f"- Date diagnostics rows: {len(diagnostics) if isinstance(diagnostics, pd.DataFrame) else 0}",
            "- Status: success",
        ],
    )
    return inventory


def run_build_universe() -> pd.DataFrame:
    workbook = locate_workbook()
    cfg = load_yaml("configs/data/universe.yaml")
    rules = UniverseRule(
        min_rows=int(cfg.get("min_rows", 30)),
        min_price_coverage=float(cfg.get("min_price_coverage", 0.7)),
        require_any_sentiment=bool(cfg.get("require_any_sentiment", False)),
    )

    manifest = build_token_universe_manifest(workbook, rules)
    safe_write_csv(manifest, output_path("outputs", "manifests", "token_universe_manifest.csv"), step="build_universe")

    exclusion = _read_csv_contract(
        "outputs/manifests/token_universe_exclusion_log.csv",
        step="build_universe",
        required_columns=["sheet_name", "final_included", "reason", "invalid_date"],
        upstream="build_token_universe_manifest",
    )

    date_excluded = int(exclusion.loc[exclusion["invalid_date"] == True, "sheet_name"].nunique())

    _write_step_report(
        "universe_report.md",
        "Token Universe Report",
        [
            f"- Included tokens: {int((manifest['inclusion_status'] == 'included').sum())}",
            f"- Excluded tokens: {int((manifest['inclusion_status'] == 'excluded').sum())}",
            f"- Excluded due to invalid_date flag: {date_excluded}",
            "- Status: success",
        ],
    )

    return manifest


def run_preprocess() -> pd.DataFrame:
    workbook = locate_workbook()
    manifest = _read_csv_contract(
        "outputs/manifests/token_universe_manifest.csv",
        step="preprocess",
        required_columns=["sheet_name", "inclusion_status", "exclusion_reason"],
        upstream="build_universe",
    )

    cfg = load_yaml("configs/data/preprocess.yaml")
    pconfig = PreprocessConfig(
        forecast_horizon_days=int(cfg.get("forecast_horizon_days", 3)),
        lookback_days=int(cfg.get("lookback_days", 14)),
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        sentiment_mode=str(cfg.get("sentiment_mode", "raw")),
    )

    dataset = preprocess_dataset(workbook, manifest, pconfig)
    if dataset.empty:
        raise DataContractError("[preprocess] processed dataset is empty")

    surviving = set(dataset["sheet_name"].astype(str).unique().tolist())
    include_mask = manifest["inclusion_status"].eq("included")
    drop_mask = include_mask & (~manifest["sheet_name"].astype(str).isin(surviving))
    if drop_mask.any():
        manifest.loc[drop_mask, "inclusion_status"] = "excluded"
        manifest.loc[drop_mask, "exclusion_reason"] = (
            manifest.loc[drop_mask, "exclusion_reason"].fillna("").astype(str).str.strip(";")
            + ";dropped_in_preprocess:no_valid_rows_after_filters"
        ).str.strip(";")
        safe_write_csv(manifest, output_path("outputs", "manifests", "token_universe_manifest.csv"), step="preprocess")

    subset_manifest = _build_modeling_subset_manifest(manifest, dataset)

    comp = (
        dataset.groupby("sheet_name", as_index=False)
        .agg(
            token=("token", "first"),
            ticker=("ticker", "first"),
            rows=("sheet_name", "size"),
            first_date=("datetime", "min"),
            last_date=("datetime", "max"),
            price_non_null=("price", lambda s: int(s.notna().sum())),
        )
    )
    safe_write_csv(comp, output_path(*TABLES_DIR, "dataset_composition.csv"), step="preprocess")

    _write_step_report(
        "preprocessing_report.md",
        "Preprocessing Report",
        [
            f"- Input workbook: {workbook}",
            f"- Forecast horizon days: {pconfig.forecast_horizon_days}",
            f"- Lookback days: {pconfig.lookback_days}",
            f"- Train ratio: {pconfig.train_ratio}",
            f"- Processed rows: {len(dataset)}",
            f"- Tokens in panel: {dataset['sheet_name'].nunique()}",
            f"- Selected for modeling: {int(subset_manifest['selected_for_modeling'].sum())}",
            "- Status: success",
        ],
    )

    return dataset


def _load_selected_dataset(step: str) -> pd.DataFrame:
    dataset = _read_parquet_contract(
        "outputs/processed/processed_panel.parquet",
        step=step,
        required_columns=["sheet_name", "datetime", "price", "target_t_plus_h", "split"],
        upstream="preprocess",
    )
    selected = set(_selected_tokens_from_manifest())
    subset = dataset[dataset["sheet_name"].astype(str).isin(selected)].copy()
    if subset.empty:
        raise DataContractError(f"[{step}] selected modeling subset is empty")
    return subset


def run_baseline_experiments() -> pd.DataFrame:
    dataset = _load_selected_dataset("train_baselines")

    cfg = load_yaml("configs/model/baselines.yaml")
    bcfg = BaselineConfig(train_ratio=float(cfg.get("train_ratio", 0.8)), seed=int(cfg.get("seed", 42)))
    set_global_seed(bcfg.seed)

    metrics_df, impl_df, pred_df = run_baselines_with_diagnostics(dataset, bcfg)
    if metrics_df.empty:
        raise DataContractError("[train_baselines] baseline metrics are empty")

    # Fail-fast duplicate check for GRU/LSTM output vectors unless explicitly distinct.
    gru = pred_df[pred_df["model"] == "gru"][["sheet_name", "prediction_hash"]].rename(columns={"prediction_hash": "gru_hash"})
    lstm = pred_df[pred_df["model"] == "lstm"][["sheet_name", "prediction_hash"]].rename(columns={"prediction_hash": "lstm_hash"})
    overlap = gru.merge(lstm, on="sheet_name", how="inner")
    if not overlap.empty and bool((overlap["gru_hash"] == overlap["lstm_hash"]).all()):
        raise DataContractError("[train_baselines] GRU and LSTM predictions are identical for all overlapping tokens")

    safe_write_csv(metrics_df, output_path(*RESULTS_DIR, "per_token_baseline_metrics.csv"), step="train_baselines")
    safe_write_csv(metrics_df, output_path("outputs", "metrics", "baseline_metrics_per_token.csv"), step="train_baselines")
    safe_write_csv(impl_df, output_path(*TABLES_DIR, "baseline_implementation_registry.csv"), step="train_baselines")

    table = (
        metrics_df.groupby("model", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
            roi_mean=("roi_proxy", "mean"),
            n_tokens=("sheet_name", "nunique"),
        )
    )
    table = table.merge(impl_df[["model_label", "actual_backend", "fallback_used"]], left_on="model", right_on="model_label", how="left")
    table["model_display"] = np.where(table["fallback_used"] == True, table["model"] + " (fallback)", table["model"])
    table = table.sort_values("mae_mean")

    safe_write_csv(table, output_path(*TABLES_DIR, "table_baseline_comparison.csv"), step="train_baselines")
    save_metric_boxplot(metrics_df, "model", "mae", "eda_baseline_mae_boxplot.png")

    _write_step_report(
        "baseline_experiment_report.md",
        "Baseline Experiment Report",
        [
            f"- Models evaluated: {metrics_df['model'].nunique()}",
            f"- Rows: {len(metrics_df)}",
            f"- Fallback models: {int(impl_df['fallback_used'].sum())}",
            "- Status: success",
        ],
    )
    return metrics_df


def run_prism_and_ablation() -> pd.DataFrame:
    dataset = _load_selected_dataset("ablation")

    cfg = load_yaml("configs/model/prism.yaml")
    ecfg = PrismExperimentConfig(
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        seed=int(cfg.get("seed", 42)),
        seeds=tuple(int(s) for s in cfg.get("seeds", [11, 17, 23])),
    )

    set_global_seed(ecfg.seed)
    metrics = run_ablation(dataset, ecfg)
    if metrics.empty:
        raise DataContractError("[ablation] ablation metrics are empty")

    safe_write_csv(metrics, output_path(*RESULTS_DIR, "per_token_ablation_metrics.csv"), step="ablation")
    safe_write_csv(metrics, output_path("outputs", "metrics", "ablation_metrics_per_token.csv"), step="ablation")

    diag = metrics.attrs.get("prism_prediction_diagnostics")
    if isinstance(diag, pd.DataFrame) and not diag.empty:
        safe_write_csv(diag, output_path(*AUDITS_DIR, "prism_prediction_diagnostics.csv"), step="ablation")
    else:
        raise DataContractError("[ablation] missing PRISM prediction diagnostics")

    table = (
        metrics.groupby("variant", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
            roi_mean=("roi_proxy", "mean"),
            n_tokens=("sheet_name", "nunique"),
        )
        .sort_values("variant")
    )
    safe_write_csv(table, output_path(*TABLES_DIR, "table_ablation_sequential.csv"), step="ablation")

    save_metric_boxplot(metrics, "variant", "mae", "eda_ablation_mae_boxplot.png")
    _write_step_report(
        "ablation_report.md",
        "Sequential Ablation Report",
        [
            f"- Variants: {sorted(metrics['variant'].unique().tolist())}",
            f"- Rows: {len(metrics)}",
            f"- Suspected PRISM scale mismatches: {int(diag['suspected_scale_mismatch'].sum())}",
            "- Status: success",
        ],
    )
    return metrics


def run_robustness() -> pd.DataFrame:
    dataset = _load_selected_dataset("robustness")

    rcfg = load_yaml("configs/experiment/robustness.yaml")
    lookbacks = [int(v) for v in rcfg.get("lookback_windows", [7, 14, 21])]
    train_ratios = [float(v) for v in rcfg.get("train_ratios", [0.7, 0.8, 0.9])]
    seeds = [11, 17, 23]

    rows: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []

    for lookback in lookbacks:
        for train_ratio in train_ratios:
            for seed in seeds:
                for sheet, gdf in dataset.groupby("sheet_name"):
                    gdf = gdf.sort_values("datetime").reset_index(drop=True)
                    for lag in range(1, lookback + 1):
                        gdf[f"rb_price_lag_{lag}"] = gdf["price"].shift(lag)
                    feature_cols = [f"rb_price_lag_{lag}" for lag in range(1, lookback + 1)]

                    split = int(len(gdf) * train_ratio)
                    if split < 10 or len(gdf) - split < 5:
                        continue

                    train = gdf.iloc[:split].dropna(subset=feature_cols + ["target_t_plus_h"])
                    test = gdf.iloc[split:].dropna(subset=feature_cols + ["target_t_plus_h"])
                    if train.empty or test.empty:
                        continue

                    from sklearn.linear_model import LinearRegression

                    reg = LinearRegression()
                    reg.fit(train[feature_cols].fillna(0.0), train["target_t_plus_h"])
                    eval_df = test[["price", "target_t_plus_h"]].copy()
                    eval_df["yhat"] = reg.predict(test[feature_cols].fillna(0.0))
                    metric = evaluate_frame(eval_df, "yhat")
                    rows.append(
                        {
                            "sheet_name": sheet,
                            "seed": seed,
                            "lookback": lookback,
                            "train_ratio": train_ratio,
                            **metric,
                        }
                    )

                    fprint = hashlib.sha256(
                        f"{sheet}|{seed}|{lookback}|{train_ratio}|{len(train)}|{len(test)}|{len(feature_cols)}".encode("utf-8")
                    ).hexdigest()
                    fingerprint_rows.append(
                        {
                            "sheet_name": sheet,
                            "seed": seed,
                            "lookback": lookback,
                            "train_ratio": train_ratio,
                            "n_features": len(feature_cols),
                            "fingerprint": fprint,
                        }
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        raise DataContractError("[robustness] no robustness rows were produced")

    safe_write_csv(result, output_path(*RESULTS_DIR, "robustness_metrics.csv"), step="robustness")
    safe_write_csv(pd.DataFrame(fingerprint_rows), output_path(*AUDITS_DIR, "robustness_config_fingerprint.csv"), step="robustness")

    table = (
        result.groupby(["lookback", "train_ratio"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            rmse_mean=("rmse", "mean"),
            smape_mean=("smape", "mean"),
            n_tokens=("sheet_name", "nunique"),
        )
    )

    # Guardrail against unwired lookback parameter.
    for tr, g in table.groupby("train_ratio"):
        if g["lookback"].nunique() > 1:
            if float(g["mae_mean"].std()) < 1e-12 and float(g["rmse_mean"].std()) < 1e-12:
                raise DataContractError(f"[robustness] lookback appears unwired for train_ratio={tr}")

    safe_write_csv(table, output_path(*TABLES_DIR, "table_robustness.csv"), step="robustness")

    _write_step_report(
        "robustness_report.md",
        "Robustness Report",
        [
            f"- Rows: {len(result)}",
            f"- Configs: lookback={lookbacks}, train_ratio={train_ratios}, seeds={seeds}",
            "- Status: success",
        ],
    )

    return result


def run_significance_tests() -> pd.DataFrame:
    baseline = _read_csv_contract(
        "outputs/results/per_token_baseline_metrics.csv",
        step="significance",
        required_columns=["sheet_name", "model", "mae"],
        upstream="train_baselines",
    )
    ablation = _read_csv_contract(
        "outputs/results/per_token_ablation_metrics.csv",
        step="significance",
        required_columns=["sheet_name", "variant", "mae"],
        upstream="ablation",
    )

    frames: list[pd.DataFrame] = []

    ablation_sig = paired_wilcoxon_table(
        frame=ablation.rename(columns={"variant": "model"}),
        model_col="model",
        token_col="sheet_name",
        metric="mae",
        comparisons=[("V0", "V1"), ("V1", "V2"), ("V2", "V3")],
    )
    if not ablation_sig.empty:
        ablation_sig["comparison_group"] = "ablation"
        frames.append(ablation_sig)

    prism = (
        ablation[ablation["variant"] == "V3"]
        .groupby("sheet_name", as_index=False)["mae"]
        .mean()
        .assign(model="PRISM")
    )
    base = baseline[["sheet_name", "model", "mae"]]
    comparisons = [("PRISM", m) for m in sorted(base["model"].unique().tolist()) if m != "PRISM"]
    baseline_sig = paired_wilcoxon_table(
        frame=pd.concat([base, prism], ignore_index=True),
        model_col="model",
        token_col="sheet_name",
        metric="mae",
        comparisons=comparisons,
    )
    if not baseline_sig.empty:
        baseline_sig["comparison_group"] = "baseline_vs_prism"
        frames.append(baseline_sig)

    if not frames:
        raise DataContractError("[significance] no valid paired comparisons")

    sig = pd.concat(frames, ignore_index=True)
    safe_write_csv(sig, output_path(*TABLES_DIR, "significance_tests.csv"), step="significance")

    supported = int(sig["significant_0_05"].fillna(False).sum())
    _write_step_report(
        "significance_report.md",
        "Significance Report",
        [
            "- Metric: MAE (lower is better)",
            "- Test: paired Wilcoxon signed-rank",
            "- Correction: Holm",
            f"- Comparisons: {len(sig)}",
            f"- Significant comparisons: {supported}",
            "- Status: success",
        ],
    )

    return sig


def run_metric_validation_audit() -> pd.DataFrame:
    baseline = _read_csv_contract(
        "outputs/results/per_token_baseline_metrics.csv",
        step="metric_validation",
        required_columns=["sheet_name", "model", "mae", "rmse", "mape", "smape", "n_obs"],
        upstream="train_baselines",
    )
    ablation = _read_csv_contract(
        "outputs/results/per_token_ablation_metrics.csv",
        step="metric_validation",
        required_columns=["sheet_name", "variant", "mae", "rmse", "mape", "smape", "n_obs"],
        upstream="ablation",
    )
    prism_diag = _read_csv_contract(
        "outputs/audits/prism_prediction_diagnostics.csv",
        step="metric_validation",
        required_columns=["sheet_name", "suspected_scale_mismatch", "contains_nan", "contains_inf"],
        upstream="ablation",
    )

    b = baseline.assign(source="baseline", model_or_variant=baseline["model"])
    a = ablation.assign(source="ablation", model_or_variant=ablation["variant"])
    m = pd.concat([b, a], ignore_index=True)
    m["invalid_metric"] = m[["mae", "rmse", "mape", "smape"]].isna().any(axis=1)
    m["non_positive_n_obs"] = m["n_obs"] <= 0

    summary = (
        m.groupby(["source", "model_or_variant"], as_index=False)
        .agg(
            rows=("sheet_name", "size"),
            invalid_metric_rows=("invalid_metric", "sum"),
            non_positive_n_obs_rows=("non_positive_n_obs", "sum"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
        )
    )
    summary = summary.merge(
        prism_diag.groupby("sheet_name", as_index=False)["suspected_scale_mismatch"].max().rename(columns={"suspected_scale_mismatch": "prism_scale_flag"}),
        on="sheet_name",
        how="left",
    ) if "sheet_name" in summary.columns else summary

    safe_write_csv(m, output_path(*AUDITS_DIR, "metric_validation_report.csv"), step="metric_validation")

    mismatch_count = int(prism_diag["suspected_scale_mismatch"].sum())
    mismatch_rate = float(mismatch_count / max(len(prism_diag), 1))
    mismatch_threshold = float(load_yaml("configs/model/prism.yaml").get("max_allowed_scale_mismatch_rate", 0.05))
    _write_step_report(
        "metric_validation_report.md",
        "Metric Validation Report",
        [
            f"- Rows validated: {len(m)}",
            f"- Invalid metric rows: {int(m['invalid_metric'].sum())}",
            f"- Non-positive n_obs rows: {int(m['non_positive_n_obs'].sum())}",
            f"- PRISM suspected scale mismatches: {mismatch_count}",
            f"- PRISM suspected scale mismatch rate: {mismatch_rate:.4f}",
            f"- Allowed mismatch threshold: {mismatch_threshold:.4f}",
            "- Status: success" if mismatch_rate <= mismatch_threshold else "- Status: warning",
        ],
    )

    if mismatch_rate > mismatch_threshold:
        raise DataContractError("[metric_validation] PRISM prediction diagnostics indicate material scale mismatch")

    return m


def export_paper_tables() -> None:
    manifest = _read_csv_contract(
        "outputs/manifests/token_universe_manifest.csv",
        step="export_tables",
        required_columns=["sheet_name", "token", "first_date", "last_date", "inclusion_status", "exclusion_reason"],
        upstream="build_universe",
    )
    dataset = _read_parquet_contract(
        "outputs/processed/processed_panel.parquet",
        step="export_tables",
        required_columns=["sheet_name", "price", "volume", "return_1d"],
        upstream="preprocess",
    )
    baseline_table = _read_csv_contract(
        "outputs/tables/table_baseline_comparison.csv",
        step="export_tables",
        required_columns=["model", "mae_mean"],
        upstream="train_baselines",
    )
    ablation = _read_csv_contract(
        "outputs/results/per_token_ablation_metrics.csv",
        step="export_tables",
        required_columns=["sheet_name", "variant", "mae", "roi_proxy"],
        upstream="ablation",
    )

    prism_row = (
        ablation[ablation["variant"] == "V3"]
        .groupby("variant", as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mae_median=("mae", "median"),
            rmse_mean=("rmse", "mean"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
            roi_mean=("roi_proxy", "mean"),
            n_tokens=("sheet_name", "nunique"),
        )
        .rename(columns={"variant": "model"})
    )
    if not prism_row.empty:
        prism_row["model_display"] = "PRISM"
        prism_row["model_label"] = "PRISM"
        prism_row["actual_backend"] = "prism_v3"
        prism_row["fallback_used"] = False
        for col in baseline_table.columns:
            if col not in prism_row.columns:
                prism_row[col] = np.nan
        baseline_table = pd.concat([baseline_table, prism_row[baseline_table.columns]], ignore_index=True)

    table1 = manifest[manifest["inclusion_status"] == "included"][
        ["sheet_name", "token", "ticker", "platform_or_chain", "row_count", "first_date", "last_date", "exclusion_reason"]
    ]
    safe_write_csv(table1, output_path(*TABLES_DIR, "table_1_dataset_composition.csv"), step="export_tables")

    summary = (
        dataset[["price", "volume", "return_1d"]]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.reset_index()
        .rename(columns={"index": "feature", "count": "n"})
    )
    safe_write_csv(summary, output_path(*TABLES_DIR, "table_2_summary_statistics.csv"), step="export_tables")

    hcfg = load_yaml("configs/model/prism.yaml")
    bcfg = load_yaml("configs/model/baselines.yaml")
    table4 = pd.DataFrame(
        [{"parameter": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in {**bcfg, **hcfg}.items()]
    )
    safe_write_csv(table4, output_path(*TABLES_DIR, "table_4_hyperparameters.csv"), step="export_tables")

    safe_write_csv(baseline_table.sort_values("mae_mean"), output_path(*TABLES_DIR, "table_baseline_comparison.csv"), step="export_tables")

    # Keep table_3 for continuity: top/bottom by ROI using PRISM V3 per token.
    roi = (
        ablation[ablation["variant"] == "V3"]
        .groupby("sheet_name", as_index=False)["roi_proxy"]
        .mean()
        .sort_values("roi_proxy")
    )
    table3 = pd.concat([roi.head(10).assign(rank_group="bottom"), roi.tail(10).assign(rank_group="top")], ignore_index=True)
    safe_write_csv(table3, output_path(*TABLES_DIR, "table_3_top_bottom_tokens.csv"), step="export_tables")

    _write_step_report(
        "descriptive_stats_report.md",
        "Descriptive Statistics Report",
        [
            f"- Table 1 rows: {len(table1)}",
            f"- Table 2 rows: {len(summary)}",
            f"- Table 3 rows: {len(table3)}",
            f"- Table 4 rows: {len(table4)}",
            "- Status: success",
        ],
    )


def generate_experiment_registry() -> None:
    registry = {
        "timestamp": _timestamp(),
        "data": {
            "universe": load_yaml("configs/data/universe.yaml"),
            "preprocess": load_yaml("configs/data/preprocess.yaml"),
        },
        "model": {
            "baselines": load_yaml("configs/model/baselines.yaml"),
            "prism": load_yaml("configs/model/prism.yaml"),
        },
        "experiment": {
            "robustness": load_yaml("configs/experiment/robustness.yaml"),
            "significance": load_yaml("configs/experiment/significance.yaml"),
            "subset_selection_rule": "eligible included tokens ranked by rows_available desc, then sheet_name asc; top max_tokens_for_modeling selected",
        },
    }

    text = ["# Experiment Registry", "", f"- Timestamp: {registry['timestamp']}", "```json", json.dumps(registry, indent=2), "```"]
    output_path(*REPORTS_DIR, "experiment_registry.md").write_text("\n".join(text), encoding="utf-8")


def generate_figures() -> None:
    baseline = _read_csv_contract(
        "outputs/results/per_token_baseline_metrics.csv",
        step="figures",
        required_columns=["model", "mae", "rmse"],
        upstream="train_baselines",
    )
    ablation = _read_csv_contract(
        "outputs/results/per_token_ablation_metrics.csv",
        step="figures",
        required_columns=["variant", "mae", "rmse"],
        upstream="ablation",
    )

    figures = [
        save_metric_boxplot(baseline, "model", "rmse", "eda_baseline_rmse_boxplot.png"),
        save_metric_boxplot(ablation, "variant", "rmse", "eda_ablation_rmse_boxplot.png"),
    ]
    lines = ["# Figure Manifest", "", f"- Timestamp: {_timestamp()}", "- Status: success"] + [f"- {p.as_posix()}" for p in figures]
    output_path(*REPORTS_DIR, "figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def generate_revision_reports() -> None:
    sig = _read_csv_contract(
        "outputs/tables/significance_tests.csv",
        step="revision_reports",
        required_columns=["left", "right", "significant_0_05", "favored_model", "left_mean", "right_mean"],
        upstream="significance",
    )
    baseline = _read_csv_contract(
        "outputs/tables/table_baseline_comparison.csv",
        step="revision_reports",
        required_columns=["model", "mae_mean"],
        upstream="export_tables",
    )

    prism_mask = baseline["model"].astype(str).eq("PRISM") | baseline["model"].astype(str).eq("V3")
    if "model_label" in baseline.columns:
        prism_mask = prism_mask | baseline["model_label"].astype(str).eq("PRISM")
    if "model_display" in baseline.columns:
        prism_mask = prism_mask | baseline["model_display"].astype(str).str.contains("PRISM", case=False, na=False)

    prism_mae = float(baseline.loc[prism_mask, "mae_mean"].iloc[0]) if prism_mask.any() else np.nan
    non_prism = baseline[baseline["model"] != "PRISM"]

    prism_sig = sig[(sig["comparison_group"] == "baseline_vs_prism") & (sig["left"] == "PRISM")]
    sig_support = prism_sig[(prism_sig["significant_0_05"] == True) & (prism_sig["favored_model"] == "left_better")]

    if np.isfinite(prism_mae) and not non_prism.empty and prism_mae < float(non_prism["mae_mean"].min()) and len(sig_support) == len(prism_sig) and len(prism_sig) > 0:
        superiority_status = "supported"
    elif np.isfinite(prism_mae) and not non_prism.empty and prism_mae < float(non_prism["mae_mean"].median()) and len(sig_support) > 0:
        superiority_status = "partially supported"
    else:
        superiority_status = "requires rewording"

    guardrail = [
        "# Claim Guardrail Report",
        "",
        f"- Timestamp: {_timestamp()}",
        f"- PRISM MAE mean: {prism_mae}",
        f"- Significant PRISM-favored baseline comparisons: {len(sig_support)}/{len(prism_sig)}",
        "",
        "## superior performance over traditional models",
        f"- {superiority_status}",
        "",
        "## statistically robust approach",
        "- partially supported" if len(sig_support) > 0 else "- requires rewording",
        "",
        "## establishing a new benchmark",
        "- requires rewording",
    ]
    output_path(*REPORTS_DIR, "claim_guardrail_report.md").write_text("\n".join(guardrail), encoding="utf-8")

    patch = [
        "# Paper Patch Notes",
        "",
        f"- Timestamp: {_timestamp()}",
        "## Results",
        "- Use table_baseline_comparison.csv, table_ablation_sequential.csv, table_robustness.csv, significance_tests.csv.",
        "## Claims",
        "- Use claim_guardrail_report.md to downgrade unsupported superiority statements.",
    ]
    output_path(*REPORTS_DIR, "paper_patch_notes.md").write_text("\n".join(patch), encoding="utf-8")


def run_manuscript_integrity_gate() -> None:
    checks: list[tuple[str, bool, str]] = []

    required_tables = [
        "outputs/tables/table_1_dataset_composition.csv",
        "outputs/tables/table_2_summary_statistics.csv",
        "outputs/tables/table_4_hyperparameters.csv",
        "outputs/tables/table_baseline_comparison.csv",
        "outputs/tables/table_ablation_sequential.csv",
        "outputs/tables/table_robustness.csv",
        "outputs/tables/significance_tests.csv",
        "outputs/tables/modeling_subset_manifest.csv",
    ]

    for path in required_tables:
        try:
            _read_csv_contract(path, step="integrity_gate", upstream="all")
            checks.append((path, True, "ok"))
        except Exception as exc:
            checks.append((path, False, str(exc)))

    sig = _read_csv_contract("outputs/tables/significance_tests.csv", step="integrity_gate", required_columns=["left", "right", "favored_model", "left_mean", "right_mean", "metric_direction"], upstream="significance")

    direction_ok = True
    for _, row in sig.iterrows():
        if row["metric_direction"] == "lower_better":
            expected = "left_better" if row["left_mean"] < row["right_mean"] else "right_better"
        else:
            expected = "left_better" if row["left_mean"] > row["right_mean"] else "right_better"
        if expected != row["favored_model"]:
            direction_ok = False
            break
    checks.append(("significance_direction_consistency", direction_ok, "direction mismatch" if not direction_ok else "ok"))

    pred = _read_csv_contract("outputs/results/per_token_baseline_metrics.csv", step="integrity_gate", required_columns=["sheet_name", "model"], upstream="baselines")
    impl = _read_csv_contract("outputs/tables/baseline_implementation_registry.csv", step="integrity_gate", required_columns=["model_label", "actual_backend", "fallback_used"], upstream="baselines")
    labels_honest = True
    for _, row in impl.iterrows():
        if bool(row["fallback_used"]) and "fallback" not in str(row["actual_backend"]).lower() and "surrogate" not in str(row["actual_backend"]).lower():
            labels_honest = False
            break
    checks.append(("baseline_label_honesty", labels_honest, "fallback not clearly labeled" if not labels_honest else "ok"))

    guardrail = output_path(*REPORTS_DIR, "claim_guardrail_report.md")
    gtext = guardrail.read_text(encoding="utf-8") if guardrail.exists() else ""
    baseline_table = _read_csv_contract("outputs/tables/table_baseline_comparison.csv", step="integrity_gate", required_columns=["model", "mae_mean"], upstream="export_tables")
    if "supported" in gtext.lower():
        prism = baseline_table[baseline_table["model"] == "PRISM"]
        if not prism.empty and float(prism["mae_mean"].iloc[0]) > float(baseline_table[baseline_table["model"] != "PRISM"]["mae_mean"].min()):
            checks.append(("claim_supported_consistency", False, "guardrail says supported while PRISM MAE is not best"))
        else:
            checks.append(("claim_supported_consistency", True, "ok"))

    all_ok = all(flag for _, flag, _ in checks)
    lines = ["# Manuscript Integrity Report", "", f"- Timestamp: {_timestamp()}", f"- Safe for manuscript rewriting: {'YES' if all_ok else 'NO'}", "", "## Checks"]
    for name, flag, detail in checks:
        lines.append(f"- {name}: {'PASS' if flag else 'FAIL'} ({detail})")
    output_path(*REPORTS_DIR, "manuscript_integrity_report.md").write_text("\n".join(lines), encoding="utf-8")

    if not all_ok:
        raise DataContractError("[integrity_gate] manuscript integrity checks failed")


def run_full_pipeline() -> dict[str, Any]:
    _remove_stale_reports()
    statuses: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    steps = [
        ("audit", run_audit),
        ("build_universe", run_build_universe),
        ("preprocess", run_preprocess),
        ("train_baselines", run_baseline_experiments),
        ("ablation", run_prism_and_ablation),
        ("robustness", run_robustness),
        ("significance", run_significance_tests),
        ("metric_validation", run_metric_validation_audit),
        ("export_tables", export_paper_tables),
        ("generate_figures", generate_figures),
        ("experiment_registry", generate_experiment_registry),
        ("revision_reports", generate_revision_reports),
        ("integrity_gate", run_manuscript_integrity_gate),
    ]

    stop = False
    for name, fn in steps:
        if stop:
            statuses.append(_step_status(name, "skipped_due_to_upstream_failure", "upstream step failed"))
            continue
        try:
            result = fn()
            if isinstance(result, pd.DataFrame):
                frames[name] = result
            statuses.append(_step_status(name, "success"))
        except Exception as exc:
            statuses.append(_step_status(name, "failed", f"{type(exc).__name__}: {exc}"))
            stop = True

    manifest_payload = {
        "step_status": statuses,
        "steps": {
            "audit_rows": int(len(frames.get("audit", pd.DataFrame()))),
            "manifest_rows": int(len(frames.get("build_universe", pd.DataFrame()))),
            "dataset_rows": int(len(frames.get("preprocess", pd.DataFrame()))),
            "baseline_rows": int(len(frames.get("train_baselines", pd.DataFrame()))),
            "ablation_rows": int(len(frames.get("ablation", pd.DataFrame()))),
            "robustness_rows": int(len(frames.get("robustness", pd.DataFrame()))),
            "significance_rows": int(len(frames.get("significance", pd.DataFrame()))),
        },
    }
    write_run_manifest(manifest_payload)
    return manifest_payload
