from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.diversification_metrics import diversification_summary
from metrics.risk_metrics import risk_metric_frame
from scie_revision.common import save_table_bundle, scie_path


def evaluate_portfolio(predictions: pd.DataFrame, selected_panel: pd.DataFrame, full_panel: pd.DataFrame, output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out = output_dir or scie_path("outputs", "scie_revision", "tables")
    risk_rows = []
    div_rows = []
    forecast_rows = []
    for model, frame in predictions.groupby("model"):
        frame = frame.sort_values(["sheet_name", "datetime"])
        forecast_rows.append(
            {
                "model": model,
                "mae_mean": float(frame["mae"].mean()),
                "mae_std": float(frame["mae"].std(ddof=0)),
                "rmse_mean": float(frame["rmse"].mean()),
                "rmse_std": float(frame["rmse"].std(ddof=0)),
                "smape_mean": float(frame["smape"].mean()),
                "smape_std": float(frame["smape"].std(ddof=0)),
                "mase_mean": float(frame["mase"].mean()) if "mase" in frame.columns else float("nan"),
                "mase_std": float(frame["mase"].std(ddof=0)) if "mase" in frame.columns else float("nan"),
                "directional_accuracy_mean": float(frame["directional_accuracy"].mean()),
                "directional_accuracy_std": float(frame["directional_accuracy"].std(ddof=0)),
                "n_tokens": int(frame["sheet_name"].nunique()),
                "n_seeds": int(frame["seed"].nunique()) if "seed" in frame.columns else 1,
            }
        )
        risk_cols = [c for c in ["cum_return_proxy", "mean_return_proxy", "sharpe_ratio_proxy", "sortino_ratio_proxy", "max_drawdown_proxy", "hit_rate", "turnover_proxy", "var_95", "cvar_95", "downside_deviation", "volatility_of_returns"] if c in frame.columns]
        if risk_cols:
            risk_rows.append({"model": model, **{c: float(frame[c].mean()) for c in risk_cols}})
        if not selected_panel.empty and not full_panel.empty:
            div_rows.append({"model": model, **diversification_summary(full_panel, selected_panel)})

    forecast = pd.DataFrame(forecast_rows)
    risk = pd.DataFrame(risk_rows)
    diversification = pd.DataFrame(div_rows)
    if not forecast.empty:
        save_table_bundle(forecast, scie_path("outputs", "scie_revision", "tables", "table_main_forecasting_metrics"), "Main Forecasting Metrics", "Forecasting comparison across models.")
    if not risk.empty:
        save_table_bundle(risk, scie_path("outputs", "scie_revision", "tables", "table_risk_adjusted_metrics"), "Risk Adjusted Metrics", "Risk and trading-proxy metrics across models.")
    if not diversification.empty:
        save_table_bundle(diversification, scie_path("outputs", "scie_revision", "tables", "table_diversification_metrics"), "Diversification Metrics", "Diversification and graph-redundancy summary.")
    return {"forecast": forecast, "risk": risk, "diversification": diversification}
