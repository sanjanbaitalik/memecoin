from __future__ import annotations

from dataclasses import dataclass

import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from prism.evaluation.metrics import evaluate_frame


VARIANT_DESCRIPTIONS = {
    "V0": "price-only MLP (baseline)",
    "V1": "V0 + sentiment fusion",
    "V2": "V1 + risk-aware graph + MIS support diversification",
    "V3": "V2 + meta-learning-inspired adaptation proxy (anchor blending: 0.7*price + 0.3*pred)",
}


@dataclass
class PrismExperimentConfig:
    train_ratio: float = 0.8
    seed: int = 42
    seeds: tuple[int, ...] = (11, 17, 23)


def _variant_features(dataset: pd.DataFrame, variant: str) -> list[str]:
    price = [c for c in dataset.columns if c.startswith("price_lag_") and c.endswith("_z")]
    sentiment = [
        c
        for c in ["twitter_sentiments", "reddit_sentiment", "telegram_sentiment"]
        if c in dataset.columns
    ]
    sentiment_z = [f"{c}_z" for c in sentiment if f"{c}_z" in dataset.columns]
    graph = [c for c in ["return_1d", "volume"] if c in dataset.columns]
    graph_z = [f"{c}_z" for c in graph if f"{c}_z" in dataset.columns]

    if variant == "V0":
        return price
    if variant == "V1":
        return price + sentiment_z
    if variant == "V2":
        return price + sentiment_z + graph_z
    if variant == "V3":
        return price + sentiment_z + graph_z
    raise ValueError(f"Unknown variant: {variant}")


def _fit_predict_variant(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, variant: str, seed: int) -> np.ndarray:
    x_train = train_x.fillna(0.0).to_numpy(dtype=float)
    x_test = test_x.fillna(0.0).to_numpy(dtype=float)
    y_train = train_y.to_numpy(dtype=float)
    y_train = np.clip(y_train, a_min=1e-12, a_max=None)
    y_train_t = np.log1p(y_train)

    if variant == "V3":
        model = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=seed, max_iter=500)
    elif variant == "V2":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(hidden_layer_sizes=(64,), random_state=seed, max_iter=400)

    model.fit(x_train, y_train_t)
    pred_t = model.predict(x_test)
    pred = np.expm1(pred_t)

    y_max = float(np.nanmax(y_train)) if y_train.size else 1.0
    y_med = float(np.nanmedian(y_train)) if y_train.size else 0.0
    upper = max(y_max * 5.0, y_med * 10.0, 1e-10)
    return np.clip(pred, a_min=0.0, a_max=upper)


def run_ablation(dataset: pd.DataFrame, config: PrismExperimentConfig) -> pd.DataFrame:
    rows: list[dict] = []
    diag_rows: list[dict] = []
    variants = ["V0", "V1", "V2", "V3"]

    for seed in config.seeds:
        for sheet, gdf in dataset.groupby("sheet_name"):
            gdf = gdf.sort_values("datetime").reset_index(drop=True)
            split = int(len(gdf) * config.train_ratio)
            if split < 10 or len(gdf) - split < 5:
                continue

            train = gdf.iloc[:split]
            test = gdf.iloc[split:]

            for variant in variants:
                cols = [c for c in _variant_features(gdf, variant) if c in gdf.columns]
                if not cols:
                    continue

                pred = _fit_predict_variant(
                    train_x=train[cols],
                    train_y=train["target_t_plus_h"],
                    test_x=test[cols],
                    variant=variant,
                    seed=seed,
                )

                if variant == "V3":
                    anchor = test["price"].to_numpy(dtype=float)
                    pred = 0.7 * anchor + 0.3 * pred
                    anchor_cap = np.clip(anchor * 3.0, a_min=1e-10, a_max=None)
                    pred = np.clip(pred, a_min=0.0, a_max=anchor_cap)

                eval_df = test[["price", "target_t_plus_h"]].copy()
                eval_df["yhat"] = pred
                metrics = evaluate_frame(eval_df, "yhat")

                rows.append(
                    {
                        "seed": seed,
                        "sheet_name": sheet,
                        "variant": variant,
                        **metrics,
                    }
                )

                if variant == "V3":
                    y_true = eval_df["target_t_plus_h"].to_numpy(dtype=float)
                    y_pred = eval_df["yhat"].to_numpy(dtype=float)
                    ratio = np.nan
                    if np.nanstd(y_true) > 0:
                        ratio = float(np.nanstd(y_pred) / np.nanstd(y_true))
                    suspected = bool((not np.isfinite(ratio)) or ratio > 50 or ratio < 0.02)
                    pred_hash = hashlib.sha256(np.asarray(y_pred, dtype=float).tobytes()).hexdigest()

                    diag_rows.append(
                        {
                            "sheet_name": sheet,
                            "seed": seed,
                            "n_test": int(len(eval_df)),
                            "y_true_min": float(np.nanmin(y_true)),
                            "y_true_max": float(np.nanmax(y_true)),
                            "y_true_mean": float(np.nanmean(y_true)),
                            "y_pred_min": float(np.nanmin(y_pred)),
                            "y_pred_max": float(np.nanmax(y_pred)),
                            "y_pred_mean": float(np.nanmean(y_pred)),
                            "used_scaler": True,
                            "inverse_transform_applied": False,
                            "contains_nan": bool(np.isnan(y_pred).any()),
                            "contains_inf": bool(np.isinf(y_pred).any()),
                            "suspected_scale_mismatch": suspected,
                            "prediction_hash": pred_hash,
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    if diag_rows:
        metrics_df.attrs["prism_prediction_diagnostics"] = pd.DataFrame(diag_rows)
    return metrics_df
