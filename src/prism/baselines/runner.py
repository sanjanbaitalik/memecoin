from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from prism.evaluation.metrics import evaluate_frame


@dataclass
class BaselineConfig:
    train_ratio: float = 0.8
    seed: int = 42


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unavailable"


def _fit_predict_sklearn(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    kind: str,
    seed: int,
) -> tuple[np.ndarray, str]:
    if kind == "random_forest":
        model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    elif kind == "gru":
        # Distinct GRU surrogate path (explicit fallback).
        model = MLPRegressor(hidden_layer_sizes=(96, 48), random_state=seed + 11, max_iter=500)
        return _fit_predict_with_model(model, train_x, train_y, test_x), "gru_fallback_mlp"
    elif kind == "lstm":
        # Distinct LSTM surrogate path (explicit fallback).
        model = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=seed + 29, max_iter=650)
        return _fit_predict_with_model(model, train_x, train_y, test_x), "lstm_fallback_mlp"
    elif kind == "xgboost":
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(random_state=seed)
            return _fit_predict_with_model(model, train_x, train_y, test_x), "xgboost_fallback_gradient_boosting"
    else:
        model = LinearRegression()

    return _fit_predict_with_model(model, train_x, train_y, test_x), "ok"


def _fit_predict_with_model(model, train_x, train_y, test_x) -> np.ndarray:
    x_train = train_x.fillna(0.0).to_numpy(dtype=float)
    y_train = train_y.to_numpy(dtype=float)
    x_test = test_x.fillna(0.0).to_numpy(dtype=float)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _arima_predict(train_y: pd.Series, test_size: int) -> tuple[np.ndarray, str]:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train_y.to_numpy(dtype=float), order=(1, 1, 1))
        fit = model.fit()
        pred = fit.forecast(steps=test_size)
        return np.asarray(pred, dtype=float), "ok"
    except Exception as exc:
        pred = np.repeat(float(train_y.iloc[-1]), test_size)
        return pred, f"arima_fallback_persistence:{type(exc).__name__}"


def _prophet_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    try:
        from prophet import Prophet

        m = Prophet()
        df = pd.DataFrame({"ds": train_df["datetime"], "y": train_df["target_t_plus_h"]})
        m.fit(df)
        future = pd.DataFrame({"ds": test_df["datetime"]})
        forecast = m.predict(future)
        return forecast["yhat"].to_numpy(dtype=float), "ok"
    except Exception as exc:
        pred = np.repeat(float(train_df["target_t_plus_h"].iloc[-1]), len(test_df))
        return pred, f"prophet_fallback_persistence:{type(exc).__name__}"


def run_baselines(dataset: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    metrics, _, _ = run_baselines_with_diagnostics(dataset, config)
    return metrics


def run_baselines_with_diagnostics(dataset: pd.DataFrame, config: BaselineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in dataset.columns if c.endswith("_z")]
    rows: list[dict] = []
    registry_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for sheet, gdf in dataset.groupby("sheet_name"):
        gdf = gdf.sort_values("datetime").reset_index(drop=True)
        split = int(len(gdf) * config.train_ratio)
        if split < 10 or len(gdf) - split < 5:
            continue

        train = gdf.iloc[:split]
        test = gdf.iloc[split:]
        train_x = train[feature_cols]
        test_x = test[feature_cols]

        candidate_models = [
            "persistence",
            "arima",
            "prophet",
            "random_forest",
            "xgboost",
            "gru",
            "lstm",
        ]

        for model_name in candidate_models:
            eval_df = test[["datetime", "sheet_name", "price", "target_t_plus_h"]].copy()
            note = "ok"
            backend = model_name
            fallback_used = False
            fallback_reason = ""
            library_version = ""
            if model_name == "persistence":
                eval_df["yhat"] = test["price"].to_numpy(dtype=float)
                backend = "naive_last_value"
                library_version = "n/a"
            elif model_name == "arima":
                pred, note = _arima_predict(train["target_t_plus_h"], len(test))
                eval_df["yhat"] = pred
                backend = "statsmodels_arima"
                library_version = _package_version("statsmodels")
                if "fallback" in note:
                    fallback_used = True
                    fallback_reason = note
            elif model_name == "prophet":
                pred, note = _prophet_predict(train, test)
                eval_df["yhat"] = pred
                backend = "prophet"
                library_version = _package_version("prophet")
                if "fallback" in note:
                    fallback_used = True
                    fallback_reason = note
            else:
                pred, note = _fit_predict_sklearn(
                    train_x=train_x,
                    train_y=train["target_t_plus_h"],
                    test_x=test_x,
                    kind=model_name,
                    seed=config.seed,
                )
                eval_df["yhat"] = pred
                if model_name == "random_forest":
                    backend = "sklearn_random_forest"
                    library_version = _package_version("scikit-learn")
                elif model_name == "xgboost":
                    if "fallback" in note:
                        backend = "sklearn_gradient_boosting"
                        fallback_used = True
                        fallback_reason = note
                        library_version = _package_version("scikit-learn")
                    else:
                        backend = "xgboost_xgbregressor"
                        library_version = _package_version("xgboost")
                elif model_name == "gru":
                    backend = "sklearn_mlp_gru_surrogate"
                    fallback_used = True
                    fallback_reason = note
                    library_version = _package_version("scikit-learn")
                elif model_name == "lstm":
                    backend = "sklearn_mlp_lstm_surrogate"
                    fallback_used = True
                    fallback_reason = note
                    library_version = _package_version("scikit-learn")

            metrics = evaluate_frame(eval_df, "yhat")
            rows.append(
                {
                    "sheet_name": sheet,
                    "model": model_name,
                    "note": note,
                    **metrics,
                }
            )

            pred_hash = hashlib.sha256(np.asarray(eval_df["yhat"], dtype=float).tobytes()).hexdigest()
            prediction_rows.append(
                {
                    "sheet_name": sheet,
                    "model": model_name,
                    "prediction_hash": pred_hash,
                    "n_test": int(len(eval_df)),
                }
            )

            registry_rows.append(
                {
                    "model_label": model_name,
                    "actual_backend": backend,
                    "fallback_used": bool(fallback_used),
                    "fallback_reason": fallback_reason,
                    "library_version": library_version,
                    "random_seed": int(config.seed),
                }
            )

    metrics_df = pd.DataFrame(rows)
    registry_df = pd.DataFrame(registry_rows).drop_duplicates(subset=["model_label", "actual_backend", "fallback_reason"])
    pred_df = pd.DataFrame(prediction_rows)
    return metrics_df, registry_df, pred_df
