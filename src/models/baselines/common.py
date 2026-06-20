from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

from prism.evaluation.metrics import evaluate_frame


@dataclass
class BaselineRunResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unavailable"


def _fit_predict(model: Any, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> np.ndarray:
    model.fit(train_x.fillna(0.0).to_numpy(dtype=float), train_y.to_numpy(dtype=float))
    return np.asarray(model.predict(test_x.fillna(0.0).to_numpy(dtype=float)), dtype=float)


def _chronological_val_split(train: pd.DataFrame, val_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(train) <= 5:
        return train.copy(), train.iloc[:0].copy()
    val_start = max(int(len(train) * (1.0 - val_ratio)), 1)
    return train.iloc[:val_start].copy(), train.iloc[val_start:].copy()


def _grid_search(
    model_factory,
    grid: list[dict[str, Any]],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x: pd.DataFrame,
    val_y: pd.Series,
) -> tuple[Any, dict[str, Any], float]:
    best_model = None
    best_params: dict[str, Any] = {}
    best_score = float("inf")
    if not grid:
        grid = [{}]
    for params in grid:
        model = model_factory(**params)
        try:
            pred = _fit_predict(model, train_x, train_y, val_x)
            score = mean_absolute_error(val_y.to_numpy(dtype=float), pred)
        except Exception:
            continue
        if np.isfinite(score) and score < best_score:
            best_model = model
            best_params = dict(params)
            best_score = float(score)
    if best_model is None:
        best_model = model_factory(**(grid[0] if grid else {}))
        best_model.fit(train_x.fillna(0.0), train_y)
        best_params = dict(grid[0] if grid else {})
        best_score = float("nan")
    return best_model, best_params, best_score


def run_sklearn_regressor(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    param_grid: list[dict[str, Any]] | None = None,
) -> BaselineRunResult:
    train_x = train[feature_cols]
    train_y = train["target_t_plus_h"]
    val_x = val[feature_cols] if not val.empty else train_x.iloc[:0].copy()
    val_y = val["target_t_plus_h"] if not val.empty else train_y.iloc[:0].copy()
    test_x = test[feature_cols]

    if model_name == "random_forest":
        factory = lambda **params: RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
        grid = param_grid or [
            {"n_estimators": 200, "max_depth": 5},
            {"n_estimators": 500, "max_depth": 7},
        ]
        backend = "sklearn_random_forest"
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor

            factory = lambda **params: XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
                n_jobs=-1,
                **params,
            )
            backend = "xgboost_xgbregressor"
            grid = param_grid or [
                {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
                {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.01},
            ]
        except Exception:
            factory = lambda **params: GradientBoostingRegressor(random_state=seed, **params)
            backend = "sklearn_gradient_boosting"
            grid = param_grid or [
                {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
                {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.1},
            ]
    elif model_name == "ridge":
        factory = lambda **params: Ridge(**params)
        grid = param_grid or [{"alpha": 1.0}, {"alpha": 0.1}, {"alpha": 10.0}]
        backend = "sklearn_ridge"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    tuned_model, best_params, val_mae = _grid_search(factory, grid, train_x, train_y, val_x, val_y)
    pred = _fit_predict(tuned_model, pd.concat([train_x, val_x]), pd.concat([train_y, val_y]), test_x) if not val.empty else _fit_predict(tuned_model, train_x, train_y, test_x)
    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat")
    prediction_frame = test[["sheet_name", "datetime", "price", "target_t_plus_h"]].copy()
    prediction_frame["yhat"] = pred
    return BaselineRunResult(
        predictions=prediction_frame,
        metrics=metrics,
        metadata={
            "model_name": model_name,
            "backend": backend,
            "best_params": best_params,
            "validation_mae": val_mae,
            "library_version": package_version("scikit-learn"),
        },
    )


def run_neural_mlp(
    model_name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    hidden_layer_sizes: tuple[int, ...],
    learning_rate_init: float = 0.001,
    max_iter: int = 400,
    alpha: float = 1e-4,
) -> BaselineRunResult:
    train_x = train[feature_cols]
    train_y = train["target_t_plus_h"]
    val_x = val[feature_cols] if not val.empty else train_x.iloc[:0].copy()
    val_y = val["target_t_plus_h"] if not val.empty else train_y.iloc[:0].copy()
    test_x = test[feature_cols]

    grid = [
        {"hidden_layer_sizes": hidden_layer_sizes, "learning_rate_init": learning_rate_init, "max_iter": max_iter, "alpha": alpha},
        {"hidden_layer_sizes": hidden_layer_sizes, "learning_rate_init": learning_rate_init / 2.0, "max_iter": max_iter + 100, "alpha": alpha},
    ]

    def factory(**params):
        return MLPRegressor(
            random_state=seed,
            solver="adam",
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.2,
            tol=1e-4,
            **params,
        )

    tuned_model, best_params, val_mae = _grid_search(factory, grid, train_x, train_y, val_x, val_y)
    pred = _fit_predict(tuned_model, pd.concat([train_x, val_x]), pd.concat([train_y, val_y]), test_x) if not val.empty else _fit_predict(tuned_model, train_x, train_y, test_x)
    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat")
    prediction_frame = test[["sheet_name", "datetime", "price", "target_t_plus_h"]].copy()
    prediction_frame["yhat"] = pred
    return BaselineRunResult(
        predictions=prediction_frame,
        metrics=metrics,
        metadata={
            "model_name": model_name,
            "backend": "sklearn_mlp_regressor",
            "best_params": best_params,
            "validation_mae": val_mae,
            "library_version": package_version("scikit-learn"),
        },
    )


def run_persistence(train: pd.DataFrame, test: pd.DataFrame) -> BaselineRunResult:
    pred = test["price"].to_numpy(dtype=float)
    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat")
    return BaselineRunResult(
        predictions=test[["sheet_name", "datetime", "price", "target_t_plus_h"]].assign(yhat=pred),
        metrics=metrics,
        metadata={"model_name": "persistence", "backend": "naive_last_value", "library_version": "n/a"},
    )


def run_arima(train: pd.DataFrame, test: pd.DataFrame) -> BaselineRunResult:
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train["target_t_plus_h"].to_numpy(dtype=float), order=(1, 1, 1))
        fit = model.fit()
        pred = np.asarray(fit.forecast(steps=len(test)), dtype=float)
        backend = "statsmodels_arima"
        lib = package_version("statsmodels")
    except Exception as exc:
        pred = np.full(len(test), np.nan)
        backend = f"arima_failed:{type(exc).__name__}"
        lib = package_version("statsmodels")

    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat") if not np.isnan(pred).all() else {}
    return BaselineRunResult(
        predictions=test[["sheet_name", "datetime", "price", "target_t_plus_h"]].assign(yhat=pred),
        metrics={k: float("nan") for k in ["mae", "mse", "rmse", "mape", "smape", "mase", "r2"]} if np.isnan(pred).all() else metrics,
        metadata={"model_name": "arima", "backend": backend, "library_version": lib, "status": "failed" if np.isnan(pred).all() else "ok"},
    )


def run_prophet(train: pd.DataFrame, test: pd.DataFrame) -> BaselineRunResult:
    try:
        from prophet import Prophet

        m = Prophet()
        df = pd.DataFrame({"ds": train["datetime"], "y": train["target_t_plus_h"]})
        m.fit(df)
        pred = m.predict(pd.DataFrame({"ds": test["datetime"]}))["yhat"].to_numpy(dtype=float)
        backend = "prophet"
        note = "ok"
    except Exception as exc:
        pred = np.full(len(test), np.nan)
        backend = f"prophet_failed:{type(exc).__name__}"
        note = f"failed:{type(exc).__name__}"
    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat") if not np.isnan(pred).all() else {}
    return BaselineRunResult(
        predictions=test[["sheet_name", "datetime", "price", "target_t_plus_h"]].assign(yhat=pred),
        metrics={k: float("nan") for k in ["mae", "mse", "rmse", "mape", "smape", "mase", "r2"]} if np.isnan(pred).all() else metrics,
        metadata={"model_name": "prophet", "backend": backend, "library_version": package_version("prophet"), "note": note, "status": "failed" if np.isnan(pred).all() else "ok"},
    )
