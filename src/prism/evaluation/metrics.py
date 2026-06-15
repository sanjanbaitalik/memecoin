from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-8


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    denom = np.where(np.abs(y_true) < eps, np.nan, y_true)
    val = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(val) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = EPS) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    val = 2.0 * np.abs(y_pred - y_true) / np.where(denom < eps, np.nan, denom)
    return float(np.nanmean(val) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, previous_price: np.ndarray) -> float:
    true_dir = np.sign(y_true - previous_price)
    pred_dir = np.sign(y_pred - previous_price)
    return float(np.mean(true_dir == pred_dir))


def roi_proxy(y_true: np.ndarray, y_pred: np.ndarray, previous_price: np.ndarray) -> float:
    signal = np.sign(y_pred - previous_price)
    realized = signal * ((y_true - previous_price) / np.where(previous_price == 0, np.nan, previous_price))
    return float(np.nanmean(realized))


def evaluate_frame(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    if "target_t_plus_h" not in frame.columns or pred_col not in frame.columns or "price" not in frame.columns:
        raise ValueError("evaluate_frame requires target_t_plus_h, price and prediction columns")

    y_true = frame["target_t_plus_h"].to_numpy(dtype=float)
    y_pred = frame[pred_col].to_numpy(dtype=float)
    prev = frame["price"].to_numpy(dtype=float)

    if len(y_true) == 0:
        raise ValueError("evaluate_frame received empty arrays")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if np.all(np.isnan(y_pred)):
        raise ValueError("All predictions are NaN")

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(prev)
    if finite_mask.sum() == 0:
        raise ValueError("No finite aligned observations for metric computation")

    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    prev = prev[finite_mask]

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": safe_mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred, prev),
        "roi_proxy": roi_proxy(y_true, y_pred, prev),
        "n_obs": int(len(y_true)),
    }
