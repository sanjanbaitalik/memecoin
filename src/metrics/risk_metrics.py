from __future__ import annotations

import numpy as np
import pandas as pd


def _returns_from_prices(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> np.ndarray:
    signal = np.sign(yhat - previous_price)
    realized = signal * ((price - previous_price) / np.where(previous_price == 0, np.nan, previous_price))
    return np.asarray(realized, dtype=float)


def cumulative_return_proxy(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    return float(np.nansum(ret))


def mean_return_proxy(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    return float(np.nanmean(ret))


def sharpe_ratio_proxy(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    sd = np.nanstd(ret, ddof=0)
    return float(np.nanmean(ret) / sd) if np.isfinite(sd) and sd > 0 else 0.0


def sortino_ratio_proxy(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    downside = ret[ret < 0]
    dd = np.nanstd(downside, ddof=0) if downside.size else 0.0
    return float(np.nanmean(ret) / dd) if np.isfinite(dd) and dd > 0 else 0.0


def max_drawdown_proxy(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    if ret.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + np.nan_to_num(ret, nan=0.0))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / np.where(peak == 0, np.nan, peak)
    return float(np.nanmin(drawdown)) if np.isfinite(drawdown).any() else 0.0


def hit_rate(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    actual = np.sign(price - previous_price)
    predicted = np.sign(yhat - previous_price)
    return float(np.mean(actual == predicted))


def turnover_proxy(yhat: np.ndarray, previous_price: np.ndarray) -> float:
    signal = np.sign(yhat - previous_price)
    if signal.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(signal))))


def downside_deviation(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    downside = ret[ret < 0]
    return float(np.nanstd(downside, ddof=0)) if downside.size else 0.0


def volatility_of_returns(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    return float(np.nanstd(ret, ddof=0))


def value_at_risk(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray, alpha: float = 0.95) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    return float(np.nanquantile(ret, 1 - alpha)) if ret.size else 0.0


def conditional_value_at_risk(price: np.ndarray, yhat: np.ndarray, previous_price: np.ndarray, alpha: float = 0.95) -> float:
    ret = _returns_from_prices(price, yhat, previous_price)
    if ret.size == 0:
        return 0.0
    cutoff = np.nanquantile(ret, 1 - alpha)
    tail = ret[ret <= cutoff]
    return float(np.nanmean(tail)) if tail.size else float(cutoff)


def risk_metric_frame(frame: pd.DataFrame, pred_col: str = "yhat") -> dict[str, float]:
    price = frame["price"].to_numpy(dtype=float)
    yhat = frame[pred_col].to_numpy(dtype=float)
    previous = frame["price"].shift(1).bfill().to_numpy(dtype=float)
    return {
        "cum_return_proxy": cumulative_return_proxy(price, yhat, previous),
        "mean_return_proxy": mean_return_proxy(price, yhat, previous),
        "sharpe_ratio_proxy": sharpe_ratio_proxy(price, yhat, previous),
        "sortino_ratio_proxy": sortino_ratio_proxy(price, yhat, previous),
        "max_drawdown_proxy": max_drawdown_proxy(price, yhat, previous),
        "hit_rate": hit_rate(price, yhat, previous),
        "turnover_proxy": turnover_proxy(yhat, previous),
        "var_95": value_at_risk(price, yhat, previous, alpha=0.95),
        "cvar_95": conditional_value_at_risk(price, yhat, previous, alpha=0.95),
        "downside_deviation": downside_deviation(price, yhat, previous),
        "volatility_of_returns": volatility_of_returns(price, yhat, previous),
    }
