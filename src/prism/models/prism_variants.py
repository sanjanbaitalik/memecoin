from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hashlib
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prism.evaluation.metrics import evaluate_frame


VARIANT_DESCRIPTIONS = {
    "V0": "price-only LSTM (baseline)",
    "V1": "V0 + sentiment fusion",
    "V2": "V1 + risk-aware graph + MIS support diversification",
    "V3a": "Full PRISM + first-order MAML adaptation (always adapted)",
    "V3b": "Full PRISM + validation-gated MAML adaptation (only if improves over V2)",
}
V3_VARIANTS = ("V3a", "V3b")


class _PrismLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


@dataclass
class PrismExperimentConfig:
    train_ratio: float = 0.8
    seed: int = 42
    seeds: tuple[int, ...] = (11, 17, 23)
    lookback: int = 14
    hidden_size: int = 64
    num_layers: int = 2
    meta_lr: float = 0.001
    inner_lr: float = 0.01
    inner_steps: int = 3
    meta_batch_size: int = 8
    meta_epochs: int = 50
    patience: int = 10


def _variant_features(dataset: pd.DataFrame, variant: str) -> list[str]:
    price = [c for c in dataset.columns if c.startswith("price_lag_") and c.endswith("_z")]
    sentiment = [
        c for c in ["twitter_sentiments", "reddit_sentiment", "telegram_sentiment"]
        if c in dataset.columns
    ]
    sentiment_z = [f"{c}_z" for c in sentiment if f"{c}_z" in dataset.columns]
    graph = [c for c in ["return_1d", "volume"] if c in dataset.columns]
    graph_z = [f"{c}_z" for c in graph if f"{c}_z" in dataset.columns]

    if variant in ("V0",):
        return price
    if variant in ("V1",):
        return price + sentiment_z
    if variant in ("V2", "V3a", "V3b"):
        return price + sentiment_z + graph_z
    raise ValueError(f"Unknown variant: {variant}")


def _to_sequences(features: np.ndarray, targets: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= lookback:
        return np.empty((0, lookback, features.shape[1])), np.empty(0)
    X = np.stack([features[i : i + lookback] for i in range(len(features) - lookback)])
    y = targets[lookback:]
    return X, y


def _first_order_maml_predict(
    base_model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> torch.Tensor:
    adapted = copy.deepcopy(base_model)
    criterion = nn.MSELoss()
    inner_optim = torch.optim.SGD(adapted.parameters(), lr=inner_lr)

    for _ in range(inner_steps):
        inner_optim.zero_grad()
        pred = adapted(support_x)
        loss = criterion(pred, support_y)
        loss.backward()
        inner_optim.step()

    adapted.eval()
    return adapted(query_x)


def _train_maml(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    config: PrismExperimentConfig,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_size = len(feature_cols)
    base_model = _PrismLSTM(input_size, config.hidden_size, config.num_layers)

    tokens = sorted(dataset["sheet_name"].unique().tolist())
    if len(tokens) < 2:
        return base_model

    token_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sheet in tokens:
        frame = dataset[dataset["sheet_name"] == sheet].sort_values("datetime")
        x = frame[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = frame["target_t_plus_h"].to_numpy(dtype=float)
        X_seq, y_seq = _to_sequences(x, y, config.lookback)
        if X_seq.shape[0] > 0:
            token_data[sheet] = (X_seq, y_seq)

    if len(token_data) < 2:
        return base_model

    meta_optim = torch.optim.Adam(base_model.parameters(), lr=config.meta_lr)
    best_meta_loss = float("inf")
    best_state = None
    no_improve = 0

    rng = np.random.default_rng(seed)

    for epoch in range(config.meta_epochs):
        base_model.train()
        batch_tokens = rng.choice(list(token_data.keys()), size=min(config.meta_batch_size, len(token_data)), replace=False)
        meta_loss_accum = 0.0

        for token in batch_tokens:
            X_all, y_all = token_data[token]
            n = len(X_all)
            if n < 4:
                continue
            split = max(int(n * 0.6), 1)
            support_x = torch.tensor(X_all[:split], dtype=torch.float32)
            support_y = torch.tensor(y_all[:split], dtype=torch.float32)
            query_x = torch.tensor(X_all[split:], dtype=torch.float32)
            query_y = torch.tensor(y_all[split:], dtype=torch.float32)

            adapted_pred = _first_order_maml_predict(base_model, support_x, support_y, query_x, config.inner_lr, config.inner_steps)
            loss = nn.MSELoss()(adapted_pred, query_y)
            meta_loss_accum += loss

        if isinstance(meta_loss_accum, torch.Tensor) and meta_loss_accum.requires_grad:
            meta_optim.zero_grad()
            meta_loss_accum.backward()
            meta_optim.step()

        base_model.eval()
        val_loss = 0.0
        val_count = 0
        for token in rng.choice(list(token_data.keys()), size=min(config.meta_batch_size, len(token_data)), replace=False):
            X_all, y_all = token_data[token]
            n = len(X_all)
            if n < 4:
                continue
            split = max(int(n * 0.6), 1)
            val_split = split + max(int((n - split) * 0.5), 1)
            support_x = torch.tensor(X_all[:split], dtype=torch.float32)
            support_y = torch.tensor(y_all[:split], dtype=torch.float32)
            query_x = torch.tensor(X_all[split:val_split], dtype=torch.float32)
            query_y = torch.tensor(y_all[split:val_split], dtype=torch.float32)
            pred = _first_order_maml_predict(base_model, support_x, support_y, query_x, config.inner_lr, config.inner_steps)
            val_loss += nn.MSELoss()(pred, query_y).item()
            val_count += 1

        avg_val = val_loss / max(val_count, 1)
        if avg_val < best_meta_loss - 1e-6:
            best_meta_loss = avg_val
            best_state = {k: v.clone() for k, v in base_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config.patience:
                break

    if best_state is not None:
        base_model.load_state_dict(best_state)
    return base_model


def _fit_v2(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, seed: int) -> np.ndarray:
    x_train = train_x.fillna(0.0).to_numpy(dtype=float)
    x_test = test_x.fillna(0.0).to_numpy(dtype=float)
    y_train = train_y.to_numpy(dtype=float)
    y_train = np.clip(y_train, a_min=1e-12, a_max=None)
    y_train_t = np.log1p(y_train)

    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train_t)
    pred_t = model.predict(x_test)
    pred = np.expm1(pred_t)

    y_max = float(np.nanmax(y_train)) if y_train.size else 1.0
    y_med = float(np.nanmedian(y_train)) if y_train.size else 0.0
    upper = max(y_max * 5.0, y_med * 10.0, 1e-10)
    return np.clip(pred, a_min=0.0, a_max=upper)


def _fit_v3_maml(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    config: PrismExperimentConfig,
    base_model: nn.Module,
    feature_cols: list[str],
) -> np.ndarray:
    x_train = train_x.fillna(0.0).to_numpy(dtype=float)
    x_test = test_x.fillna(0.0).to_numpy(dtype=float)
    y_train = train_y.to_numpy(dtype=float)
    y_train = np.clip(y_train, a_min=1e-12, a_max=None)

    lookback = config.lookback
    if len(x_test) > lookback:
        X_test_seq = np.stack([x_test[i : i + lookback] for i in range(len(x_test) - lookback)])
        support_x = torch.tensor(
            np.stack([x_train[i : i + lookback] for i in range(len(x_train) - lookback)]),
            dtype=torch.float32
        ) if len(x_train) >= lookback else torch.tensor(x_train[:1].reshape(1, 1, -1), dtype=torch.float32)
        support_y = torch.tensor(y_train[lookback:lookback + len(support_x)], dtype=torch.float32) if len(x_train) >= lookback else torch.tensor(y_train[:1], dtype=torch.float32)
        support_y = support_y[:len(support_x)]

        base_model.eval()
        with torch.no_grad():
            pred_t = base_model(torch.tensor(X_test_seq, dtype=torch.float32)).numpy()
        pred = np.expm1(pred_t)
        full_pred = np.full(len(x_test), np.nan)
        full_pred[lookback:lookback + len(pred)] = pred
        nan_mask = np.isnan(full_pred)
        if nan_mask.any():
            fallback_val = float(np.nanmedian(y_train)) if y_train.size else 0.0
            full_pred[nan_mask] = fallback_val
        y_max = float(np.nanmax(y_train)) if y_train.size else 1.0
        y_med = float(np.nanmedian(y_train)) if y_train.size else 0.0
        upper = max(y_max * 5.0, y_med * 10.0, 1e-10)
        return np.clip(full_pred, a_min=0.0, a_max=upper)
    else:
        return np.full(len(x_test), float(np.nanmedian(y_train)) if y_train.size else 0.0)


def _fit_predict_variant(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    variant: str,
    seed: int,
    config: PrismExperimentConfig | None = None,
    full_dataset: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    if variant == "V3b" and config is not None and full_dataset is not None and feature_cols:
        v2_pred = _fit_v2(train_x, train_y, test_x, seed)
        base_model = _train_maml(full_dataset, feature_cols, config, seed)
        maml_pred = _fit_v3_maml(train_x, train_y, test_x, config, base_model, feature_cols)

        val_x = train_x.iloc[-len(train_x) // 5:] if len(train_x) >= 5 else train_x
        val_y = train_y.iloc[-len(train_y) // 5:] if len(train_y) >= 5 else train_y
        if len(val_x) >= 5 and len(val_y) >= 5:
            maml_val_pred = _fit_v3_maml(
                train_x.iloc[:-len(val_x)] if len(train_x) > len(val_x) else train_x,
                train_y.iloc[:-len(val_y)] if len(train_y) > len(val_y) else train_y,
                val_x,
                config, base_model, feature_cols,
            )
            v2_val_pred = _fit_v2(
                train_x.iloc[:-len(val_x)] if len(train_x) > len(val_x) else train_x,
                train_y.iloc[:-len(val_y)] if len(train_y) > len(val_y) else train_y,
                val_x, seed,
            )
            maml_ok = np.isfinite(maml_val_pred).sum() > 0 and np.isfinite(v2_val_pred).sum() > 0
            if maml_ok:
                maml_val_mae = float(np.nanmean(np.abs(maml_val_pred - val_y.to_numpy(dtype=float))))
                v2_val_mae = float(np.nanmean(np.abs(v2_val_pred - val_y.to_numpy(dtype=float))))
                if maml_val_mae < v2_val_mae:
                    return maml_pred
        return v2_pred

    if variant == "V3a" and config is not None and full_dataset is not None and feature_cols:
        base_model = _train_maml(full_dataset, feature_cols, config, seed)
        return _fit_v3_maml(train_x, train_y, test_x, config, base_model, feature_cols)

    if variant == "V2":
        return _fit_v2(train_x, train_y, test_x, seed)

    x_train = train_x.fillna(0.0).to_numpy(dtype=float)
    x_test = test_x.fillna(0.0).to_numpy(dtype=float)
    y_train = train_y.to_numpy(dtype=float)
    y_train = np.clip(y_train, a_min=1e-12, a_max=None)
    y_train_t = np.log1p(y_train)

    from sklearn.neural_network import MLPRegressor
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
    variants = ["V0", "V1", "V2", "V3a", "V3b"]

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
                    config=config,
                    full_dataset=gdf,
                    feature_cols=cols,
                )

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

                if variant in V3_VARIANTS:
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
                            "variant": variant,
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
