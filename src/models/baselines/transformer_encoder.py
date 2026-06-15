from __future__ import annotations

import math

import pandas as pd
import torch
import torch.nn as nn

from .common import BaselineRunResult, package_version
from .sequence_models import train_sequence_model, _to_sequences

from prism.evaluation.metrics import evaluate_frame


class _TransformerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_enc[:, :seq_len, :]
        out = self.transformer(h)
        return self.fc(out[:, -1, :]).squeeze(-1)


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], seed: int) -> BaselineRunResult:
    result = train_sequence_model(
        model_class=_TransformerModel,
        model_name="torch_transformer",
        train=train,
        val=val,
        test=test,
        feature_cols=feature_cols,
        seed=seed,
        hidden_size=64,
        num_layers=2,
        lookback=14,
        lr=1e-4,
        epochs=100,
        batch_size=32,
        patience=15,
    )

    pred = result["predictions"]
    eval_df = test[["price", "target_t_plus_h"]].copy()
    eval_df["yhat"] = pred
    metrics = evaluate_frame(eval_df, "yhat")
    prediction_frame = test[["sheet_name", "datetime", "price", "target_t_plus_h"]].copy()
    prediction_frame["yhat"] = pred

    return BaselineRunResult(
        predictions=prediction_frame,
        metrics=metrics,
        metadata={
            "model_name": "transformer_encoder",
            "backend": result["backend"],
            "fallback_used": result.get("fallback_used", False),
            "best_params": {"hidden_size": result["hidden_size"], "num_layers": result["num_layers"], "lookback": result["lookback"]},
            "library_version": package_version("torch"),
        },
    )
