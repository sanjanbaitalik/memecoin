from __future__ import annotations

import pandas as pd

from .common import BaselineRunResult, package_version
from .sequence_models import _GRUModel, train_sequence_model

from prism.evaluation.metrics import evaluate_frame


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], seed: int) -> BaselineRunResult:
    result = train_sequence_model(
        model_class=_GRUModel,
        model_name="torch_gru",
        train=train,
        val=val,
        test=test,
        feature_cols=feature_cols,
        seed=seed,
        hidden_size=48,
        num_layers=2,
        lookback=14,
        lr=1e-3,
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
            "model_name": "gru",
            "backend": result["backend"],
            "best_params": {"hidden_size": result["hidden_size"], "num_layers": result["num_layers"], "lookback": result["lookback"]},
            "library_version": package_version("torch"),
            "status": "failed" if "error" in result else "ok",
            "error": result.get("error"),
        },
    )
