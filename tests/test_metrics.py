from __future__ import annotations

import numpy as np

from prism.evaluation.metrics import mae, rmse, safe_mape


def test_metric_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 4.0])

    assert mae(y_true, y_pred) == 2.0 / 3.0
    assert round(rmse(y_true, y_pred), 6) == round((2.0 / 3.0) ** 0.5, 6)
    assert safe_mape(y_true, y_pred) > 0
