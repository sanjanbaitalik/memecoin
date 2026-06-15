from __future__ import annotations

import pandas as pd

from prism.data.workbook import summarize_sheet


def test_schema_validation_detects_missing_required_columns() -> None:
    df = pd.DataFrame({"token": ["x"], "ticker": ["X"], "date": ["2024-01-01"]})
    out = summarize_sheet("x_sheet", df)
    assert out.valid_schema is False
    assert "price" in out.missing_required_columns
