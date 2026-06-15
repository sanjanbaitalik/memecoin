from __future__ import annotations

import pandas as pd

from prism.data.workbook import coerce_datetime_with_diagnostics


def test_date_not_destroyed_by_zero_time_column() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-11", "2024-11-12", "2024-11-13"]),
            "time": [0.0, 0.0, 0.0],
        }
    )
    dt = coerce_datetime_with_diagnostics(df, sheet_name="s")
    assert dt.notna().sum() == 3
