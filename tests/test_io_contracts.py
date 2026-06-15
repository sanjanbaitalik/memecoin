from __future__ import annotations

import pandas as pd
import pytest

from prism.utils.io_contracts import DataContractError, safe_write_csv


def test_safe_write_csv_rejects_empty(tmp_path) -> None:
    target = tmp_path / "x.csv"
    with pytest.raises(DataContractError):
        safe_write_csv(pd.DataFrame(), target, step="t", allow_empty=False)
