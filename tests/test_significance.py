from __future__ import annotations

import pandas as pd

from prism.stats.significance import paired_wilcoxon_table


def test_paired_wilcoxon_table_runs() -> None:
    df = pd.DataFrame(
        {
            "sheet_name": ["a", "b", "a", "b"],
            "model": ["m1", "m1", "m2", "m2"],
            "mae": [0.9, 0.8, 0.7, 0.6],
        }
    )
    out = paired_wilcoxon_table(
        frame=df,
        model_col="model",
        token_col="sheet_name",
        metric="mae",
        comparisons=[("m1", "m2")],
    )
    assert len(out) == 1
    assert "p_value_holm" in out.columns
