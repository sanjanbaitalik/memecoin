from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graph.risk_aware_graph import build_risk_aware_graph, greedy_maximal_independent_set, graph_statistics
from scie_revision.common import save_table_bundle, scie_path


def generate_graph_threshold_sensitivity(panel: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    out = output_dir or scie_path("outputs", "scie_revision", "reports")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    quantiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    rows = []
    for q in quantiles:
        try:
            nodes, edges = build_risk_aware_graph(panel, threshold=0.5, quantile_threshold=q)
            mis = greedy_maximal_independent_set(nodes["token"].astype(str).tolist(), edges)
            stats = graph_statistics(panel, edges, mis, threshold=q)
            stats_dict = {row["metric"]: row["value"] for _, row in stats.iterrows()}
            rows.append({
                "quantile_threshold": q,
                "n_edges": stats_dict.get("number_of_edges", 0),
                "graph_density": stats_dict.get("graph_density", 0),
                "redundancy_reduction_pct": stats_dict.get("redundancy_reduction_percentage", 0),
                "selected_mis_size": stats_dict.get("selected_mis_support_size", 0),
                "avg_degree": stats_dict.get("average_degree", 0),
            })
        except Exception:
            rows.append({"quantile_threshold": q, "n_edges": 0, "graph_density": 0, "redundancy_reduction_pct": 0, "selected_mis_size": 0, "avg_degree": 0})

    table = pd.DataFrame(rows)
    (out / "graph_threshold_sensitivity.md").write_text(
        "# Graph Threshold Sensitivity Analysis\n\n"
        "This report shows how different quantile thresholds affect graph structure and MIS diversification.\n\n"
        + table.to_markdown(index=False),
        encoding="utf-8",
    )
    return table
