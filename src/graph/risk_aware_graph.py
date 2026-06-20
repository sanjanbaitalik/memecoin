from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from scie_revision.common import pairwise_correlation_mean


@dataclass
class GraphBuildResult:
    edges: pd.DataFrame
    selected_tokens: list[str]
    stats: pd.DataFrame


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    merged = pd.concat([a, b], axis=1).dropna()
    if merged.shape[0] < 3:
        return 0.0
    return float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))


def _compute_token_features(token_frame: pd.DataFrame) -> pd.DataFrame:
    frame = token_frame.sort_values("datetime").copy()
    price = frame["price"].to_numpy(dtype=float)
    returns = np.diff(price) / np.where(price[:-1] == 0, np.nan, price[:-1])
    returns = np.concatenate([[0.0], returns])

    features = pd.DataFrame(index=frame.index)
    features["return_1d"] = returns
    features["return_3d"] = pd.Series(returns).rolling(3, min_periods=1).mean().to_numpy()
    features["return_7d"] = pd.Series(returns).rolling(7, min_periods=1).mean().to_numpy()
    features["volatility_7d"] = pd.Series(returns).rolling(7, min_periods=1).std().to_numpy()
    features["volatility_14d"] = pd.Series(returns).rolling(14, min_periods=1).std().to_numpy()

    if "volume" in frame.columns:
        vol = frame["volume"].to_numpy(dtype=float)
        features["log_volume"] = np.log1p(np.abs(vol))
        features["volume_ma7"] = pd.Series(vol).rolling(7, min_periods=1).mean().to_numpy()
    else:
        features["log_volume"] = 0.0
        features["volume_ma7"] = 0.0

    for prefix in ["twitter_sentiments", "reddit_sentiment", "telegram_sentiment"]:
        if prefix in frame.columns:
            features[f"{prefix}_mean7"] = pd.Series(frame[prefix].to_numpy(dtype=float)).rolling(7, min_periods=1).mean().to_numpy()
        else:
            features[f"{prefix}_mean7"] = 0.0

    features["price_level"] = np.log1p(np.abs(price))
    cum_max = np.maximum.accumulate(price)
    features["drawdown"] = (price - cum_max) / np.where(cum_max == 0, np.nan, cum_max)
    features["drawdown"] = features["drawdown"].fillna(0.0)

    neg_rets = returns[returns < 0]
    cvar_95 = float(np.mean(neg_rets)) if len(neg_rets) > 0 else 0.0
    features["cvar_95"] = cvar_95

    return features.fillna(0.0)


def edge_score(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_features = _compute_token_features(left)
    right_features = _compute_token_features(right)

    common_cols = [c for c in left_features.columns if c in right_features.columns]
    if not common_cols:
        return _safe_corr(left["price"], right["price"])

    corrs = []
    for col in common_cols:
        c = _safe_corr(left_features[col], right_features[col])
        if np.isfinite(c):
            corrs.append(abs(c))

    price_corr = abs(_safe_corr(left["price"], right["price"]))
    if np.isfinite(price_corr):
        corrs.append(price_corr)

    return float(np.mean(corrs)) if corrs else 0.0


def build_risk_aware_graph(
    panel: pd.DataFrame,
    threshold: float = 0.5,
    quantile_threshold: float | None = 0.85,
    k_neighbors: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_frames = {token: frame.sort_values("datetime") for token, frame in panel.groupby("sheet_name")}
    nodes = sorted(token_frames)

    if len(nodes) < 2:
        return pd.DataFrame({"token": nodes}), pd.DataFrame(columns=["source", "target", "score", "abs_score"])

    all_scores: list[dict] = []
    for left, right in combinations(nodes, 2):
        left_frame = token_frames[left]
        right_frame = token_frames[right]
        score = edge_score(left_frame, right_frame)
        all_scores.append({"source": left, "target": right, "score": score, "abs_score": abs(score)})

    if not all_scores:
        return pd.DataFrame({"token": nodes}), pd.DataFrame(columns=["source", "target", "score", "abs_score"])

    scores_df = pd.DataFrame(all_scores)

    if k_neighbors is not None and k_neighbors > 0:
        edge_rows = []
        for node in nodes:
            node_scores = scores_df[(scores_df["source"] == node) | (scores_df["target"] == node)].copy()
            if node_scores.empty:
                continue
            node_scores["neighbor"] = node_scores.apply(lambda r: r["target"] if r["source"] == node else r["source"], axis=1)
            top_k = node_scores.nlargest(k_neighbors, "abs_score")
            for _, row in top_k.iterrows():
                edge_rows.append({"source": min(node, row["neighbor"]), "target": max(node, row["neighbor"]), "score": row["score"], "abs_score": row["abs_score"]})
        if edge_rows:
            edges = pd.DataFrame(edge_rows).drop_duplicates(subset=["source", "target"])
        else:
            edges = pd.DataFrame(columns=["source", "target", "score", "abs_score"])
    elif quantile_threshold is not None:
        q = float(scores_df["abs_score"].quantile(quantile_threshold))
        edges = scores_df[scores_df["abs_score"] >= q].copy()
    else:
        edges = scores_df[scores_df["abs_score"] >= threshold].copy()

    return pd.DataFrame({"token": nodes}), edges


def greedy_maximal_independent_set(nodes: list[str], edges: pd.DataFrame) -> list[str]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for _, row in edges.iterrows():
        adjacency[str(row["source"])].add(str(row["target"]))
        adjacency[str(row["target"])].add(str(row["source"]))
    ordered = sorted(nodes, key=lambda n: (len(adjacency.get(n, set())), n))
    chosen: list[str] = []
    excluded: set[str] = set()
    for node in ordered:
        if node in excluded:
            continue
        chosen.append(node)
        excluded.add(node)
        excluded.update(adjacency.get(node, set()))
    return chosen


def connected_components(nodes: list[str], edges: pd.DataFrame) -> int:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for _, row in edges.iterrows():
        a = str(row["source"])
        b = str(row["target"])
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[str] = set()
    components = 0
    for node in nodes:
        if node in seen:
            continue
        components += 1
        queue = deque([node])
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(adjacency.get(cur, set()) - seen)
    return components


def graph_statistics(
    panel: pd.DataFrame,
    edges: pd.DataFrame,
    selected_tokens: list[str],
    threshold: float,
) -> pd.DataFrame:
    nodes = sorted(panel["sheet_name"].astype(str).unique().tolist())
    selected_panel = panel[panel["sheet_name"].astype(str).isin(selected_tokens)].copy()
    full_corr = pairwise_correlation_mean(panel.pivot_table(index="datetime", columns="sheet_name", values="price", aggfunc="last").ffill().bfill())
    selected_corr = pairwise_correlation_mean(selected_panel.pivot_table(index="datetime", columns="sheet_name", values="price", aggfunc="last").ffill().bfill()) if not selected_panel.empty else float("nan")
    n_nodes = len(nodes)
    n_edges = int(len(edges))
    density = float((2 * n_edges) / max(n_nodes * (n_nodes - 1), 1))
    selected_edge_count = int(sum(
        1 for _, row in edges.iterrows() if str(row["source"]) in selected_tokens and str(row["target"]) in selected_tokens
    ))
    avg_degree = float((2 * n_edges) / max(n_nodes, 1))
    selected_avg_degree = float((2 * selected_edge_count) / max(len(selected_tokens), 1))
    redundancy_reduction = float(100.0 * (1.0 - (selected_corr / full_corr))) if np.isfinite(full_corr) and full_corr not in (0.0,) and np.isfinite(selected_corr) else 0.0

    full_prices = panel.pivot_table(index="datetime", columns="sheet_name", values="price", aggfunc="last").ffill().bfill()
    full_returns = full_prices.pct_change().dropna()
    full_cvar = float(full_returns.quantile(0.05).mean()) if not full_returns.empty else 0.0

    sel_prices = selected_panel.pivot_table(index="datetime", columns="sheet_name", values="price", aggfunc="last").ffill().bfill()
    sel_returns = sel_prices.pct_change().dropna()
    sel_cvar = float(sel_returns.quantile(0.05).mean()) if not sel_returns.empty else 0.0

    return pd.DataFrame(
        [
            {"metric": "number_of_nodes", "value": n_nodes},
            {"metric": "number_of_edges", "value": n_edges},
            {"metric": "graph_density", "value": density},
            {"metric": "average_degree", "value": avg_degree},
            {"metric": "connected_components", "value": connected_components(nodes, edges)},
            {"metric": "selected_mis_size", "value": len(selected_tokens)},
            {"metric": "selected_edge_count", "value": selected_edge_count},
            {"metric": "average_pairwise_correlation_before_mis", "value": full_corr},
            {"metric": "average_pairwise_correlation_after_mis", "value": selected_corr},
            {"metric": "redundancy_reduction_percentage", "value": redundancy_reduction},
            {"metric": "average_cvar_before_mis", "value": full_cvar},
            {"metric": "average_cvar_after_mis", "value": sel_cvar},
            {"metric": "similarity_threshold_or_quantile", "value": threshold},
        ]
    )


def calibrate_threshold(
    panel: pd.DataFrame,
    candidate_thresholds: list[float] | None = None,
    use_quantile: bool = False,
    min_density: float = 0.05,
    max_density: float = 0.60,
) -> tuple[float, dict[str, float]]:
    if candidate_thresholds is None:
        candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    if use_quantile:
        candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_score = -float("inf")
    best_threshold = candidate_thresholds[0]
    best_info: dict[str, float] = {}

    for thresh in candidate_thresholds:
        if use_quantile:
            _, edges = build_risk_aware_graph(panel, quantile_threshold=thresh)
        else:
            _, edges = build_risk_aware_graph(panel, threshold=thresh)

        params = {}
        stat_df = graph_statistics(panel, edges, [], thresh)
        params["threshold"] = thresh
        params["density"] = float(stat_df[stat_df["metric"] == "graph_density"]["value"].iloc[0])
        params["mis_size"] = 0
        mis_size_total = len(panel["sheet_name"].unique())

        if len(edges) > 0:
            mis = greedy_maximal_independent_set(
                sorted(panel["sheet_name"].astype(str).unique().tolist()),
                edges,
            )
            params["mis_size"] = len(mis)

        dens = params["density"]
        mis_size = params["mis_size"]

        if dens > 0 and mis_size > 0:
            mis_size_norm = mis_size / max(mis_size_total, 1)
            redundancy_norm = 1.0 - min(dens, 1.0)
            density_penalty = abs(dens - 0.25)
            score = mis_size_norm + redundancy_norm - density_penalty
        else:
            score = -float("inf")

        params["score"] = score

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_info = params

    return best_threshold, best_info
