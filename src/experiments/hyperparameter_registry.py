from __future__ import annotations

from itertools import product


REGISTRY = {
    "prism": {
        "lookback": [7, 14, 21],
        "horizon": [3],
        "graph_threshold": [0.3, 0.5, 0.7],
        "volatility_window": [7, 14, 21],
        "sentiment_weight": [0.25, 0.50, 0.75, 1.00],
        "maml_inner_steps": [1, 3, 5],
        "inner_learning_rate": [0.001, 0.005, 0.01],
        "hidden_size": [32, 64, 128],
    },
    "neural_baselines": {
        "hidden_size": [48, 64, 128],
        "num_layers": [1, 2],
        "lookback": [7, 14, 21],
        "dropout": [0.0, 0.1],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
    },
    "tree_baselines": {
        "n_estimators": [200, 500],
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}


def grid(model_name: str):
    params = REGISTRY.get(model_name, {})
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, values)) for values in product(*[params[k] for k in keys])]
