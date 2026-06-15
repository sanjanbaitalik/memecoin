# Experiment Registry

- Timestamp: 2026-03-22T11:59:54.065418+00:00
```json
{
  "timestamp": "2026-03-22T11:59:54.065418+00:00",
  "data": {
    "universe": {
      "min_rows": 30,
      "min_price_coverage": 0.7,
      "require_any_sentiment": false
    },
    "preprocess": {
      "forecast_horizon_days": 3,
      "lookback_days": 14,
      "train_ratio": 0.8,
      "sentiment_mode": "raw"
    }
  },
  "model": {
    "baselines": {
      "train_ratio": 0.8,
      "seed": 42,
      "max_tokens_for_modeling": 200
    },
    "prism": {
      "train_ratio": 0.8,
      "seed": 42,
      "seeds": [
        11,
        17,
        23
      ],
      "optimizer": "adam",
      "learning_rate": 0.001,
      "batch_size": 64,
      "epochs": 200,
      "early_stopping_patience": 20,
      "maml_inner_steps": 5,
      "maml_outer_steps": 200,
      "maml_inner_lr": 0.01,
      "graph_volatility_window": 14,
      "graph_threshold": 0.5,
      "forecast_horizon_days": 3,
      "max_allowed_scale_mismatch_rate": 0.05
    }
  },
  "experiment": {
    "robustness": {
      "train_ratios": [
        0.7,
        0.8,
        0.9
      ],
      "lookback_windows": [
        7,
        14,
        21
      ]
    },
    "significance": {
      "metric": "mae",
      "correction": "holm",
      "alpha": 0.05
    },
    "subset_selection_rule": "eligible included tokens ranked by rows_available desc, then sheet_name asc; top max_tokens_for_modeling selected"
  }
}
```