# Ablation Study

The ablation study evaluates incremental PRISM components across V0 (price-only), V1 (V0 + sentiment), V2 (V1 + graph/MIS), V3a (V2 + always-adapted MAML), and V3b (V2 + validation-gated MAML). See `tables/table_ablation.csv` for per-variant summary metrics.

V3b uses validation-gated adaptation: MAML weights replace V2 weights only when token-level validation MAE improves, otherwise V2 weights are retained. This avoids counterproductive adaptation on tokens where MAML does not help.