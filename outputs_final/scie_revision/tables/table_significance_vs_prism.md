# Significance vs PRISM

Paired Wilcoxon / Holm-corrected comparison against PRISM.

| left | right | metric | n_pairs | wilcoxon_stat | p_value | effect_rank_biserial | left_mean | right_mean | metric_direction | favored_model | p_value_holm | significant_0_05 | comparison_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRISM | arima | mae | 200 | 7162.0000 | 0.0004 | 0.2874 | 0.1231 | 0.3140 | lower_better | left_better | 0.0009 | TRUE | baseline_vs_prism |
| PRISM | gru | mae | 200 | 112.0000 | 0.0000 | -0.9889 | 0.1231 | 0.3964 | lower_better | left_better | 0.0000 | TRUE | baseline_vs_prism |
| PRISM | lstm | mae | 200 | 174.0000 | 0.0000 | -0.9827 | 0.1231 | 0.4026 | lower_better | left_better | 0.0000 | TRUE | baseline_vs_prism |
| PRISM | persistence | mae | 200 | 252.0000 | 0.0000 | 0.9749 | 0.1231 | 0.1056 | lower_better | right_better | 0.0000 | TRUE | baseline_vs_prism |
| PRISM | prophet | mae | 200 | 6429.0000 | 0.0000 | 0.3603 | 0.1231 | 0.0882 | lower_better | right_better | 0.0000 | TRUE | baseline_vs_prism |
| PRISM | random_forest | mae | 200 | 6746.0000 | 0.0001 | 0.3288 | 0.1231 | 0.2619 | lower_better | left_better | 0.0002 | TRUE | baseline_vs_prism |
| PRISM | xgboost | mae | 200 | 8069.0000 | 0.0156 | 0.1971 | 0.1231 | 0.3719 | lower_better | left_better | 0.0156 | TRUE | baseline_vs_prism |