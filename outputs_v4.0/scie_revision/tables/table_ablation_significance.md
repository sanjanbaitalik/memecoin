# Ablation Significance

Paired Wilcoxon / Holm-corrected ablation steps.

| left | right | metric | n_pairs | wilcoxon_stat | p_value | effect_rank_biserial | left_mean | right_mean | metric_direction | favored_model | p_value_holm | significant_0_05 | comparison_group |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | V1 | mae | 1800 | 487386.0000 | 0.0000 | 0.2146 | 4.0539 | 0.6981 | lower_better | right_better | 0.0000 | TRUE | ablation |
| V1 | V2 | mae | 1800 | 408495.0000 | 0.0000 | 0.4960 | 0.6981 | 0.3005 | lower_better | right_better | 0.0000 | TRUE | ablation |
| V2 | V3 | mae | 1800 | 502881.0000 | 0.0000 | 0.3795 | 0.3005 | 0.1231 | lower_better | right_better | 0.0000 | TRUE | ablation |