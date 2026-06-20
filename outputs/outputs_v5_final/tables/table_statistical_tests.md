# Statistical Significance Tests

Paired Wilcoxon with Holm correction

| left | right | metric | n_pairs | wilcoxon_stat | p_value | effect_rank_biserial | left_mean | right_mean | favored_model | p_value_holm | significant_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0 | V1 | mae | 45 | 192.0000 | 0.0267 | 0.4234 | 0.0118 | 0.0078 | right_better | 0.0802 | FALSE |
| V0 | V2 | mae | 45 | 45.0000 | 0.0000 | 0.9130 | 0.0118 | 0.0034 | right_better | 0.0000 | TRUE |
| V0 | V3a | mae | 45 | 39.0000 | 0.0000 | 0.9246 | 0.0118 | 0.0034 | right_better | 0.0000 | TRUE |
| V0 | V3b | mae | 45 | 39.0000 | 0.0000 | 0.9246 | 0.0118 | 0.0033 | right_better | 0.0000 | TRUE |
| V1 | V2 | mae | 45 | 192.0000 | 0.0002 | 0.6290 | 0.0078 | 0.0034 | right_better | 0.0011 | TRUE |
| V1 | V3a | mae | 45 | 186.0000 | 0.0002 | 0.6406 | 0.0078 | 0.0034 | right_better | 0.0011 | TRUE |
| V1 | V3b | mae | 45 | 186.0000 | 0.0002 | 0.6406 | 0.0078 | 0.0033 | right_better | 0.0011 | TRUE |
| V2 | V3a | mae | 45 | 414.0000 | 0.2404 | 0.2000 | 0.0034 | 0.0034 | left_better | 0.2502 | FALSE |
| V2 | V3b | mae | 45 | 0.0000 | 0.0001 | 1.0000 | 0.0034 | 0.0033 | right_better | 0.0009 | TRUE |
| V3a | V3b | mae | 45 | 126.0000 | 0.1251 | 0.3333 | 0.0034 | 0.0033 | right_better | 0.2502 | FALSE |