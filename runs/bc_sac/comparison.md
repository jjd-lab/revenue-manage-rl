# BC → SAC evaluation comparison

demand.kind: `tree_elastic` | held-out months 6 & 12 | n=30 | `score = mean_true_revenue - 200 * mean_capacity_shortfall`

| policy | n_episodes | mean_true_revenue | mean_load_factor | mean_remain_inv | oversell_rate | undersell_gt1500_rate | sellout_rate | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_sac_final | 30 | 1114825.17 | 0.8964 | 1036.19 | 0.4 | 0.4333 | 0.4 | 870575.61 |
| rl_best_sac@200k | 30 | 1085378.54 | 0.8697 | 1302.7 | 0.1 | 0.4333 | 0.1 | 823018.55 |
| myopic_greedy | 30 | 1049488.47 | 0.8404 | 1595.92 | 0.0 | 0.4333 | 0.0 | 730304.77 |
| fixed_price_80 | 30 | 1024314.44 | 0.9431 | 568.7 | 0.5667 | 0.4333 | 0.5667 | 673174.45 |
| bc_only | 30 | 1033867.8 | 0.8167 | 1833.27 | 0.0 | 0.4333 | 0.0 | 667213.47 |
| bc_sac_bestckpt | 30 | 1070975.71 | 0.7711 | 2289.32 | 0.0 | 0.5333 | 0.0 | 613112.58 |

## Artifact paths

- `rl_best_sac@200k`: `artifacts/tree_long/best/rl_best.zip`
- `bc_only`: `artifacts/bc_sac/bc_only.zip`
- `bc_sac_final`: `artifacts/bc_sac/rl_bc_sac_final.zip`
- `bc_sac_bestckpt`: `artifacts/bc_sac/rl_bc_sac_best.zip`
