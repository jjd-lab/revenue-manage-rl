# Pace + soft-day price-only PPO — held-out comparison

demand.kind: `tree_elastic` | held-out months 6 & 12 | n=30 | `score = mean_true_revenue - 200 * mean_capacity_shortfall`

Training used `pace_reward` + `soft_day_upweight` (shaped return only). Business metrics below are **unshaped**.

| policy | n_episodes | mean_true_revenue | mean_load_factor | mean_remain_inv | oversell_rate | undersell_gt1500_rate | sellout_rate | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_sac_final | 30 | 1114825.22 | 0.8964 | 1036.19 | 0.4 | 0.4333 | 0.4 | 870575.76 |
| pace_ppo@200k | 30 | 1072382.06 | 0.8797 | 1202.7 | 0.0 | 0.4333 | 0.0 | 831842.17 |
| price_only_ppo_analytic@200k | 30 | 1087614.96 | 0.8696 | 1303.79 | 0.0 | 0.4667 | 0.0 | 826856.83 |
| rl_best_sac@200k | 30 | 1085362.93 | 0.8697 | 1302.87 | 0.1 | 0.4333 | 0.1 | 823061.29 |
| myopic_greedy | 30 | 1049488.47 | 0.8404 | 1595.92 | 0.0 | 0.4333 | 0.0 | 730304.77 |
| fixed_price_80 | 30 | 961979.13 | 0.88 | 1199.63 | 0.0 | 0.4333 | 0.0 | 722054.0 |

## Artifact paths

- `pace_ppo@200k`: `artifacts/pace_ppo/rl_pace_ppo.zip`
- `price_only_ppo_analytic@200k`: `artifacts/price_only_long/rl_ppo_analytic.zip`
- `rl_best_sac@200k`: `artifacts/tree_long/best/rl_best.zip`
- `bc_sac_final`: `artifacts/bc_sac/rl_bc_sac_final.zip`
