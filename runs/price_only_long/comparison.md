# Price-only RL long campaign — held-out comparison

demand.kind: `tree_elastic` | held-out months 6 & 12 | n=30 | `score = mean_true_revenue - 200 * mean_capacity_shortfall`

Price-only agents/baselines: 1D price + selling-limit controller. Joint agents (`bc_sac_final`, `rl_best`): full (price, SL) action. Same held-out seeds/months and business metrics for apples-to-apples comparison.

| policy | n_episodes | mean_true_revenue | mean_load_factor | mean_remain_inv | oversell_rate | undersell_gt1500_rate | sellout_rate | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_sac_final | 30 | 1114825.17 | 0.8964 | 1036.19 | 0.4 | 0.4333 | 0.4 | 870575.61 |
| price_only_ppo_analytic@200k | 30 | 1087614.96 | 0.8696 | 1303.79 | 0.0 | 0.4667 | 0.0 | 826856.83 |
| rl_best_sac@200k | 30 | 1085378.54 | 0.8697 | 1302.7 | 0.1 | 0.4333 | 0.1 | 823018.55 |
| price_only_sac_optimize1d@150k | 30 | 1071552.63 | 0.8733 | 1267.1 | 0.0 | 0.4333 | 0.0 | 818132.08 |
| myopic_greedy@joint | 30 | 1049488.47 | 0.8404 | 1595.92 | 0.0 | 0.4333 | 0.0 | 730304.77 |
| myopic_greedy@price_only | 30 | 1049488.47 | 0.8404 | 1595.92 | 0.0 | 0.4333 | 0.0 | 730304.77 |
| fixed_price_80@price_only | 30 | 961979.13 | 0.88 | 1199.63 | 0.0 | 0.4333 | 0.0 | 722054.0 |
| fixed_price_100@price_only | 30 | 1079306.79 | 0.7981 | 2018.7 | 0.0 | 0.4333 | 0.0 | 675567.16 |
| fixed_price_80@joint | 30 | 1024314.44 | 0.9431 | 568.7 | 0.5667 | 0.4333 | 0.5667 | 673174.45 |
| heuristic_booking_limit@price_only | 30 | 1101105.58 | 0.7385 | 2614.63 | 0.0 | 0.7 | 0.0 | 578180.5 |

## Artifact paths

- `price_only_ppo_analytic@200k`: `artifacts/price_only_long/rl_ppo_analytic.zip`
- `price_only_sac_optimize1d@150k`: `artifacts/price_only_long/rl_sac_optimize1d.zip`
- `rl_best_sac@200k`: `artifacts/tree_long/best/rl_best.zip`
- `bc_sac_final`: `artifacts/bc_sac/rl_bc_sac_final.zip`

## Verdict vs `bc_sac_final`

| Question | Answer |
| --- | --- |
| price_only PPO beat `bc_sac_final` on score? | **NO** (826,857 vs 870,576) |
| price_only SAC beat `bc_sac_final` on score? | **NO** (818,132 vs 870,576) |
| price_only PPO beat `rl_best` on score? | **YES** (826,857 vs 823,019), oversell 0 vs 0.1 |
| Prefer for zero-oversell constraint? | **price_only PPO** (or `rl_best`) over `bc_sac_final` |
| Prefer for max score? | **`bc_sac_final`** |

See `NOTES.md` for training details and caveats.

