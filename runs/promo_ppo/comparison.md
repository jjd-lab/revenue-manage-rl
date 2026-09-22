# Early-promo + pace price-only PPO — held-out comparison

demand.kind: `tree_elastic` | held-out months 6 & 12 | n=30 | `score = mean_true_revenue - 200 * mean_capacity_shortfall`

Training: `early_promo` price post-process + `pace_reward` + `soft_day_upweight`. Business metrics below are **unshaped**.

| policy | n_episodes | mean_true_revenue | mean_load_factor | mean_remain_inv | oversell_rate | undersell_gt1500_rate | sellout_rate | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc_sac_final | 30 | 1114825.17 | 0.8964 | 1036.19 | 0.4 | 0.4333 | 0.4 | 870575.61 |
| pace_ppo@200k | 30 | 1072480.34 | 0.8799 | 1200.81 | 0.0 | 0.4333 | 0.0 | 832318.92 |
| price_only_ppo_analytic@200k | 30 | 1087614.96 | 0.8696 | 1303.79 | 0.0 | 0.4667 | 0.0 | 826856.83 |
| rl_best_sac@200k | 30 | 1085378.54 | 0.8697 | 1302.7 | 0.1 | 0.4333 | 0.1 | 823018.55 |
| promo_ppo@200k | 30 | 1061456.31 | 0.8793 | 1206.56 | 0.0 | 0.4333 | 0.0 | 820144.9 |
| promo_ppo_bestckpt | 30 | 1055452.32 | 0.853 | 1470.19 | 0.0 | 0.4333 | 0.0 | 761413.73 |
| myopic_greedy | 30 | 1049488.47 | 0.8404 | 1595.92 | 0.0 | 0.4333 | 0.0 | 730304.77 |
| fixed_price_80 | 30 | 961979.13 | 0.88 | 1199.63 | 0.0 | 0.4333 | 0.0 | 722054.0 |

## Artifact paths

- `promo_ppo@200k`: `artifacts/promo_ppo/rl_promo_ppo.zip`
- `promo_ppo_bestckpt`: `artifacts/promo_ppo/rl_promo_ppo_best.zip`
- `pace_ppo@200k`: `artifacts/pace_ppo/rl_pace_ppo.zip`
- `price_only_ppo_analytic@200k`: `artifacts/price_only_long/rl_ppo_analytic.zip`
- `rl_best_sac@200k`: `artifacts/tree_long/best/rl_best.zip`
- `bc_sac_final`: `artifacts/bc_sac/rl_bc_sac_final.zip`


## Success criteria (honest)

| Question | Answer |
| --- | --- |
| undersell>1500 improved vs **pace_ppo**? | **NO** (tied 0.433) |
| undersell>1500 improved vs **price_only_ppo**? | **YES** (0.467 → 0.433; same as pace) |
| score vs **pace_ppo**? | **NO** (820.1k < 832.3k, −12.2k) |
| score vs **bc_sac_final**? | **NO** (820.1k < 870.6k) |

Smoke: `test_early_promo_triggers_on_soft_state` green. Prefer **pace_ppo** among price-only; early promo taxes revenue without extra fill.

