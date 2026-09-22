# Soft-day-aware evaluation comparison

Stratifies episodes into **soft** (structurally weak demand) vs **peak/rich** so we do not over-penalize undersell when demand cannot fill even at floor price.

demand.kind: `tree_elastic` | n=30

## Classification

- rule: `structural`
- peak_months: `(7, 8, 11, 12)`
- weekend_dow: `(5, 6)`
- base_demand_threshold: `90.0`
- soft_focus_months: `(6,)`
- soft_oracle: `min_price`

Soft if (`structural`): mid-horizon tree base < threshold **OR** (weekday AND month in soft_focus_months). Alternative `rule=any`: weekday OR off-peak month OR low base.

## Selection scores

```
score_peak  = mean_rev_peak  - λ * mean_shortfall_peak
score_soft  = mean_rev_soft  - gap_to_oracle   # or - μ * mean_oversell_amount
score_aware = score_peak + score_soft
```

λ (lambda_peak)=200.0, μ (mu_soft)=200.0, soft_score_mode=`gap_to_oracle`

**Primary soft KPIs:** revenue, gap to soft oracle, frac days at/near floor, oversell.

**Primary peak KPIs:** revenue, load/remain, oversell, score (undersell matters here).

**Secondary / diagnostic:** overall `undersell>1500` (do not lead with this).

## Summary table

| policy | n_soft | n_peak | score_aware | score_peak | score_soft | soft_rev | soft_gap_to_oracle | soft_frac_at_floor | soft_oversell | peak_rev | peak_remain | peak_load | peak_oversell | peak_score | overall_rev | overall_score | overall_oversell | undersell_gt1500_diag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| myopic_greedy | 13 | 17 | 1972305.33 | 1101619.76 | 870685.56 | 870685.56 | 0.0 | 0.0 | 0.0 | 1186220.11 | 423.0 | 0.9577 | 0.0 | 1101619.76 | 1049488.47 | 730304.77 | 0.0 | 0.4333 |
| fixed_price_80 | 13 | 17 | 1816170.12 | 959401.94 | 856768.18 | 856768.18 | 0.0 | 1.0 | 0.0 | 1042434.56 | 415.16 | 0.9585 | 0.0 | 959401.94 | 961979.13 | 722054.0 | 0.0 | 0.4333 |
| rl_sac | 13 | 17 | 2081399.96 | 1221614.84 | 859785.12 | 859785.12 | 0.0 | 0.3623 | 0.0 | 1280208.74 | 292.97 | 0.9707 | 0.0 | 1221614.84 | 1098025.17 | 863219.56 | 0.0 | 0.4333 |
| joint_bc_sac_raw | 13 | 17 | 2094381.48 | 1234596.36 | 859785.12 | 859785.12 | 0.0 | 0.3623 | 0.0 | 1309855.88 | 49.73 | 0.995 | 0.7059 | 1234596.36 | 1114825.22 | 870575.76 | 0.4 | 0.4333 |
| joint_sac_rl_best | 13 | 17 | 2033263.9 | 1170178.12 | 863085.79 | 863085.79 | 0.0 | 0.15 | 0.0 | 1255339.56 | 410.56 | 0.9589 | 0.1765 | 1170178.12 | 1085362.93 | 823061.29 | 0.1 | 0.4333 |
| joint_ppo_long_007 | 13 | 17 | 2010885.26 | 1150494.04 | 860391.21 | 860391.21 | 0.0 | 0.5292 | 0.0 | 1265207.18 | 284.56 | 0.9715 | 0.3529 | 1150494.04 | 1089786.93 | 816644.33 | 0.2 | 0.4333 |
| price_only_pace_ppo | 13 | 17 | 2009913.95 | 1153145.77 | 856768.18 | 856768.18 | 0.0 | 1.0 | 0.0 | 1237263.27 | 420.59 | 0.9579 | 0.0 | 1153145.77 | 1072382.06 | 831842.17 | 0.0 | 0.4333 |
| price_only_ppo | 13 | 17 | 2003091.18 | 1145959.32 | 857131.86 | 857131.86 | 0.0 | 0.9446 | 0.0 | 1263866.74 | 589.54 | 0.941 | 0.0 | 1145959.32 | 1087614.96 | 826856.83 | 0.0 | 0.4667 |

## Soft slice (primary)

| policy | soft_rev | soft_gap_to_oracle | soft_frac_at_floor | soft_oversell | score_soft |
| --- | --- | --- | --- | --- | --- |
| myopic_greedy | 870685.56 | 0.0 | 0.0 | 0.0 | 870685.56 |
| fixed_price_80 | 856768.18 | 0.0 | 1.0 | 0.0 | 856768.18 |
| rl_sac | 859785.12 | 0.0 | 0.3623 | 0.0 | 859785.12 |
| joint_bc_sac_raw | 859785.12 | 0.0 | 0.3623 | 0.0 | 859785.12 |
| joint_sac_rl_best | 863085.79 | 0.0 | 0.15 | 0.0 | 863085.79 |
| joint_ppo_long_007 | 860391.21 | 0.0 | 0.5292 | 0.0 | 860391.21 |
| price_only_pace_ppo | 856768.18 | 0.0 | 1.0 | 0.0 | 856768.18 |
| price_only_ppo | 857131.86 | 0.0 | 0.9446 | 0.0 | 857131.86 |

## Peak slice (primary)

| policy | peak_rev | peak_remain | peak_load | peak_oversell | peak_score | score_peak |
| --- | --- | --- | --- | --- | --- | --- |
| myopic_greedy | 1186220.11 | 423.0 | 0.9577 | 0.0 | 1101619.76 | 1101619.76 |
| fixed_price_80 | 1042434.56 | 415.16 | 0.9585 | 0.0 | 959401.94 | 959401.94 |
| rl_sac | 1280208.74 | 292.97 | 0.9707 | 0.0 | 1221614.84 | 1221614.84 |
| joint_bc_sac_raw | 1309855.88 | 49.73 | 0.995 | 0.7059 | 1234596.36 | 1234596.36 |
| joint_sac_rl_best | 1255339.56 | 410.56 | 0.9589 | 0.1765 | 1170178.12 | 1170178.12 |
| joint_ppo_long_007 | 1265207.18 | 284.56 | 0.9715 | 0.3529 | 1150494.04 | 1150494.04 |
| price_only_pace_ppo | 1237263.27 | 420.59 | 0.9579 | 0.0 | 1153145.77 | 1153145.77 |
| price_only_ppo | 1263866.74 | 589.54 | 0.941 | 0.0 | 1145959.32 | 1145959.32 |

## Diagnostic (secondary)

Overall `undersell_gt1500_diag` is kept for continuity — do **not** lead with it when soft demand cannot fill at floor.

| policy | overall_rev | overall_score | overall_oversell | undersell_gt1500_diag | score_aware |
| --- | --- | --- | --- | --- | --- |
| myopic_greedy | 1049488.47 | 730304.77 | 0.0 | 0.4333 | 1972305.33 |
| fixed_price_80 | 961979.13 | 722054.0 | 0.0 | 0.4333 | 1816170.12 |
| rl_sac | 1098025.17 | 863219.56 | 0.0 | 0.4333 | 2081399.96 |
| joint_bc_sac_raw | 1114825.22 | 870575.76 | 0.4 | 0.4333 | 2094381.48 |
| joint_sac_rl_best | 1085362.93 | 823061.29 | 0.1 | 0.4333 | 2033263.9 |
| joint_ppo_long_007 | 1089786.93 | 816644.33 | 0.2 | 0.4333 | 2010885.26 |
| price_only_pace_ppo | 1072382.06 | 831842.17 | 0.0 | 0.4333 | 2009913.95 |
| price_only_ppo | 1087614.96 | 826856.83 | 0.0 | 0.4667 | 2003091.18 |
