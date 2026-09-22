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
| fixed_price_80 | 13 | 17 | 1729912.1 | 873143.92 | 856768.18 | 856768.18 | 0.0 | 1.0 | 0.0 | 1152438.06 | -698.24 | 1.0698 | 1.0 | 873143.92 | 1024314.44 | 673174.45 | 0.5667 | 0.4333 |
| bc_sac+safe_sl | 13 | 17 | 2081399.96 | 1221614.84 | 859785.12 | 859785.12 | 0.0 | 0.3623 | 0.0 | 1280208.74 | 292.97 | 0.9707 | 0.0 | 1221614.84 | 1098025.17 | 863219.56 | 0.0 | 0.4333 |
| pace_ppo | 13 | 17 | 2009913.95 | 1153145.77 | 856768.18 | 856768.18 | 0.0 | 1.0 | 0.0 | 1237263.27 | 420.59 | 0.9579 | 0.0 | 1153145.77 | 1072382.06 | 831842.17 | 0.0 | 0.4333 |

## Soft slice (primary)

| policy | soft_rev | soft_gap_to_oracle | soft_frac_at_floor | soft_oversell | score_soft |
| --- | --- | --- | --- | --- | --- |
| myopic_greedy | 870685.56 | 0.0 | 0.0 | 0.0 | 870685.56 |
| fixed_price_80 | 856768.18 | 0.0 | 1.0 | 0.0 | 856768.18 |
| bc_sac+safe_sl | 859785.12 | 0.0 | 0.3623 | 0.0 | 859785.12 |
| pace_ppo | 856768.18 | 0.0 | 1.0 | 0.0 | 856768.18 |

## Peak slice (primary)

| policy | peak_rev | peak_remain | peak_load | peak_oversell | peak_score | score_peak |
| --- | --- | --- | --- | --- | --- | --- |
| myopic_greedy | 1186220.11 | 423.0 | 0.9577 | 0.0 | 1101619.76 | 1101619.76 |
| fixed_price_80 | 1152438.06 | -698.24 | 1.0698 | 1.0 | 873143.92 | 873143.92 |
| bc_sac+safe_sl | 1280208.74 | 292.97 | 0.9707 | 0.0 | 1221614.84 | 1221614.84 |
| pace_ppo | 1237263.27 | 420.59 | 0.9579 | 0.0 | 1153145.77 | 1153145.77 |

## Diagnostic (secondary)

Overall `undersell_gt1500_diag` is kept for continuity — do **not** lead with it when soft demand cannot fill at floor.

| policy | overall_rev | overall_score | overall_oversell | undersell_gt1500_diag | score_aware |
| --- | --- | --- | --- | --- | --- |
| myopic_greedy | 1049488.47 | 730304.77 | 0.0 | 0.4333 | 1972305.33 |
| fixed_price_80 | 1024314.44 | 673174.45 | 0.5667 | 0.4333 | 1729912.1 |
| bc_sac+safe_sl | 1098025.17 | 863219.56 | 0.0 | 0.4333 | 2081399.96 |
| pace_ppo | 1072382.06 | 831842.17 | 0.0 | 0.4333 | 2009913.95 |
