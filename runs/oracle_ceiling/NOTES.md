# Oracle soft-day fill ceiling

**Demand:** `tree_elastic` with **noise std = 0** (deterministic mean ceiling)
**Held-out:** months 6 & 12, seeds 0..29
**Soft definition:** mid-horizon tree base < 90 **or** June weekday
**SL:** max selling limit (15000) — inventory not the binding constraint

## Soft-episode counts

- Soft episodes: **13** / 30
- Soft episodes where **all** oracles still end with remain > 1500: **13** (1.000)
- Soft episodes where some oracle can get remain ≤ 1500: **0**

## Soft-slice aggregate (noise-free)

| policy | mean remain | mean load | undersell>1500 | oversell | mean revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| always_min_price_80 | 2250.0 | 0.775 | 1.000 | 0.000 | 858100 |
| myopic_grid_max_sl | 3150.0 | 0.685 | 1.000 | 0.000 | 872217 |
| best_constant_price | 2250.0 | 0.775 | 1.000 | 0.000 | 858100 |

## Full held-out aggregate (noise-free)

| policy | mean remain | mean load | undersell>1500 | oversell | mean revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| always_min_price_80 | -339.1 | 1.034 | 0.433 | 0.567 | 1113640 |
| myopic_grid_max_sl | 468.3 | 0.953 | 0.433 | 0.300 | 1175435 |
| best_constant_price | 795.5 | 0.920 | 0.433 | 0.200 | 1177507 |

## Verdict

**YES — structural ceiling:** even always-min-price / best-constant / myopic with max SL cannot clear remain≤1500 on essentially all soft days (best soft undersell>1500 rate = 1.000 via `always_min_price_80`). Undersell>1500 on soft days is **unavoidable** given current tree base + elasticity=-1.2 and price≥$80.

Prior RL / myopic / bc_sac all tied at **undersell>1500 ≈ 0.433** on noisy held-out (13/30 episodes). Soft count here is **13**.

### Unavoidable soft seeds (best oracle remain > 1500)

- seed 1 (2022-06-22, base50=89.3): best_remain=2356
- seed 2 (2022-06-01, base50=89.3): best_remain=2356
- seed 3 (2022-06-08, base50=89.3): best_remain=2356
- seed 5 (2022-06-21, base50=89.3): best_remain=2356
- seed 9 (2022-06-03, base50=93.3): best_remain=1895
- seed 11 (2022-06-27, base50=89.3): best_remain=2356
- seed 14 (2022-06-20, base50=89.3): best_remain=2356
- seed 16 (2022-06-07, base50=89.3): best_remain=2356
- seed 17 (2022-06-17, base50=93.3): best_remain=1895
- seed 19 (2022-06-03, base50=93.3): best_remain=1895
- seed 21 (2022-06-20, base50=89.3): best_remain=2356
- seed 24 (2022-06-16, base50=89.3): best_remain=2356
- seed 26 (2022-06-29, base50=89.3): best_remain=2356

## Implication for goals

- **(B) better soft-day undersell:** if structural ceiling binds, MPC / promo / pace shaping **cannot** beat the 0.433 floor on this demand; treat undersell improvements as only possible on the avoidable subset (if any).
- **(A) score + low oversell:** still addressable via safe-SL projection on bc_sac (cut oversell without needing more soft fill).

## Artifacts

- `oracle_all_episodes.csv`, `oracle_soft_episodes.csv`, `oracle_summary.json`
