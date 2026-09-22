# Both goals: safe SL + soft-day MPC — NOTES

**Demand:** `tree_elastic`  
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29  
**Score:** `mean_true_revenue - 200 * mean_capacity_shortfall`  
**Date:** 2026-09-17, refreshed 2026-09-21 and 2026-09-22 (see notes below)  
**Threads:** torch and OpenMP capped at 1 (the trainer's default)

## Goals

| | Goal | Target |
| --- | --- | --- |
| **(A)** | High score **without** ~40% oversell | oversell ≪ 0.40; score near bc_sac ~871k |
| **(B)** | Better soft-day undersell if physically possible | undersell>1500 < 0.433 |

## What we built (config-driven, no retrain)

1. **`control.safe_sl`** — joint post-process (`OversellGuardEnv`) projects policy SL down with chance/analytic cap from keep-rate + remain (`controls/oversell_cap.py`, renamed from `safe_sl.py` on 2026-09-21; the config key itself is unchanged). Knobs: `kind`, `overbook_factor`, `activate_remain_frac`, `mix_alpha`.
2. **`control.mpc`** — short-horizon price grid MPC on soft states inside `PriceOnlyWrapper` (`controls/price_mpc.py`). Soft-only + prefer-min-when-soft.
3. **Oracle soft-day ceiling** — `runs/oracle_ceiling/` (noise-free always-$80 / myopic / best-constant).

Configs: `configs/experiment_bc_sac_safe_sl.yaml`, `configs/experiment_pace_mpc.yaml`.

## Comparison table (held-out 30 eps)

> **2026-09-21 refresh.** `pace_ppo` / `pace/mpc` moved 832,319 → 831,842 (−0.06%):
> the price-only selling-limit controller was computing its low-remain check one
> day ahead of the day `ReservationEnv.step()` actually charges it against
> (`envs/price_only.py`), which occasionally changed a booking accept/reject at
> the margin. `bc_sac*` and `rl_best` did not go through that code path — their
> sub-$100 movement here is ordinary run-to-run inference noise from a torch/BLAS
> version difference between machines, the same noise the README already flags
> for training. Verdict below is unaffected; see `docs/EXPERIMENT_LOG.md` §5c.
>
> **2026-09-22 refresh.** The price MPC now sees the same decision day as the
> other controllers (`envs/price_only.py`). Rerun: every row, `pace/mpc`
> included, is unchanged to the cent — the MPC never overrides a price the pace
> policy has not already put at the floor.

| policy | score | oversell | undersell>1500 | mean remain | mean revenue | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **bc_sac raw** | **870,576** | **0.40** | 0.433 | 1036 | 1,114,825 | baseline high score |
| **bc_sac+safe_sl** (mix=0.25) | **863,220** | **0.00** | 0.433 | 1174 | 1,098,025 | **recommended (A)** |
| bc_sac+safe_sl_mix0.4 | 874,293 | 0.233 | 0.433 | 1103 | 1,106,687 | higher score, residual oversell |
| bc_sac+safe_sl_hard (mix=0) | 827,798 | 0.00 | 0.433 | 1286 | 1,085,094 | safer, larger score drop |
| **pace_ppo** | 831,842 | 0.00 | 0.433 | 1203 | 1,072,382 | zero oversell price-only |
| **pace/mpc** | 831,842 | 0.00 | 0.433 | 1203 | 1,072,382 | = pace_ppo (already @ $80 soft) |
| rl_best | 823,061 | 0.10 | 0.433 | 1303 | 1,085,363 | prior joint SAC |
| oracle soft (always $80) | — | 0.00 | **1.000** on soft | **2250** soft mean | — | structural ceiling |

Oracle detail: `runs/oracle_ceiling/NOTES.md` — **all 13/13 soft episodes** stay remain>1500 even at min price + max SL (noise-free).

## Verdict

| Goal | Result | Detail |
| --- | --- | --- |
| **(A) score + low oversell** | **PARTIAL → strong YES on safety** | Recommended `bc_sac+safe_sl` (`mix_alpha=0.25`, `activate_remain_frac=0.20`): **oversell 0.40 → 0.00**, score **870.6k → 863.2k (−7.4k, −0.8%)**. Beats pace_ppo (+31k) and rl_best (+40k) with zero oversell. Hard project (mix=0) also zero oversell but −43k score. |
| **(B) soft-day undersell** | **NO — structural ceiling** | Oracle: undersell>1500 on soft days is **unavoidable** under current tree base + elasticity −1.2 and price≥$80 (best soft remain ≈1895–2356). MPC / promo / pace cannot beat the **0.433** floor (= 13/30 soft episodes). `pace/mpc` is a no-op vs pace_ppo (policy already min-prices soft states). |

### Which lever moved which metric?

| Lever | Oversell | Undersell>1500 | Score |
| --- | --- | --- | --- |
| **safe_sl on bc_sac** | **↓ 0.40 → 0** | unchanged 0.433 | small drop (−7k @ mix=0.25) |
| **MPC on pace_ppo** | unchanged 0 | unchanged 0.433 | unchanged (redundant) |
| **Oracle / demand** | — | **proves floor** | — |

## Recommended artifacts

- Policy weights unchanged: `artifacts/bc_sac/rl_bc_sac_final.zip` + config `configs/experiment_bc_sac_safe_sl.yaml`
- Eval tables: `runs/both_goals/comparison.md`
- Oracle: `runs/oracle_ceiling/NOTES.md`

## Smoke / docs

- `pytest tests -q` — green (safe_sl + mpc tests added with this campaign)
- Docs: `docs/DESIGN.md`, `docs/EXTENDING.md` updated for `safe_sl` / `mpc`
