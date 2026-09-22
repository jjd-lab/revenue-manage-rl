# Pace + soft-day price-only PPO — NOTES

**Demand:** `tree_elastic` (default)  
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29  
**Score:** `mean_true_revenue - 200 * mean_capacity_shortfall`  
**Date:** 2026-09-17  
**Threads:** torch and OpenMP capped at 1 (the trainer's default)

## What changed

Training-only reward shaping in `ReservationEnv` (business metrics stay unshaped):

1. **Pace / booking-curve reward** (`pace_reward: true`) — each step, if load
   factor is behind a linear target path to `pace_target_final=0.95`, subtract
   `pace_penalty * gap` (`pace_penalty=5.0`).
2. **Soft-day upweight** (`soft_day_upweight: true`) — multiply shaped return by
   weekday (×1.5) and/or off-peak (×1.25) factors; also when tree base mid-horizon
   `< 90` (×1.25). Training date sampling boosts soft calendar days (`sample_boost=2`).

Config: `configs/experiment_price_only_pace_ppo.yaml`.  
Docs: `docs/DESIGN.md` (Reward vs metrics), `docs/EVALUATING_POLICIES.md`.

## What we trained

| run | algo | SL | timesteps | seed | wall clock |
| --- | --- | --- | ---: | ---: | --- |
| `ppo_pace` | PPO (`n_envs=2`) | analytic (`overbook_factor=1.05`) | 200 000 | 42 | ~3.4 min |

EvalCallback shaped-reward “best” peaked @40k then dipped (same pattern as prior
price-only / BC→SAC). **Final** @200k used for primary verdict.

## Held-out results (30 eps)

| policy | mode | mean revenue | load | remain | oversell | undersell>1500 | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
> **2026-09-21 refresh.** `pace_ppo` moved 832,319 → 831,842 (−0.06%) when the
> price-only selling-limit controller was fixed to see the decision day
> (`envs/price_only.py`); every other row was unchanged to the cent. The
> `pace_ppo_bestckpt` row (score 726,962, remain 1645) is gone: that EvalCallback
> checkpoint was never kept, so it cannot be re-evaluated. Verdict unaffected.

| policy | mode | mean revenue | load | remain | oversell | undersell>1500 | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **bc_sac_final** | joint | **1,114,825** | 0.896 | 1036 | 0.40 | 0.433 | **870,576** |
| **pace_ppo@200k** | price_only | 1,072,382 | **0.880** | **1203** | **0.00** | **0.433** | **831,842** |
| price_only_ppo_analytic@200k | price_only | 1,087,615 | 0.870 | 1304 | 0.00 | 0.467 | 826,857 |
| rl_best_sac@200k | joint | 1,085,363 | 0.870 | 1303 | 0.10 | 0.433 | 823,061 |
| myopic_greedy | price_only | 1,049,488 | 0.840 | 1596 | 0.00 | 0.433 | 730,305 |
| fixed_price_80 | price_only | 961,979 | 0.880 | 1200 | 0.00 | 0.433 | 722,054 |

Full CSV/JSON: `runs/pace_ppo/comparison_table.csv`, `comparison_summary.json`, `episode_metrics.csv`.

## Verdict (success criteria)

| Question | Answer |
| --- | --- |
| Did **undersell>1500** drop vs prior price-only PPO? | **YES** — 0.467 → **0.433** |
| Did undersell>1500 drop below the common floor (myopic / bc_sac / rl_best)? | **NO** — still tied at **0.433** |
| Did **score** beat prior price-only PPO? | **YES** — 831.8k vs 826.9k (**+5.0k**) |
| Did score approach / beat **bc_sac_final**? | **NO** — still −38.7k (831.8k vs 870.6k); closed a little of the prior −43.7k gap |
| Did score beat **rl_best**? | **YES** — +8.8k with zero oversell |
| Prefer final or EvalCallback best? | **Final** — the shaped best@40k undersold more (remain 1645 when it was last evaluated) |

**Bottom line:** Pace + soft-day shaping is a **small but real** win over prior
price-only PPO: better fill (remain 1304→1203), undersell>1500 down to the
baseline floor, score +5.0k, still **zero oversell**. It does **not** close the
gap to **bc_sac_final** (which still leads on revenue/score by filling harder and
paying with 40% oversell episodes). Soft-day undersell is **not solved** — the
0.433 rate matches myopic / bc_sac on this held-out set.

## Artifacts

- `artifacts/pace_ppo/rl_pace_ppo.zip` ← final @200k (recommended)
- `artifacts/pace_ppo/rl_pace_ppo_best.zip` ← EvalCallback best (shaped; not preferred; not kept)
- `runs/pace_ppo/ppo_pace/` — EvalCallback curve and `train_meta.json`

## Ops / smoke

- `pytest tests/test_smoke.py -q` — green (incl. pace-penalty unit check)
- Cap Torch/OpenMP threads to 1
