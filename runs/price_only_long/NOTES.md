# Price-only RL long campaign — NOTES

**Demand:** `tree_elastic` (default)  
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29  
**Score:** `mean_true_revenue - 200 * mean_capacity_shortfall`  
**Date:** 2026-09-22  
**Threads:** torch and OpenMP capped at 1 (the trainer's default)

The PPO row is the shipped checkpoint, scored again. The SAC row is a new training at seed 42: the original weights were never kept, so this number is a retrain, not a replay.

## What we trained

| run | algo | SL controller | timesteps | seed | wall clock |
| --- | --- | --- | ---: | ---: | --- |
| `ppo_analytic` | PPO (`n_envs=2`) | **analytic** (`overbook_factor=1.05`) | 200 000 | 42 | ~3.5 min |
| `sac_optimize1d` | SAC (`n_envs=1`) | **optimize_1d** (`n_grid=41`) | 150 000 | 42 | ~5.8 min |

**optimize_1d choice:** kept for SAC. Env-step probe showed ~0.5 ms/step (vs ~0.4 ms analytic) — not a bottleneck vs SAC gradient updates. No analytic-SAC fallback needed.

Configs: `configs/experiment_price_only_ppo_long.yaml`, `configs/experiment_price_only_sac_long.yaml`.

EvalCallback shaped-reward “best” peaked early (PPO @40k, SAC @30k) then dipped — same pattern as BC→SAC. **Final** checkpoints used for the table below.

## Held-out results (30 eps)

| policy | mode | mean revenue | load | remain | oversell | undersell>1500 | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **bc_sac_final** | joint | **1,114,825** | 0.896 | 1036 | **0.40** | 0.433 | **870,576** |
| price_only_ppo_analytic@200k | price_only | 1,087,615 | 0.870 | 1304 | **0.00** | 0.467 | 826,857 |
| rl_best_sac@200k | joint | 1,085,363 | 0.870 | 1303 | 0.10 | 0.433 | 823,061 |
| price_only_sac_optimize1d@150k | price_only | 1,058,931 | 0.871 | 1287 | **0.00** | 0.433 | 801,591 |
| myopic_greedy | both* | 1,049,488 | 0.840 | 1596 | 0.00 | 0.433 | 730,305 |
| fixed_price_80@price_only | price_only | 961,979 | 0.880 | 1200 | 0.00 | 0.433 | 722,054 |
| fixed_price_100@price_only | price_only | 1,079,307 | 0.798 | 2019 | 0.00 | 0.433 | 675,567 |
| fixed_price_80@joint | joint | 1,024,314 | 0.943 | 569 | 0.57 | 0.433 | 673,174 |
| heuristic@price_only | price_only | 1,101,106 | 0.739 | 2615 | 0.00 | 0.700 | 578,181 |

\*myopic@joint and myopic@price_only matched exactly on this seed set (price trajectory dominates; SL differences did not move aggregates).

Full CSV/JSON: `runs/price_only_long/comparison_table.csv`, `comparison_summary.json`, `episode_metrics.csv`.

## Verdict vs `bc_sac_final`

| Question | Answer |
| --- | --- |
| Does **price_only PPO** beat `bc_sac_final` on **score**? | **NO** (−43.7k; 826.9k vs 870.6k) |
| Does **price_only SAC (optimize_1d)** beat `bc_sac_final` on **score**? | **NO** (−69.0k; 801.6k vs 870.6k) |
| Does either price_only beat `bc_sac_final` on **revenue**? | **NO** |
| Does price_only beat `bc_sac_final` on **oversell rate**? | **YES** — both price_only runs have **oversell=0** vs 0.40 |
| Does **price_only PPO** beat `rl_best` on score? | **YES** (+3.8k; 826.9k vs 823.0k) with zero oversell |
| Does price_only beat myopic / fixed_80 on score? | **YES** (both PPO and SAC) |

**Bottom line:** Price-only RL is a **clear improvement over classical baselines**, with a **safer oversell profile**. PPO sits just above joint `rl_best` (+3.8k) at zero oversell. The SAC retrain is 21.5k below `rl_best` and 69.0k below BC→SAC, still at zero oversell. Neither closes the gap to **BC→SAC final**, which still leads on revenue/score by filling harder (paying for that with 40% oversell episodes).

If the product constraint is **zero/low oversell**, price_only PPO is competitive with `rl_best` and preferable to `bc_sac_final`. If the objective is **raw score**, keep `bc_sac_final`.

## Artifacts

- `artifacts/price_only_long/rl_ppo_analytic.zip` ← PPO final @200k (recommended price_only)
- `artifacts/price_only_long/rl_ppo_analytic_best.zip` ← EvalCallback best (shaped; early)
- `artifacts/price_only_long/rl_sac_optimize1d.zip` ← copy of `sac_optimize1d/final_model.zip` @150k
- `artifacts/price_only_long/rl_sac_optimize1d_best.zip` ← EvalCallback best (shaped; early)
- `runs/price_only_long/ppo_analytic/`, `sac_optimize1d/` — EvalCallback curves and `train_meta.json`

## Ops / smoke

- Smoke: `pytest tests/test_smoke.py -q` — green before and after campaign.
- Cap Torch/OpenMP threads to 1 (same CPU pathology as tree_long / bc_sac).
