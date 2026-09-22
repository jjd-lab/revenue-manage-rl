# BC → SAC experiment NOTES

**Demand:** `tree_elastic` (default)  
**Protocol:** clone `myopic_greedy` on **train** months → warm SAC actor + seed replay → SAC fine-tune 150k (seed 42)  
**Selection score:** `mean_true_revenue - 200 * mean_capacity_shortfall`  
**Held-out eval:** 30 episodes, months 6 & 12  
**Date:** 2026-09-17

## Pipeline

1. Collect 300 expert episodes (train months) → 30 000 transitions (written to `runs/bc_sac/expert_dataset.npz` at train time; not shipped).
2. Behavioral clone SAC actor: MSE on `tanh(μ(obs))` vs expert actions, 50 epochs → **BC MSE ≈ 8.4e-4**.
3. Seed SAC replay with expert data; 1 000 critic warm steps.
4. Fine-tune SAC 150 000 timesteps.

Code: `src/reservation_pricing/algorithms/bc.py`, `train/bc_finetune.py`, config `configs/experiment_bc_sac.yaml`, CLI `rprl-bc-sac`.

## Held-out results (30 eps)

| policy | mean revenue | load | remain | oversell rate | undersell>1500 | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **bc_sac_final** | **1,114,825** | 0.896 | 1036 | 0.40 | 0.433 | **870,576** |
| rl_best_sac@200k | 1,085,379 | 0.870 | 1303 | 0.10 | 0.433 | 823,019 |
| myopic_greedy | 1,049,488 | 0.840 | 1596 | 0.00 | 0.433 | 730,305 |
| fixed_price_80 | 1,024,314 | 0.943 | 569 | 0.57 | 0.433 | 673,174 |
| bc_only | 1,033,868 | 0.817 | 1833 | 0.00 | 0.433 | 667,213 |
| bc_sac_bestckpt (Eval@30k) | 1,070,976 | 0.771 | 2289 | 0.00 | 0.533 | 613,113 |

`rl_best` score matches the tree_long leaderboard exactly (823 019).

## Verdict

| Question | Answer |
| --- | --- |
| Does **BC→SAC** beat `rl_best` on **score**? | **YES** (+47.6k; 870.6k vs 823.0k) |
| Does it beat `rl_best` on **undersell>1500 rate**? | **NO** — tied at 0.433 |
| Does it beat myopic / fixed_80 on score? | **YES** |
| Prefer EvalCallback “best” or final? | **Final** — shaped-reward best@30k undersells badly |

**Tradeoff:** BC→SAC final wins revenue/score by filling harder, but **oversell rate rises** (0.40 vs 0.10 for `rl_best`). If zero/low oversell is a hard constraint, keep `rl_best` or pure BC (`oversell=0`, score below myopic here).

**BC-only** clones myopic’s zero-oversell habit but does **not** beat myopic on this tree_elastic held-out set (unlike an earlier run on the linear demand model). Fine-tuning is necessary to beat `rl_best`.

## Artifacts

- `artifacts/bc_sac/rl_bc_sac_final.zip` ← recommended BC→SAC
- `artifacts/bc_sac/rl_bc_sac_best.zip` ← EvalCallback best (shaped reward; **not** preferred)
- `artifacts/bc_sac/bc_only.zip` ← pure imitation
- `artifacts/tree_long/best/rl_best.zip` ← prior best (SAC@200k, no BC)

## Ops notes

- Cap Torch/OpenMP threads (`RPRL_TORCH_THREADS`, default 1) — same CPU pathology as tree_long.
- Warm-critic before `learn()` must install SB3 logger (`algorithms/bc.py:warm_critic`).
