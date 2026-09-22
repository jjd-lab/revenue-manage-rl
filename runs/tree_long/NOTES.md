# Tree-long campaign NOTES

**Demand:** default `tree_elastic`  
**Selection:** `score = mean_true_revenue - 200 * mean_capacity_shortfall`  
**Date:** 2026-09-17

## Bug fix (critical)

`merge_algo_train_cfg` previously let `algorithm:` defaults overwrite grid/`train` HPs.
Fixed so **train/grid wins**. Trainer also caps Torch/OMP threads; `n_epochs` is wired.

## Screen (10 PPO @ 50k)

Best screen: trial_003 @ **754.7k** (30-ep). Already beat myopic/fixed_80.

## Long training

| model | timesteps | score (30-ep) |
| --- | ---: | ---: |
| **SAC final** | 200k | **823.0k** ← best overall |
| PPO long_007 final | 300k | 816.6k |
| PPO long_003 best ckpt | 300k | 800.7k |
| PPO long_003 final | 300k | 797.8k |
| PPO long_000 final | 300k | 794.4k |
| PPO screen_003 | 50k | 754.7k |

## Baselines (30 eps)

| policy | score |
| --- | ---: |
| fixed_price_80 | 590.2k |
| myopic_greedy | 578.0k |

## Does RL beat baselines?

- Beat **myopic**? **YES** (823.0k vs 578.0k)
- Beat **fixed_80**? **YES** (823.0k vs 590.2k)

Best artifact: `artifacts/tree_long/best/rl_best.zip` (= SAC @ 200k final)

## What won

1. HP-merge fix so screens actually vary.
2. Longer budgets (50k → 200–300k) improve fill/score.
3. On tree_elastic, **SAC @ 200k** edges out best PPO @ 300k on selection score (slightly lower revenue, better shortfall/sellout tradeoff). Best pure PPO is `long_007` (ent_coef=0, n_steps=2048, γ=1.0).
4. Parallel multi-thread CPU training was pathological on the CPU-only machine used; the trainer caps torch and OpenMP at one thread by default (`RPRL_TORCH_THREADS`).
