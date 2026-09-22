# Tree-demand sanity comparison

An early 30k-step check that the tree demand model, the baselines, and PPO all
run end-to-end. Superseded by the campaigns in `docs/EXPERIMENT_LOG.md` §2
onward; kept because it is the one table that shows how PPO looks *before* the
HP-merge fix and a real training budget.

**Demand:** `tree_elastic` (default) — GBT base + multiplicative elasticity `-1.2` @ `ref_price=100`  
**Eval:** 20 held-out episodes (months 6 & 12), `shortfall_weight=200`  
**PPO:** 30k timesteps, seed 42, `configs/experiment_tree_ppo.yaml` merged on default  
**Date:** 2026-09-17

## Results

| policy | mean true revenue | load factor | remain inv | sellout rate | score |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_price_100 | 1,035,267 | 0.764 | 2356 | 0.20 | 487,910 |
| fixed_price_80 | 967,861 | 0.887 | 1127 | 0.40 | **583,466** |
| fixed_price_120 | 972,863 | 0.601 | 3987 | 0.05 | 150,425 |
| myopic_greedy (1D grid) | 989,829 | 0.789 | 2107 | 0.00 | 568,341 |
| heuristic_booking_limit | 1,013,278 | 0.690 | 3098 | 0.00 | 393,664 |
| rl_ppo @ 30k | 1,027,478 | 0.781 | 2190 | 0.20 | 525,878 |

## Notes

- Myopic uses **grid search** over `demand_model.predict_mean` (no closed-form free lunch under tree base).
- At 30k steps PPO is **near myopic on revenue** but trails `fixed_price_80` / myopic on **score** (fill vs shortfall). Longer budgets (100k–200k) and reward tuning are expected before claiming RL wins — same lesson as the legacy linear sim.
- Neither the 30k checkpoint nor the raw tables were kept; the numbers above are the record.

## Reproduce

```bash
source .venv/bin/activate
rprl-baselines -c configs/default.yaml --episodes 20 --out-dir runs/scratch/tree_sanity_baselines
rprl-train -c configs/experiment_tree_ppo.yaml --timesteps 30000 --seed 42 --run-name tree_sanity_ppo
rprl-eval -c configs/default.yaml --model artifacts/tree_sanity_ppo/final_model.zip --episodes 20 --out-dir runs/scratch/tree_sanity_eval
```

## Follow-up: longer training

See [`runs/tree_long/comparison.md`](tree_long/comparison.md) and [`runs/tree_long/NOTES.md`](tree_long/NOTES.md).

After HP-merge fix + 50k screen + 300k PPO / 200k SAC, **best RL = SAC @ 200k score ≈823k** on 30 held-out episodes — **beats** myopic (≈578k) and fixed_80 (≈590k). Best PPO ≈817k (`long_007`).
