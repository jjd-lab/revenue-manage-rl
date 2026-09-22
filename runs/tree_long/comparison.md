# Tree-elastic long campaign comparison

**Demand:** `tree_elastic` (default) — GBT base + elasticity `-1.2` @ `ref_price=100`
**Eval:** 30 held-out episodes (months 6 & 12), `shortfall_weight=200`
**Date:** 2026-09-17

## Results (30 held-out episodes)

| policy | mean true revenue | load factor | remain inv | sellout rate | score |
| --- | ---: | ---: | ---: | ---: | ---: |
| rl_sac_long@200k | 1,085,379 | 0.870 | 1303 | 0.10 | 823,019 ** |
| rl_ppo_long_007@300k | 1,089,787 | 0.880 | 1202 | 0.20 | 816,644 |
| rl_ppo_long_003_best@300k | 1,137,315 | 0.881 | 1191 | 0.27 | 800,673 |
| rl_ppo_long_003@300k | 1,144,387 | 0.892 | 1078 | 0.40 | 797,844 |
| rl_ppo_long_000_best@300k | 1,136,801 | 0.909 | 905 | 0.37 | 794,499 |
| rl_ppo_long_000@300k | 1,139,318 | 0.913 | 866 | 0.40 | 794,351 |
| rl_sac_long_best@200k | 1,080,600 | 0.855 | 1451 | 0.07 | 787,876 |
| rl_ppo_long_007_best@300k | 1,075,376 | 0.850 | 1501 | 0.03 | 772,056 |
| rl_ppo_screen_003@50k | 1,131,240 | 0.861 | 1391 | 0.30 | 754,654 |
| rl_ppo_screen_000@50k | 1,126,153 | 0.854 | 1456 | 0.30 | 743,141 |
| rl_ppo_screen_007@50k | 1,133,958 | 0.863 | 1368 | 0.30 | 725,498 |
| fixed_price_80 | 973,459 | 0.890 | 1101 | 0.40 | 590,213 |
| myopic_greedy | 995,325 | 0.791 | 2086 | 0.00 | 578,039 |
| fixed_price_100 | 1,042,169 | 0.767 | 2325 | 0.20 | 498,486 |
| heuristic_booking_limit | 1,021,645 | 0.694 | 3060 | 0.00 | 409,592 |
| fixed_price_120 | 985,191 | 0.608 | 3922 | 0.07 | 167,029 |

## Does best RL beat baselines?

- Beat **myopic** on score? **YES** (823,019 vs 578,039)
- Beat **fixed_price_80** on score? **YES** (823,019 vs 590,213)

**Best RL:** `rl_sac_long@200k` → `artifacts/tree_long/long/long_sac/final_model.zip` (also `artifacts/tree_long/best/rl_best.zip`)

## Training notes

- Fixed HP-merge bug (`train`/grid overrides were overwritten by `algorithm` defaults).
- Screen: 10 PPO @ 50k; promoted top-3 to **300k** + SAC @ **200k** (sequential, torch capped at one thread).
- Longer PPO training improved fill and selection score vs 50k screen.

## Model paths

- Best overall: `artifacts/tree_long/long/long_sac/final_model.zip`
- `artifacts/tree_long/best/`
- Long PPO: `artifacts/tree_long/long/long_{000,003,007}/`
- SAC: `artifacts/tree_long/long/long_sac/`
