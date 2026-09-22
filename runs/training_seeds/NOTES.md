# Does the BC→SAC lead repeat? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, seeds 0–29, the same nights as §7
**Score:** `score_aware`, recomputed inside each paired resample
**Trainings:** seed 42 is the shipped zips and was not retrained. Seeds 43, 44, and 46 are new. Pace is `configs/experiment_price_only_pace_ppo.yaml` (standard trainer). BC→SAC is `configs/experiment_bc_sac.yaml` (behaviour-clone warm start). The cap is eval-only (`configs/experiment_bc_sac_safe_sl.yaml`) on that same BC zip.
**Regenerate:** `python runs/training_seeds/train.py` then `python runs/training_seeds/eval.py`. The driver skips a seed whose `final_model.zip` is already there. It refuses seed 42, nights 0–29, the 100–129 block, and `artifacts/bc_sac/` / `artifacts/pace_ppo/`.

## The question

§7's paired interval says capped BC→SAC beats pace PPO by +71.5k (+31.2k to +105.8k) on these thirty nights. Both of those checkpoints are one training, at seed 42. The question is whether that lead sits above zero for every retraining of the pair.

A repeat means every capped-minus-matched-pace interval sits above zero. One interval that covers zero means the published lead is training-seed sensitive. The night list stays 0–29, so the soft/peak split stays 13/17 and the score levels stay comparable to §7.

## Results

Seed 42's matched interval reproduces §7 (+71.5k, +31.2k to +105.8k). Source of truth: `paired_intervals.csv`.

| training seed | capped − matched pace | 95% interval | tie |
| --- | ---: | --- | --- |
| 42 (shipped) | +71.5k | +31.2k to +105.8k | no |
| 43 | +107.1k | +83.6k to +128.3k | no |
| 44 | +224.8k | +168.9k to +283.4k | no |
| 46 | +8.0k | −7.9k to +25.2k | **yes** |

Against the published pace zip, rather than the pace run from the same seed:

| capped BC→SAC | minus published pace | 95% interval | tie |
| --- | ---: | --- | --- |
| seed 43 | +48.5k | +8.2k to +86.6k | no |
| seed 44 | +93.4k | +68.5k to +114.3k | no |
| seed 46 | −50.3k | −78.8k to −24.1k | no — the interval sits below zero |

`score_aware` for the capped policy across the four seeds is 2,081,400 / 2,058,454 / 2,103,280 / 1,959,620 (seeds 42, 43, 44, 46). The spread is 144k, from 1.960M to 2.103M. The night-level interval on the published pair runs from +31.2k to +105.8k, a width of 75k. Training-seed spread is wider than that night-level interval.

The matched pace scores move too: 2,009,914 / 1,951,377 / 1,878,515 / 1,951,624. Seed 44's +225k is a strong BC run against a weak pace run. Seed 46 is the other end: capped BC at 1,959,620 ties its own pace run and loses to the published pace checkpoint.

The cap does not take every retrain to zero peak denied admission. Peak oversell under the wrapper is 0.00, 0.24, 0.47, 0.00 on seeds 42, 43, 44, 46. Seed 46 was already at 0.00 before the cap. The published "zero denied admission" result is the seed-42 checkpoint.

## Verdict

The lead does **not** repeat. Three of the four matched intervals sit above zero; seed 46 covers zero, and that same checkpoint is below the published pace policy. The §7 lead is training-seed sensitive. The point estimates and `runs/joint_vs_price_only_soft_aware/paired_intervals.csv` stay the seed-42 record.
