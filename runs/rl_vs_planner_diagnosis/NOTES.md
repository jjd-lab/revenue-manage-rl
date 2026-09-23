# Why does RL trail the planner? — NOTES

**Demand:** `tree_elastic` (default), true forecast
**Held-out:** 30 episodes, seeds 0–29, the same nights as §7 and §14
**Score:** `score_aware`
**View:** every policy sees the operator's show-up estimate, bookings × usual keep rate (`env.operator_view`); nothing from the night's realized draw reaches a learned policy or the cap. Reference rows match `runs/show_up_leak/` "operator, true rates".
**Regenerate:** `rprl-train -c configs/experiment_score_sac.yaml`, `rprl-bc-sac -c configs/experiment_score_bc_dp_sac.yaml`, then `python runs/rl_vs_planner_diagnosis/run_diagnosis.py`.

## The question

§14–§16 left the DP planner 5.9–8.4% above every learned policy. Candidate reasons:

1. **No headroom.** The planner is handed the simulator's own demand model and the score's own costs, and solves one night by backward induction. Predicted and realized scores agree within 0.5%.
2. **Objective mismatch.** The learned policies train on a reward that is not the score. It charges empty seats on soft nights too.
3. **Missing inputs.** The observation has no night type. Training never samples June or December (`use_held_out=False`), yet all thirty test nights are in those months, so their month one-hot slots first switch on at test time.
4. **Unstable training.** One seed, noisy curves, and five-night checkpoint selection on the test months.
5. **Not enough data.** Not in volume: RL learns from an unlimited simulator (200k steps ≈ 2,000 nights), not from history. In coverage, yes: none of those nights is in June or December, the only test months (reason 3). For a real venue this means history covering every season, at least a year and ideally two, not more nights of the seasons already seen.

## The runs

Both runs use joint SAC, seed 7 and 200k steps, the same budget as `cu200`. Both use the reward revenue − $200 per unsold seat on **peak nights only** − $400 per denied admission, with no bonus. That is the planner's terminal charge. Soft/peak comes from the forecast.

The observation adds the forecast's mid-horizon base demand and soft flag (`env.night_features`). The kept checkpoint is picked on 30 **training-month** nights every 20k steps (`train.eval_held_out: false`).

- **R1** `score_sac`: from scratch. Tests reasons 2 and 3.
- **R2** `score_bc_dp_sac`: clone the planner on 300 training-month nights, then fine-tune on R1's reward. Meant to test reason 1.

## Results (held-out)

| policy | score_aware | peak denied nights (seats avg) | peak unsold avg | soft mean price |
| --- | ---: | --- | ---: | ---: |
| **DP planner** | **2,219,272** | 8 of 17 (32) | 56 | $92.0 |
| DP planner + cap | 2,216,741 | 7 of 17 (24) | 67 | $92.0 |
| R2 clone + SAC + cap | 2,134,485 | 4 of 17 (16) | 190 | $86.3 |
| R2 clone + SAC, final | 2,128,208 | 14 of 17 (245) | 48 | $89.4 |
| R1 score SAC, final | 2,112,693 | 8 of 17 (143) | 146 | $87.9 |
| R1 score SAC (kept checkpoint) | 2,111,489 | 8 of 17 (144) | 149 | $87.9 |
| R1 score SAC + cap | 2,094,687 | 1 of 17 (3) | 328 | $87.9 |
| joint SAC `cu200` | 2,086,503 | 11 of 17 (220) | 162 | $83.2 |
| joint BC→SAC | 2,086,140 | 11 of 17 (120) | 174 | $81.7 |
| joint BC→SAC + cap | 2,079,572 | 1 of 17 (3) | 296 | $81.7 |
| R2 clone + SAC (kept checkpoint) | 2,058,278 | 14 of 17 (377) | 117 | $86.3 |
| myopic | 1,963,710 | 0 of 17 | 450 | $92.0 |
| R2 planner clone (before SAC) | 1,927,144 | 14 of 17 (708) | 186 | $96.7 |

Paired intervals:

| comparison | difference | 95% interval |
| --- | ---: | --- |
| planner + cap − R2 + cap | **+82,255** | [61,654, 104,549] |
| planner + cap − R1 + cap | +122,054 | [103,229, 139,245] |
| planner − R1 | +107,783 | [87,198, 132,058] |
| planner − R2, final (neither capped) | +91,063 | [72,993, 108,594] |
| planner − joint BC→SAC (neither capped) | +133,131 | [105,494, 164,135] |
| R1 − `cu200` | +24,986 | [−18,636, 64,475] (tie) |
| R2 − its own clone | +131,133 | [95,527, 168,407] |

Full CSVs: `diagnosis_table.csv`, `paired_intervals.csv`, `training_curves.csv`, `training_month_table.csv`.

## The same policies on training months

Sixty training-month nights (seeds 1000–1059, `use_held_out=False`), the same draw for every policy:

| policy | held-out score | gap to planner | training-month score | gap to planner |
| --- | ---: | ---: | ---: | ---: |
| DP planner | 2,219,272 | — | 1,916,252 | — |
| R2 clone + SAC, final | 2,128,208 | −4.1% | 1,891,457 | −1.3% |
| R1 score SAC, final | 2,112,693 | −4.8% | 1,863,784 | −2.7% |
| joint SAC `cu200` | 2,086,503 | −6.0% | 1,870,398 | −2.4% |

No paired intervals on this table.

## Reading

- **The gap roughly doubles on held-out months, which puts about half or more of the blame on the month shift.** On months they trained on, the learned policies come within 1.3–2.7% of the planner. On June and December they trail by 4.1–6.0%. The night-type feature was meant to cover this, but the June and December one-hot slots are still inputs the network never saw switched on. This finding is new and comes from one training seed.
- **The objective matters, but less than expected.** Training on the score's own charge and showing the night type (R1) moves soft-night prices from $83 toward the optimal $92 and cuts peak denied admission from 220 to 143 seats. The gain over `cu200`, +25k, is a tie.
- **Cloning the planner does not reproduce it.**
  - The clone fits its training actions closely (MSE 0.0016 on actions scaled to [−1, 1], about ±100 seats of selling limit).
  - On held-out nights it overbooks 14 of 17 peak nights, by 708 seats on average, and scores 1.93M.
  - Small errors on the selling limit compound, and the months are unseen.
  - So R2 never started *at* the planner, and it cannot say whether RL could improve on one.
  - Fine-tuning then added 131k and made R2, with the cap, the best learned policy on these nights. It is still 82k below the planner with the cap.
- **Checkpoint selection is noisy.** Picking the best of ten evals on 30 training-month nights chose a checkpoint 70k *below* R2's final model. R1's pick was level with its final model. Thirty nights are too few to rank checkpoints this close.
- **Undertraining is not the main story.** The training-month reward rose from 849k to 1,021k per night over 200k steps, with swings of about 100k between evals, and its last point was the highest. But on training months the policies are already within 1.3–2.7% of the planner. A longer R1 run (the conditional R3) was not launched.

## What is left

In order of expected influence:

1. **Month shift, i.e. seasonal coverage.** Retrain R1 on all twelve months and hold out nights instead, or drop the month one-hot (the night-type feature already carries the calendar signal the planner uses). One run. This is also the direct test of whether data that covers every season closes the gap.
2. **Headroom.** A clone that actually reproduces the planner, e.g. DAgger with the planner as the online expert, before asking whether SAC can beat it.
3. **Seeds.** All of this is one training seed. §7's seed check found a 130–144k spread, as large as the gaps here.

## Caveats

- One training seed per run.
- The planner is handed the simulator's own model. On this simulator it is close to the best any non-clairvoyant policy can do: reading the night's true show-up draw adds only 1.1% (§14).
- The observation change and the reward change were made together in R1, so their separate effects are not identified.
