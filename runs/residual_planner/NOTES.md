# Can RL improve on the planner by correcting it? — NOTES

**Demand:** `tree_elastic`. Test years move *when* demand arrives by the same number of days on every night (`demand.night_variation.timing_shift`: +10 earlier, 0, −10 later); the forecast keeps the usual booking curve.
**Held-out:** 30 episodes, seeds 0–29 (June and December), the same nights as §7 and §14–§18.
**Score:** `score_aware`. Nothing is capped.
**View:** every policy sees the operator's show-up estimate (`env.operator_view`).
**Regenerate:** `rprl-train -c configs/experiment_residual_sac.yaml`, then `python runs/residual_planner/run_residual.py`.

## The question

§18 found the planner absorbs a year whose demand *level* misses the forecast, because it re-plans from its own bookings. Two questions remain:

- Can a forecast wrong about *timing* fool it? Early demand looks like a hot year.
- Can RL add value by correcting the planner rather than replacing it?

## The run

**residual** (`configs/experiment_residual_sac.yaml`, `control.residual`):

- Each day the DP planner proposes a price and selling limit. Joint SAC adds a correction bounded to ±$10 and ±500 seats. A zero correction is the planner, checked in `tests/test_year_drift.py`.
- The agent also sees the planner's proposal and the pickup ratio: booking requests over forecast at the prices charged.
- **Training:** all twelve months, R1's reward and view, seed 7, 200k steps.
- **Training nights:** each draws its own timing shift (normal, sd 7 days).

Comparisons, no training:

- **planner:** the plain planner.
- **planner + pickup:** rescales its forecast from booking requests on days 70, 50, 30 and 15 out (§18).
- **all months:** plain joint SAC from `runs/year_drift/`.

## Results

| policy | demand 10 days early | on time | 10 days late |
| --- | ---: | ---: | ---: |
| **planner** | **2,203,010** | 2,219,272 | **2,209,082** |
| planner + pickup | 2,185,959 | **2,220,195** | 2,206,797 |
| residual | 2,169,204 | 2,192,369 | 2,194,711 |
| residual, kept checkpoint | 2,167,966 | 2,201,634 | 2,193,183 |
| all months (no planner) | 2,155,916 | 2,155,496 | 2,153,821 |

Paired intervals:

| year | comparison | difference | 95% interval |
| --- | --- | ---: | --- |
| early | residual − planner | −33,807 | [−45,095, −21,720] |
| on time | residual − planner | −26,902 | [−38,087, −16,050] |
| late | residual − planner | −14,371 | [−27,208, −1,714] |
| early | pickup − planner | −17,052 | [−24,303, −9,704] |
| late | pickup − planner | −2,284 | [−4,004, −345] |

Peak nights, on time: the planner leaves 56 seats empty and denies 32. The residual leaves 144 empty and denies 11.

Full CSVs: `residual_table.csv`, `paired_intervals.csv`, `training_curves.csv`.

**Correction (2026-09-23).** The first version of the timing shift clipped the booking curve at the ends of the window. That also changed each night's total demand: +3.3% at 10 days early and −5.8% at 10 days late on seed 4. The shift now rescales each night so total demand matches the forecast. The tables above use the corrected shift. The residual was trained on the first version, whose nights mixed a small level change into the timing. It was not retrained; the evaluation was rerun.

## Reading

- **Learned corrections make the planner worse in every year**, by 0.7–1.5%, and the gap is real each time. The kept checkpoint trails in all three too (point estimates; no interval computed for it).
- **The corrections trade denied admission for empty seats.** On time, the residual denies 21 fewer seats per peak night and leaves 88 more empty. At $400 against $200 a seat, that trade loses. The planner already sits at the balance the score rewards.
- **The training curve is flat.** The training-month reward starts at the planner's level, about 1.0M per night, and stays there for 200k steps, with swings of about 70k between evals. SAC found no correction that pays.
- **A timing error barely hurts the planner.** Demand arriving 10 days early or late costs it 0.7% and 0.5%. It re-plans from its own bookings throughout. Neither a pickup rule nor a learned correction beats it in any year.
- **The pickup adjustment backfires under timing errors,** as expected. It reads early demand as a hot year: −17k early, −2k late. A pickup rule that cannot tell timing from level should not be the default.

## Conclusion across §17–§19

On this simulator the DP planner is hard to beat. Learning comes within about 3% of it once it sees every month (§18). It does not overtake it when the demand level drifts (§18), and it cannot improve it by correcting it when demand timing drifts (§19). The honest recommendation: plan with a model; use learning where no model can be built.

## Caveats

- One training seed.
- The correction bounds (±$10, ±500 seats) and the timing spread in training (7 days) were chosen once, not tuned.
- Training used the first, uncorrected timing shift (see Correction above); only the evaluation was rerun.
