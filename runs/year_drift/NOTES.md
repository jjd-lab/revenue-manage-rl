# Seasonal coverage, and a year that misses the forecast — NOTES

**Demand:** `tree_elastic`. Test worlds scale every night's demand by the same `level_shift` (0.8, 1.0, 1.2) via `demand.night_variation`; the forecast stays the usual model.
**Held-out:** 30 episodes, seeds 0–29 (June and December), the same nights as §7 and §14–§17.
**Score:** `score_aware`. Nothing is capped on either side.
**View:** every policy sees the operator's show-up estimate (`env.operator_view`).
**Regenerate:** `rprl-train -c configs/experiment_score_allmonths_sac.yaml`, `rprl-train -c configs/experiment_year_drift_sac.yaml`, then `python runs/year_drift/run_drift.py`.

## The questions

§17 found the learned policies within 1.3–2.7% of the planner on months they trained on and 4.1–6.0% behind on June and December, which training never samples.

1. **Coverage.** Does training on all twelve months close that gap?
2. **A year that runs off forecast.** Real years differ. When a whole year runs above or below the forecast, does a policy trained across such years, reading the year from booking pace, beat a planner with a stale forecast?

## The runs

Both use joint SAC, seed 7 and 200k steps, R1's reward and observation (`configs/experiment_score_sac.yaml`), and train on all twelve months (`env.held_out_months: []`).

- **all months** (`experiment_score_allmonths_sac.yaml`): the usual demand on every training night.
- **drift-trained** (`experiment_year_drift_sac.yaml`): each training night draws its own demand level (lognormal, sd 0.15, about 95% within ±30%) and price sensitivity (−1.2 ± 0.3).

Rule-based comparisons, no training:

- **planner**: the DP on the usual forecast (stale in a shifted year).
- **planner + pickup**: on days 70, 50, 30 and 15 out, it compares booking requests seen so far with what its forecast expected at the prices it charged, scales the forecast by that ratio, and re-plans (`dp_policy(pickup_days=...)`).

## Results

| policy | demand 20% below forecast | forecast right | demand 20% above forecast |
| --- | ---: | ---: | ---: |
| planner + pickup | **1,723,236** | **2,220,195** | **2,518,226** |
| planner | 1,684,378 | 2,219,272 | 2,501,119 |
| drift-trained | 1,657,887 | 2,141,761 | 2,407,057 |
| all months | 1,556,596 | 2,155,496 | 2,429,566 |
| R1 (no June or December, §17) | 1,643,276 | 2,112,692 | 2,333,653 |

Paired intervals:

| year | comparison | difference | 95% interval |
| --- | --- | ---: | --- |
| forecast right | all months − R1 | **+42,804** | [2,770, 85,622] |
| forecast right | planner − all months | +63,776 | [37,727, 89,215] |
| 20% below | planner − drift-trained | +26,490 | [2,543, 50,021] |
| forecast right | planner − drift-trained | +77,511 | [52,918, 102,296] |
| 20% above | planner − drift-trained | +94,062 | [64,293, 126,964] |
| 20% below | pickup − planner | +38,859 | [21,728, 57,653] |
| forecast right | pickup − planner | +924 | [−31, 2,000] (tie) |
| 20% above | pickup − planner | +17,107 | [10,581, 23,369] |

Peak nights (17 in every world):

| year | policy | mean price | unsold avg | denied nights (seats avg) |
| --- | --- | ---: | ---: | --- |
| 20% below | planner | $97.0 | 811 | 0 |
| 20% below | drift-trained | $93.6 | 821 | 1 (0) |
| 20% above | planner | $110.3 | 55 | 8 (30) |
| 20% above | drift-trained | $107.7 | 154 | 5 (35) |

Full CSVs: `drift_table.csv`, `paired_intervals.csv`, `training_curves.csv`.

## Reading

- **Coverage matters, and the gap is real.** Training on all twelve months adds +42.8k over R1 on the same test nights and cuts the planner's lead from 4.8% to 2.9%. That confirms §17's month-shift reading: about 40% of the gap was months never seen in training.
- **The planner wins every year tested, even with a stale forecast.** The drift-trained policy comes closest in a cold year, 1.6% behind (+26k, a real gap). With the forecast right it is 3.5% behind, in a hot year 3.8%.
- **Why a stale forecast does not sink the planner: it is closed-loop.** It re-plans every day from the bookings it has actually taken. In a hot year it finds itself ahead of plan and raises prices ($110 on peak nights against $104 in a normal year), and still fills to within 55 seats. Reading the booking pace, the advantage RL was meant to have, is already built into a dynamic program that tracks its own state.
- **A pickup adjustment helps the planner a little more.** +39k in a cold year, +17k in a hot one, a tie when the forecast is right. A fair planner baseline should have it.
- **Training across drifting years helps RL in cold years only.** Drift-trained beats all-months by about 101k at 0.8, but trails it by 14k and 23k at 1.0 and 1.2.

## What is left

- **Errors a closed loop cannot absorb.** A level shift shows up in the pace and the planner reacts. A forecast wrong about *when* demand arrives (booking curve earlier or later), or about the price response, is harder for a planner that trusts its curve. That is the next place RL could win.
- **Seeds.** One training seed per policy. §7's seed spread, 130–144k, is larger than several gaps here.

## Caveats

- One training seed per policy.
- The shift is the same on every test night and known to be constant within a year; real drift is messier.
- Pickup uses booking requests, including those the selling limit refused. A venue sees these as turned-away requests; if it could not, the adjustment would read low whenever the limit binds.
