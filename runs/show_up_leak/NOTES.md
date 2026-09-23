# What is the true show-up count worth? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29 (13 soft, 17 peak)
**Score:** `score_aware`, paired bootstrap (`evaluate/intervals.py`)
**Tool:** `src/reservation_pricing/envs/operator_view.py`, evaluation only
**Regenerate:** `python runs/show_up_leak/run_leak.py`

## The leak

`ReservationEnv` builds `cumulative_mat_boh` (expected show-ups so far) and
`remain_inv` from the true cancellation and no-show process, including the
night's realized no-show draw. Several parts of decision code read them:
- **The learned joint policies**, in observation slots 6 and 7.
- **The oversell cap**, which uses them to decide how far to lower the limit.
- **Myopic's late tighten**, which cuts the limit once `remain_inv` falls under 5%
  of capacity.

An operator sees none of this. It knows the bookings it took, and it has to
estimate how many will show up from cancellation and no-show rates measured
from history.

`OperatorViewEnv` sits directly on the env and replaces both values with that
estimate, bookings taken × assumed keep rate, wherever decision code looks. The
env still generates the night from the truth, and scoring is unchanged. Tests
(`tests/test_operator_view.py`) check that with noise off and true rates the
view equals the truth, and that only slots 6 and 7 of the observation change.

The DP planner (`runs/dp_baseline/`) already counts show-ups itself, so its
rows repeat that run. It is also scored behind the cap, with the cap using the
same assumed rates.

## Results

Score change against the published score (env count), per show-up model:

| policy | published | operator, true rates | no-shows 12% | no-shows 20% | cancellations higher (ρ −0.1) | cancellations lower (ρ +0.1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP planner | 2,219,272 | 0.00% | −4.15% | −5.80% | **−11.76%** | −5.03% |
| DP planner + cap | 2,216,741 | 0.00% | −4.48% | −1.47% | −5.89% | −5.26% |
| joint BC→SAC + cap | 2,081,400 | −0.09% | −2.52% | +1.13% | −2.69% | −3.25% |
| joint SAC `cu200` | 2,087,750 | −0.06% | +0.02% | −0.46% | −1.09% | +0.28% |
| joint SAC `rl_best` | 2,033,264 | −0.01% | −2.59% | +1.21% | +2.01% | −3.20% |
| myopic | 1,972,305 | −0.44% | −7.01% | +4.28% | +3.19% | −8.39% |

Scores:

| policy | operator, true rates | no-shows 12% | no-shows 20% | cancellations higher | cancellations lower |
| --- | ---: | ---: | ---: | ---: | ---: |
| **DP planner + cap** | **2,216,741** | **2,117,486** | **2,184,048** | **2,086,143** | **2,100,037** |
| DP planner | 2,219,272 | 2,127,177 | 2,090,587 | 1,958,178 | 2,107,606 |
| joint BC→SAC + cap | 2,079,572 | 2,028,992 | 2,104,947 | 2,025,494 | 2,013,853 |
| joint SAC `cu200` | 2,086,503 | 2,088,175 | 2,078,160 | 2,065,039 | 2,093,596 |
| joint SAC `rl_best` | 2,033,126 | 1,980,558 | 2,057,790 | 2,074,109 | 1,968,170 |
| myopic | 1,963,710 | 1,834,022 | 2,056,784 | 2,035,154 | 1,806,887 |

Peak nights with denied admission (of 17), and mean denied seats per peak night:

| policy | published | operator, true rates | no-shows 12% | no-shows 20% | cancellations higher | cancellations lower |
| --- | --- | --- | --- | --- | --- | --- |
| DP planner + cap | 7 (24) | 7 (24) | 0 | 11 (67) | 15 (326) | 0 |
| joint BC→SAC + cap | **0** | **1 (3)** | 0 | 8 (43) | 14 (337) | 0 |
| joint SAC `cu200` | 11 (221) | 11 (220) | 8 (166) | 13 (273) | 14 (315) | 8 (144) |
| joint SAC `rl_best` | 3 (5) | 2 (9) | 0 | 6 (53) | 9 (112) | 0 |

Full CSVs: `leak_table.csv`, `paired_intervals.csv`.

## Verdict

**With the true rates, the leak is worth almost nothing.** Replacing the env's
count with the operator's estimate moves every learned policy by less than 0.1%.
The realized no-show draw the policies were seeing is too small to matter. The
planner's lead holds: it is ahead of capped BC→SAC by +139,700 [117,670,
164,437], and every interval excludes zero.

**With wrong rates, the learned policies are exposed too.** Their earlier
immunity came from being shown the true count. Given a wrong count, `rl_best`
moves by −3.2% to +2.0% and capped BC→SAC by −3.3% to +1.1%. Some errors
*raise* a score: `rl_best` underfills peak nights, so a count that tells it to
sell more happens to help. `cu200` barely moves (−1.1% to +0.3%), because it
already overbooks heavily regardless.

**The cap's zero-denied-admission guarantee depends on the rates.** Capped
BC→SAC turns people away on 1 peak night with the true rates, 8 with no-shows
overestimated, and 14 with cancellations overestimated. The guarantee held in
`runs/oversell_cap_transfer/` because the cap was reading the true count.

**A cap fixes the planner's worst case.** Behind the cap, the planner's two
overbooking disasters shrink: no-shows at 20% goes from −5.8% to −1.5%, and
overestimated cancellations from −11.8% to −5.9%. The planner plus cap scores at
or above the best learned policy in every scenario. In two of them the margin is
6–12k, and no interval was computed for those. Under the cap the planner still
turns people away on 11–15 peak nights when it overestimates no-shows or
cancellations. So do the learned policies.

## What it means

- The comparison is now fair: every policy decides from bookings taken and
  assumed rates. On that footing a forecast-and-optimize planner with a cap is
  the best policy in every scenario tried. Its lead over RL runs from 130k with
  the true rates to a tie when cancellations are misjudged.
- How well a venue knows its cancellation and no-show rates matters more than
  which policy it runs. Every policy, learned or not, overbooks when those rates
  are overestimated.

## Caveats

- One error at a time, at one size each. Demand errors were not combined with
  show-up errors.
- The "best learned policy" in each scenario is the best of three, picked after
  the fact, which favours RL.
- The learned policies were trained on the true count. A policy trained on the
  operator's estimate might behave differently. Checking that means retraining.
- Ties are point estimates. No paired intervals were computed for the
  wrong-rate scenarios.
