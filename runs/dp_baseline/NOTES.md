# Is RL better than the textbook planner? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29 (13 soft, 17 peak)
**Score:** `score_aware`, paired bootstrap (`evaluate/intervals.py`)
**Planner:** `src/reservation_pricing/baselines/dp.py`, no training
**Regenerate:** `python runs/dp_baseline/run_dp.py`

## The question

Every "traditional" comparison so far was myopic: each day it picks the price
that is best for that day alone, and its selling limit never binds. A competent
revenue-management team would plan the whole booking window against a forecast.
This run builds that planner and asks three things:

1. With the true model, how far are the learned policies from it?
2. With a wrong **demand** forecast, does its lead survive?
3. With a wrong **show-up** model (cancellation or no-show rates off), does it?

The learned policies consult no demand forecast and no show-up model, so their
scores don't change in 2 or 3. That is where RL was expected to win.

## The planner

Backward induction over one night. The state is expected show-ups so far. A
booking made `d` days out shows up at a keep rate that depends only on `d`, so no
other part of the booking history changes the outcome. Each day the planner picks
a price ($80–$120 in $1 steps) and a cap on today's bookings (0, 25, 50, 75 or
100% of expected demand). The cap is carried out as selling limit = bookings on
hand + cap. Demand noise is averaged over at 5 quadrature points. The terminal
charge is the score's own: $200 per unsold seat and $400 per denied admission on
peak nights, and on soft nights only the $400.

What it is allowed to know:
- **Demand:** `decision_model(env)`, so a wrong forecast reaches it as it reaches
  myopic. It also classifies the night as soft or peak from that forecast.
- **Keep rates:** `estimate_keep_rate`, the disclosed access the cap and the
  analytic limit use (§11a), or wrong values via `keep_overrides`.
- **Bookings taken and bookings on hand:** a real operator knows both.
- **Not the env's `cumulative_mat_boh`.** The planner counts expected show-ups
  itself, as bookings taken times its own keep rates. The env's count is built
  from the true rates and the night's realized no-show draw, which no operator
  sees. A first version read it and scored 24k higher (2,243,368). A test now
  checks that the planner's actions don't change when that count is corrupted.

It solves once per night type and forecast, in about a second.

Checks (`tests/test_dp_baseline.py`):
- The solver matches brute-force enumeration on a small case.
- With noise off, its predicted score for the night matches the rollout within 0.5%.
- No cap it wanted was blocked by the selling-limit floor on any of the 30 nights.

## Results: true model

| policy | score_aware | DP ahead by (95% interval) | behind DP | peak unsold | peak nights with denied admission (mean seats) | soft price |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| **DP planner** | **2,219,272** | — | — | 56 | 8 of 17 (31.8) | $92.00 |
| joint SAC `cu200` | 2,087,750 | +131,521 [102,665, 161,232] | 5.9% | 158 | 11 of 17 (221.4) | $83.22 |
| joint BC→SAC + cap | 2,081,400 | +137,872 [116,864, 162,003] | 6.2% | 293 | 0 | $81.65 |
| joint SAC `rl_best` | 2,033,264 | +186,008 [155,301, 216,605] | 8.4% | 416 | 3 of 17 (5.1) | $83.73 |
| myopic | 1,972,305 | +246,966 [171,173, 322,969] | 11.1% | 423 | 0 | $92.00 |

Full CSVs: `dp_table.csv`, `paired_intervals.csv`. myopic, `rl_best` and capped
BC→SAC reproduce their published scores.

**Peak nights carry the gain.** The planner aims at capacity and misses by about
90 seats per peak night on average (56 empty, 32 turned away). It can't hit
exactly because it doesn't see the night's no-show draw. The learned policies
leave 150–420 seats empty or overbook by hundreds.

**Soft nights: +11k.** The planner prices at $92, the price that maximizes
revenue when a night can't fill. That is myopic's price. The learned policies sit
at $82–84, pulled down by a training reward that charges about $730 per empty
seat (`runs/objective/`).

## Results: wrong demand forecast

| forecast | DP planner | change | myopic | change |
| --- | ---: | ---: | ---: | ---: |
| true | 2,219,272 | — | 1,972,305 | — |
| elasticity −0.9 (thinks demand less price-sensitive) | 2,189,337 | −1.35% | 1,903,796 | −3.47% |
| elasticity −1.5 (thinks demand more price-sensitive) | 2,211,998 | −0.33% | 1,857,432 | −5.82% |
| weekend/peak level 25% low | 2,164,334 | −2.48% | 1,972,305 | 0.00% |

Full CSV: `dp_forecast.csv`. **A wrong demand forecast barely hurts it.** Its
worst case, 2,164,334, is still about 77k above the best learned policy.

## Results: wrong show-up model

True values: no-show base 16%; Weibull cancellation shape ρ 0.4 on weekdays and
0.5 on weekends. A lower ρ means the planner expects more cancellations.

| planner's show-up model | score_aware | change | peak unsold | peak nights with denied admission (mean seats) |
| --- | ---: | ---: | ---: | --- |
| true | 2,219,272 | — | 56 | 8 of 17 (31.8) |
| no-shows 12% | 2,127,177 | −4.1% | 453 | 0 |
| cancellations lower (ρ +0.1) | 2,107,606 | −5.0% | 543 | 0 |
| no-shows 20% | 2,090,587 | −5.8% | 0 | **17 of 17 (441.6)** |
| cancellations higher (ρ −0.1) | 1,958,178 | **−11.8%** | 0 | **17 of 17 (786.1)** |

Full CSV: `dp_show_up.csv`.

**This is the planner's weak spot.** It plans the whole night around how many
bookings will turn into show-ups, and it has no way to see that it's wrong until
the night itself.
- **It expects too few show-ups** (too many no-shows or cancellations): it
  overbooks and turns people away on every peak night. With cancellations
  overestimated, it falls below myopic.
- **It expects too many show-ups:** it holds back and leaves 450–540 seats
  empty per peak night, but stays just ahead of the learned policies.

Against the best learned policy (2,087,750): two of the four errors leave the
planner ahead by 20–39k, one ties it, and one puts it 130k behind. These are
moderate errors, a 4-point no-show miss or a cancellation curve a little off,
of the kind an operator estimating from history could make.

## Verdict

**With a correct model, the textbook planner beats every learned policy by
6–8%. A wrong demand forecast barely dents that. A wrong show-up model can erase
it.** A 4-point no-show miss in the overbooking direction roughly ties it, and a cancellation miss that makes it
overbook puts it below myopic, with denied admission on every peak night.

So the practical question is not "RL or a planner". It is how well the venue
knows its own cancellations and no-shows. If they are well measured, plan. If
they are uncertain or drifting, the planner needs a guard against overbooking,
such as the cap, or a policy that doesn't depend on them.

## Caveats

- **The learned policies see a true show-up count.** Their observation includes
  `cumulative_mat_boh` and `remain_inv`, which the env builds from the true
  cancellation and no-show process and the night's realized draw. So their
  immunity to a wrong show-up model is partly because they are shown the answer.
  This is the same leak the planner no longer uses. It isn't priced here, and
  fixing it would change the observation and mean retraining every checkpoint.
- **Mild demand noise.** Demand varies by ±8 bookings a day, so a night is close
  to deterministic, which suits a planner. Heavier noise was not tested.
- **One error at a time.** Demand and show-up errors were not combined.
- **The planner has no cap of its own.** Adding one to the overbooking scenarios
  was not tried.
- **Part of the gap is the reward.** The learned policies trained 200k steps on a
  reward that disagrees with the score (`runs/objective/`). How much of the gap
  is the reward and how much is RL is not separated here.
