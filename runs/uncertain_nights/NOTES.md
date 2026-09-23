# RL against the planner when every night differs — NOTES

**Demand:** `tree_elastic`, varied per night (`demand.night_variation`)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29 (17 peak nights)
**Score:** `score_aware`, paired bootstrap (`evaluate/intervals.py`)
**Retrained:** joint SAC, seed 7, 200k steps, the $200/$400 reward, one per setting (`artifacts/uncertain_nights/`, untracked)
**Regenerate:** `python runs/uncertain_nights/run_uncertain.py`

## The question

`runs/dp_baseline/` and `runs/show_up_leak/` tested one wrong forecast at a
time, wrong the same way on every night. A real venue knows only its *usual*
demand and show-up rates, and every night misses them a little, in a different
direction. Does an RL policy trained on such nights beat the planner with the
cap?

## The setup

Each night draws its own true values once, at the start. About 95% of nights
fall within:

| what varies | usual value | range |
| --- | --- | --- |
| demand level | the tree's forecast | ±25% |
| price sensitivity | −1.2 | −0.9 to −1.5 |
| no-show rate | about 10–12% on these nights | ±4 points |
| cancellation curve ρ | 0.4 weekday / 0.5 weekend | ±0.1 |

- Every policy sees only what a venue sees (`env.operator_view`): bookings, and
  show-ups estimated with the usual rates.
- The planner and the cap plan with the usual demand model and rates. They are
  never told a night's draw (`tests/test_uncertain_nights.py`).
- The draws come from the night's seed, so every policy faces the same nights.
- The noisy setting also raises daily demand noise from ±8 to ±24 bookings.

An RL policy was retrained in each setting, on the same kind of nights and with
the same view.

## Results

| policy | uncertain | uncertain, noisy | peak nights with denied admission (uncertain) |
| --- | ---: | ---: | --- |
| **DP planner + cap** | **2,130,168** | **2,119,459** | 4 of 17 (69 seats avg) |
| DP planner | 2,121,114 | 2,115,736 | 5 of 17 (129) |
| joint BC→SAC + cap | 2,027,321 | 2,021,304 | 4 of 17 (33) |
| **RL retrained here** | **2,022,087** | **2,018,903** | 6 of 17 (117) |
| joint SAC `cu200` | 1,999,148 | 1,996,636 | 8 of 17 (251) |
| joint SAC `rl_best` | 1,977,462 | 1,970,164 | 3 of 17 (31) |
| myopic | 1,880,067 | 1,873,232 | 4 of 17 (42) |

Paired intervals:

| setting | comparison | difference | 95% interval |
| --- | --- | ---: | --- |
| uncertain | planner + cap − RL retrained | **+108,081** | [64,394, 154,985] |
| uncertain | planner − RL retrained | +99,027 | [57,312, 142,222] |
| uncertain | RL retrained − `cu200` | +22,938 | [−45,881, 91,279] |
| noisy | planner + cap − RL retrained | **+100,555** | [34,648, 164,831] |
| noisy | planner − RL retrained | +96,832 | [38,664, 152,197] |
| noisy | RL retrained − `cu200` | +22,268 | [−71,088, 117,351] |

Full CSVs: `uncertain_table.csv`, `paired_intervals.csv`.

## Verdict

**The planner with the cap still wins, by about 5%, and the interval excludes
zero in both settings.** It only knows the usual values, and every night is
off. Its lead shrank from about 130k with perfect knowledge to about 100–108k.
But the retrained RL policy didn't close the gap.

**Training RL on uncertain nights did not clearly help it.** The retrained
policy scores about 22k above the $200/$400 policy trained on ordinary nights,
and that interval covers zero. It lands next to capped BC→SAC (2,022,087 against
2,027,321; not paired).

**Noisier daily demand changed almost nothing.** Tripling the daily noise moved
every score by under 1%. Over 100 days the extra noise mostly averages out.
What matters is the night-level miss in demand and show-ups, and that was
already present in both settings.

**Every policy turns some people away** on 3–8 of 17 peak nights, because none
of them knows the night's true show-up rate.

## What it means

On this simulator, RL's real advantage, needing no forecast, is not enough. A
planner that re-plans from the bookings it sees absorbs forecast and show-up
errors of this size and still leads by about 5%. RL's case would need something
this simulator doesn't have. Examples: errors much larger than these, errors
shared across a whole season, or a demand shape no simple model captures.

## Caveats

- One training seed per setting, and one draw of 30 nights.
- The spreads are my guess at plausible forecasting error.
- Errors are independent per night. Shared errors, like a whole season running
  quiet, were not tested.
- The RL reward is the $200/$400 rule, not the score itself. The planner
  optimizes the score's own costs.
