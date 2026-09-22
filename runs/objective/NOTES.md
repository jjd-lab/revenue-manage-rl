# What does the training objective do to the policy? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29 (13 soft, 17 peak)
**Score:** `score_aware`, paired bootstrap (`evaluate/intervals.py`)
**Arms:** joint SAC, seed 7, 200k steps, same hyperparameters (`tree_long_sac.yaml`); only the reward differs
**Cap:** `configs/experiment_bc_sac_safe_sl.yaml`, as in `runs/oversell_cap_transfer/`
**Regenerate:** `python runs/objective/run_objective.py`

## The question

`runs/price_monotone_up/` found that SAC optimizes its reward faithfully and
that the reward and `score_aware` price seats differently. How much of what a
policy does comes from its objective?

The objective has to be one a venue could state. `score_aware` switches on a
soft/peak label drawn from the simulator's true base demand, which no operator
knows, so the new reward uses one rule for every night:

| per night | `rl_best` (default reward) | `cu200` |
| --- | --- | --- |
| unsold seat | −$650, plus $80 of utilization bonus not earned | −$200 |
| denied admission | −$450, **and the whole bonus (up to $800,000) lost on any oversold night** | −$400 |
| config | `configs/default.yaml` | `configs/experiment_objective_cu200_sac.yaml` |

`cu200` sets `undersell_penalty: 200`, `oversell_penalty: 400` and
`utilization_bonus: 0`. At `revenue_scale` 1e-4 and capacity 10,000, those are
dollars per seat. `default.yaml` is untouched. `cu200` matches the score exactly
on peak nights, and charges soft nights $200 per empty seat where the score
charges nothing.

## Results

| | `rl_best` | `cu200` |
| --- | ---: | ---: |
| **score_aware** | 2,033,264 | **2,087,750** |
| paired vs `rl_best` | — | **+54,486** [26,377, 80,566] |
| score_aware behind the cap | 2,008,716 | **2,095,566** |
| paired vs `rl_best`, both capped | — | **+86,851** [47,135, 128,012] |
| peak nights with denied admission | 3 of 17 | **11 of 17** |
| … behind the cap | 0 | **2** |
| peak: revenue | 1,255,340 | 1,345,165 |
| peak: mean price | $97.11 | $100.61 |
| peak: unsold seats | 416 | 158 |
| peak: denied seats (mean over 17) | 5 | 221 |
| soft: revenue | 863,086 | 862,771 |
| soft: mean price | $83.73 | $83.22 |
| soft: unsold seats | 2,470 | 2,460 |
| weekend path: open → crest (days out) → close | $108.50 → $117.30 (36) → $89.08 | $112.43 → $118.07 (28) → $95.49 |

Each policy wins on the objective it was trained on (mean episode return):

| | default reward | `cu200` reward |
| --- | ---: | ---: |
| `rl_best` | **85.09** | 82.31 |
| `cu200` | 74.27 | **85.48** |

Full CSVs: `objective_table.csv`, `cross_objective.csv`, `paired_intervals.csv`.
`rl_best` reproduces its published 2,033,264, its capped 2,008,716 and F4's
weekend path. Every episode's summed return matches its closed form from the
final `info`.

## Verdict

**The objective decides the overbooking, and the overbooking is where the score
is.** With the cliff gone and a denied admission priced at $400, SAC books past
capacity on 11 of 17 peak nights (about 340 denied admissions on each of them).
It charges more ($100.61 against $97.11) and still leaves 258 fewer seats empty
per peak night. Peak revenue rises $89,800 a night. The score's peak term rises
$54,800 after it charges for the denied admissions. That is the whole gain.

**Soft nights did not move.** The expected effect of a cheaper empty seat, a
higher soft-night price, did not happen: $83.22 against $83.73, the same seats
unsold. `rl_best` was already near the $80 floor on soft nights, and on this
demand model soft nights can't fill at any price, so a $200 charge and a $730
charge lead to the same answer. The likely driver on peak nights is the removed
cliff: a cheaper empty seat on its own would make the policy fill *less*, and it
filled more. This run changes both at once, so it can't separate them.

**Behind the cap the gap widens, but the cap no longer finishes the job.** The
cap removes the worst of `cu200`'s overbooking and its score *rises* to
2,095,566, in the range of capped Joint BC→SAC (2,081,400, not paired here). But
2 of 17 peak nights still deny admission. The cap reached zero on every policy
in `runs/oversell_cap_transfer/`. A policy trained to overbook pushes past it.

**The weekend markdown shrinks.** The late fall is $22.58 from crest to close,
against $28.22, and the crest comes 8 days later. This objective alone does not
remove the markdown.

## The costs in the objective set the answer

Each reward weight is a price the policy is told to pay, and the policy follows
those prices:

| cost in the reward | default | `cu200` | what the policy did |
| --- | --- | --- | --- |
| an empty seat | ~$730 | $200 | soft nights: nothing, because they can't fill at any price |
| a denied admission | $450 + the whole fill bonus on that night | $400 flat | peak nights: overbooked 11 of 17, up from 3 |

The score is a set of prices too, and it prices a denied admission at $400,
the same as `cu200`. The ranking depends on that price. `cu200` denies 216.3 more
seats per peak night than `rl_best` (221.4 against 5.1), so each extra dollar
charged per denied admission takes 216.3 off its lead of 54,486:

> break-even = $400 + 54,486 / 216.3 ≈ **$652 per denied admission**

If a venue thinks turning a ticket holder away costs less than about $652 (the
refund, the compensation, the lost customer), `cu200` is the better policy.
Above that, `rl_best` is. This is arithmetic on the rollouts above, not a new
evaluation. The capped break-even was not worked out.

## Caveats

- One training seed. `runs/training_seeds/` shows seed variance can flip
  rankings within a family.
- This is not a same-day matched retrain. `rl_best` re-scores exactly today, so
  the simulator and scoring are unchanged. But retrains don't match bit for bit,
  so part of the gap could be run-to-run training noise rather than the
  objective. A retrain of the default reward at seed 7 would settle it.
- The score values a denied admission at $400, the same price `cu200` trains
  on. See the break-even above.
