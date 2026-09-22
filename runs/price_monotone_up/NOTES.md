# What does a non-decreasing price cost, and can a policy learn under it? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29
**Score:** `score_aware`, paired bootstrap against the unconstrained arm (`evaluate/intervals.py`)
**Constraint:** `direction: up`. `mode: clamp` raises an offending action to the floor. `mode: ratchet` reads the action as a move from the floor instead. `mode: penalty` does not constrain; the wrapper subtracts `penalty * (violation_dollars / 40)` from the step reward.
**Selection:** penalty weights on seeds 100–129 only (`penalty_selection.csv`, `penalty_hw_selection.csv`). Those scores are not results.
**Reference:** frozen `artifacts/tree_long/best/rl_best.zip` (joint SAC, seed 7, 200k)
**Retrains:** same hyperparameters, `artifacts/price_monotone/`, seed 7, 200k. Untracked.
**Regenerate:** `python runs/price_monotone_up/run_monotone.py`

## The question

F4's unconstrained weekend path opens near $109, crests near $117 around day 37,
then marks down to about $89 by day 1. A later buyer paying less than an earlier
one is the move a venue cannot make. The first four arms ask what forbidding it
costs. The rest ask why the retrain answered so badly.

## Results

| arm | score_aware | paired vs unconstrained | 95% interval | charged decreases | clamped steps |
| --- | ---: | ---: | --- | ---: | ---: |
| unconstrained `rl_best` | 2,033,264 | 0 | — | 1,672 | 0 |
| **clamp, frozen** | **2,077,976** | **+44,712** | [20,945, 66,164] | **0** | 2,610 |
| penalty_hw 1, retrained | 2,048,563 | +15,299 | [−39,344, 74,833] | 1,377 | 0 |
| **bc_clone** | **2,038,966** | **+5,702** | [−42,250, 51,133] | **0** | 1,812 |
| bc_clone_ratchet | 2,024,192 | −9,072 | [−63,648, 38,982] | **0** | 0 |
| peak_only | 1,964,902 | −68,362 | [−124,006, −12,670] | 528 | 0 |
| penalty 10, retrained | 1,951,555 | −81,709 | [−134,174, −32,539] | 1,323 | 0 |
| ratchet_pace | 1,935,085 | −98,179 | [−162,419, −31,938] | **0** | 0 |
| bc_ratchet | 1,935,036 | −98,228 | [−230,925, 16,571] | **0** | 0 |
| clamp, retrained | 1,923,106 | −110,158 | [−146,418, −73,687] | **0** | 0 |
| ratchet | 1,921,207 | −112,057 | [−144,200, −80,287] | **0** | 0 |
| bc_clamp | 1,911,249 | −122,015 | [−147,486, −94,954] | **0** | 1,851 |

Full CSV: `monotone_table.csv`. Intervals: `paired_intervals.csv`.

## Verdict

**Clamping a trained policy is free. Training under the clamp is not, and no
parameterization fixed that.**

The guarantee itself costs nothing: `clamp_frozen` takes the published
checkpoint, overrides it on **2,610 of 3,000 decisions**, removes every markdown,
and *scores higher* than leaving it alone. The late plunge was losing money.

Everything trained under the constraint lands 100k below it, and the reason is
not the one the mechanism suggested:

- **The clamp does alias the action space, and fixing that changed nothing.**
  Under `clamp` every action below the floor charges the floor, so reward and
  dynamics are flat over that region and the gradient there is exactly zero.
  `ratchet` removes the aliasing by construction — the action becomes a move,
  so no two actions collide. It scored **1,921,207**, statistically identical to
  the clamp retrain it was meant to rescue. The aliasing was real and was not
  the binding constraint.
- **What the retrains actually do is pin low.** The clamp retrain holds $116.31
  on weekend nights from day 100 to day 1 and sits on the $80 floor on weekdays.
  The ratchet opens at $80 and peaks at $82. Under an irreversible ratchet,
  opening at `min_price` is the move that keeps the most options, and on soft
  nights — where the revenue-maximizing path genuinely *falls* — it is close to
  right. Pace shaping (`ratchet_pace`) does buy back the shape, lifting the path
  to open $80.9 and climb to $94.5, but only ~14k of score.
- **Most of the loss is soft nights.** `peak_only` keeps the guarantee on
  weekends and peak months and frees the rest; it recovers 42k of the gap and
  gives up the guarantee on 528 steps across 13 nights.

## Cloning works; fine-tuning is what destroys it

Behaviour cloning of `clamp_frozen` lands at **2,038,966** with zero markdowns —
a tie with the unconstrained reference, and 116k above the clamp retrain. Then
fine-tuning undoes it, and no gentler setting helps:

| | score_aware |
| --- | ---: |
| BC-only, no fine-tuning | **2,038,966** |
| + 200k steps, lr 3e-4 | 1,966,735 |
| + 200k steps, lr 3e-5 | 1,937,339 |
| + 20k steps, lr 3e-4 | 1,808,182 |

SAC walks away from a policy worth 2.04M toward the same floor-pinning basin
every from-scratch arm found. The training reward and `score_aware` disagree
about soft nights, and under the constraint the policy cannot mark down to
recover, so the reward-optimal answer is to open low. **The recipe that works is
clone and stop.**

## The penalty does not buy a guarantee at any weight

Charged against yesterday's price, a slow staircase down is a series of small
fines: weight 10 marks down on all 30 nights. Charging against the episode's
high-water mark makes sitting below the peak cost that much every remaining
step, and selected properly (weight 1, on seeds 100–129) it is the best-scoring
retrained arm at 2,048,563 — **and it still marks down 1,377 times**. Every
weight tried leaves four-figure decrease counts. A soft penalty is not a
guarantee; if the promise is "later buyers never pay less", it has to be a clamp.

## What an operator should do

Train unconstrained, deploy behind the clamp. That is `clamp_frozen`: the best
score in the table, zero markdowns, and no retraining. If a policy that has
internalised the rule is wanted, clone the clamped one and do not fine-tune it.

**Reproducing the fine-tune sweep.** The three rows above come from
`experiment_monotone_up_bc_clamp_sac.yaml` and its two variants,
`configs/experiment_monotone_up_bc_short_sac.yaml` (20k steps) and `configs/experiment_monotone_up_bc_gentle_sac.yaml` (lr 3e-5),
each run with `rprl-bc-sac`. The BC-only number is `bc_only_model.zip`, written
before any RL step, and is the `bc_clone` arm in the table above.
