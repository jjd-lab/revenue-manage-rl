# What does a non-decreasing price cost? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0–29
**Score:** `score_aware`, paired bootstrap against the unconstrained arm (`evaluate/intervals.py`)
**Constraint:** `direction: up`. `mode: project` clamps. `mode: penalty` does not; the wrapper subtracts `penalty * (violation_dollars / 40)` from the step reward.
**Penalty:** grid `{1, 10, 100}`, ranked on seeds 100–129, reported weight **10**. The selection scores are not a result.
**Reference:** frozen `artifacts/tree_long/best/rl_best.zip` (joint SAC, seed 7, 200k)
**Retrains:** same hyperparameters, `artifacts/price_monotone/`, seed 7, 200k. Untracked.
**Regenerate:** `python runs/price_monotone_up/run_monotone.py`
**Path:** `python scripts/explain_rl_best.py --config configs/experiment_monotone_up_sac.yaml --model artifacts/price_monotone/monotone_up_project/best/best_model.zip --out runs/price_monotone_up/explain`

## The question

F4's unconstrained weekend path opens near $109, crests near $117 around day 37, then marks down to about $89 by day 1. A later buyer paying less than an earlier one is the move a venue cannot make. Forbidding it removes the clearance mechanism. The question is what that costs per night, not whether the constrained policy wins.

## Results

| arm | score_aware | paired vs unconstrained | 95% interval | charged decreases |
| --- | ---: | ---: | --- | --- |
| unconstrained `rl_best` | 2,033,264 | 0 | — | 1,672 steps, 30/30 episodes |
| project, frozen checkpoint | **2,077,976** | **+44,712** | [20,945, 66,164] | **0** |
| project, retrained | 1,923,106 | −110,158 | [−146,418, −73,687] | **0** |
| penalty 10, retrained | 1,951,555 | −81,709 | [−134,174, −32,539] | 1,323 steps, 30/30 episodes |

Full CSV: `monotone_table.csv`. Intervals: `paired_intervals.csv`. None of the three gaps covers zero.

## Verdict

| Question | Answer |
| --- | --- |
| Does the unconstrained policy mark down? | **YES** — every one of the 30 nights |
| Does `mode: project` remove the markdown? | **YES** — zero charged decreases, frozen or retrained |
| Does the penalty? | **NO** — it still marks down on every night |
| Is the retrain cheaper than the frozen policy? | **NO** — the frozen projection is the only arm above the reference |

**Clamping the published policy helps. Training under the clamp does not.**
Projecting `rl_best` after the fact, with no retrain, raises `score_aware` by about 45k and deletes the markdown. Retraining under the same clamp costs about 110k. The retrained charged path does not rise toward the date: on weekend nights the mean stays at $116.31 from day 100 through day 1, and weekday nights sit on the $80 floor (`explain/rollouts.csv`). The late plunge is gone because the path is flat, not because a rising curve replaced it.

The penalty weight that won the selection still buys neither the guarantee nor the frozen arm's score. It lands between the two project arms, about 82k below the reference, and marks down on all 30 nights.
