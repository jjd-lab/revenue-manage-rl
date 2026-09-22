# What is a policy worth when the forecast is wrong? — NOTES

**Demand:** `tree_elastic` (default) — the env always generates from the true model
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29
**Score:** `score_aware`, same scoring as `docs/EXPERIMENT_LOG.md` §7
**Regenerate:** `python runs/forecast_misspecification/run_misspecification.py`

## The question

Every other table here shares a property worth naming: **nothing is ever wrong
about the world.** The env generates demand from the tree model, and every
component that consults a model consults *that same object* — the myopic
baseline, the price MPC, the `optimize_1d` limit. Real operations are defined by
the opposite condition: the forecast was fitted on history and meets a future
that moved.

`demand.forecast` (see `docs/DESIGN.md` § Forecast vs truth) splits the two. The
env keeps generating from the true model; only decision code sees the wrong one.
Nothing is retrained — a drifting forecast is something a deployed system meets,
not something it gets to prepare for.

## Choosing the error: why the obvious one does nothing

The first version of this run degraded the demand **level** and returned +0.00%
for every policy, including myopic. That was a design flaw, not a null result.

Demand enters as `base * (1 + elasticity * (price - ref) / ref)`. The base level
is a **multiplicative constant**, so it cancels out of the price argmax:

```
p* = ref(1 - e) / (-2e) = 100 * 2.2 / 2.4 = $91.67     (independent of base)
```

Verified directly — the argmax is $92 with base scaled to 1.00, 0.75 and 0.50.
This is also the real reason myopic prices "a flat $92": it is a closed form, not
a quirk of the demand curve. **A level error cannot move any pricing decision.**
It reaches only components that use the level itself: the MPC's trigger and
lookahead, and the `optimize_1d` limit.

So both kinds are injected — `elasticity` to move pricing, and a level error kept
deliberately to *demonstrate* the invariance rather than hide it.

## Results

| policy | consults the forecast for | true | elasticity −0.9 | elasticity −1.5 | level stale |
| --- | --- | ---: | ---: | ---: | ---: |
| myopic | price | 1,972,305 | **−3.47%** ($106) | **−5.82%** ($83) | 0.00% ($92) |
| myopic @ `optimize_1d` | price + limit | 1,951,069 | **−2.83%** | **−4.69%** | **−1.06%** |
| pace PPO + MPC | MPC trigger + lookahead | 2,009,914 | +0.69% | 0.00% | **−0.99%** |
| pace PPO (analytic limit) | nothing | 2,009,914 | 0.00% | 0.00% | 0.00% |
| joint SAC `rl_best` | nothing | 2,033,264 | 0.00% | 0.00% | 0.00% |
| joint BC→SAC raw | nothing | 2,094,381 | 0.00% | 0.00% | 0.00% |

Prices in brackets are the policy's mean price under that forecast. Full CSV:
`misspecification_table.csv`.

## Verdict

| Question | Answer |
| --- | --- |
| Do model-consuming policies degrade? | **YES** — myopic loses 3.5–5.8% when its elasticity is wrong |
| Are the joint RL policies affected? | **NO** — 0.00%, and their mean price is identical to the cent ($91.31, $92.08) under every forecast |
| Does a demand-*level* error move pricing? | **NO** — 0.00% for myopic, exactly as the argmax algebra predicts |
| Where does a level error land? | On the level consumers only: `optimize_1d` −1.06%, MPC −0.99% |

**The result the project was missing.** Under a perfect forecast the lead §7 can
support is capped BC→SAC over pace, 3.6%, with the interval above zero. The 1.2%
point gap is a tie. Give the forecast a plausible elasticity error and myopic
alone gives up 3.5–5.8% — about that whole lead — while the joint policies do
not move at all, because their observation is booking state plus calendar
one-hots (§8) and they consult no model.

That is the property an operator actually buys: **not a one-percent lead, but
"does not need your forecast to be right."** It is also the strongest argument in this
repo for the model-free approach, and it was invisible until the forecast and the
world were allowed to differ.

Underpricing hurts more than overpricing here: elasticity −1.5 (price $83) costs
myopic 5.82% against −0.9 (price $106) at 3.47%.

## Caveats

- **The RL policies are indifferent by construction, not by merit.** They never
  had a forecast to lose. This measures exposure, not robustness that was learned.
  A forecast-fed policy trained *under* misspecification might do better than
  either — untested.
- pace PPO + MPC at elasticity −0.9 is **+0.69%**: a wrong forecast helped slightly.
  The MPC fires only on soft days, and a less-elastic forecast nudged its floor
  price up ($92.84 vs $89.43) in a way that happened to pay. Noise, not a finding.
- Only the demand forecast is degraded here. The cancellation-model counterpart is
  `runs/keep_rate_dependence/`, priced at ≤1.11%.
- One demand family, one misspecification shape. "Wrong elasticity" is a plausible
  error but not the only one.
