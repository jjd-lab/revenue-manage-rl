# Design: config-driven reservation revenue management + RL

## Goal

A **production-like**, **config-driven** framework for single-product amphitheater reservation
dynamic pricing (price + selling limit), with:

- Pluggable **demand models** (`demand.kind`)
- Pluggable **algorithms** (`algorithm.name`: ppo | sac | td3 | baseline)
- Classical **baselines** that call the same demand API as production pricing would
- Clear extension points for multi-product / cross-price (not implemented yet)

## Package layout

```
src/reservation_pricing/
  config/           # YAML load / merge / validate
  configs/          # packaged copy of default.yaml (kept identical by a test)
  demand/           # DemandModel protocol + linear_legacy + tree_elastic + assets/
  envs/             # ReservationEnv, PriceOnlyWrapper, OversellGuardEnv, make_env(cfg)
  algorithms/       # SB3 registry (PPO/SAC/TD3) + behaviour cloning
  baselines/        # fixed / myopic (grid) / heuristic / dynamic program (dp.py)
  controls/         # selling_limit, early_promo, oversell_cap, price_mpc
  metrics.py        # business metrics, soft/peak classification, run_episode
  evaluate/         # compare (tables), soft_aware (stratified report), report (runs/ writers)
  train/            # train_from_config, bc_finetune, shared helpers
  tune/             # grid / Optuna
  cli.py            # rprl-train | eval | baselines | tune | fit-demand | bc-sac
configs/
  default.yaml              # full schema: tree_elastic + PPO, every control off
  demand_linear_legacy.yaml # A/B old simulator
  algo_ppo.yaml / algo_sac.yaml / algo_td3.yaml
  experiment_*.yaml, tree_long_*.yaml
docs/
  DESIGN.md  EXTENDING.md  EVALUATING_POLICIES.md  EXPERIMENT_LOG.md  EXPERIMENTS.md
```

Registries:

- `get_demand_model(cfg)` — `demand.kind` → instance
- `make_env(cfg)` — env + demand from config (+ optional PriceOnlyWrapper / OversellGuardEnv)
- `get_selling_limit(cfg)` — `control.selling_limit.kind` → controller
- `get_algorithm(cfg)` / `train_from_config(cfg)` — algo + hyperparams from config

## Glossary

The same object goes by several names across code, tables, and prose:

| Canonical | Also written as | Meaning |
| --- | --- | --- |
| selling limit (`selling_limit`) | SL, availability control, booking limit, authorization level | The most bookings on hand the venue will accept; above capacity it is the overbooking buffer |
| bookings on hand (`current_boh`) | BOH | Accepted bookings not yet cancelled |
| materialized (`cumulative_mat_boh`) | mat | Bookings expected to survive cancellation and no-show and occupy a seat |
| remaining inventory (`remain_inv`) | remain | `capacity − materialized`; negative means denied admission |
| oversell | denied admission | `remain_inv < 0` at the performance date |
| undersell / spoilage | `remain > 1500` | Seats left unsold; structural on soft nights |
| soft night | soft day, low-demand day | Held-out night the floor-price oracle cannot fill (`metrics.classify_soft`) |
| joint policy | two-lever, joint 2D | Agent sets price and selling limit |
| `rl_best` | joint SAC, `long_sac`, `rl_best_sac@200k`, `joint_sac_rl_best` | The shipped joint SAC checkpoint |
| BC→SAC + safe SL | `rl_sac` (row name in the headline CSV), `bc_sac+safe_sl` | The BC→SAC checkpoint under the oversell cap |

## Demand models

### `tree_elastic` (default) — production alignment

**Base (price-unaware):** shipped sklearn `GradientBoostingRegressor` over features:

`days_prior`, `dow`, `month`, `is_weekend`, `is_peak_month`, `booking_curve_bin`

Fitted on a **synthetic** venue-like corpus (weekends/peaks + booking-curve shape) and
shipped under `demand/assets/tree_base_demand.joblib`. Refit later with
`rprl-fit-demand --csv ...` or `fit_base_demand_from_csv`.

**Price effect (multiplicative linear elasticity):**

```
expected_gross = max(0, base * (1 + elasticity * (price - ref_price) / ref_price))
```

Defaults: `elasticity: -1.2`, `ref_price: 100`. Residual Gaussian noise via
`demand_noise_std`. Optional `price_mode: additive` with `beta`.

### `linear_legacy` — prototype A/B

```
mean = 80 - 0.2 * price + 60 * dow_eff + 40 * month_eff
     - 0.1 * (days_prior > 30) * days_prior
```

Select with `demand.kind: linear_legacy` or `-c configs/demand_linear_legacy.yaml`.

### Forecast vs truth

By default there is **one** demand model: the env generates bookings from it, and
every component that consults a model consults that same object. That is what all
published tables assume, and it is worth naming, because it means nothing in the
default setup is ever *wrong* about the world — unlike a real operator.

`demand.forecast` splits the two. When set, the env carries a second, deliberately
imperfect model in `env.forecast_model`, and decision code reaches it through
`demand.protocol.decision_model(env)`:

```yaml
demand:
  kind: tree_elastic
  elasticity: -1.2          # the world
  forecast:                 # what the operator believes; inherits the keys above
    elasticity: -0.9
```

| Reads the **forecast** | Reads the **truth** |
| --- | --- |
| myopic baseline (price grid / closed form) | `ReservationEnv.gross_fn`, `expected_gross` — the world |
| `optimize_1d` selling limit | soft/peak classification (`metrics.classify_soft`) |
| early promo trigger | the soft-day oracle in `evaluate/soft_aware.py` |
| short-horizon price MPC | reward, penalties, every business metric |

The split is deliberate: misspecifying the env would change the world, and
misspecifying the classifier or the oracle would change the scoreboard. Only the
operator's *information* is degraded. The joint RL policies appear in neither
column — their observation is booking state plus calendar one-hots, so they
consult no model at all, which is what `runs/forecast_misspecification/` measures.

One structural caveat that experiment exposed: demand enters as
`base * (1 + elasticity * (price - ref) / ref)`, so a purely multiplicative error
in **base** cancels out of the price argmax. Myopic's price is
`ref(1-e)/(-2e) = $91.67` no matter how wrong the demand *level* is. Level errors
reach only the components that use the level itself — the MPC and `optimize_1d`.
To misspecify *pricing*, perturb `elasticity`.

The analytic selling limit and the oversell cap have a second, narrower channel:
`estimate_keep_rate` reads the env's own cancel / no-show parameters. That is
privileged too, and `runs/keep_rate_dependence/` prices it — see that directory
before assuming it matters.

### Cancel / no-show

Weibull cancel + DOW/month no-show rates remain; parameters are **config-driven**
(`cancel_lambda`, `cancel_rho_*`, `noshow_*`).

## Myopic baseline (honest under tree demand)

Myopic **must not** assume a closed-form from a simple linear `gross_fn` when using
`tree_elastic`. It maximizes `price * predict_mean(features, price)` on a **1D price
grid** via the demand model API. For `linear_legacy` a closed form is still available
as a fast path. Production systems would use the same `predict_mean` API.

## Reward vs metrics

**Training reward (shaped):** per-step `accepted * price * revenue_scale` plus terminal
undersell / oversell penalties and a utilization bonus.

Read the weights as prices. At `revenue_scale` 1e-4 and capacity 10,000, the
default charges about $730 per empty seat ($650 penalty plus $80 of bonus not
earned) and $450 per denied admission, and it forfeits the whole bonus (up to
$800,000) on any night with a denied admission. The policy follows those
prices. An experiment config can override them under `env:` to train on
different costs, leaving `default.yaml` untouched (EXPERIMENT_LOG §13,
`runs/objective/`).

Optional denser shaping (config under `env.reward` or flat `env.*`):

- **`pace_reward`** — each step, if load factor is behind a target booking-curve
  schedule (`linear` or `concave` in elapsed fraction of horizon), subtract
  `pace_penalty * gap`. Gives soft-day undersell a signal before terminal.
- **`soft_day_upweight`** — multiply the shaped return by weekday / off-peak
  factors (and optional low tree-base threshold). Training date sampling can
  also boost soft calendar days via `soft_day_sample_boost`.

**Business metrics stay unshaped** (`true_revenue`, `remain_inv`, oversell /
undersell rates in `info` / eval). Only the SB3 return is paced / upweighted.

**Selection / tuning score:**

```
score = mean_true_revenue - shortfall_weight * mean_capacity_shortfall
```

Example: `configs/experiment_price_only_pace_ppo.yaml`.

## Control layer

Five config-driven controls can take a lever off the agent or post-process its
action. This section says what each does; the YAML to switch one on is in
`docs/EXTENDING.md`, and `configs/default.yaml` lists every knob with the
controllers' defaults.

Every controller is evaluated on the **decision day**: `ReservationEnv.step()`
decrements `days_prior` before it settles the day's bookings, so
`PriceOnlyWrapper` shows each controller `days_prior − 1`, the day the env will
actually charge the decision against.

### Early promo (soft-day price post-process)

When `control.early_promo.enabled` (alias `control.promo`), `PriceOnlyWrapper`
**overrides the RL price** after the agent acts if:

1. predicted price-unaware tree **base** demand `< base_demand_threshold`, and
2. `days_prior >= apply_when_days_prior_ge` (early/mid only),

then `mode: set` forces `promo_price` (or `mode: clip` caps at
`max_price_when_soft`). Analytic SL is unchanged — still price-only, not joint
2D. Prefer combining with pace / soft-day reward shaping.
Example: `configs/experiment_price_only_promo_ppo.yaml`.

### Safe SL projection (joint policies)

When `control.safe_sl.enabled`, `OversellGuardEnv` projects the joint policy's
selling limit **down** after the agent acts so expected show-ups stay within
`capacity * overbook_factor` (chance/analytic cap from keep-rate + remain).
Optional `activate_remain_frac` only projects near capacity; `mix_alpha` softens
the hard clip for score/oversell tradeoff. No retrain — wraps a frozen joint
policy (e.g. BC→SAC). Example: `configs/experiment_bc_sac_safe_sl.yaml`.

### Short-horizon price MPC

When `control.mpc.enabled` (price-only), soft / behind-pace states replace the
RL price with a short-horizon grid search over `predict_mean` (+ remain/score
dynamics). Soft-only mode avoids harming peak days. Example:
`configs/experiment_pace_mpc.yaml`.

### Monotone price (non-decreasing path)

When `control.price_monotone.enabled`, `MonotonePriceEnv` is the outermost
wrapper and keeps the agent's action shape, joint or price-only.
`direction: up` forbids a later day from charging less than the previous
charged price. `mode: clamp` clamps the action. `mode: penalty` leaves the
action and subtracts `penalty * (violation_dollars / price_span)` from the
step reward, outside the frozen `env.reward` weights. `mode: ratchet` reads
the action as a *move* from the reference rather than an absolute price, so
the feasible set is covered without two actions charging the same thing — the
mode to train under, since `clamp` makes every sub-floor action identical and
leaves a learner no gradient there. It requires `max_step`.

`reference` picks what the bounds are measured from: `last` is yesterday's
charged price, `high_water` the episode's running extreme. They coincide under
`clamp` with `tolerance: 0`. Elsewhere `high_water` is the correct one — under
`last` a `tolerance` of *t* permits a *t*-per-day glide without limit, because
the floor re-tracks each lowered price, and a penalty charged against yesterday
makes a slow staircase a series of small fines rather than a bar on leaving the
peak. `apply_on: peak_only` restricts the guarantee to weekends and peak months,
on the calendar alone so an operator knows in advance which nights carry it.

The first step of an episode is exempt. `reset()` writes `min_price` as a
placeholder, and the wrapper clears the last charged price so one episode's
close cannot floor the next. An optional `apply_when_days_prior_le` window is
judged on the decision day `days_prior − 1` inside the controller, because
this wrapper sits outside `PriceOnlyWrapper` and is not shown the shifted day.
The constraint is rejected when early promo (including the `promo` alias) or
MPC is also enabled: both replace the price inside `PriceOnlyWrapper` after
this wrapper has already projected. Example:
`configs/experiment_monotone_up_clamp_sac.yaml`.


## Algorithms

Config block `algorithm: { name: ppo|sac|td3, ...hyperparams }` merged with `train:`.

**BC → SAC:** `bc:` config + `rprl-bc-sac` collects expert trajectories
(`myopic_greedy` by default on train months), fits the SAC actor with MSE, seeds
the replay buffer, then fine-tunes (`algorithms/bc.py`, `train/bc_finetune.py`).

Legacy `train.algo` still works. Adding an algo = register in `algorithms/registry.py`
+ a config snippet.


## Price-only RL + selling-limit controller

Joint mode (default): action is ``(price, selling_limit)`` in ``[-1, 1]^2``.

**Price-only mode** (`control.price_only: true`):

- RL / SB3 sees a **1D** price action (`Box(shape=(1,))`).
- Each step, a **selling-limit controller** fills SL before dynamics via
  `PriceOnlyWrapper` around `ReservationEnv` (config: `docs/EXTENDING.md`).
- Registry: `get_selling_limit(cfg)` in `controls/selling_limit.py` (same style as
  demand). Kinds:
  - **analytic** — `SL = clip(capacity * overbook_factor / keep_rate)`; tighten to
    `min_selling_limit` when `remain_inv < low_remain_frac * capacity`.
  - **optimize_1d** — grid-search SL maximizing
    `price * E[accepted] - oversell_weight * overshoot/C - undersell_weight * shortfall/C`
    with `E[accepted] = min(predict_mean, max(0, SL - boh))` and keep-rate
    materialization (see module docstring).
- Baselines still work: myopic/fixed emit price; wrapper supplies SL when
  `price_only`. Joint mode is unchanged when `price_only` is false/absent.
- Example configs: `configs/experiment_price_only_ppo.yaml`,
  `configs/experiment_price_only_sac.yaml`.

## Multi-product / cross-price

**Not implemented.** See `docs/EXTENDING.md` for the migration sketch.

## Evaluation

`rprl-eval` / `rprl-baselines` roll out held-out months by default and write
comparison tables under `runs/`.

## Tradeoffs

- Synthetic tree base is a stand-in until real booking data is available.
- Single product only; multi-day products and cross-elasticity are design-only.
- Shaped reward still imperfect vs pure terminal revenue.

## History

This repo replaced an earlier prototype; section 1 of
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) describes what was rebuilt.

