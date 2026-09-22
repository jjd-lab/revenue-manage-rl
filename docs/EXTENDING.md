# Extending the framework

## Add a demand model

1. Implement a class with `kind`, `predict_mean(features, price)`, `sample_gross(...)`.
2. Register in `demand/registry.py` (`register_demand("my_kind", MyClass)`).
3. Add a config example:

```yaml
demand:
  kind: my_kind
  # ... model knobs
```

4. Smoke-test: `make_env(cfg)` steps; myopic uses `predict_mean` automatically.

## Add an algorithm

1. Add a branch in `algorithms/registry.py::build_model` (or register a factory).
2. Add `configs/algo_<name>.yaml` with `algorithm.name` + hyperparams.
3. Train: `rprl-train -c configs/algo_<name>.yaml`.

Baselines are named policies under `baselines/policies.py` (`BASELINE_FACTORY`),
not SB3 algos. Use `rprl-baselines` or `algorithm.name: baseline` for eval-only flows.

## Switch algo / env / demand from one YAML

```yaml
demand:
  kind: tree_elastic   # or linear_legacy
  elasticity: -1.2
env:
  capacity: 10000
  # cancel / noshow knobs...
algorithm:
  name: sac            # ppo | sac | td3
  learning_rate: 3.0e-4
train:
  total_timesteps: 100000
  seed: 42
eval:
  n_episodes: 30
```

```bash
rprl-train -c configs/experiment_tree_ppo.yaml
rprl-baselines -c configs/demand_linear_legacy.yaml --episodes 20
```

## Refit the tree base demand

```bash
# Re-synthesize + ship asset
rprl-fit-demand --n-samples 8000 --seed 7

# Or fit from your own booking CSV (columns: days_prior,dow,month,is_weekend,
# is_peak_month,booking_curve_bin,base_demand)
rprl-fit-demand --csv path/to/base_demand.csv --out path/to/model.joblib
```

Point config at a custom asset:

```yaml
demand:
  kind: tree_elastic
  model_path: path/to/model.joblib
```

## Future: multi-product & cross-price elasticity

Production pattern (sketch):

1. **Products** — e.g. 1-day, 2-day, lodging+ticket bundles; each with its own
   base-demand tree (or a shared multi-output model).
2. **Service rates / fare classes** — price vector `p[product, rate]`.
3. **Cross-price** — replace scalar elasticity with either:
   - a cross-elasticity matrix: `q = q0 * (1 + E @ ((p - p_ref)/p_ref))`, or
   - nested logit / attraction model over products × rates.
4. **Env** — action space becomes multi-dimensional prices (+ optional booking
   limits per product); observation grows with product-level bookings on hand / remaining.
5. **Capacity** — shared resource constraints across products that consume the
   same service-date inventory.

There is deliberately no multi-product stub in the package: a placeholder that
raises is not a design. Do **not** fake a multi-product env until real product
definitions and capacity coupling are specified.

Suggested migration steps:

1. Keep single-product `DemandModel` as the default path.
2. Introduce `product_ids` + `predict_mean_vector` behind a feature flag.
3. Extend `ReservationEnv` only after observation/action schemas are agreed.
4. Teach myopic / heuristics to optimize over the price vector (coordinate-wise
   grid or projected gradient), still via the demand API.


## Price-only control + selling-limit kinds

How each control works is in `docs/DESIGN.md` § Control layer; this section is
the YAML. `configs/default.yaml` carries every block switched off with the
controllers' defaults, so an experiment only names what it changes.

1. Implement a class with `kind` and `compute(env, price) -> float`.
2. Register via `register_selling_limit("my_kind", MyClass)` in
   `controls/selling_limit.py`.
3. Enable in config:

```yaml
control:
  price_only: true
  selling_limit:
    kind: analytic   # or optimize_1d / my_kind
    overbook_factor: 1.05
```

4. `make_env(cfg)` wraps `ReservationEnv` in `PriceOnlyWrapper` so SB3 sees a 1D
   action. Leave `price_only` false/omitted for legacy joint `(price, SL)`.

Train: `rprl-train -c configs/experiment_price_only_ppo.yaml`

## Give decision code a wrong forecast

`demand.forecast` builds a second model that only baselines and controllers see;
the env keeps generating from the block above it. Use it to ask what a policy is
worth when the forecast is stale or misfitted. Mechanism and the full
reads-forecast / reads-truth split: `docs/DESIGN.md` § Forecast vs truth.

```yaml
demand:
  kind: tree_elastic
  elasticity: -1.2            # the world
  forecast:                   # inherits every key above; override what is wrong
    enabled: true
    elasticity: -0.9          # operator underestimates price sensitivity
    # model_path: artifacts/forecasts/stale_tree.joblib   # or a different tree
```

Programmatically, `make_env(cfg, forecast_model=model)` takes any `DemandModel`.
Perturb `elasticity` to move pricing decisions; perturb the base tree to move the
MPC and `optimize_1d` limit. A multiplicative error in the base level does *not*
move myopic's price — it cancels out of the argmax.

## Config validation

`config.validate_config` checks known `demand.kind`, `algorithm.name`,
`control.selling_limit.kind`, and, when enabled, `price_monotone.direction`
and `price_monotone.mode`. It rejects an enabled monotone block combined with
an enabled `early_promo`, `promo`, or `mpc` block. Extend `KNOWN_DEMAND_KINDS`
/ `KNOWN_ALGOS` / `KNOWN_SL_KINDS` / `KNOWN_MONOTONE_DIRECTIONS` /
`KNOWN_MONOTONE_MODES` when registering new plugins.


## Early promo (soft-day price override)

Config under `control.early_promo` (or `control.promo`). Applied inside
`PriceOnlyWrapper` after RL price, before SL + dynamics. See
`controls/early_promo.py` and `configs/experiment_price_only_promo_ppo.yaml`.

```yaml
control:
  price_only: true
  early_promo:
    enabled: true
    base_demand_threshold: 90.0
    promo_price: 80.0
    mode: set
    apply_when_days_prior_ge: 30
```


## Safe SL projection (joint)

```yaml
control:
  safe_sl:
    enabled: true
    kind: chance          # analytic | chance
    overbook_factor: 1.02
    activate_remain_frac: 0.20
    mix_alpha: 0.25
```

Implemented in `controls/oversell_cap.py` + `envs/oversell_guard.py`; wired in
`make_env` when `price_only` is false.

## Short-horizon price MPC

```yaml
control:
  price_only: true
  mpc:
    enabled: true
    horizon: 5
    n_grid: 21
    soft_only: true
    prefer_min_when_soft: true
    base_demand_threshold: 90.0
```

Implemented in `controls/price_mpc.py`; hooked in `PriceOnlyWrapper` after
early_promo, before SL.

## Monotone price

Config under `control.price_monotone`. Applied by `MonotonePriceEnv`, the
outermost wrapper in `make_env`, on either action shape. `mode: clamp`
clamps; `mode: penalty` charges the violation in the wrapper reward. Do not
enable it together with early promo or MPC. See `controls/price_monotone.py`
and `configs/experiment_monotone_up_sac.yaml`.

```yaml
control:
  price_monotone:
    enabled: true
    direction: up          # up | down
    mode: clamp            # clamp | penalty | ratchet
    reference: last        # last | high_water
    apply_on: always       # always | peak_only
    tolerance: 0
    max_step: null         # required by ratchet: dollars per day
    penalty: 10
    apply_when_days_prior_le: null
```
