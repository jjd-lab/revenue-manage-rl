---
type: convention
title: Repo conventions
description: What each top-level directory owns, which files are generated, and which values are pinned by the shipped checkpoints.
tags: [convention, layout, configs]
timestamp: 2026-09-25
---

# Repo conventions

## Directory ownership

| Path | Owns | Hand-edited? |
| --- | --- | --- |
| `configs/` | Experiment YAMLs; one file per experiment | Yes |
| `src/reservation_pricing/` | The package | Yes |
| `tests/` | Smoke coverage | Yes |
| `docs/` | Reader-facing narrative and design docs | Yes |
| `artifacts/` | SB3 checkpoints (`.zip`) | No — training output, **untracked** |
| `runs/` | Eval tables, CSVs, figures, `NOTES.md` | Mixed, see below |
| `site/` | The public page and its figures (`figures/` is generated) | Page yes, figures no |
| `wiki/` | Durable internal knowledge | Yes |
| `private/` | Pre-framework notebooks and notes | Yes — **untracked**, never ship |

## Findings vs raw material

`runs/` is **findings** and is tracked; `artifacts/` is **raw material** and is
gitignored. The reasoning and the retrain recipe for each checkpoint are in the
README's *Model checkpoints* section; the rule itself is in `CLAUDE.md`.

## `runs/` is mixed

Each `runs/<experiment>/` holds **generated** output (`*.csv`, `*.png`,
`comparison.md`, `soft_aware_*.md`) alongside **hand-written** commentary
(`NOTES.md`, `README.md`) and the script that regenerates the rest
(`run_headline.py`, `final_eval.py`, `run_oracle.py`). The `final_eval.py` scripts
keep only their candidate lists; the rollout and the four output files come from
`evaluate/report.py`, so every table has the same columns and repo-relative paths.

Never hand-edit a generated table to correct it — rerun the script. A table that
disagrees with the checkpoints is a signal, not a typo. All runs use held-out
seeds 0–29; a table on a different seed set is a bug, not a variant.

The festival env draws the whole season from the reset seed, so "held-out" is a
seed set, not a set of months. Training envs (`use_held_out=False`, which both
trainers and the BC expert rollout pass) shift every seed by
`TRAIN_SEED_OFFSET` (1,000,000) so no training or demonstration season is a test
season. The first festival run lacked the offset and leaked — its correction is
in `runs/festival/NOTES.md`. Any new seed-driven env needs the same split.

Name run directories for what they found (`joint_vs_price_only_soft_aware`,
`oversell_cap_transfer`), not what they ran (`soft_aware_report_demo` is the
one deliberate exception — it names itself a demo because it is one). Driver
scripts are `run_*.py`/`final_eval.py` in lowercase, never SCREAMING_CASE.

## Config layering

The control layer has six blocks: `selling_limit`, `early_promo`, `safe_sl`,
`mpc`, `price_monotone`, and `residual` (§19; joint only — the DP planner proposes
and the agent's action is a bounded correction, `price_scale` / `limit_scale`,
plus three observation slots, so its checkpoints need its config to be scored). A new one needs adding to `_CONTROL_OVERRIDE_KEYS`
in `envs/factory.py` and to the bare-block exclusion tuple in
`controls/oversell_cap.py`, which is hardcoded rather than derived from it —
miss either and the block is silently ignored with no error.
`demand.night_variation` and `env.operator_view` are not control blocks: they
are read from the `demand:` and `env:` blocks in `make_env`, so they need
neither registration. A new plain `env:` key does need adding to `_ENV_KEYS` in
`envs/factory.py`, or `make_env` drops it. Inside `demand.night_variation`,
`level_sd` / `elasticity_sd` draw a fresh error per night, while `level_shift` /
`elasticity_shift` (§18; defaults 1.0 / 0.0) move every night the same way — a
year that runs off forecast. `timing_shift` / `timing_sd` (§19; days, positive =
demand arrives earlier; defaults 0) do the same for *when* demand arrives, against
the forecast's booking curve. The shifts consume no random draws and `timing_sd`
draws only when above 0, so an unshifted config reproduces the default nights
exactly; the forecast never sees any of them.

A top-level `festival:` block (§20) bypasses all of the above: `make_env` hands
the config to `make_festival_env`, which builds the multi-product `FestivalEnv`
from `FestivalConfig` and reads nothing from `env:`, `demand:` or `control:`
(they are still merged in from `default.yaml`, and ignored). Any `FestivalConfig`
field — nights, passes, appeal, prices, costs — can be overridden inside the
block; `festival: {}` means the defaults. The package stays separate from
`ReservationEnv` so the single-night code, and its frozen values, are untouched.

`configs/default.yaml` carries the full schema: `demand`, `env`, `control`,
`algorithm`, `train`, `eval`, `tune`, with every control switched off.
Evaluation-only switches that default to off in code
(`demand.night_variation`, `env.operator_view`, `env.cancel_rho_night_std`)
live only in the experiment configs that use them, so the frozen file stays
untouched. The same holds for the §17 training switches: `env.undersell_on_soft:
false` skips the unsold-seat charge on nights the *forecast* calls soft (the DP
planner's terminal charge), and `env.night_features: true` appends the forecast's
mid-horizon base demand and soft flag, growing the observation from 27 to 29
slots — a checkpoint trained with it must be evaluated with it. Under `train:`,
`eval_held_out: false` picks the kept checkpoint on training months so June and
December never influence it; `eval_episodes` and `eval_freq` size that
selection (`eval_settings` in `train/common.py`). All default to the old
behaviour.

`default.yaml` sets no `eval.seeds`, so `rprl-eval` and `rprl-baselines` score
seeds `0..n_episodes-1` — 0–29, the published held-out nights. A config that
lists fewer seeds than `n_episodes` is padded with seeds from 1000 up, which are
not the published nights. A
byte-identical copy ships inside the package (`src/reservation_pricing/configs/`)
so a non-editable install still resolves it; a test keeps the two in step. Every
other config overrides a subset of it:

- `algo_*.yaml` — swap the algorithm only
- `demand_*.yaml` — swap the demand model only
- `experiment_*.yaml` — a full named experiment, usually the thing you want
- `tree_long_*.yaml` — the sweep that produced the section-2 table in
  `docs/EXPERIMENT_LOG.md`; `_007` and `_sac` trained the two curated joint
  checkpoints, the rest are retrain recipes for rows whose weights were not kept.
  Provenance, not entry points.

## Experiment identity

Every runnable config sets `train.run_name` to its experiment ID, so training
writes `<model_dir>/<ID>/` and `<log_dir>/<ID>/` and a run corresponds to its
checkpoint by construction. Without it the trainer falls back to a timestamped
name, which is what forced the older curated copies to be renamed by hand.
Filenames are a family prefix (`tree_long_*`, `experiment_price_only_*`), not the
ID; `docs/EXPERIMENTS.md` is the mapping, and gains a row per new experiment.
Fragments (`algo_*`, `demand_*`, `default.yaml`) deliberately set no ID.

## Values pinned by the shipped checkpoints

The five §7 headline checkpoints in `artifacts/`, and every later retrain, were trained against these `env` values, and
every table in `runs/` assumes them:

```yaml
capacity: 10000
booking_horizon: 100
min_price: 80.0
max_price: 120.0
min_selling_limit: 10000.0
max_selling_limit: 15000.0
```

Changing any of them **invalidates every checkpoint and every result table**,
because observations are normalized against these ranges. Treat them as frozen
unless you intend to retrain and regenerate. Reward-shaping weights
(`undersell_penalty`, `oversell_penalty`, `utilization_bonus`) are likewise baked
into the trained policies. An experiment config may override them under `env:`
to train a *new* arm on a different objective (`experiment_objective_*`); that
leaves `default.yaml` and the shipped checkpoints untouched, and its weights go
to their own `artifacts/` subdirectory.

The festival checkpoints (`artifacts/festival/`) are pinned the same way by
`FestivalConfig`'s defaults — capacity, price band, selling-limit bounds, costs
and the pass list set its observation and action shapes and scales — not by
`default.yaml`.

`FestivalConfig.shape_reward` (off by default; `configs/festival_sac_shaped.yaml`
turns it on) adds potential-based shaping: `FestivalEnv.potential` is minus the
end-of-season charge if selling stopped now, from the operator's show-up
estimate, and is zero at season end. With gamma 1 it adds only a constant per
season, so the optimal policy is unchanged (`test_festival.py` checks this). That
constant does show up in the EvalCallback's training curve (+600 in scaled
units, $6M), so subtract it before comparing a shaped run's curve with an
unshaped one; the evaluation scripts score the unshaped objective.

## Presenting comparisons

- **Planner vs RL is uncapped on both sides.** The oversell cap is a separate
  control layer, not part of the question "does RL beat a planner", so the site,
  README and F6 compare the DP planner with uncapped learned policies. Capped
  rows belong to the cap's own findings (§10, §15, §16), labelled as capped.
- **The festival experiments (§20 on) stay off the site and README.** They are
  recorded only in `docs/EXPERIMENT_LOG.md`, `runs/festival/NOTES.md` and this
  wiki; `site/index.html` and `README.md` do not mention them (decision
  2026-09-25).
- **Describe the reward as it is built:** $650 per unsold seat, plus an $80
  bonus per filled seat that is lost on any night someone is turned away. Do not
  fold the two into one per-seat figure; the bonus is conditional, so the sum
  misstates both.

## Naming

The package name (`reservation_pricing`), CLI prefix (`rprl-`), and class names
(`ReservationEnv`) are deliberately domain-neutral: the code describes the
mechanism, not a particular venue.
