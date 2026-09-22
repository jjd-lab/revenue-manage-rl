---
type: convention
title: Repo conventions
description: What each top-level directory owns, which files are generated, and which values are pinned by the shipped checkpoints.
tags: [convention, layout, configs]
timestamp: 2026-09-22
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

Name run directories for what they found (`joint_vs_price_only_soft_aware`,
`oversell_cap_transfer`), not what they ran (`soft_aware_report_demo` is the
one deliberate exception — it names itself a demo because it is one). Driver
scripts are `run_*.py`/`final_eval.py` in lowercase, never SCREAMING_CASE.

## Config layering

`configs/default.yaml` carries the full schema: `demand`, `env`, `control`,
`algorithm`, `train`, `eval`, `tune`, with every control switched off. A
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

The five checkpoints in `artifacts/` were trained against these `env` values, and
every table in `runs/` assumes them:

```yaml
capacity: 10000
booking_horizon: 100
min_price: 80.0
max_price: 120.0
min_selling_limit: 10000.0
max_selling_limit: 15000.0
```

Changing any of them **invalidates all five checkpoints and every result table**,
because observations are normalized against these ranges. Treat them as frozen
unless you intend to retrain and regenerate. Reward-shaping weights
(`undersell_penalty`, `oversell_penalty`, `utilization_bonus`) are likewise baked
into the trained policies.

## Naming

The package name (`reservation_pricing`), CLI prefix (`rprl-`), and class names
(`ReservationEnv`) are deliberately domain-neutral: the code describes the
mechanism, not a particular venue.
