---
type: ops
title: Dev commands
description: The operational constraints behind the README's commands - venv location, what each extra enables, what the tests cover, how tables and figures are regenerated.
tags: [ops, cli, setup]
timestamp: 2026-09-22
---

# Dev commands

The README is the command reference: *Install* has the venv and pip lines, the
table under it lists every `rprl-*` entry point, and *Quick start* has a runnable
example of each. This page holds only what the README does not say.

## The venv

`scripts/run_smoke.sh` sources `.venv/bin/activate` by path, so the venv must sit
at the repo root.

If the checkout lives in a cloud-synced folder (iCloud, Dropbox, OneDrive), create
the venv under a name the sync client skips and symlink `.venv` to it:

```bash
python3 -m venv .venv.nosync
ln -s .venv.nosync .venv
source .venv/bin/activate && pip install -e ".[dev]"
```

The hazard is specific: sync can set the macOS hidden flag on files, and
**Python 3.13 skips hidden `.pth` files**, which silently disables an editable
install. The giveaway is `import reservation_pricing` failing while `pytest`
still passes — `pythonpath` in `pyproject.toml` means the tests never touch the
install. Check and fix:

```bash
ls -lO .venv/lib/python3.13/site-packages/*.pth      # look for "hidden"
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
```

`.gitignore` covers `.venv`, `.venv.nosync/`, and any other `.venv*`, plus
`artifacts/`, `private/` (unpublished notebooks), and `CLAUDE.local.md`.
`CLAUDE.md` is shared and tracked.

## Editor setup

`.vscode/settings.json` points the interpreter at `${workspaceFolder}/.venv/bin/python`
(the symlink, so the same setting works with a plain `.venv`), enables pytest,
adds `src` to the analysis path as a hedge for a clone that skips
`pip install -e`, and keeps the venv and `artifacts/` out of the file watcher.

## What each extra buys

| Extra | Adds | Without it |
| --- | --- | --- |
| `dev` | `pytest`, `optuna`, `matplotlib`, `ruff` | No tests, no tuning, no figures, no lint |
| `tune` | `optuna` only | `rprl-tune --method optuna` raises on import |

`optuna` is imported lazily inside `run_optuna`, so the grid search works without
either extra.

## `requirements.txt` versus `pyproject.toml`

`pyproject.toml` is the source of truth for *what* this package depends on and
which versions are allowed; `requirements.txt` pins one point inside those ranges
— the versions the shipped `runs/` tables were produced with — and is the file to
reach for when a resolver difference is a plausible cause of a numeric
discrepancy. It carries the `dev` extra too, and `-e .`, so one
`pip install -r requirements.txt` gives the same environment as
`pip install -e ".[dev]"`.

Adding or dropping a dependency means editing `pyproject.toml`; `requirements.txt`
follows only when the environment behind `runs/` is deliberately refreshed. Do not
regenerate it from `pip freeze` — that would pull in transitive packages and the
local editable path, and drift the two files apart for no benefit.

## Lint, format, CI

`ruff check .` and `ruff format .` (settings in `pyproject.toml`);
`pre-commit install` runs both before each commit. `.github/workflows/test.yml`
runs ruff plus `pytest -m "not slow"` on Python 3.10 and 3.13, and the full suite
on 3.13.

## What the tests are

`tests/test_smoke.py` is the original smoke coverage: envs build, baselines
produce positive revenue, controllers trigger, config keys reach the env, the
metric / soft-aware plumbing runs. The other files cover the edges: every config
loads and keeps observations inside the space (`test_configs.py`), the console
entry points (`test_cli.py`), the table writers behind `runs/` (`test_reports.py`),
the controller kinds the shipped experiments do not use plus the decision-day
convention (`test_controls_extra.py`), the synthetic corpus and the CSV refit
path (`test_demand_synthesize.py`), and the figure script (`test_scripts.py`).

It is **not** a correctness suite for learned policies: nothing asserts that a
policy reaches a given score. Three tests train a tiny agent and are marked
`slow`; `pytest -m "not slow" -q` runs the rest in a few seconds.

## Regenerating tables and figures

- `python runs/joint_vs_price_only_soft_aware/intervals.py` writes
  `paired_intervals.csv` from the saved episode table and
  `soft_aware_summary.json`. It does not roll policies and does not rewrite the
  point-estimate CSV. Fresh evals get the same columns from
  `rprl-eval --soft-aware --interval --baseline-policy <name>`. The rule for
  reading them is in `docs/EVALUATING_POLICIES.md`.
- `python runs/training_seeds/train.py` trains any experiment config at extra
  seeds (`-c` repeats; `--label` once per config). A `bc:` block uses the
  behaviour-clone warm start; anything else uses the standard trainer. Both
  write `artifacts/training_seeds/<label>_s<seed>/final_model.zip`. It refuses
  seed 42, nights 0–29, the 100–129 block, and the published checkpoint
  directories. `python runs/training_seeds/eval.py` scores the pace and BC→SAC
  sweep. The reading is `runs/training_seeds/NOTES.md`.
- `runs/*/final_eval.py` and `run_headline.py` read checkpoints from `artifacts/`,
  which is untracked, so they fail on a fresh clone until you retrain — see
  [[conventions]] and the README's *Model checkpoints* table. All four
  `final_eval.py` scripts share `evaluate/report.py`, which writes checkpoint
  paths relative to the repo root.
- `python scripts/explain_rl_best.py` — figures 01–07 and the two CSVs in
  `runs/explain_rl_best/`; needs the joint SAC checkpoint.
- `python scripts/build_site_figures.py` — the five PNGs under `site/figures/`,
  drawn from committed CSVs only. `site/index.html` is published with GitHub
  Pages by `.github/workflows/pages.yml`. Branch-deploy only serves `/` or
  `/docs`, so the workflow uploads `site/` as the Pages artifact instead.

`runs/bc_sac/` has no `final_eval.py`; evaluate that family with `rprl-eval`
directly. Two rows of its published table used intermediate checkpoints that were
never kept, so it cannot be reproduced in full.
