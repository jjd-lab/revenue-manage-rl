# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # optional: ruff on every commit
```

## Before you open a pull request

```bash
ruff check . && ruff format --check .
pytest -m "not slow" -q     # seconds
pytest -q                   # also trains three tiny agents; a minute or so
```

CI runs the same on Python 3.10 and 3.13.

## What is safe to change, and what is not

- `env.capacity`, the price band, the selling-limit bounds, and the reward-shaping
  weights in `configs/default.yaml` are frozen: observations are normalized
  against them, so changing one invalidates every checkpoint and every table under
  `runs/`. If you need different values, retrain and regenerate.
- The myopic baseline's overbook/keep constants (`baselines/policies.py`) are a
  softer constraint: its selling limit is non-binding, so they change no published
  table, but they are what behaviour cloning imitates — change them and
  `rprl-bc-sac` stops reproducing the shipped BC→SAC checkpoint.
- `runs/` is findings and is tracked; `artifacts/` is trained weights and is not.
  Never hand-edit a generated file under `runs/` (`*.csv`, `*.png`,
  `comparison.md`, `soft_aware_*.md`) — rerun the script beside it. `NOTES.md` and
  `README.md` there are hand-written.
- `configs/default.yaml` and `src/reservation_pricing/configs/default.yaml` must
  stay byte-identical; a test checks.
- Every evaluation uses held-out seeds 0–29 so tables stay comparable.
- All data is synthetic. Do not add real booking data, and keep the domain
  wording generic.

## Adding an experiment

Follow `docs/EXPERIMENTS.md`: pick an ID, set `train.run_name`, add a row. New
demand models, algorithms, and controllers register the way `docs/EXTENDING.md`
shows.

## Docs

Reader-facing docs live in `docs/`; maintainer notes live in `wiki/` under the
contract in `wiki/SCHEMA.md`. A behavior or convention change should update
both and append one line to `wiki/log.md`.
