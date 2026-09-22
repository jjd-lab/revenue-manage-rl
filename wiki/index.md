# Wiki index

Start here. See [SCHEMA.md](SCHEMA.md) for the maintenance contract and
[log.md](log.md) for the chronological record.

## Internal knowledge (`wiki/docs/`)

| Page | What it covers |
| --- | --- |
| [dev-commands](docs/dev-commands.md) | Env setup, extras, lint and CI, what the tests cover, regenerating tables and figures |
| [conventions](docs/conventions.md) | Directory ownership, config layering, experiment identity, values frozen by the checkpoints |

## Reader-facing docs (repo root `docs/`)

The wiki does not mirror these; they are the narrative layer for someone reading
the project cold.

| Page | What it covers |
| --- | --- |
| [DESIGN.md](../docs/DESIGN.md) | Architecture, demand formulas, reward vs metrics, glossary |
| [EXTENDING.md](../docs/EXTENDING.md) | Adding a demand model, algorithm, or controller; multi-product path |
| [EVALUATING_POLICIES.md](../docs/EVALUATING_POLICIES.md) | Soft-day-aware evaluation and the policy-quality checklist |
| [EXPERIMENT_LOG.md](../docs/EXPERIMENT_LOG.md) | Full experiment arc and final comparison table |
| [EXPERIMENTS.md](../docs/EXPERIMENTS.md) | Experiment ID ↔ config ↔ run directory ↔ checkpoint |

## Public page

[site/index.html](../site/index.html) — the chart-led reading for someone who
will not open the experiment log; deployed to GitHub Pages by
`.github/workflows/pages.yml` (branch-deploy cannot serve `/site`).
Figures come from `scripts/build_site_figures.py` (see [dev-commands](docs/dev-commands.md)).

## Results (`runs/`)

Generated, not documented here. Where to look:

| Directory | Holds |
| --- | --- |
| `runs/joint_vs_price_only_soft_aware/` | The headline soft-aware table (§7 of the experiment log) |
| `runs/explain_rl_best/` | Why the winning joint SAC behaves as it does — figures and a guide |
| `runs/ablate_selling_limit/`, `runs/oversell_cap_transfer/` | Is the second lever load-bearing (§9), and does the oversell cap transfer (§10) |
| `runs/forecast_misspecification/`, `runs/keep_rate_dependence/` | What the system is allowed to know, and what that knowledge is worth (§11) |
| `runs/both_goals/`, `runs/oracle_ceiling/` | Oversell cap, price MPC, and the soft-day fill ceiling (§5c) |
| `runs/tree_long/`, `runs/bc_sac/`, `runs/price_only_long/`, `runs/pace_ppo/`, `runs/promo_ppo/` | One directory per training campaign (§2–§5b) |
| `runs/soft_aware_eval/` | Demo of the soft-aware report on two policies |
