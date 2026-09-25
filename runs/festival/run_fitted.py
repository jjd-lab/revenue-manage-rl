#!/usr/bin/env python3
"""Phase 2: the planner fits its demand from N past seasons instead of being handed it.

History is played at random prices with limits wide open (``festival.fit``), on
training seeds. For each N the fit is repeated on three independent histories;
the smaller histories are the first seasons of the larger ones. The fitted
planner is scored on held-out seeds 0-29 against the true-model planner and
the learned policies in ``per_season.csv``, which trained on 2,000 simulated
seasons each -- far more than any N here.

Needs ``per_season.csv`` from ``run_festival.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.festival.evaluate import evaluate, paired_interval
from reservation_pricing.festival.fit import collect_history, fit_demand
from reservation_pricing.festival.planner import planner_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(30))
N_SEASONS = (2, 5, 20, 100)
REPEATS = 3
FIRST_HISTORY_SEED = 20_000
COMPARE = ("planner", "SAC", "BC -> SAC", "fixed prices")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "festival_sac.yaml")
    base = pd.read_csv(OUT / "per_season.csv")
    base = base[base.policy.isin(COMPARE)]
    true_cfg = make_env(cfg).cfg

    rows, per_season = [], []
    for rep in range(REPEATS):
        first = FIRST_HISTORY_SEED + rep * 1000
        history = collect_history(lambda: make_env(cfg, use_held_out=False), max(N_SEASONS), first)
        for n in N_SEASONS:
            head = {k: v[: n * true_cfg.horizon] for k, v in history.items()}
            fitted = fit_demand(true_cfg, head, n)
            name = f"fitted, {n} seasons, history {rep}"
            scored = evaluate(
                lambda: make_env(cfg), {name: planner_policy(model=fitted)}, SEEDS
            ).assign(n_seasons=n, history=rep)
            per_season.append(scored)
            both = pd.concat([base, scored])
            row = {
                "n_seasons": n,
                "history": rep,
                "score": scored.score.mean(),
                "unsold": scored.unsold.mean(),
                "denied": scored.denied.mean(),
                "market_size": fitted.market_size,
                "beta_early": fitted.beta_early,
                "beta_late": fitted.beta_late,
                "arrival_decay_days": fitted.arrival_decay_days,
                **{f"night_appeal_{i}": a for i, a in enumerate(fitted.night_appeal)},
                **{f"pass_appeal_{i}": a for i, a in enumerate(fitted.pass_appeal)},
            }
            for other in COMPARE:
                iv = paired_interval(both, name, other)
                row[f"vs {other}"] = iv["mean_diff"]
                row[f"vs {other} low"] = iv["ci_low"]
                row[f"vs {other} high"] = iv["ci_high"]
            rows.append(row)
            print(
                pd.DataFrame(rows)[["n_seasons", "history", "score", "vs planner", "vs SAC"]]
                .round(0)
                .to_string(index=False),
                flush=True,
            )

    pd.DataFrame(rows).round(3).to_csv(OUT / "fitted.csv", index=False)
    pd.concat(per_season).to_csv(OUT / "fitted_per_season.csv", index=False)


if __name__ == "__main__":
    main()
