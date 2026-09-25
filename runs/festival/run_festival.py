#!/usr/bin/env python3
"""Festival passes: does RL beat a re-solving planner once products share seats?

Three nights, six passes (every run of consecutive nights), logit demand, and a
per-night selling limit; ``reservation_pricing.festival`` has the scenario.
Held-out seasons: seeds 0-29. Score: revenue minus $200 per empty seat and $400
per denied admission, summed over the three nights. Nothing is capped.

- fixed prices: the planner solved once for one price per pass all season
- planner: re-solves the fluid program every day, 10 blocks
- planner + pickup: also rescales the market size from booking requests
- SAC and BC -> SAC (and the clone alone): ``configs/festival_sac.yaml``,
  ``configs/festival_bc_sac.yaml``
- DAgger clone (round 5, ``run_dagger.py``) and SAC fine-tuned from it
  (``train_dagger_sac.py``)
- the same two trained on the shaped reward: ``configs/festival_sac_shaped.yaml``,
  ``train_dagger_sac.py --shaped``

Train, then rerun this script:

    rprl-train -c configs/festival_sac.yaml
    rprl-bc-sac -c configs/festival_bc_sac.yaml
    python runs/festival/run_dagger.py        # needs this script's per_season.csv
    python runs/festival/train_dagger_sac.py
    rprl-train -c configs/festival_sac_shaped.yaml
    python runs/festival/train_dagger_sac.py --shaped
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.festival.evaluate import evaluate, paired_interval
from reservation_pricing.festival.planner import planner_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(30))
ART = ROOT / "artifacts" / "festival"
CHECKPOINTS = {
    "SAC": ART / "festival_sac/final_model.zip",
    "SAC, kept checkpoint": ART / "festival_sac/best/best_model.zip",
    "BC -> SAC": ART / "festival_bc_sac/final_model.zip",
    "BC -> SAC, kept checkpoint": ART / "festival_bc_sac/best/best_model.zip",
    "clone only": ART / "festival_bc_sac/bc_only.zip",
    "DAgger clone": ART / "festival_dagger/dagger_round5.zip",
    "DAgger -> SAC": ART / "festival_dagger_sac/final_model.zip",
    "SAC, shaped": ART / "festival_sac_shaped/final_model.zip",
    "DAgger -> SAC, shaped": ART / "festival_dagger_sac_shaped/final_model.zip",
}


def main() -> None:
    cfg = load_config(ROOT / "configs" / "festival_sac.yaml")
    policies = {
        "fixed prices": planner_policy(n_buckets=1, resolve=False),
        "planner": planner_policy(),
        "planner + pickup": planner_policy(pickup=True),
    }
    for name, path in CHECKPOINTS.items():
        if path.exists():
            policies[name] = sb3_policy(load_sb3_model(str(path), algo="sac"))
        else:
            print(f"skip {name}: no {path.relative_to(ROOT)}")

    per_season = evaluate(lambda: make_env(cfg), policies, SEEDS)
    per_season.to_csv(OUT / "per_season.csv", index=False)

    table = per_season.drop(columns="seed").groupby("policy", sort=False).mean().round(1)
    table.to_csv(OUT / "table.csv")
    intervals = pd.DataFrame(
        [paired_interval(per_season, p, "planner") for p in policies if p != "planner"]
    ).round(0)
    intervals.to_csv(OUT / "paired_intervals.csv", index=False)

    cols = ["score", "revenue", "unsold", "denied", "nights_denied"]
    print(table[cols].to_string())
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
