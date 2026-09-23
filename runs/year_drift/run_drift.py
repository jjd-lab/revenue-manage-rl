#!/usr/bin/env python3
"""Seasonal coverage, and a year that runs above or below the forecast.

Held-out seeds 0-29 (June and December), ``score_aware``, nothing capped, and
every policy sees the operator's show-up estimate (``env.operator_view``).

Step 1: does training on all twelve months close §17's gap? ``R1`` never saw
June or December; ``all months`` (``configs/experiment_score_allmonths_sac.yaml``)
is the same setup trained on every month.

Step 2: a year that misses the forecast. Every test night's demand is scaled by
the same ``level_shift`` (0.8, 1.0, 1.2), while the forecast, and everything
decision code reads from it, stays the usual model. ``drift-trained``
(``configs/experiment_year_drift_sac.yaml``) trained on nights that each draw
their own level and price sensitivity, so it has to read the year from booking
pace. The planner plans on the stale forecast, with or without a pickup
adjustment that rescales the forecast from booking requests seen so far.

Train, then rerun this script:

    rprl-train -c configs/experiment_score_allmonths_sac.yaml
    rprl-train -c configs/experiment_year_drift_sac.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines import dp_policy
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.evaluate.intervals import episode_from_mapping, paired_difference
from reservation_pricing.evaluate.soft_aware import (
    collect_soft_oracle_revenues,
    evaluate_policy_soft_aware,
)
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = list(range(30))
SHIFTS = (0.8, 1.0, 1.2)
PICKUP_DAYS = (70, 50, 30, 15)
# Every learned policy shares R1's observation; its config builds the test env.
VIEW_CONFIG = ROOT / "configs" / "experiment_score_sac.yaml"
ART = ROOT / "artifacts"
ARMS = (
    ("planner", None),
    ("planner + pickup", None),
    ("R1 (no June or December)", ART / "rl_vs_planner/score_sac/final_model.zip"),
    ("all months", ART / "year_drift/score_allmonths_sac/final_model.zip"),
    ("drift-trained", ART / "year_drift/year_drift_sac/final_model.zip"),
)
PAIRS = (
    ("all months", "R1 (no June or December)"),
    ("planner", "all months"),
    ("planner", "drift-trained"),
    ("planner + pickup", "drift-trained"),
    ("planner + pickup", "planner"),
)
CURVES = (("all months", "score_allmonths_sac"), ("drift-trained", "year_drift_sac"))


def world(shift: float) -> dict:
    cfg = copy.deepcopy(load_config(str(VIEW_CONFIG)))
    cfg["env"]["held_out_months"] = [6, 12]
    cfg["demand"]["night_variation"] = {"level_shift": shift}
    return cfg


def main() -> None:
    soft_cfg = SoftAwareConfig.from_dict(load_config()["eval"]["soft_aware"])
    models = {
        name: load_sb3_model(str(ckpt), algo="sac")
        for name, ckpt in ARMS
        if ckpt is not None and ckpt.is_file()
    }
    for name, ckpt in ARMS:
        if ckpt is not None and name not in models:
            print(f"SKIP missing {name}: {ckpt}", flush=True)

    def policy_for(name):
        if name in models:
            return sb3_policy(models[name], deterministic=True)
        return dp_policy(soft_cfg, pickup_days=PICKUP_DAYS if "pickup" in name else ())

    rows, intervals = [], []
    for shift in SHIFTS:
        cfg = world(shift)

        def factory(cfg=cfg):
            return make_env(cfg, use_held_out=True)

        oracle = collect_soft_oracle_revenues(factory, SEEDS, soft_cfg)
        by_arm = {}
        for name, ckpt in ARMS:
            if ckpt is not None and name not in models:
                continue
            summary, eps = evaluate_policy_soft_aware(
                factory,
                policy_for(name),
                n_episodes=len(SEEDS),
                seeds=SEEDS,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            by_seed = {int(r["seed"]): episode_from_mapping(r) for r in eps}
            by_arm[name] = by_seed
            peak = [e for e in by_seed.values() if not e.is_soft]
            rows.append(
                {
                    "level_shift": shift,
                    "policy": name,
                    "score_aware": round(float(summary["score_aware"]), 2),
                    "peak_nights": len(peak),
                    "peak_mean_price": round(float(np.mean([e.mean_price for e in peak])), 2),
                    "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
                    "peak_denied_seats": round(
                        sum(max(-e.remain_inv, 0.0) for e in peak) / len(peak), 1
                    ),
                    "peak_unsold_seats": round(
                        sum(max(e.remain_inv, 0.0) for e in peak) / len(peak), 1
                    ),
                }
            )
            print(f"  {shift:.1f}  {name:<26} {rows[-1]['score_aware']:>14,.0f}", flush=True)
        for pol, base in PAIRS:
            if pol not in by_arm or base not in by_arm:
                continue
            gap = paired_difference(
                by_arm[pol],
                by_arm[base],
                policy=pol,
                baseline=base,
                seeds=SEEDS,
                cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            intervals.append(
                {
                    "level_shift": shift,
                    "policy": pol,
                    "baseline": base,
                    "point_diff": round(gap.point_diff, 2),
                    "ci_low": round(gap.ci_low, 2),
                    "ci_high": round(gap.ci_high, 2),
                    "covers_zero": bool(gap.covers_zero),
                }
            )

    curves = []
    for arm, run in CURVES:
        path = OUT / run / "eval" / "evaluations.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        for step, res in zip(data["timesteps"], data["results"]):
            # Mean training reward per night in dollars (revenue_scale 1e-4).
            curves.append(
                {
                    "policy": arm,
                    "timesteps": int(step),
                    "train_month_reward": round(float(res.mean()) * 1e4, 0),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "drift_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
    print(table.pivot(index="policy", columns="level_shift", values="score_aware").to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    main()
