#!/usr/bin/env python3
"""Can RL improve on the planner by correcting it, when demand arrives off schedule?

Held-out seeds 0-29 (June and December), ``score_aware``, nothing capped, and
every policy sees the operator's show-up estimate (``env.operator_view``).

Each test year moves *when* demand arrives by the same number of days on every
night (``demand.night_variation.timing_shift``: +10 earlier, 0, -10 later). The
forecast keeps the usual booking curve, so the planner misreads early pace as a
level change.

- planner, and planner + pickup (rescales its forecast from booking requests)
- residual: the planner proposes, SAC adds a bounded correction
  (``configs/experiment_residual_sac.yaml``, trained on random timing shifts)
- all months: plain joint SAC from ``runs/year_drift/``, no planner underneath

Train, then rerun this script:

    rprl-train -c configs/experiment_residual_sac.yaml
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
SHIFTS = (10.0, 0.0, -10.0)
PICKUP_DAYS = (70, 50, 30, 15)
VIEW = ROOT / "configs" / "experiment_score_sac.yaml"
RESIDUAL = ROOT / "configs" / "experiment_residual_sac.yaml"
ART = ROOT / "artifacts"
# (arm, config the policy acts through, checkpoint or None for a planner)
ARMS = (
    ("planner", VIEW, None),
    ("planner + pickup", VIEW, None),
    ("residual", RESIDUAL, ART / "residual_planner/residual_sac/final_model.zip"),
    (
        "residual, kept checkpoint",
        RESIDUAL,
        ART / "residual_planner/residual_sac/best/best_model.zip",
    ),
    ("all months", VIEW, ART / "year_drift/score_allmonths_sac/final_model.zip"),
)
PAIRS = (
    ("residual", "planner"),
    ("residual", "planner + pickup"),
    ("planner", "all months"),
    ("planner + pickup", "planner"),
)


def world(path: Path, shift: float) -> dict:
    cfg = copy.deepcopy(load_config(str(path)))
    cfg["env"]["held_out_months"] = [6, 12]
    cfg["demand"]["night_variation"] = {"timing_shift": shift}
    return cfg


def main() -> None:
    soft_cfg = SoftAwareConfig.from_dict(load_config()["eval"]["soft_aware"])
    models = {}
    for name, _cfg, ckpt in ARMS:
        if ckpt is None:
            continue
        if ckpt.is_file():
            models[name] = load_sb3_model(str(ckpt), algo="sac")
        else:
            print(f"SKIP missing {name}: {ckpt}", flush=True)

    def policy_for(name):
        if name in models:
            return sb3_policy(models[name], deterministic=True)
        return dp_policy(soft_cfg, pickup_days=PICKUP_DAYS if "pickup" in name else ())

    rows, intervals = [], []
    for shift in SHIFTS:
        oracle = collect_soft_oracle_revenues(
            lambda s=shift: make_env(world(VIEW, s), use_held_out=True), SEEDS, soft_cfg
        )
        by_arm = {}
        for name, path, ckpt in ARMS:
            if ckpt is not None and name not in models:
                continue
            cfg = world(path, shift)
            summary, eps = evaluate_policy_soft_aware(
                lambda cfg=cfg: make_env(cfg, use_held_out=True),
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
                    "timing_shift_days": shift,
                    "policy": name,
                    "score_aware": round(float(summary["score_aware"]), 2),
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
            print(f"  {shift:+.0f}  {name:<26} {rows[-1]['score_aware']:>14,.0f}", flush=True)
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
                    "timing_shift_days": shift,
                    "policy": pol,
                    "baseline": base,
                    "point_diff": round(gap.point_diff, 2),
                    "ci_low": round(gap.ci_low, 2),
                    "ci_high": round(gap.ci_high, 2),
                    "covers_zero": bool(gap.covers_zero),
                }
            )

    curves = []
    path = OUT / "residual_sac" / "eval" / "evaluations.npz"
    if path.is_file():
        data = np.load(path)
        for step, res in zip(data["timesteps"], data["results"]):
            # Mean training reward per night in dollars (revenue_scale 1e-4).
            curves.append(
                {"timesteps": int(step), "train_month_reward": round(float(res.mean()) * 1e4, 0)}
            )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "residual_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
    print(
        table.pivot(index="policy", columns="timing_shift_days", values="score_aware").to_string()
    )
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    main()
