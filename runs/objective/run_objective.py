#!/usr/bin/env python3
"""What does the training objective do to the policy?

Held-out seeds 0–29. Two joint SAC policies with the same algorithm,
hyperparameters, seed 7 and 200k steps, differing only in the reward:

- rl_best — frozen ``artifacts/tree_long/best/rl_best.zip``, trained on the
  default reward: about $730 per unsold seat, $450 per denied admission, and a
  utilization bonus (up to $800,000) forfeited on any oversold night
- cu200 — ``configs/experiment_objective_cu200_sac.yaml``: $200 per unsold
  seat and $400 per denied admission on every night, no bonus

Both are also scored behind the published oversell cap (``control.safe_sl``
from ``configs/experiment_bc_sac_safe_sl.yaml``, as in
``runs/oversell_cap_transfer/``), since the objective moves denied admission.

Train, then rerun this script:

    rprl-train -c configs/experiment_objective_cu200_sac.yaml

Rewards do not feed back into dynamics, so each policy is rolled out once per
objective on the same seeds and must produce the same trajectory both times.
Each summed return is checked against its closed form from the final ``info``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
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
REPORT_SEEDS = list(range(30))
ARMS = (
    ("rl_best", ROOT / "artifacts" / "tree_long" / "best" / "rl_best.zip"),
    ("cu200", ROOT / "artifacts" / "objective" / "cu200" / "final_model.zip"),
)
OBJECTIVES = (
    ("default", None),
    ("cu200", ROOT / "configs" / "experiment_objective_cu200_sac.yaml"),
)
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"
RETURN_TOL = 1e-6


def _closed_form_return(env_cfg: dict, info: dict) -> float:
    """The episode's summed reward, rebuilt from its final ``info``."""
    cap = float(env_cfg["capacity"])
    remain = float(info["remain_inv"])
    unsold = max(0.0, remain)
    oversold = max(0.0, -remain)
    total = float(info["true_revenue"]) * float(env_cfg["revenue_scale"])
    total -= float(env_cfg["undersell_penalty"]) * unsold / cap
    total -= float(env_cfg["oversell_penalty"]) * oversold / cap
    if remain >= 0:
        total += float(env_cfg["utilization_bonus"]) * float(np.clip(1.0 - remain / cap, 0.0, 1.0))
    return total


def _rollouts(cfg: dict, policy) -> list[dict]:
    rows = []
    for seed in REPORT_SEEDS:
        env = make_env(cfg, use_held_out=True)
        obs, _info = env.reset(seed=seed)
        terminated = truncated = False
        state: dict = {}
        ret = 0.0
        prices = []
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(policy(obs, env, state))
            ret += float(reward)
            prices.append(float(info["price"]))
        expected = _closed_form_return(cfg["env"], info)
        if abs(ret - expected) > RETURN_TOL:
            raise SystemExit(f"seed {seed}: return {ret} != closed form {expected}")
        rows.append(
            {
                "seed": seed,
                "return": ret,
                "revenue": float(info["true_revenue"]),
                "remain": float(info["remain_inv"]),
                "dow": int(info["dow"]),
                "prices": prices,
            }
        )
    return rows


def main() -> None:
    for name, path in ARMS:
        if not path.is_file():
            raise SystemExit(f"missing checkpoint for {name}: {path}")

    cfgs = {obj: load_config(None if path is None else str(path)) for obj, path in OBJECTIVES}
    plain = cfgs["default"]
    soft_cfg = SoftAwareConfig.from_dict(plain["eval"]["soft_aware"])

    def plain_factory():
        return make_env(plain, use_held_out=True)

    capped = load_config(str(CAP_CONFIG))

    def capped_factory():
        return make_env(capped, use_held_out=True)

    oracle = collect_soft_oracle_revenues(plain_factory, REPORT_SEEDS, soft_cfg)

    scored = {}
    scored_capped = {}
    rollouts = {}
    for name, path in ARMS:
        policy = sb3_policy(load_sb3_model(str(path), algo="sac"), deterministic=True)
        summary, rows = evaluate_policy_soft_aware(
            plain_factory,
            policy,
            n_episodes=len(REPORT_SEEDS),
            seeds=REPORT_SEEDS,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        by_seed = {int(row["seed"]): episode_from_mapping(row) for row in rows}
        scored[name] = (float(summary["score_aware"]), by_seed)
        summary, rows = evaluate_policy_soft_aware(
            capped_factory,
            policy,
            n_episodes=len(REPORT_SEEDS),
            seeds=REPORT_SEEDS,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        by_seed = {int(row["seed"]): episode_from_mapping(row) for row in rows}
        scored_capped[name] = (float(summary["score_aware"]), by_seed)
        rollouts[name] = {obj: _rollouts(cfgs[obj], policy) for obj, _ in OBJECTIVES}
        a, b = (rollouts[name][obj] for obj, _ in OBJECTIVES)
        if any(x["revenue"] != y["revenue"] or x["prices"] != y["prices"] for x, y in zip(a, b)):
            raise SystemExit(f"{name}: trajectories differ between objectives")

    intervals = []
    for label, runs in (("uncapped", scored), ("capped", scored_capped)):
        gap = paired_difference(
            runs["cu200"][1],
            runs["rl_best"][1],
            policy="cu200",
            baseline="rl_best",
            seeds=REPORT_SEEDS,
            cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        intervals.append(
            {
                "comparison": label,
                "policy": "cu200",
                "baseline": "rl_best",
                "point_diff": gap.point_diff,
                "ci_low": gap.ci_low,
                "ci_high": gap.ci_high,
                "covers_zero": bool(gap.covers_zero),
            }
        )

    weekend = set(soft_cfg.weekend_dow)
    table = []
    cross = []
    for name, _path in ARMS:
        score, by_seed = scored[name]
        episodes = rollouts[name]["default"]
        for obj, _ in OBJECTIVES:
            cross.append(
                {
                    "policy": name,
                    "objective": obj,
                    "mean_return": round(
                        float(np.mean([r["return"] for r in rollouts[name][obj]])), 4
                    ),
                }
            )
        capped_eps = scored_capped[name][1].values()
        row = {
            "arm": name,
            "score_aware": round(score, 2),
            "score_aware_capped": round(scored_capped[name][0], 2),
            "peak_denied_nights_capped": sum(
                1 for e in capped_eps if not e.is_soft and e.remain_inv < 0
            ),
        }
        for label, soft in (("soft", True), ("peak", False)):
            eps = [r for r in episodes if bool(by_seed[r["seed"]].is_soft) is soft]
            remain = np.array([r["remain"] for r in eps])
            row[f"{label}_nights"] = len(eps)
            row[f"{label}_revenue"] = round(float(np.mean([r["revenue"] for r in eps])), 2)
            row[f"{label}_mean_price"] = round(
                float(np.mean([np.mean(r["prices"]) for r in eps])), 2
            )
            row[f"{label}_unsold"] = round(float(np.mean(np.maximum(remain, 0.0))), 1)
            row[f"{label}_denied_nights"] = int(np.sum(remain < 0))
            row[f"{label}_denied_seats"] = round(float(np.mean(np.maximum(-remain, 0.0))), 1)
        path = np.mean([r["prices"] for r in episodes if r["dow"] in weekend], axis=0)
        crest = int(np.argmax(path))
        row["weekend_open"] = round(float(path[0]), 2)
        row["weekend_crest"] = round(float(path[crest]), 2)
        row["weekend_crest_days_prior"] = len(path) - 1 - crest
        row["weekend_close"] = round(float(path[-1]), 2)
        table.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(table)
    frame.to_csv(OUT / "objective_table.csv", index=False)
    pd.DataFrame(cross).to_csv(OUT / "cross_objective.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    print(frame.T.to_string(header=False))
    print(pd.DataFrame(cross).to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    main()
