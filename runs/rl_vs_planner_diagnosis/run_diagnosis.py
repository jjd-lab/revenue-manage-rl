#!/usr/bin/env python3
"""Why does RL trail the planner? Two retrains that remove its handicaps.

Held-out seeds 0-29, ``score_aware``. Every policy sees what an operator sees:
bookings x usual keep rate in place of the true show-up count
(``env.operator_view``), so nothing from the night's realized draw leaks in.

- R1 ``configs/experiment_score_sac.yaml``: joint SAC trained on the score's
  own terminal charge (the planner's objective) and shown the forecast's base
  demand and soft flag. Tests objective mismatch and missing night type.
- R2 ``configs/experiment_score_bc_dp_sac.yaml``: clone the DP planner, then
  fine-tune with SAC on R1's reward and view. Tests headroom above the planner:
  if fine-tuning cannot beat its own clone, RL has nothing to find.

Both keep the checkpoint that scored best on 30 training-month nights, never on
June or December. Train, then rerun this script:

    rprl-train -c configs/experiment_score_sac.yaml
    rprl-bc-sac -c configs/experiment_score_bc_dp_sac.yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines import dp_policy, myopic_greedy_policy
from reservation_pricing.config import load_config
from reservation_pricing.controls.oversell_cap import get_oversell_cap
from reservation_pricing.envs import make_env
from reservation_pricing.envs.operator_view import OperatorViewEnv
from reservation_pricing.envs.oversell_guard import OversellGuardEnv
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
# Training-month nights, same draw for every policy: is the gap there too?
TRAIN_SEEDS = list(range(1000, 1060))
TRAIN_ARMS = ("DP planner", "joint SAC cu200", "R1 score SAC, final", "R2 clone + SAC, final")
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"
R1_CONFIG = ROOT / "configs" / "experiment_score_sac.yaml"
R2_CONFIG = ROOT / "configs" / "experiment_score_bc_dp_sac.yaml"
R1_DIR = ROOT / "artifacts" / "rl_vs_planner" / "score_sac"
R2_DIR = ROOT / "artifacts" / "rl_vs_planner" / "score_bc_dp_sac"
# (arm, config: None = default, checkpoint or None for a rule, capped)
ARMS = (
    ("DP planner", None, None, False),
    ("DP planner + cap", None, None, True),
    ("myopic", None, None, False),
    ("joint SAC cu200", None, ROOT / "artifacts/objective/cu200/final_model.zip", False),
    ("joint BC->SAC", None, ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip", False),
    ("joint BC->SAC + cap", None, ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip", True),
    ("R1 score SAC", R1_CONFIG, R1_DIR / "best/best_model.zip", False),
    ("R1 score SAC + cap", R1_CONFIG, R1_DIR / "best/best_model.zip", True),
    ("R1 score SAC, final", R1_CONFIG, R1_DIR / "final_model.zip", False),
    ("R2 planner clone", R2_CONFIG, R2_DIR / "bc_only_model.zip", False),
    ("R2 clone + SAC", R2_CONFIG, R2_DIR / "best/best_model.zip", False),
    ("R2 clone + SAC + cap", R2_CONFIG, R2_DIR / "best/best_model.zip", True),
    ("R2 clone + SAC, final", R2_CONFIG, R2_DIR / "final_model.zip", False),
)
# (policy, baseline) pairs for paired intervals
PAIRS = (
    ("DP planner + cap", "R1 score SAC + cap"),
    ("DP planner + cap", "R2 clone + SAC + cap"),
    ("DP planner", "R1 score SAC"),
    ("R1 score SAC", "joint SAC cu200"),
    ("R2 clone + SAC", "R2 planner clone"),
    ("DP planner", "R2 clone + SAC"),
    ("DP planner", "R2 clone + SAC, final"),
    ("DP planner", "joint BC->SAC"),
)
CURVES = (("R1 score SAC", "score_sac"), ("R2 clone + SAC", "score_bc_dp_sac"))


def main() -> None:
    plain = load_config()
    cap_block = load_config(str(CAP_CONFIG))["control"]
    soft_cfg = SoftAwareConfig.from_dict(plain["eval"]["soft_aware"])

    def factory(cfg_path, capped, held_out=True):
        def _make():
            if cfg_path is None:
                env = OperatorViewEnv(make_env(plain, use_held_out=held_out))
            else:
                env = make_env(load_config(str(cfg_path)), use_held_out=held_out)
            if capped:
                env = OversellGuardEnv(env, get_oversell_cap(cap_block))
            return env

        return _make

    oracle = collect_soft_oracle_revenues(factory(None, False), SEEDS, soft_cfg)

    def policy_for(name, ckpt):
        if ckpt is not None:
            return sb3_policy(load_sb3_model(str(ckpt), algo="sac"), deterministic=True)
        if name == "myopic":
            return myopic_greedy_policy()
        return dp_policy(soft_cfg)

    rows, by_arm = [], {}
    for name, cfg_path, ckpt, capped in ARMS:
        if ckpt is not None and not Path(ckpt).is_file():
            print(f"SKIP missing {name}: {ckpt}", flush=True)
            continue
        summary, eps = evaluate_policy_soft_aware(
            factory(cfg_path, capped),
            policy_for(name, ckpt),
            n_episodes=len(SEEDS),
            seeds=SEEDS,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        by_seed = {int(r["seed"]): episode_from_mapping(r) for r in eps}
        by_arm[name] = by_seed
        peak = [e for e in by_seed.values() if not e.is_soft]
        soft = [e for e in by_seed.values() if e.is_soft]
        rows.append(
            {
                "policy": name,
                "score_aware": round(float(summary["score_aware"]), 2),
                "score_peak": round(float(summary["score_peak"]), 2),
                "score_soft": round(float(summary["score_soft"]), 2),
                "soft_mean_price": round(float(np.mean([e.mean_price for e in soft])), 2),
                "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
                "peak_denied_seats": round(
                    sum(max(-e.remain_inv, 0.0) for e in peak) / len(peak), 1
                ),
                "peak_unsold_seats": round(
                    sum(max(e.remain_inv, 0.0) for e in peak) / len(peak), 1
                ),
            }
        )
        print(f"  {name:<24} {rows[-1]['score_aware']:>14,.0f}", flush=True)

    intervals = []
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
                "policy": pol,
                "baseline": base,
                "point_diff": round(gap.point_diff, 2),
                "ci_low": round(gap.ci_low, 2),
                "ci_high": round(gap.ci_high, 2),
                "covers_zero": bool(gap.covers_zero),
            }
        )

    train_rows = []
    train_oracle = collect_soft_oracle_revenues(factory(None, False, False), TRAIN_SEEDS, soft_cfg)
    for name, cfg_path, ckpt, capped in ARMS:
        if name not in TRAIN_ARMS or (ckpt is not None and not Path(ckpt).is_file()):
            continue
        summary, _ = evaluate_policy_soft_aware(
            factory(cfg_path, capped, held_out=False),
            policy_for(name, ckpt),
            n_episodes=len(TRAIN_SEEDS),
            seeds=TRAIN_SEEDS,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=train_oracle,
        )
        train_rows.append(
            {
                "policy": name,
                "held_out_score": next(r["score_aware"] for r in rows if r["policy"] == name),
                "training_month_score": round(float(summary["score_aware"]), 2),
            }
        )

    curves = []
    for arm, run in CURVES:
        path = OUT / run / "eval" / "evaluations.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        for step, res in zip(data["timesteps"], data["results"]):
            # Mean training reward per night in dollars (revenue_scale 1e-4): the score's
            # terminal charge, without the soft-night floor-price term.
            curves.append(
                {
                    "policy": arm,
                    "timesteps": int(step),
                    "train_month_reward": round(float(res.mean()) * 1e4, 0),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "diagnosis_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "training_curves.csv", index=False)
    pd.DataFrame(train_rows).to_csv(OUT / "training_month_table.csv", index=False)
    print(table.to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))
    print(pd.DataFrame(curves).to_string(index=False))
    print(pd.DataFrame(train_rows).to_string(index=False))


if __name__ == "__main__":
    main()
