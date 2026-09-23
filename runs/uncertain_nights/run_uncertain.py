#!/usr/bin/env python3
"""RL against the planner when every night differs from the usual model.

Each night draws its own demand level, price sensitivity, no-show rate and
cancellation curve (``demand.night_variation``, ``env.noshow_noise_std``,
``env.cancel_rho_night_std``). Every policy sees only what a venue sees
(``env.operator_view``): bookings, and show-ups estimated with the *usual*
rates. The planner and the cap plan with the usual demand model and rates, so
they are a little wrong in a different way on every night.

Two settings, each with an RL policy retrained in it (seed 7, 200k, the
$200/$400 reward):

    rprl-train -c configs/experiment_uncertain_cu200_sac.yaml
    rprl-train -c configs/experiment_uncertain_noisy_cu200_sac.yaml

Held-out seeds 0–29, ``score_aware``. The world's draws come from the night's
seed, so every policy faces the same thirty nights.
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
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"
SETTINGS = (
    ("uncertain", "experiment_uncertain_cu200_sac.yaml", "uncertain_cu200"),
    ("uncertain, noisy", "experiment_uncertain_noisy_cu200_sac.yaml", "uncertain_noisy_cu200"),
)
RETRAINED = "RL retrained here"
# (arm, checkpoint under artifacts/, capped); None marks the retrained arm
LEARNED = (
    (RETRAINED, None, False),
    ("joint SAC cu200", "objective/cu200/final_model.zip", False),
    ("joint SAC rl_best", "tree_long/best/rl_best.zip", False),
    ("joint BC->SAC + cap", "bc_sac/rl_bc_sac_final.zip", True),
)
PAIRS = (
    ("DP planner + cap", RETRAINED),
    ("DP planner", RETRAINED),
    (RETRAINED, "joint SAC cu200"),
)


def main() -> None:
    cap_block = load_config(str(CAP_CONFIG))["control"]
    table, intervals = [], []
    for setting, config, run_name in SETTINGS:
        cfg = load_config(str(ROOT / "configs" / config))
        soft_cfg = SoftAwareConfig.from_dict(cfg["eval"]["soft_aware"])

        def factory(capped, cfg=cfg):
            def _make():
                env = make_env(cfg, use_held_out=True)
                return OversellGuardEnv(env, get_oversell_cap(cap_block)) if capped else env

            return _make

        oracle = collect_soft_oracle_revenues(factory(False), SEEDS, soft_cfg)
        arms = []
        for name, ckpt, capped in LEARNED:
            rel = f"uncertain_nights/{run_name}/final_model.zip" if ckpt is None else ckpt
            path = ROOT / "artifacts" / rel
            if not path.is_file():
                print(f"SKIP missing {name}: {path}", flush=True)
                continue
            model = load_sb3_model(str(path), algo="sac")
            arms.append((name, sb3_policy(model, deterministic=True), capped))
        arms += [
            ("myopic", myopic_greedy_policy(), False),
            ("DP planner", dp_policy(soft_cfg), False),
            ("DP planner + cap", dp_policy(soft_cfg), True),
        ]

        scored = {}
        for name, policy, capped in arms:
            summary, rows = evaluate_policy_soft_aware(
                factory(capped),
                policy,
                n_episodes=len(SEEDS),
                seeds=SEEDS,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            by_seed = {int(r["seed"]): episode_from_mapping(r) for r in rows}
            scored[name] = by_seed
            peak = [e for e in by_seed.values() if not e.is_soft]
            table.append(
                {
                    "setting": setting,
                    "policy": name,
                    "score_aware": round(float(summary["score_aware"]), 2),
                    "peak_nights": len(peak),
                    "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
                    "peak_denied_seats": round(
                        float(np.mean([max(-e.remain_inv, 0.0) for e in peak])), 1
                    ),
                    "peak_unsold_seats": round(
                        float(np.mean([max(e.remain_inv, 0.0) for e in peak])), 1
                    ),
                }
            )
            print(f"  {setting:<17} {name:<22} {table[-1]['score_aware']:>14,.0f}", flush=True)

        for a, b in PAIRS:
            if a not in scored or b not in scored:
                continue
            gap = paired_difference(
                scored[a],
                scored[b],
                policy=a,
                baseline=b,
                seeds=SEEDS,
                cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            intervals.append(
                {
                    "setting": setting,
                    "policy": a,
                    "baseline": b,
                    "point_diff": gap.point_diff,
                    "ci_low": gap.ci_low,
                    "ci_high": gap.ci_high,
                    "covers_zero": bool(gap.covers_zero),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(table)
    frame.to_csv(OUT / "uncertain_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    print(frame.to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    main()
