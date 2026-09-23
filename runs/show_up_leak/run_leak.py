#!/usr/bin/env python3
"""What is the true show-up count worth to the policies that read it?

``ReservationEnv`` builds ``cumulative_mat_boh`` and ``remain_inv`` from the true
cancellation and no-show process and the night's realized no-show draw. The
learned joint policies read both in their observation, and the oversell cap and
myopic's late tighten read them as attributes. An operator only knows the
bookings it took. ``envs/operator_view.py`` replaces both with the operator's
estimate, bookings taken x assumed keep rate, everywhere decision code looks.

Held-out seeds 0–29, ``score_aware``. Every policy is scored three ways:

- env count: as published, reading the env's true count
- operator, true rates: the estimate, with the true cancellation and no-show rates
- operator, wrong rates: the four show-up errors from ``runs/dp_baseline/``

The DP planner never read the env's count, so its rows here match
``runs/dp_baseline/``. It is also scored behind the cap, with the cap using the
same assumed rates. Nothing is trained.
"""

from __future__ import annotations

from pathlib import Path

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
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"
# Same scenarios as runs/dp_baseline/run_dp.py. None = the env's own count.
SCENARIOS = (
    ("env count", None),
    ("operator, true rates", {}),
    ("no-shows 12%", {"noshow_base": 0.12}),
    ("no-shows 20%", {"noshow_base": 0.20}),
    ("cancellations higher (rho -0.1)", {"cancel_rho_weekday": 0.3, "cancel_rho_weekend": 0.4}),
    ("cancellations lower (rho +0.1)", {"cancel_rho_weekday": 0.5, "cancel_rho_weekend": 0.6}),
)
# (arm, checkpoint under artifacts/ or None for a planner, capped)
ARMS = (
    ("joint SAC rl_best", "tree_long/best/rl_best.zip", False),
    ("joint SAC cu200", "objective/cu200/final_model.zip", False),
    ("joint BC->SAC + cap", "bc_sac/rl_bc_sac_final.zip", True),
    ("myopic", None, False),
    ("DP planner", None, False),
    ("DP planner + cap", None, True),
)


def main() -> None:
    plain = load_config()
    cap_block = load_config(str(CAP_CONFIG))["control"]
    soft_cfg = SoftAwareConfig.from_dict(plain["eval"]["soft_aware"])

    def factory(overrides, capped):
        def _make():
            env = make_env(plain, use_held_out=True)
            if overrides is not None:
                env = OperatorViewEnv(env, overrides)
            if capped:
                env = OversellGuardEnv(env, get_oversell_cap(cap_block))
            return env

        return _make

    oracle = collect_soft_oracle_revenues(factory(None, False), SEEDS, soft_cfg)

    models = {}
    for name, ckpt, _capped in ARMS:
        if ckpt is None:
            continue
        path = ROOT / "artifacts" / ckpt
        if not path.is_file():
            print(f"SKIP missing {name}: {path}", flush=True)
            continue
        models[name] = load_sb3_model(str(path), algo="sac")

    def policy_for(name, overrides):
        if name in models:
            return sb3_policy(models[name], deterministic=True)
        if name == "myopic":
            return myopic_greedy_policy()
        return dp_policy(soft_cfg, keep_overrides=overrides or {})

    rows, fair = [], {}
    for label, overrides in SCENARIOS:
        for name, ckpt, capped in ARMS:
            if ckpt is not None and name not in models:
                continue
            summary, eps = evaluate_policy_soft_aware(
                factory(overrides, capped),
                policy_for(name, overrides),
                n_episodes=len(SEEDS),
                seeds=SEEDS,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            by_seed = {int(r["seed"]): episode_from_mapping(r) for r in eps}
            peak = [e for e in by_seed.values() if not e.is_soft]
            rows.append(
                {
                    "scenario": label,
                    "policy": name,
                    "score_aware": round(float(summary["score_aware"]), 2),
                    "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
                    "peak_denied_seats": round(
                        sum(max(-e.remain_inv, 0.0) for e in peak) / len(peak), 1
                    ),
                    "peak_unsold_seats": round(
                        sum(max(e.remain_inv, 0.0) for e in peak) / len(peak), 1
                    ),
                }
            )
            if label == "operator, true rates":
                fair[name] = by_seed
            print(f"  {label:<34} {name:<22} {rows[-1]['score_aware']:>14,.0f}", flush=True)

    intervals = []
    for name in fair:
        if name.startswith("DP planner"):
            continue
        gap = paired_difference(
            fair["DP planner"],
            fair[name],
            policy="DP planner",
            baseline=name,
            seeds=SEEDS,
            cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        intervals.append(
            {
                "scenario": "operator, true rates",
                "policy": "DP planner",
                "baseline": name,
                "point_diff": gap.point_diff,
                "ci_low": gap.ci_low,
                "ci_high": gap.ci_high,
                "covers_zero": bool(gap.covers_zero),
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "leak_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    print(table.pivot(index="policy", columns="scenario", values="score_aware").to_string())
    print(table.pivot(index="policy", columns="scenario", values="peak_denied_nights").to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == "__main__":
    main()
