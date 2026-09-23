#!/usr/bin/env python3
"""How far is RL from the textbook planner, and how wrong must its forecast be?

Held-out seeds 0–29, ``score_aware``. The dynamic program
(``baselines/dp.py``) plans each night by backward induction on the demand
forecast, charging the score's own costs. Two questions:

1. With the true forecast, where do the learned policies sit against it?
   Paired against: myopic, joint SAC ``rl_best``, capped joint BC→SAC, and the
   ``cu200`` retrain from ``runs/objective/``.
2. With the wrong forecasts from ``runs/forecast_misspecification/`` (elasticity
   −0.9 and −1.5, weekend/peak level 25% low), how much does the planner lose?
   The learned joint policies consult no forecast, so their scores do not move
   (EXPERIMENT_LOG §11b); myopic is rescored alongside for reference.
3. With a wrong show-up model, i.e. cancellation or no-show rates off, how much
   does it lose? The planner counts expected show-ups itself, from the bookings
   it took and its own keep rates, so a wrong rate is never corrected by the
   env's true count. The learned policies read no show-up model.

Nothing is trained. Checkpoints come from ``artifacts/``; missing ones are
skipped with a note.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines import dp_policy, myopic_greedy_policy
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
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"
# The true values are noshow_base 0.16 and cancel rho 0.4 weekday / 0.5 weekend.
# A lower rho means more cancellations are expected, so the planner overbooks.
KEEP_SCENARIOS = (
    ("no-shows 12%", {"noshow_base": 0.12}),
    ("no-shows 20%", {"noshow_base": 0.20}),
    ("cancellations higher (rho -0.1)", {"cancel_rho_weekday": 0.3, "cancel_rho_weekend": 0.4}),
    ("cancellations lower (rho +0.1)", {"cancel_rho_weekday": 0.5, "cancel_rho_weekend": 0.6}),
)
# (arm, checkpoint under artifacts/, capped)
LEARNED = (
    ("joint SAC rl_best", "tree_long/best/rl_best.zip", False),
    ("joint BC->SAC + cap", "bc_sac/rl_bc_sac_final.zip", True),
    ("joint SAC cu200", "objective/cu200/final_model.zip", False),
)


def _build_forecasts(demand_cfg: dict) -> dict:
    path = ROOT / "runs" / "forecast_misspecification" / "run_misspecification.py"
    spec = importlib.util.spec_from_file_location("run_misspecification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_forecasts(demand_cfg)


def _diagnostics(factory, policy) -> dict:
    """Denied admission, cap honouring and the planner's own forecast of its score."""
    predicted, unhonoured = [], 0
    for seed in SEEDS:
        env = factory()
        obs, _info = env.reset(seed=seed)
        state: dict = {}
        done = False
        while not done:
            obs, _r, terminated, truncated, _info = env.step(policy(obs, env, state))
            done = terminated or truncated
        if "dp_value" in state:
            predicted.append(state["dp_value"])
            unhonoured += state["dp_unhonoured"]
    return {
        "predicted_mean": float(np.mean(predicted)) if predicted else np.nan,
        "unhonoured_caps": unhonoured,
    }


def main() -> None:
    plain = load_config()
    capped = load_config(str(CAP_CONFIG))
    soft_cfg = SoftAwareConfig.from_dict(plain["eval"]["soft_aware"])

    def factory(cfg, forecast=None):
        return lambda: make_env(cfg, use_held_out=True, forecast_model=forecast)

    oracle = collect_soft_oracle_revenues(factory(plain), SEEDS, soft_cfg)

    def score(fac, policy):
        summary, rows = evaluate_policy_soft_aware(
            fac,
            policy,
            n_episodes=len(SEEDS),
            seeds=SEEDS,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        return summary, {int(r["seed"]): episode_from_mapping(r) for r in rows}

    arms = [
        ("DP planner", factory(plain), dp_policy(soft_cfg)),
        ("myopic", factory(plain), myopic_greedy_policy()),
    ]
    for name, ckpt, is_capped in LEARNED:
        path = ROOT / "artifacts" / ckpt
        if not path.is_file():
            print(f"SKIP missing {name}: {path}", flush=True)
            continue
        policy = sb3_policy(load_sb3_model(str(path), algo="sac"), deterministic=True)
        arms.append((name, factory(capped if is_capped else plain), policy))

    scored = {}
    table = []
    for name, fac, policy in arms:
        summary, by_seed = score(fac, policy)
        scored[name] = by_seed
        peak = [e for e in by_seed.values() if not e.is_soft]
        soft = [e for e in by_seed.values() if e.is_soft]
        row = {
            "policy": name,
            "score_aware": round(float(summary["score_aware"]), 2),
            "score_peak": round(float(summary["score_peak"]), 2),
            "score_soft": round(float(summary["score_soft"]), 2),
            "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
            "peak_denied_seats": round(float(np.mean([max(-e.remain_inv, 0) for e in peak])), 1),
            "peak_unsold_seats": round(float(np.mean([max(e.remain_inv, 0) for e in peak])), 1),
            "peak_mean_price": round(float(np.mean([e.mean_price for e in peak])), 2),
            "soft_mean_price": round(float(np.mean([e.mean_price for e in soft])), 2),
        }
        if name == "DP planner":
            row.update(_diagnostics(fac, dp_policy(soft_cfg)))
        table.append(row)
        print(f"  {name:<22} {row['score_aware']:>14,.0f}", flush=True)

    intervals = []
    for name in scored:
        if name == "DP planner":
            continue
        gap = paired_difference(
            scored["DP planner"],
            scored[name],
            policy="DP planner",
            baseline=name,
            seeds=SEEDS,
            cfg=soft_cfg,
            oracle_revenue_by_seed=oracle,
        )
        intervals.append(
            {
                "policy": "DP planner",
                "baseline": name,
                "point_diff": gap.point_diff,
                "ci_low": gap.ci_low,
                "ci_high": gap.ci_high,
                "covers_zero": bool(gap.covers_zero),
            }
        )

    print("rescoring under wrong forecasts ...", flush=True)
    forecasts = _build_forecasts(dict(plain["demand"]))
    wrong = []
    for label, make_policy in (
        ("DP planner", lambda: dp_policy(soft_cfg)),
        ("myopic", myopic_greedy_policy),
    ):
        row = {"policy": label}
        for fname, forecast in forecasts.items():
            summary, by_seed = score(factory(plain, forecast), make_policy())
            row[fname] = round(float(summary["score_aware"]), 2)
            row[f"{fname}_peak_denied_nights"] = sum(
                e.remain_inv < 0 for e in by_seed.values() if not e.is_soft
            )
        wrong.append(row)
        print(f"  {label:<22} " + "  ".join(f"{k} {row[k]:,.0f}" for k in forecasts), flush=True)

    print("rescoring under wrong show-up models ...", flush=True)
    keep_rows = []
    for label, overrides in (("true", {}),) + KEEP_SCENARIOS:
        summary, by_seed = score(factory(plain), dp_policy(soft_cfg, keep_overrides=overrides))
        peak = [e for e in by_seed.values() if not e.is_soft]
        keep_rows.append(
            {
                "show_up_model": label,
                "score_aware": round(float(summary["score_aware"]), 2),
                "peak_unsold_seats": round(float(np.mean([max(e.remain_inv, 0) for e in peak])), 1),
                "peak_denied_nights": sum(e.remain_inv < 0 for e in peak),
                "peak_denied_seats": round(
                    float(np.mean([max(-e.remain_inv, 0) for e in peak])), 1
                ),
            }
        )
        print(f"  {label:<34} {keep_rows[-1]['score_aware']:>14,.0f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keep_rows).to_csv(OUT / "dp_show_up.csv", index=False)
    pd.DataFrame(table).to_csv(OUT / "dp_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    pd.DataFrame(wrong).to_csv(OUT / "dp_forecast.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))
    print(pd.DataFrame(wrong).to_string(index=False))
    print(pd.DataFrame(keep_rows).to_string(index=False))


if __name__ == "__main__":
    main()
