#!/usr/bin/env python3
"""Day-by-day price and selling limit on two peak nights: planner vs capped BC→SAC.

Writes ``night_paths.csv`` for ``scripts/build_site_figures.py`` (F6). Seed 4 is
a December weekend night, seed 0 a December weekday; both are peak nights in the
held-out set. ``show_ups`` is the env's expected show-ups so far, the number that
has to land on the 10,000 seats.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines import dp_policy
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = (4, 0)
BC_SAC = ROOT / "artifacts" / "bc_sac" / "rl_bc_sac_final.zip"
CAP_CONFIG = ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"


def _path(cfg, policy, seed: int, label: str) -> list[dict]:
    env = make_env(cfg, use_held_out=True)
    obs, _info = env.reset(seed=seed)
    u = env.unwrapped
    rows, state, done = [], {}, False
    while not done:
        obs, _r, terminated, truncated, info = env.step(policy(obs, env, state))
        done = terminated or truncated
        rows.append(
            {
                "policy": label,
                "seed": seed,
                "dow": int(u.dow),
                "month": int(u.month),
                "days_prior": int(info["days_prior"]),
                "price": float(info["price"]),
                "selling_limit": float(info["selling_limit"]),
                "accepted": float(info["accepted_booking"]),
                "show_ups": float(u.cumulative_mat_boh),
                "revenue": float(info["true_revenue"]),
            }
        )
    return rows


def main() -> None:
    if not BC_SAC.is_file():
        raise SystemExit(f"missing checkpoint: {BC_SAC}")
    plain = load_config()
    capped = load_config(str(CAP_CONFIG))
    soft_cfg = SoftAwareConfig.from_dict(plain["eval"]["soft_aware"])
    learned = sb3_policy(load_sb3_model(str(BC_SAC), algo="sac"), deterministic=True)
    rows = []
    for seed in SEEDS:
        rows += _path(plain, dp_policy(soft_cfg), seed, "Planner")
        rows += _path(capped, learned, seed, "Joint BC to SAC + cap")
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "night_paths.csv", index=False)
    last = frame.groupby(["seed", "policy"]).tail(1)
    print(last[["seed", "policy", "show_ups", "revenue"]].to_string(index=False))


if __name__ == "__main__":
    main()
