#!/usr/bin/env python3
"""Demo: soft-aware eval for bc_sac+safe_sl vs pace_ppo vs myopic.

Writes runs/soft_aware_report_demo/soft_aware_comparison.md.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("RPRL_TORCH_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import yaml

from reservation_pricing.evaluate.soft_aware import run_soft_aware_comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30


def main() -> None:
    # Primary config: tree demand + soft_aware (no control wrappers — for baselines/oracle)
    overlay = {
        "demand": {"kind": "tree_elastic"},
        "eval": {
            "n_episodes": EPISODES,
            "soft_aware": {
                "rule": "structural",
                "peak_months": [7, 8, 11, 12],
                "weekend_dow": [5, 6],
                "soft_focus_months": [6],
                "base_demand_threshold": 90.0,
                "soft_oracle": "min_price",
                "floor_price_tol": 1.0,
                "undersell_threshold": 1500.0,
                "lambda_peak": 200.0,
                "mu_soft": 200.0,
                "soft_score_mode": "gap_to_oracle",
            },
        },
        "tune": {"shortfall_weight": 200.0},
    }
    cfg_path = OUT / "config_soft_aware.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)

    extra = [
        {
            "name": "bc_sac+safe_sl",
            "model_path": str(ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip"),
            "algo": "sac",
            "config_path": str(ROOT / "configs/experiment_bc_sac_safe_sl.yaml"),
        },
        {
            "name": "pace_ppo",
            "model_path": str(ROOT / "artifacts/pace_ppo/rl_pace_ppo.zip"),
            "algo": "ppo",
            "config_path": str(ROOT / "configs/experiment_price_only_pace_ppo.yaml"),
        },
    ]
    for e in extra:
        if not Path(e["model_path"]).exists():
            raise FileNotFoundError(e["model_path"])

    out = run_soft_aware_comparison(
        config_path=str(cfg_path),
        model_path=None,
        n_episodes=EPISODES,
        seeds=list(range(EPISODES)),
        held_out=True,
        out_dir=str(OUT),
        include_baselines=True,
        baseline_names=["myopic_greedy", "fixed_price_80"],
        extra_policies=extra,
    )
    print(out["table"].to_string(index=False))
    print(f"\nWrote {OUT / 'soft_aware_comparison.md'}")


if __name__ == "__main__":
    main()
