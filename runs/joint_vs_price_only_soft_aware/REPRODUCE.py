"""Reproduce the soft-aware joint vs price-only table (the headline comparison).

Same thirty held-out seeds (0..29) as every other run under ``runs/``.
"""

from __future__ import annotations

from pathlib import Path

from reservation_pricing.evaluate.soft_aware import run_soft_aware_comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30


def _policy(name: str, model: str, algo: str, config: str) -> dict:
    return {
        "name": name,
        "model_path": str(ROOT / "artifacts" / model),
        "algo": algo,
        "config_path": str(ROOT / "configs" / config),
    }


if __name__ == "__main__":
    run_soft_aware_comparison(
        config_path=str(ROOT / "configs/experiment_bc_sac_safe_sl.yaml"),
        model_path=str(ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip"),
        algo="sac",
        n_episodes=EPISODES,
        seeds=list(range(EPISODES)),
        held_out=True,
        out_dir=str(OUT),
        include_baselines=True,
        baseline_names=["myopic_greedy", "fixed_price_80"],
        extra_policies=[
            _policy("joint_bc_sac_raw", "bc_sac/rl_bc_sac_final.zip", "sac", "default.yaml"),
            _policy("joint_sac_rl_best", "tree_long/best/rl_best.zip", "sac", "default.yaml"),
            _policy(
                "joint_ppo_long_007", "tree_long/best/rl_ppo_long_007.zip", "ppo", "default.yaml"
            ),
            _policy(
                "price_only_pace_ppo",
                "pace_ppo/rl_pace_ppo.zip",
                "ppo",
                "experiment_price_only_pace_ppo.yaml",
            ),
            _policy(
                "price_only_ppo",
                "price_only_long/rl_ppo_analytic.zip",
                "ppo",
                "experiment_price_only_ppo.yaml",
            ),
        ],
    )
    print("Wrote", OUT)
