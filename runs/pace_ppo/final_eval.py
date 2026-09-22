#!/usr/bin/env python3
"""Held-out comparison: pace+soft-day price-only PPO vs baselines / prior PPO / bc_sac / rl_best."""

from __future__ import annotations

from pathlib import Path

from reservation_pricing.baselines import fixed_price_policy, myopic_greedy_policy
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.report import run_checkpoint, run_labelled, write_comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30


def main() -> None:
    cfg_pace = load_config(str(ROOT / "configs" / "experiment_price_only_pace_ppo.yaml"))
    cfg_po = load_config(str(ROOT / "configs" / "experiment_price_only_ppo_long.yaml"))
    cfg_joint = load_config(str(ROOT / "configs" / "experiment_bc_sac.yaml"))
    cfg_joint.setdefault("control", {})["price_only"] = False

    results: list[dict] = []
    all_eps: list[dict] = []

    def env_pace():
        return make_env(cfg_pace, use_held_out=True)

    def env_po():
        return make_env(cfg_po, use_held_out=True)

    def env_joint():
        return make_env(cfg_joint, use_held_out=True)

    def keep(got):
        if got is not None:
            results.append(got[0])
            all_eps.extend(got[1])

    print("=== Baselines (price_only + analytic SL) ===", flush=True)
    for name, policy in [
        ("myopic_greedy", myopic_greedy_policy()),
        ("fixed_price_80", fixed_price_policy(80.0, selling_limit=12000.0)),
    ]:
        keep(run_labelled(name, env_po, policy, episodes=EPISODES))

    print("=== Price-only RL ===", flush=True)
    keep(
        run_checkpoint(
            "pace_ppo@200k",
            env_pace,
            ROOT / "artifacts/pace_ppo/rl_pace_ppo.zip",
            "ppo",
            episodes=EPISODES,
        )
    )
    # EvalCallback best (shaped) for honesty, when it was kept
    pace_best = ROOT / "artifacts/pace_ppo/rl_pace_ppo_best.zip"
    if pace_best.exists():
        keep(run_checkpoint("pace_ppo_bestckpt", env_pace, pace_best, "ppo", episodes=EPISODES))
    keep(
        run_checkpoint(
            "price_only_ppo_analytic@200k",
            env_po,
            ROOT / "artifacts/price_only_long/rl_ppo_analytic.zip",
            "ppo",
            episodes=EPISODES,
        )
    )

    print("=== Joint mode (prior best) ===", flush=True)
    keep(
        run_checkpoint(
            "rl_best_sac@200k",
            env_joint,
            ROOT / "artifacts/tree_long/best/rl_best.zip",
            "sac",
            episodes=EPISODES,
        )
    )
    keep(
        run_checkpoint(
            "bc_sac_final",
            env_joint,
            ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip",
            "sac",
            episodes=EPISODES,
        )
    )

    write_comparison(
        OUT,
        results,
        all_eps,
        title="Pace + soft-day price-only PPO — held-out comparison",
        episodes=EPISODES,
        preamble=[
            "Training used `pace_reward` + `soft_day_upweight` (shaped return only). "
            "Business metrics below are **unshaped**.",
        ],
    )


if __name__ == "__main__":
    main()
