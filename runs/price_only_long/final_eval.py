#!/usr/bin/env python3
"""Held-out comparison: price-only PPO/SAC vs baselines vs joint bc_sac / rl_best.

Apples-to-apples business metrics on the same tree_elastic held-out seeds/months:
- price_only agents + baselines use price_only env (analytic SL for baselines/PPO;
  SAC trained with optimize_1d — eval uses that same controller).
- joint agents (bc_sac_final, rl_best) use joint (price, SL) action space.

Reads the curated copies under `artifacts/price_only_long/`. Skips a row
when its zip is absent (`artifacts/` is untracked).
"""

from __future__ import annotations

from pathlib import Path

from reservation_pricing.baselines import (
    fixed_price_policy,
    heuristic_booking_limit_policy,
    myopic_greedy_policy,
)
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.report import run_checkpoint, run_labelled, write_comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[-1]


def main() -> None:
    cfg_po_analytic = load_config(str(ROOT / "configs" / "experiment_price_only_ppo_long.yaml"))
    cfg_po_opt1d = load_config(str(ROOT / "configs" / "experiment_price_only_sac_long.yaml"))
    cfg_joint = load_config(str(ROOT / "configs" / "experiment_bc_sac.yaml"))
    cfg_joint.setdefault("control", {})["price_only"] = False

    results: list[dict] = []
    all_eps: list[dict] = []

    def env_po_analytic():
        return make_env(cfg_po_analytic, use_held_out=True)

    def env_po_opt1d():
        return make_env(cfg_po_opt1d, use_held_out=True)

    def env_joint():
        return make_env(cfg_joint, use_held_out=True)

    def keep(got):
        if got is not None:
            results.append(got[0])
            all_eps.extend(got[1])

    print("=== Baselines (price_only + analytic SL) ===", flush=True)
    for name, policy in [
        ("myopic_greedy@price_only", myopic_greedy_policy()),
        ("fixed_price_80@price_only", fixed_price_policy(80.0, selling_limit=12000.0)),
        ("fixed_price_100@price_only", fixed_price_policy(100.0, selling_limit=12000.0)),
        ("heuristic_booking_limit@price_only", heuristic_booking_limit_policy()),
    ]:
        keep(run_labelled(name, env_po_analytic, policy, episodes=EPISODES))

    print("=== Price-only RL ===", flush=True)
    art = ROOT / "artifacts/price_only_long"
    ppo_path = _first_existing(art / "rl_ppo_analytic.zip", art / "ppo_analytic/final_model.zip")
    sac_path = _first_existing(
        art / "rl_sac_optimize1d.zip", art / "sac_optimize1d/final_model.zip"
    )
    sac_analytic = _first_existing(
        art / "rl_sac_analytic.zip", art / "sac_analytic/final_model.zip"
    )
    keep(
        run_checkpoint(
            "price_only_ppo_analytic@200k", env_po_analytic, ppo_path, "ppo", episodes=EPISODES
        )
    )
    keep(
        run_checkpoint(
            "price_only_sac_optimize1d@150k", env_po_opt1d, sac_path, "sac", episodes=EPISODES
        )
    )
    if sac_analytic.exists():
        keep(
            run_checkpoint(
                "price_only_sac_analytic", env_po_analytic, sac_analytic, "sac", episodes=EPISODES
            )
        )

    print("=== Joint mode (prior best) ===", flush=True)
    for name, policy in [
        ("myopic_greedy@joint", myopic_greedy_policy()),
        ("fixed_price_80@joint", fixed_price_policy(80.0, selling_limit=12000.0)),
    ]:
        keep(run_labelled(name, env_joint, policy, episodes=EPISODES))
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
        title="Price-only RL long campaign — held-out comparison",
        episodes=EPISODES,
        preamble=[
            "Price-only agents/baselines: 1D price + selling-limit controller. "
            "Joint agents (`bc_sac_final`, `rl_best`): full (price, SL) action. "
            "Same held-out seeds/months and business metrics for apples-to-apples comparison.",
        ],
    )


if __name__ == "__main__":
    main()
