#!/usr/bin/env python3
"""Held-out comparison for both_goals: safe SL + MPC vs baselines."""

from __future__ import annotations

from pathlib import Path

from reservation_pricing.config import deep_merge, load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.report import run_checkpoint, write_comparison

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30

BC_SAC = ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip"
PACE_PPO = ROOT / "artifacts/pace_ppo/rl_pace_ppo.zip"
RL_BEST = ROOT / "artifacts/tree_long/best/rl_best.zip"


def _safe_sl(mix_alpha: float) -> dict:
    return {
        "control": {
            "price_only": False,
            "safe_sl": {
                "enabled": True,
                "kind": "chance",
                "overbook_factor": 1.02,
                "low_remain_frac": 0.05,
                "activate_remain_frac": 0.20,
                "mix_alpha": mix_alpha,
            },
        }
    }


# (label, config path or None for default.yaml, checkpoint, algo, config overlay)
CANDIDATES = [
    ("bc_sac_raw", None, BC_SAC, "sac", {"control": {"price_only": False}}),
    ("bc_sac+safe_sl", "configs/experiment_bc_sac_safe_sl.yaml", BC_SAC, "sac", None),
    ("bc_sac+safe_sl_hard", None, BC_SAC, "sac", _safe_sl(0.0)),
    ("bc_sac+safe_sl_mix0.4", None, BC_SAC, "sac", _safe_sl(0.4)),
    ("pace_ppo", "configs/experiment_price_only_pace_ppo.yaml", PACE_PPO, "ppo", None),
    ("pace/mpc", "configs/experiment_pace_mpc.yaml", PACE_PPO, "ppo", None),
    ("rl_best", None, RL_BEST, "sac", {"control": {"price_only": False}}),
]


def main() -> None:
    results: list[dict] = []
    all_eps: list[dict] = []

    for label, cfg_path, path, algo, overlay in CANDIDATES:
        cfg = load_config(str(ROOT / cfg_path)) if cfg_path else load_config()
        if overlay:
            cfg = deep_merge(cfg, overlay)

        def env_factory(cfg=cfg):
            return make_env(cfg, use_held_out=True)

        got = run_checkpoint(label, env_factory, path, algo, episodes=EPISODES)
        if got is None:
            continue
        d, rows = got
        results.append(d)
        all_eps.extend(rows)

    # Recommended safe-SL variant: oversell < 0.05 first, then best score.
    safe_rows = [r for r in results if "safe_sl" in r["policy"]]
    footer: list[str] = []
    if safe_rows:
        rec = sorted(
            safe_rows, key=lambda r: (r["oversell_rate"] > 0.05, r["oversell_rate"], -r["score"])
        )[0]
        footer.append(
            f"**Recommended safe-SL variant:** `{rec['policy']}` "
            f"(score={rec['score']:.0f}, oversell={rec['oversell_rate']:.3f})"
        )

    write_comparison(
        OUT,
        results,
        all_eps,
        title="Both goals evaluation comparison",
        episodes=EPISODES,
        footer=footer,
    )


if __name__ == "__main__":
    main()
