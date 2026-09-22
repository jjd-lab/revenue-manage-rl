#!/usr/bin/env python3
"""Does the oversell cap transfer, or was it a fix for one policy?

`docs/EXPERIMENT_LOG.md` §5c and §7 only ever apply the cap (`control.safe_sl`)
to BC→SAC, where it turns 0.71 peak denied admission into 0.00 for 0.6% of score.
The site generalises that into a method claim — keep overbooking risk in a
projection rather than in the reward. Joint SAC and joint PPO also deny admission
(0.18 and 0.35) and have never been capped. This runs every joint policy through
the same wrapper, uncapped and capped, on the same held-out seeds.

The cap only ever projects a limit **down**, so it cannot invent revenue; the
question is how much score each policy gives up to reach zero denied admission.

Needs the three joint checkpoints (README "Model checkpoints"); rows whose
checkpoint is missing are skipped. No training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.evaluate.soft_aware import (
    collect_soft_oracle_revenues,
    evaluate_policy_soft_aware,
)
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EPISODES = 30
CAP_CONFIG = "configs/experiment_bc_sac_safe_sl.yaml"

POLICIES = {
    "joint_sac_rl_best": ("tree_long/best/rl_best.zip", "sac"),
    "joint_bc_sac_raw": ("bc_sac/rl_bc_sac_final.zip", "sac"),
    "joint_ppo_long_007": ("tree_long/best/rl_ppo_long_007.zip", "ppo"),
}


def main() -> None:
    cfg_plain = load_config()
    cfg_capped = load_config(str(ROOT / CAP_CONFIG))
    soft_cfg = SoftAwareConfig.from_dict(cfg_plain["eval"]["soft_aware"])
    seeds = list(range(EPISODES))
    cap = cfg_capped["control"]["safe_sl"]
    print(
        f"cap: kind={cap['kind']} overbook={cap['overbook_factor']} "
        f"activate_remain_frac={cap.get('activate_remain_frac')} mix_alpha={cap.get('mix_alpha')}",
        flush=True,
    )

    def plain_factory():
        return make_env(cfg_plain, use_held_out=True)

    def capped_factory():
        return make_env(cfg_capped, use_held_out=True)

    print("collecting the soft-day oracle ...", flush=True)
    oracle = collect_soft_oracle_revenues(plain_factory, seeds, soft_cfg)

    rows: list[dict] = []
    for name, (rel, algo) in POLICIES.items():
        path = ROOT / "artifacts" / rel
        if not path.exists():
            print(f"SKIP missing {name}: {path}", flush=True)
            continue
        policy = sb3_policy(load_sb3_model(str(path), algo=algo), deterministic=True)
        scored = {}
        for arm, factory in (("uncapped", plain_factory), ("capped", capped_factory)):
            sa, _ = evaluate_policy_soft_aware(
                factory,
                policy,
                n_episodes=EPISODES,
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            scored[arm] = (float(sa["score_aware"]), float(sa["peak"]["oversell_rate"]))
            print(
                f"  {name:<20} {arm:<9} {scored[arm][0]:>13,.0f}  oversell {scored[arm][1]:.4f}",
                flush=True,
            )
        (s0, o0), (s1, o1) = scored["uncapped"], scored["capped"]
        rows.append(
            {
                "policy": name,
                "score_uncapped": round(s0, 2),
                "score_capped": round(s1, 2),
                "score_given_up_pct": round(100.0 * (1.0 - s1 / s0), 3),
                "peak_oversell_uncapped": round(o0, 4),
                "peak_oversell_capped": round(o1, 4),
            }
        )

    if not rows:
        raise SystemExit("no joint checkpoints found; see README 'Model checkpoints'")

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "cap_transfer_table.csv", index=False)
    print(f"\nWrote {OUT / 'cap_transfer_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
