#!/usr/bin/env python3
"""Is the joint advantage the second lever, or just a better price policy?

Each joint policy is scored three ways on the same held-out seeds, with the same
soft-aware scoring as the headline table:

    learned       the selling limit the policy chose (as published)
    open          limit pinned wide open at max_selling_limit (lever disabled)
    fixed         limit pinned to the myopic baseline's constant rule

Only the limit is overridden; the price the policy asks for is untouched. If the
three arms tie, the second lever contributes nothing and the joint-vs-price-only
gap in section 7 is a price-policy difference.

Caveat the table cannot express: the pinned arms are **off-distribution**. Each
policy chose its prices expecting the limit it learned, so this measures how
tightly the two levers are coupled, not what a policy purpose-trained for a fixed
limit would score. The price-only policies are that comparison, and they live in
the section 7 table.

Needs the three joint checkpoints (README "Model checkpoints"); rows whose
checkpoint is missing are skipped. No training, no demand-model changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines.policies import MYOPIC_KEEP_RATE, MYOPIC_OVERBOOK_FACTOR
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

POLICIES = {
    "joint_sac_rl_best": ("tree_long/best/rl_best.zip", "sac"),
    "joint_bc_sac_raw": ("bc_sac/rl_bc_sac_final.zip", "sac"),
    "joint_ppo_long_007": ("tree_long/best/rl_ppo_long_007.zip", "ppo"),
}


def pin(policy, value_norm: float):
    """Same policy, but its selling limit is replaced by a constant."""

    def _policy(obs, env, state):
        a = np.asarray(policy(obs, env, state), dtype=np.float32).copy()
        a[1] = value_norm
        return a

    return _policy


def main() -> None:
    cfg = load_config()
    soft_cfg = SoftAwareConfig.from_dict(cfg["eval"]["soft_aware"])
    seeds = list(range(EPISODES))
    env_cfg = cfg["env"]
    lo, hi = float(env_cfg["min_selling_limit"]), float(env_cfg["max_selling_limit"])
    myopic_sl = float(env_cfg["capacity"]) * MYOPIC_OVERBOOK_FACTOR / MYOPIC_KEEP_RATE

    def to_norm(physical: float) -> float:
        return 2.0 * (physical - lo) / (hi - lo) - 1.0

    def factory():
        return make_env(cfg, use_held_out=True)

    print("collecting the soft-day oracle ...", flush=True)
    oracle = collect_soft_oracle_revenues(factory, seeds, soft_cfg)

    rows: list[dict] = []
    for name, (rel, algo) in POLICIES.items():
        path = ROOT / "artifacts" / rel
        if not path.exists():
            print(f"SKIP missing {name}: {path}", flush=True)
            continue
        base = sb3_policy(load_sb3_model(str(path), algo=algo), deterministic=True)
        arms = [
            ("learned", base),
            (f"open_{hi:.0f}", pin(base, 1.0)),
            (f"fixed_{myopic_sl:.0f}", pin(base, to_norm(myopic_sl))),
        ]
        learned_score = None
        for arm, policy in arms:
            sa, eps = evaluate_policy_soft_aware(
                factory,
                policy,
                n_episodes=EPISODES,
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            score = float(sa["score_aware"])
            if learned_score is None:
                learned_score = score
            rows.append(
                {
                    "policy": name,
                    "limit_arm": arm,
                    "score_aware": round(score, 2),
                    "peak_oversell": round(float(sa["peak"]["oversell_rate"]), 4),
                    "mean_selling_limit": round(
                        float(np.mean([e["mean_selling_limit"] for e in eps])), 1
                    ),
                    "delta_vs_learned": round(score - learned_score, 2),
                }
            )
            print(f"  {name:<20} {arm:<16} {score:>13,.0f}", flush=True)

    if not rows:
        raise SystemExit("no joint checkpoints found; see README 'Model checkpoints'")

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ablation_table.csv", index=False)
    print(f"\nWrote {OUT / 'ablation_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
