#!/usr/bin/env python3
"""How much do the controllers depend on knowing the true cancellation model?

`controls.selling_limit.estimate_keep_rate` reads the environment's **own**
generating parameters — `cancel_lambda`, `cancel_rho_weekday/weekend`,
`noshow_base`, `noshow_dow_coef`, `noshow_month_coef` — and replays the same
Weibull the env uses in `cancel_fn`. So the analytic selling limit and the
oversell cap operate with a perfectly specified cancellation model, which a real
venue would have to estimate and would get somewhat wrong.

That is privileged information. This prices it: replace the estimator with fixed
guesses (`control.*.keep_rate`), including deliberately wrong ones, and see what
moves. Two places it is used:

    cap        the oversell cap on BC->SAC (control.safe_sl.keep_rate)
    limit      the analytic selling limit the price-only policies get every day
               (control.selling_limit.keep_rate)

Needs the BC->SAC, pace PPO and price-only PPO checkpoints; missing rows skip.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import deep_merge, load_config
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

# None = the exact estimator that reads the env's true parameters.
GUESSES = [("exact_model", None), ("fixed_0.85", 0.85), ("fixed_0.70", 0.70), ("fixed_0.95", 0.95)]

# (label, which control block holds keep_rate, config, checkpoint, algo)
SUBJECTS = [
    (
        "bc_sac + cap",
        "safe_sl",
        "configs/experiment_bc_sac_safe_sl.yaml",
        "bc_sac/rl_bc_sac_final.zip",
        "sac",
    ),
    (
        "pace PPO limit",
        "selling_limit",
        "configs/experiment_price_only_pace_ppo.yaml",
        "pace_ppo/rl_pace_ppo.zip",
        "ppo",
    ),
    (
        "price-only PPO limit",
        "selling_limit",
        "configs/experiment_price_only_ppo.yaml",
        "price_only_long/rl_ppo_analytic.zip",
        "ppo",
    ),
]


def main() -> None:
    base = load_config()
    soft_cfg = SoftAwareConfig.from_dict(base["eval"]["soft_aware"])
    seeds = list(range(EPISODES))
    oracle = collect_soft_oracle_revenues(
        lambda: make_env(base, use_held_out=True), seeds, soft_cfg
    )

    rows: list[dict] = []
    for label, block, cfg_path, ckpt, algo in SUBJECTS:
        path = ROOT / "artifacts" / ckpt
        if not path.exists():
            print(f"SKIP missing {label}: {path}", flush=True)
            continue
        policy = sb3_policy(load_sb3_model(str(path), algo=algo), deterministic=True)
        cfg0 = load_config(str(ROOT / cfg_path))
        row: dict[str, object] = {"subject": label, "control_block": block}
        for name, keep in GUESSES:
            cfg = deep_merge(cfg0, {"control": {block: {"keep_rate": keep}}})
            sa, _ = evaluate_policy_soft_aware(
                lambda cfg=cfg: make_env(cfg, use_held_out=True),
                policy,
                n_episodes=EPISODES,
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            row[name] = round(float(sa["score_aware"]), 2)
            row[f"{name}_oversell"] = round(float(sa["peak"]["oversell_rate"]), 4)
        scores = [row[n] for n, _ in GUESSES]
        row["spread_pct"] = round(100.0 * (max(scores) - min(scores)) / row["exact_model"], 3)
        row["exact_is_best"] = bool(row["exact_model"] == max(scores))
        rows.append(row)
        print(
            f"  {label:<22} spread {row['spread_pct']:.2f}%   exact-is-best {row['exact_is_best']}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no checkpoints found; see README 'Model checkpoints'")

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "keep_rate_table.csv", index=False)
    print(f"\nWrote {OUT / 'keep_rate_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
