#!/usr/bin/env python3
"""What happens when the demand forecast is wrong?

Everything else published here shares one property: nothing is ever wrong about
the world. The env generates demand from the tree model, and every component that
consults a model — the myopic baseline, the price MPC, the `optimize_1d` selling
limit — consults *that same model*. Real operations are the opposite: the forecast
is fitted on history and meets a future that moved.

This degrades the forecast and leaves the world alone (`demand.forecast`, see
`demand.protocol.decision_model`), then re-scores policies on the usual seeds.

Which error to inject matters, and the first version of this script got it wrong.
Demand enters as ``base * (1 + elasticity * (price - ref) / ref)``, so **base is a
multiplicative constant and cancels out of the price argmax**: myopic's price is
``ref(1-e)/(-2e) = $91.67`` however wrong the demand *level* is. A level error can
only reach components that use the level itself — the MPC's trigger and lookahead,
and the `optimize_1d` limit. So both kinds are injected:

    elasticity_-0.9   forecast thinks demand is less price-sensitive (price -> ~$106)
    elasticity_-1.5   forecast thinks it is more price-sensitive     (price -> ~$83)
    level_stale       weekend/peak demand fitted 25% low - included to *demonstrate*
                      that a pure level error moves nothing that prices

Subjects are chosen to span what a policy consults, and the joint RL policies are
the control group: their observation is booking state plus calendar one-hots (§8),
so they should be exactly indifferent.

Nothing is retrained. A drifting forecast is something a deployed system meets,
not something it gets to prepare for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines import myopic_greedy_policy
from reservation_pricing.config import load_config
from reservation_pricing.demand.registry import get_demand_model
from reservation_pricing.demand.synthesize import (
    FEATURE_COLS,
    fit_gradient_boosted,
    synthesize_corpus,
)
from reservation_pricing.demand.tree_elastic import TreeElasticDemand
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


def build_forecasts(demand_cfg: dict) -> dict[str, object]:
    """Wrong models for decision code. The env keeps generating from the true one."""
    forecasts: dict[str, object] = {"true": None}

    # Price-response error: same tree, wrong elasticity. This is what moves prices.
    for e in (-0.9, -1.5):
        forecasts[f"elasticity_{e}"] = get_demand_model({**demand_cfg, "elasticity": e})

    # Level error: weekend/peak demand fitted 25% low, elasticity left correct.
    corpus = synthesize_corpus(n_samples=8000, seed=7)
    lift = 1.0 - 0.25 * ((corpus["is_weekend"] == 1) | (corpus["is_peak_month"] == 1))
    corpus = corpus.assign(base_demand=corpus["base_demand"] * lift)
    payload = {
        "model": fit_gradient_boosted(corpus, random_state=7),
        "feature_cols": FEATURE_COLS,
        "meta": {"source": "forecast misspecification probe"},
    }
    knobs = {k: v for k, v in demand_cfg.items() if k not in ("kind", "model_path", "fitted")}
    forecasts["level_stale"] = TreeElasticDemand(fitted=payload, **knobs)
    return forecasts


# (label, config, checkpoint or None, algo, what the policy consults the forecast for)
SUBJECTS = [
    ("myopic", None, None, None, "price"),
    (
        "myopic @ optimize_1d limit",
        "configs/experiment_price_only_sac.yaml",
        None,
        None,
        "price + limit",
    ),
    (
        "pace PPO + MPC",
        "configs/experiment_pace_mpc.yaml",
        "pace_ppo/rl_pace_ppo.zip",
        "ppo",
        "MPC trigger + lookahead",
    ),
    (
        "pace PPO (analytic limit)",
        "configs/experiment_price_only_pace_ppo.yaml",
        "pace_ppo/rl_pace_ppo.zip",
        "ppo",
        "nothing",
    ),
    ("joint SAC rl_best", None, "tree_long/best/rl_best.zip", "sac", "nothing"),
    ("joint BC->SAC raw", None, "bc_sac/rl_bc_sac_final.zip", "sac", "nothing"),
]


def main() -> None:
    base = load_config()
    soft_cfg = SoftAwareConfig.from_dict(base["eval"]["soft_aware"])
    seeds = list(range(EPISODES))

    print("building wrong forecasts ...", flush=True)
    forecasts = build_forecasts(dict(base["demand"]))
    names = list(forecasts)

    print("collecting the soft-day oracle (always on the true model) ...", flush=True)
    oracle = collect_soft_oracle_revenues(
        lambda: make_env(base, use_held_out=True), seeds, soft_cfg
    )

    rows: list[dict] = []
    for label, cfg_path, ckpt, algo, consults in SUBJECTS:
        cfg = load_config(str(ROOT / cfg_path)) if cfg_path else base
        if ckpt is not None:
            path = ROOT / "artifacts" / ckpt
            if not path.exists():
                print(f"SKIP missing {label}: {path}", flush=True)
                continue
            policy = sb3_policy(load_sb3_model(str(path), algo=algo), deterministic=True)
        else:
            policy = myopic_greedy_policy()

        row: dict[str, object] = {"policy": label, "consults_forecast_for": consults}
        for name in names:
            sa, eps = evaluate_policy_soft_aware(
                lambda cfg=cfg, f=forecasts[name]: make_env(
                    cfg, use_held_out=True, forecast_model=f
                ),
                policy,
                n_episodes=EPISODES,
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            row[name] = round(float(sa["score_aware"]), 2)
            row[f"{name}_price"] = round(float(np.mean([e["mean_price"] for e in eps])), 2)
        for name in names[1:]:
            row[f"{name}_pct"] = round(100.0 * (row[name] / row["true"] - 1.0), 3)
        rows.append(row)
        print(
            f"  {label:<28} true {row['true']:>12,.0f}  "
            + "  ".join(
                f"{n.replace('elasticity_', 'e=')} {row[f'{n}_pct']:+6.2f}%" for n in names[1:]
            ),
            flush=True,
        )

    if not rows:
        raise SystemExit("no checkpoints found; see README 'Model checkpoints'")

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "misspecification_table.csv", index=False)
    print(f"\nWrote {OUT / 'misspecification_table.csv'}")
    cols = ["policy", "consults_forecast_for", "true"] + [
        c for n in names[1:] for c in (f"{n}_pct", f"{n}_price")
    ]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
