"""Evaluate baselines and trained RL policies on business metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model, resolve_algo_name
from reservation_pricing.baselines import evaluate_baselines
from reservation_pricing.config import load_config
from reservation_pricing.envs import ReservationEnv, make_env
from reservation_pricing.metrics import (
    AggregateMetrics,
    PolicyFn,
    aggregate,
    metrics_table,
    run_episode,
)


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def sb3_policy(model, deterministic: bool = True) -> PolicyFn:
    def _policy(obs: np.ndarray, env: ReservationEnv, state: dict) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return _policy


def evaluate_policy(
    env_factory: Callable[[], ReservationEnv],
    policy: PolicyFn,
    n_episodes: int = 30,
    seeds: Optional[list[int]] = None,
    shortfall_weight: float = 50.0,
) -> tuple[AggregateMetrics, list[dict]]:
    seeds = seeds or list(range(n_episodes))
    episodes = []
    rows = []
    for seed in seeds[:n_episodes]:
        env = env_factory()
        ep = run_episode(env, policy, seed=int(seed))
        episodes.append(ep)
        row = ep.to_dict()
        row["seed"] = int(seed)
        rows.append(row)
    return aggregate(episodes, shortfall_weight=shortfall_weight), rows


def run_comparison(
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    algo: Optional[str] = None,
    n_episodes: int = 30,
    seeds: Optional[list[int]] = None,
    held_out: bool = True,
    out_dir: Optional[str] = None,
    include_baselines: bool = True,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    eval_cfg = cfg.get("eval", {})
    n_episodes = int(n_episodes or eval_cfg.get("n_episodes", 30))
    seeds = seeds or list(eval_cfg.get("seeds", list(range(n_episodes))))
    if len(seeds) < n_episodes:
        seeds = list(seeds) + list(range(1000, 1000 + n_episodes - len(seeds)))

    shortfall_weight = float(cfg.get("tune", {}).get("shortfall_weight", 50.0))
    algo_name = (algo or resolve_algo_name(cfg)).lower()

    def env_factory():
        return make_env(cfg, use_held_out=held_out)

    aggregates: dict[str, AggregateMetrics] = {}
    all_episodes: list[dict] = []

    if include_baselines:
        base = evaluate_baselines(
            env_factory,
            n_episodes=n_episodes,
            seeds=seeds,
            shortfall_weight=shortfall_weight,
        )
        aggregates.update(base["aggregates"])
        all_episodes.extend(base["episodes"])

    if model_path:
        model = load_sb3_model(model_path, algo=algo_name)
        policy = sb3_policy(model, deterministic=True)
        agg, rows = evaluate_policy(
            env_factory,
            policy,
            n_episodes=n_episodes,
            seeds=seeds,
            shortfall_weight=shortfall_weight,
        )
        name = f"rl_{algo_name}"
        aggregates[name] = agg
        for r in rows:
            r["policy"] = name
        all_episodes.extend(rows)

    table = metrics_table(aggregates)
    out = {
        "table": table,
        "aggregates": {k: v.to_dict() for k, v in aggregates.items()},
        "episodes": all_episodes,
        "demand_kind": (cfg.get("demand") or {}).get("kind"),
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path / "comparison_table.csv", index=False)
        pd.DataFrame(all_episodes).to_csv(out_path / "episode_metrics.csv", index=False)
        with (out_path / "comparison_summary.json").open("w") as f:
            json.dump(out["aggregates"], f, indent=2)
        md_lines = [
            "# Evaluation comparison",
            "",
            f"demand.kind: `{(cfg.get('demand') or {}).get('kind')}`",
            "",
            _df_to_markdown(table),
            "",
        ]
        (out_path / "comparison.md").write_text("\n".join(md_lines))

    return out
