#!/usr/bin/env python3
"""What does a non-decreasing price cost?

Four arms, held-out seeds 0–29, ``score_aware``:

- unconstrained — frozen ``artifacts/tree_long/best/rl_best.zip``
- clamp_frozen — the same checkpoint under ``mode: clamp`` (no retrain)
- clamp_retrained — SAC trained under ``mode: clamp``
- penalty — SAC trained under ``mode: penalty``

The penalty weight is chosen from ``{1, 10, 100}`` on seeds 100–129. Only the
winner is scored on 0–29. Those selection scores are printed and not written.

Train (seed 7, 200k steps), then rerun this script:

    rprl-train -c configs/experiment_monotone_up_sac.yaml
    rprl-train -c configs/experiment_monotone_up_penalty_sac.yaml
    # penalty 1 and 100: same config, override penalty and run_name
    #   monotone_up_penalty_1 / monotone_up_penalty_100

Decreases are counted on successive ``info["price"]`` values. ``reset()``'s
placeholder is not a prior decision. Episode metrics only store the mean price.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.evaluate.intervals import episode_from_mapping, paired_difference
from reservation_pricing.evaluate.soft_aware import (
    collect_soft_oracle_revenues,
    evaluate_policy_soft_aware,
)
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORT_SEEDS = list(range(30))
SELECT_SEEDS = list(range(100, 130))
PENALTY_WEIGHTS = (1, 10, 100)
# Float32 action round-trip, not the controller tolerance. A real markdown is dollars.
DECREASE_SLACK = 1e-4
RL_BEST = ROOT / "artifacts" / "tree_long" / "best" / "rl_best.zip"


def _by_seed(rows: list[dict]) -> dict:
    return {int(row["seed"]): episode_from_mapping(row) for row in rows}


def _best_zip(run_name: str) -> Path | None:
    folder = ROOT / "artifacts" / "price_monotone" / run_name
    for path in (folder / "best" / "best_model.zip", folder / "final_model.zip"):
        if path.is_file():
            return path
    return None


def _decreases(factory, policy, seeds: list[int]) -> tuple[int, int]:
    """``(decreasing steps, episodes with a decrease)`` on charged prices."""
    steps = 0
    episodes = 0
    for seed in seeds:
        env = factory()
        obs, _info = env.reset(seed=int(seed))
        prev: float | None = None
        hit = False
        terminated = truncated = False
        state: dict = {}
        while not (terminated or truncated):
            obs, _reward, terminated, truncated, info = env.step(policy(obs, env, state))
            price = float(info["price"])
            if prev is not None and price < prev - DECREASE_SLACK:
                steps += 1
                hit = True
            prev = price
        if hit:
            episodes += 1
    return steps, episodes


def _score(factory, policy, seeds, soft_cfg, oracle):
    summary, rows = evaluate_policy_soft_aware(
        factory,
        policy,
        n_episodes=len(seeds),
        seeds=seeds,
        soft_cfg=soft_cfg,
        oracle_revenue_by_seed=oracle,
    )
    return summary, _by_seed(rows)


def main() -> None:
    if not RL_BEST.is_file():
        raise SystemExit(f"missing reference checkpoint: {RL_BEST}")

    cfg_plain = load_config()
    cfg_clamp = load_config(str(ROOT / "configs" / "experiment_monotone_up_sac.yaml"))
    soft_cfg = SoftAwareConfig.from_dict(cfg_plain["eval"]["soft_aware"])

    clamp_zip = _best_zip("monotone_up_clamp")
    penalty_zips = {
        weight: _best_zip(f"monotone_up_penalty_{weight}") for weight in PENALTY_WEIGHTS
    }
    missing = [f"monotone_up_penalty_{w}" for w, path in penalty_zips.items() if path is None]
    if clamp_zip is None:
        missing.append("monotone_up_clamp")
    if missing:
        raise SystemExit(
            "missing checkpoints under artifacts/price_monotone/: "
            + ", ".join(missing)
            + ". Train with rprl-train; see the docstring."
        )

    def plain_factory():
        return make_env(cfg_plain, use_held_out=True)

    def clamp_factory():
        return make_env(cfg_clamp, use_held_out=True)

    def penalty_factory(weight: float):
        cfg = load_config(str(ROOT / "configs" / "experiment_monotone_up_penalty_sac.yaml"))
        cfg["control"]["price_monotone"]["penalty"] = float(weight)

        def _factory(cfg=cfg):
            return make_env(cfg, use_held_out=True)

        return _factory

    print("selecting the penalty weight on seeds 100-129 ...", flush=True)
    oracle_select = collect_soft_oracle_revenues(plain_factory, SELECT_SEEDS, soft_cfg)
    ranking: list[tuple[float, int]] = []
    for weight in PENALTY_WEIGHTS:
        policy = sb3_policy(
            load_sb3_model(str(penalty_zips[weight]), algo="sac"), deterministic=True
        )
        summary, _rows = _score(
            penalty_factory(weight), policy, SELECT_SEEDS, soft_cfg, oracle_select
        )
        score = float(summary["score_aware"])
        ranking.append((score, weight))
        print(f"  penalty {weight:<3}  score_aware {score:,.0f}  (not a result)", flush=True)
    ranking.sort(reverse=True)
    winner = ranking[0][1]
    print(f"penalty winner: {winner}", flush=True)

    print("scoring the four arms on seeds 0-29 ...", flush=True)
    oracle = collect_soft_oracle_revenues(plain_factory, REPORT_SEEDS, soft_cfg)
    frozen = sb3_policy(load_sb3_model(str(RL_BEST), algo="sac"), deterministic=True)
    retrained = sb3_policy(load_sb3_model(str(clamp_zip), algo="sac"), deterministic=True)
    penalised = sb3_policy(
        load_sb3_model(str(penalty_zips[winner]), algo="sac"), deterministic=True
    )

    arms = [
        ("unconstrained", frozen, plain_factory, None),
        ("clamp_frozen", frozen, clamp_factory, None),
        ("clamp_retrained", retrained, clamp_factory, None),
        ("penalty", penalised, penalty_factory(winner), winner),
    ]

    scored = {}
    decreases = {}
    for name, policy, factory, _weight in arms:
        summary, by_seed = _score(factory, policy, REPORT_SEEDS, soft_cfg, oracle)
        scored[name] = (float(summary["score_aware"]), by_seed)
        decreases[name] = _decreases(factory, policy, REPORT_SEEDS)
        print(
            f"  {name:<20} {scored[name][0]:>13,.0f}  "
            f"decreases {decreases[name][0]} steps / {decreases[name][1]} episodes",
            flush=True,
        )

    ref_name = "unconstrained"
    rows = []
    intervals = []
    for name, _policy, _factory, weight in arms:
        score, by_seed = scored[name]
        if name == ref_name:
            point = ci_low = ci_high = 0.0
        else:
            gap = paired_difference(
                by_seed,
                scored[ref_name][1],
                policy=name,
                baseline=ref_name,
                seeds=REPORT_SEEDS,
                cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            point, ci_low, ci_high = gap.point_diff, gap.ci_low, gap.ci_high
            intervals.append(
                {
                    "policy": name,
                    "baseline": ref_name,
                    "point_diff": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "covers_zero": bool(gap.covers_zero),
                }
            )
        step_hits, episode_hits = decreases[name]
        rows.append(
            {
                "arm": name,
                "score_aware": round(score, 2),
                "paired_vs_unconstrained": round(point, 2),
                "ci_low": round(ci_low, 2),
                "ci_high": round(ci_high, 2),
                "n_decrease_steps": step_hits,
                "n_episodes_with_decrease": episode_hits,
                "penalty": "" if weight is None else int(weight),
            }
        )

    if decreases["unconstrained"][0] <= 0:
        raise SystemExit("unconstrained arm never decreased price; the constraint is not binding")
    for name in ("clamp_frozen", "clamp_retrained"):
        if decreases[name][0] != 0:
            raise SystemExit(f"{name} charged a price decrease ({decreases[name][0]} steps)")

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "monotone_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    print(f"\nWrote {OUT / 'monotone_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
