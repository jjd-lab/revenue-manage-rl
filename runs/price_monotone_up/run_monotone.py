#!/usr/bin/env python3
"""What does a non-decreasing price cost, and can a policy learn under it?

Held-out seeds 0–29, ``score_aware``, every arm paired against the
unconstrained reference.

The first four arms ask what the guarantee costs:

- unconstrained — frozen ``artifacts/tree_long/best/rl_best.zip``
- clamp_frozen — the same checkpoint under ``mode: clamp`` (no retrain)
- clamp_retrained — SAC trained under ``mode: clamp``
- penalty — SAC trained under ``mode: penalty``, weight chosen on 100–129

``clamp_frozen`` beat the reference and ``clamp_retrained`` lost badly, so the
rest ask *why the retrain failed*, one fix at a time:

- ratchet — the action is a move, not a price, so no two actions collide
- ratchet_pace — the same, plus pace shaping for the scarcity signal
- bc_clamp — clone ``clamp_frozen`` and fine-tune under the clamp
- bc_ratchet — clone it and fine-tune in the ratchet parameterization
- peak_only — the ratchet, but the guarantee only on weekends and peak months
- penalty_hw — the penalty charged against the high-water mark, not yesterday

Train (seed 7, 200k steps each), then rerun this script:

    rprl-train  -c configs/experiment_monotone_up_sac.yaml
    rprl-train  -c configs/experiment_monotone_up_penalty_sac.yaml
    # penalty 1 and 100: same config, override penalty and run_name
    rprl-train  -c configs/experiment_monotone_up_ratchet_sac.yaml
    rprl-train  -c configs/experiment_monotone_up_ratchet_pace_sac.yaml
    rprl-train  -c configs/experiment_monotone_up_peak_only_sac.yaml
    rprl-train  -c configs/experiment_monotone_up_penalty_hw_sac.yaml
    rprl-bc-sac -c configs/experiment_monotone_up_bc_sac.yaml
    rprl-bc-sac -c configs/experiment_monotone_up_bc_ratchet_sac.yaml

Decreases are counted on successive ``info["price"]`` values. ``reset()``'s
placeholder is not a prior decision. Episode metrics only store the mean price.
Clamped steps come from ``info["price_monotone_clamped"]``: a decrease count of
zero says the guarantee held, and the clamped count says how hard it had to
work to hold it.
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
# The high-water penalty accumulates, so it needs its own grid: a weight tuned
# against yesterday's price is far too strong measured against the peak.
PENALTY_HW_WEIGHTS = ((0.1, "01"), (0.5, "05"), (1, "1"), (10, ""))
# (arm, config, run_name, trainer). Missing checkpoints are skipped with a note,
# so the four original arms still report before the rest have been trained.
EXTRA_ARMS = (
    ("ratchet", "experiment_monotone_up_ratchet_sac.yaml", "monotone_up_ratchet", None),
    (
        "ratchet_pace",
        "experiment_monotone_up_ratchet_pace_sac.yaml",
        "monotone_up_ratchet_pace",
        None,
    ),
    # The clone before any RL: the arm that answers the question.
    ("bc_clone", "experiment_monotone_up_bc_sac.yaml", "monotone_up_bc_clamp", "bc_only_model.zip"),
    (
        "bc_clone_ratchet",
        "experiment_monotone_up_bc_ratchet_sac.yaml",
        "monotone_up_bc_ratchet",
        "bc_only_model.zip",
    ),
    ("bc_clamp", "experiment_monotone_up_bc_sac.yaml", "monotone_up_bc_clamp", None),
    ("bc_ratchet", "experiment_monotone_up_bc_ratchet_sac.yaml", "monotone_up_bc_ratchet", None),
    ("peak_only", "experiment_monotone_up_peak_only_sac.yaml", "monotone_up_peak_only", None),
)
# peak_only deliberately leaves soft nights free, so it is the one arm whose
# charged path may fall. Every other constrained arm must show zero decreases.
GUARANTEE_EVERYWHERE = (
    "clamp_frozen",
    "clamp_retrained",
    "ratchet",
    "ratchet_pace",
    "bc_clamp",
    "bc_ratchet",
)
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


def _decreases(factory, policy, seeds: list[int]) -> tuple[int, int, int]:
    """``(decreasing steps, episodes with a decrease, clamped steps)``."""
    steps = 0
    episodes = 0
    clamped = 0
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
            if bool(info.get("price_monotone_clamped", False)):
                clamped += 1
            if prev is not None and price < prev - DECREASE_SLACK:
                steps += 1
                hit = True
            prev = price
        if hit:
            episodes += 1
    return steps, episodes, clamped


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
    # Written so the choice is auditable. These are hyperparameter-selection
    # scores on seeds 100-129 and are never a reported result.
    pd.DataFrame(
        [
            {"penalty": w, "score_aware_seeds_100_129": round(sc, 2), "chosen": w == winner}
            for sc, w in sorted(ranking, key=lambda r: r[1])
        ]
    ).to_csv(OUT / "penalty_selection.csv", index=False)

    print("selecting the high-water penalty weight on seeds 100-129 ...", flush=True)
    hw_cfg_path = ROOT / "configs" / "experiment_monotone_up_penalty_hw_sac.yaml"
    hw_ranking: list[tuple[float, float, Path]] = []
    for weight, tag in PENALTY_HW_WEIGHTS:
        run_name = f"monotone_up_penalty_hw_{tag}" if tag else "monotone_up_penalty_hw"
        zip_path = _best_zip(run_name)
        if zip_path is None:
            print(f"  SKIP high-water penalty {weight}: no checkpoint at {run_name}")
            continue
        hw_cfg = load_config(str(hw_cfg_path))
        hw_cfg["control"]["price_monotone"]["penalty"] = float(weight)

        def _hw_factory(cfg=hw_cfg):
            return make_env(cfg, use_held_out=True)

        policy = sb3_policy(load_sb3_model(str(zip_path), algo="sac"), deterministic=True)
        summary, _rows = _score(_hw_factory, policy, SELECT_SEEDS, soft_cfg, oracle_select)
        score = float(summary["score_aware"])
        hw_ranking.append((score, float(weight), zip_path))
        print(f"  high-water penalty {weight:<5} score_aware {score:,.0f}  (not a result)")
    hw_winner = max(hw_ranking)[1] if hw_ranking else None
    if hw_ranking:
        print(f"high-water penalty winner: {hw_winner}", flush=True)
        pd.DataFrame(
            [
                {
                    "reference": "high_water",
                    "penalty": w,
                    "score_aware_seeds_100_129": round(sc, 2),
                    "chosen": w == hw_winner,
                }
                for sc, w, _z in sorted(hw_ranking, key=lambda r: r[1])
            ]
        ).to_csv(OUT / "penalty_hw_selection.csv", index=False)

    print("scoring the arms on seeds 0-29 ...", flush=True)
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

    for arm, config_name, run_name, explicit in EXTRA_ARMS:
        if explicit is None:
            zip_path = _best_zip(run_name)
        else:
            candidate = ROOT / "artifacts" / "price_monotone" / run_name / explicit
            zip_path = candidate if candidate.is_file() else None
        if zip_path is None:
            print(f"  SKIP {arm}: no checkpoint at artifacts/price_monotone/{run_name}")
            continue
        arm_cfg = load_config(str(ROOT / "configs" / config_name))

        def _factory(cfg=arm_cfg):
            return make_env(cfg, use_held_out=True)

        arms.append(
            (
                arm,
                sb3_policy(load_sb3_model(str(zip_path), algo="sac"), deterministic=True),
                _factory,
                arm_cfg["control"]["price_monotone"].get("penalty")
                if arm_cfg["control"]["price_monotone"]["mode"] == "penalty"
                else None,
            ),
        )

    if hw_ranking:
        hw_zip = max(hw_ranking)[2]
        hw_cfg = load_config(str(hw_cfg_path))
        hw_cfg["control"]["price_monotone"]["penalty"] = float(hw_winner)

        def _hw_report_factory(cfg=hw_cfg):
            return make_env(cfg, use_held_out=True)

        arms.append(
            (
                "penalty_hw",
                sb3_policy(load_sb3_model(str(hw_zip), algo="sac"), deterministic=True),
                _hw_report_factory,
                hw_winner,
            )
        )

    scored = {}
    decreases = {}
    for name, policy, factory, _weight in arms:
        summary, by_seed = _score(factory, policy, REPORT_SEEDS, soft_cfg, oracle)
        scored[name] = (float(summary["score_aware"]), by_seed)
        decreases[name] = _decreases(factory, policy, REPORT_SEEDS)
        print(
            f"  {name:<20} {scored[name][0]:>13,.0f}  "
            f"decreases {decreases[name][0]} steps / {decreases[name][1]} episodes"
            f"  clamped {decreases[name][2]}",
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
        step_hits, episode_hits, clamped_hits = decreases[name]
        rows.append(
            {
                "arm": name,
                "score_aware": round(score, 2),
                "paired_vs_unconstrained": round(point, 2),
                "ci_low": round(ci_low, 2),
                "ci_high": round(ci_high, 2),
                "n_decrease_steps": step_hits,
                "n_episodes_with_decrease": episode_hits,
                "n_clamped_steps": clamped_hits,
                "penalty": "" if weight is None else int(weight),
            }
        )

    if decreases["unconstrained"][0] <= 0:
        raise SystemExit("unconstrained arm never decreased price; the constraint is not binding")
    for name in GUARANTEE_EVERYWHERE:
        if name in decreases and decreases[name][0] != 0:
            raise SystemExit(f"{name} charged a price decrease ({decreases[name][0]} steps)")

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "monotone_table.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "paired_intervals.csv", index=False)
    print(f"\nWrote {OUT / 'monotone_table.csv'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
