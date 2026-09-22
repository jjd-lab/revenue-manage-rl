"""Score the training-seed sweep on held-out nights 0–29.

Seed 42 uses the shipped zips. Seeds 43, 44, and 46 use the weights written by
``train.py``. Each BC→SAC checkpoint is scored raw and under the oversell cap.
Does not retrain and does not rewrite the §7 point estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reservation_pricing.evaluate.intervals import (
    N_DRAWS,
    RNG_SEED,
    index_by_policy,
    paired_difference,
)
from reservation_pricing.evaluate.soft_aware import run_soft_aware_comparison
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44, 46)
HELD_OUT = list(range(30))


def checkpoint(seed: int, kind: str) -> Path:
    """``kind`` is ``pace`` or ``bc``.

    New seeds use ``<label>_s<seed>/final_model.zip``, the path both trainers
    write. ``rl_bc_sac_final.zip`` is accepted as the same file under its
    curated name.
    """
    if seed == 42 and kind == "pace":
        return ROOT / "artifacts" / "pace_ppo" / "rl_pace_ppo.zip"
    if seed == 42 and kind == "bc":
        return ROOT / "artifacts" / "bc_sac" / "rl_bc_sac_final.zip"
    label = "pace" if kind == "pace" else "bc_sac"
    directory = ROOT / "artifacts" / "training_seeds" / f"{label}_s{seed}"
    for name in ("final_model.zip", "rl_bc_sac_final.zip"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return directory / "final_model.zip"


def _entry(name: str, model: Path, algo: str, config: str) -> dict:
    if not model.is_file():
        raise FileNotFoundError(
            f"missing {model}. Seed 42 uses the shipped zips; 43, 44, and 46 come from train.py."
        )
    return {
        "name": name,
        "model_path": str(model),
        "algo": algo,
        "config_path": str(ROOT / "configs" / config),
    }


def policies() -> list[dict]:
    rows = []
    for seed in SEEDS:
        rows.append(
            _entry(
                f"pace_s{seed}",
                checkpoint(seed, "pace"),
                "ppo",
                "experiment_price_only_pace_ppo.yaml",
            )
        )
        bc = checkpoint(seed, "bc")
        rows.append(_entry(f"bc_raw_s{seed}", bc, "sac", "default.yaml"))
        rows.append(_entry(f"bc_cap_s{seed}", bc, "sac", "experiment_bc_sac_safe_sl.yaml"))
    return rows


def _intervals(by_policy: dict) -> pd.DataFrame:
    summary_cfg = SoftAwareConfig()
    # The comparison writes the config it actually used; prefer that when present.
    summary_path = OUT / "soft_aware_summary.json"
    oracle: dict[int, float] | None = None
    cfg = summary_cfg
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        cfg = SoftAwareConfig.from_dict(summary.get("soft_cfg"))
        oracle = {
            int(seed): float(revenue) for seed, revenue in summary["oracle_revenue_by_seed"].items()
        }
    records = []

    def add(policy: str, baseline: str, comparison: str) -> None:
        interval = paired_difference(
            by_policy[policy],
            by_policy[baseline],
            policy=policy,
            baseline=baseline,
            seeds=HELD_OUT,
            cfg=cfg,
            oracle_revenue_by_seed=oracle,
            n_draws=N_DRAWS,
            rng_seed=RNG_SEED,
        )
        records.append(
            {
                "comparison": comparison,
                "policy": policy,
                "baseline": baseline,
                "point_diff": interval.point_diff,
                "paired_diff": interval.mean_diff,
                "paired_ci_low": interval.ci_low,
                "paired_ci_high": interval.ci_high,
                "tie": interval.covers_zero,
                "n_draws": interval.n_draws,
                "n_seeds": interval.n_seeds,
            }
        )

    for seed in SEEDS:
        add(f"bc_cap_s{seed}", f"pace_s{seed}", "matched")
    for seed in SEEDS:
        if seed == 42:
            continue
        add(f"bc_cap_s{seed}", "pace_s42", "published_pace")
    return pd.DataFrame.from_records(records)


def main() -> None:
    run_soft_aware_comparison(
        config_path=str(ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml"),
        model_path=None,
        n_episodes=len(HELD_OUT),
        seeds=HELD_OUT,
        held_out=True,
        out_dir=str(OUT),
        include_baselines=False,
        extra_policies=policies(),
    )
    episodes = pd.read_csv(OUT / "episode_metrics.csv")
    frame = _intervals(index_by_policy(episodes.to_dict(orient="records")))
    frame.to_csv(OUT / "paired_intervals.csv", index=False)
    print(frame.to_string(index=False))
    print("Wrote", OUT / "paired_intervals.csv")


if __name__ == "__main__":
    main()
