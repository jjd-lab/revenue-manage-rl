"""Paired bootstrap intervals for ``score_aware``.

The interval is over nights. ``score_aware`` is a stratified sum of a peak
slice and a soft slice, so a draw resamples the seed list with replacement
and recomputes the score on that multiset. Bootstrapping the finished number
would treat a soft night and a peak night as interchangeable.

A comparison is paired: both policies are scored on the same resampled seeds.
The generator is ``numpy.random.default_rng(0)`` unless a test passes another
seed. The reported interval is the 2.5 and 97.5 percentiles of
``score_aware(policy) - score_aware(baseline)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from reservation_pricing.evaluate.soft_aware import aggregate_soft_aware
from reservation_pricing.metrics import EpisodeMetrics, SoftAwareConfig

N_DRAWS = 10_000
RNG_SEED = 0


@dataclass(frozen=True)
class PairedInterval:
    """One paired comparison of ``score_aware``."""

    policy: str
    baseline: str
    point_diff: float
    mean_diff: float
    ci_low: float
    ci_high: float
    n_draws: int
    n_seeds: int

    @property
    def covers_zero(self) -> bool:
        return self.ci_low <= 0.0 <= self.ci_high


def episode_from_mapping(row: Mapping) -> EpisodeMetrics:
    """Build an episode from a CSV row or a ``to_dict`` record."""

    def _optional_float(key: str) -> Optional[float]:
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return None
        return float(value)

    def _optional_int(key: str) -> Optional[int]:
        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return None
        return int(value)

    is_soft = row.get("is_soft")
    is_peak = row.get("is_peak")
    service = row.get("service_date")
    if service is None or service == "" or pd.isna(service):
        service_date = None
    else:
        service_date = str(service)
    return EpisodeMetrics(
        true_revenue=float(row["true_revenue"]),
        shaped_return=float(row.get("shaped_return", 0.0)),
        load_factor=float(row.get("load_factor", 0.0)),
        remain_inv=float(row["remain_inv"]),
        sellout_day=_optional_int("sellout_day"),
        capacity=int(row.get("capacity", 0)),
        service_date=service_date,
        month=_optional_int("month"),
        dow=_optional_int("dow"),
        mean_price=float(row.get("mean_price", 0.0) or 0.0),
        mean_selling_limit=float(row.get("mean_selling_limit", 0.0) or 0.0),
        n_steps=int(row.get("n_steps", 0) or 0),
        base_demand=_optional_float("base_demand"),
        frac_at_floor=float(row.get("frac_at_floor", 0.0) or 0.0),
        is_soft=None if is_soft is None or pd.isna(is_soft) else bool(is_soft),
        is_peak=None if is_peak is None or pd.isna(is_peak) else bool(is_peak),
    )


def index_by_policy(
    rows: Sequence[Mapping],
) -> dict[str, dict[int, EpisodeMetrics]]:
    """``{policy: {seed: episode}}``. Later rows for the same pair replace earlier ones."""
    indexed: dict[str, dict[int, EpisodeMetrics]] = {}
    for row in rows:
        name = str(row["policy"])
        seed = int(row["seed"])
        indexed.setdefault(name, {})[seed] = episode_from_mapping(row)
    return indexed


def _score(
    by_seed: Mapping[int, EpisodeMetrics],
    seeds: Sequence[int],
    cfg: SoftAwareConfig,
    oracle: Optional[dict[int, float]],
) -> float:
    episodes = [by_seed[int(seed)] for seed in seeds]
    scored = aggregate_soft_aware(
        episodes,
        cfg,
        oracle_revenue_by_seed=oracle,
        seed_by_index=[int(seed) for seed in seeds],
    )
    return float(scored["score_aware"])


def _draws(seeds: Sequence[int], n_draws: int, rng_seed: int) -> np.ndarray:
    population = np.asarray(list(seeds), dtype=np.int64)
    rng = np.random.default_rng(rng_seed)
    return rng.choice(population, size=(n_draws, population.shape[0]), replace=True)


def _aligned_seeds(
    by_seed_a: Mapping[int, EpisodeMetrics],
    by_seed_b: Mapping[int, EpisodeMetrics],
    seeds: Optional[Sequence[int]],
) -> list[int]:
    shared = set(by_seed_a) & set(by_seed_b)
    if seeds is None:
        ordered = sorted(shared)
    else:
        ordered = [int(seed) for seed in seeds if int(seed) in shared]
        missing = [int(seed) for seed in seeds if int(seed) not in shared]
        if missing:
            raise ValueError(f"seeds missing from one or both policies: {missing}")
    if not ordered:
        raise ValueError("no shared seeds to resample")
    return ordered


def bootstrap_scores(
    by_policy: Mapping[str, Mapping[int, EpisodeMetrics]],
    seeds: Sequence[int],
    *,
    cfg: Optional[SoftAwareConfig] = None,
    oracle_revenue_by_seed: Optional[Mapping[int, float]] = None,
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Point ``score_aware`` and one bootstrap sample per policy, on the same draws."""
    cfg = cfg or SoftAwareConfig()
    oracle = (
        None
        if oracle_revenue_by_seed is None
        else {int(k): float(v) for k, v in oracle_revenue_by_seed.items()}
    )
    seed_list = [int(seed) for seed in seeds]
    draws = _draws(seed_list, n_draws, rng_seed)
    point: dict[str, float] = {}
    samples: dict[str, np.ndarray] = {}
    for name, by_seed in by_policy.items():
        missing = [seed for seed in seed_list if seed not in by_seed]
        if missing:
            raise ValueError(f"{name} is missing seeds {missing}")
        point[name] = _score(by_seed, seed_list, cfg, oracle)
        sampled = np.empty(n_draws, dtype=np.float64)
        for i in range(n_draws):
            sampled[i] = _score(by_seed, draws[i], cfg, oracle)
        samples[name] = sampled
    return point, samples


def _interval_from_samples(
    policy: str,
    baseline: str,
    point: Mapping[str, float],
    samples: Mapping[str, np.ndarray],
    n_seeds: int,
) -> PairedInterval:
    diff = samples[policy] - samples[baseline]
    return PairedInterval(
        policy=policy,
        baseline=baseline,
        point_diff=float(point[policy] - point[baseline]),
        mean_diff=float(np.mean(diff)),
        ci_low=float(np.percentile(diff, 2.5, method="linear")),
        ci_high=float(np.percentile(diff, 97.5, method="linear")),
        n_draws=int(diff.shape[0]),
        n_seeds=n_seeds,
    )


def paired_difference(
    by_seed_a: Mapping[int, EpisodeMetrics],
    by_seed_b: Mapping[int, EpisodeMetrics],
    *,
    policy: str = "policy",
    baseline: str = "baseline",
    seeds: Optional[Sequence[int]] = None,
    cfg: Optional[SoftAwareConfig] = None,
    oracle_revenue_by_seed: Optional[Mapping[int, float]] = None,
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
) -> PairedInterval:
    """Paired bootstrap of ``score_aware(policy) - score_aware(baseline)``.

    The same episode map may be passed for both sides. That comparison is
    paired with itself, so every draw has difference 0 and the interval has
    width 0.
    """
    seed_list = _aligned_seeds(by_seed_a, by_seed_b, seeds)
    point, samples = bootstrap_scores(
        {policy: by_seed_a, baseline: by_seed_b},
        seed_list,
        cfg=cfg,
        oracle_revenue_by_seed=oracle_revenue_by_seed,
        n_draws=n_draws,
        rng_seed=rng_seed,
    )
    return _interval_from_samples(policy, baseline, point, samples, len(seed_list))


def all_pair_intervals(
    by_policy: Mapping[str, Mapping[int, EpisodeMetrics]],
    *,
    order: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    cfg: Optional[SoftAwareConfig] = None,
    oracle_revenue_by_seed: Optional[Mapping[int, float]] = None,
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
) -> list[PairedInterval]:
    """Every unordered pair, in ``order``, as ``score(earlier) - score(later)``."""
    names = list(order) if order is not None else list(by_policy)
    unknown = [name for name in names if name not in by_policy]
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if seeds is None:
        seed_sets = [set(by_policy[name]) for name in names]
        seed_list = sorted(set.intersection(*seed_sets)) if seed_sets else []
    else:
        seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("no seeds to resample")
    point, samples = bootstrap_scores(
        {name: by_policy[name] for name in names},
        seed_list,
        cfg=cfg,
        oracle_revenue_by_seed=oracle_revenue_by_seed,
        n_draws=n_draws,
        rng_seed=rng_seed,
    )
    rows: list[PairedInterval] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            rows.append(_interval_from_samples(left, right, point, samples, len(seed_list)))
    return rows


def intervals_vs_baseline(
    by_policy: Mapping[str, Mapping[int, EpisodeMetrics]],
    baseline: str,
    *,
    order: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    cfg: Optional[SoftAwareConfig] = None,
    oracle_revenue_by_seed: Optional[Mapping[int, float]] = None,
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
) -> list[PairedInterval]:
    """Each policy against ``baseline``, including a zero-width row for the baseline itself."""
    if baseline not in by_policy:
        raise ValueError(f"baseline {baseline!r} is not in {sorted(by_policy)}")
    names = list(order) if order is not None else list(by_policy)
    if baseline not in names:
        names.append(baseline)
    if seeds is None:
        seed_list = sorted(set.intersection(*(set(by_policy[name]) for name in names)))
    else:
        seed_list = [int(seed) for seed in seeds]
    point, samples = bootstrap_scores(
        {name: by_policy[name] for name in names},
        seed_list,
        cfg=cfg,
        oracle_revenue_by_seed=oracle_revenue_by_seed,
        n_draws=n_draws,
        rng_seed=rng_seed,
    )
    return [
        _interval_from_samples(name, baseline, point, samples, len(seed_list)) for name in names
    ]


def intervals_frame(rows: Sequence[PairedInterval]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "policy": row.policy,
                "baseline": row.baseline,
                "point_diff": row.point_diff,
                "paired_diff": row.mean_diff,
                "paired_ci_low": row.ci_low,
                "paired_ci_high": row.ci_high,
                "tie": row.covers_zero,
                "n_draws": row.n_draws,
                "n_seeds": row.n_seeds,
            }
        )
    return pd.DataFrame.from_records(records)


def attach_interval_columns(table: pd.DataFrame, rows: Sequence[PairedInterval]) -> pd.DataFrame:
    """Add ``paired_diff``, ``paired_ci_low``, ``paired_ci_high`` matched on ``policy``."""
    by_name = {row.policy: row for row in rows}
    out = table.copy()
    diffs: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for policy in out["policy"]:
        row = by_name[str(policy)]
        diffs.append(round(row.mean_diff, 2))
        lows.append(round(row.ci_low, 2))
        highs.append(round(row.ci_high, 2))
    out["paired_diff"] = diffs
    out["paired_ci_low"] = lows
    out["paired_ci_high"] = highs
    return out


def load_saved_run(
    run_dir: str | Path,
) -> tuple[
    dict[str, dict[int, EpisodeMetrics]], SoftAwareConfig, dict[int, float], list[int], list[str]
]:
    """Episodes, the soft-aware config, oracle revenue, seeds, and policy order.

    Oracle revenue is read from ``soft_aware_summary.json``, where a fresh
    soft-aware eval already stores it next to the episode table.
    """
    directory = Path(run_dir)
    episodes = pd.read_csv(directory / "episode_metrics.csv")
    by_policy = index_by_policy(episodes.to_dict(orient="records"))
    summary = json.loads((directory / "soft_aware_summary.json").read_text())
    cfg = SoftAwareConfig.from_dict(summary.get("soft_cfg"))
    oracle = {
        int(seed): float(revenue) for seed, revenue in summary["oracle_revenue_by_seed"].items()
    }
    seeds = [int(seed) for seed in summary["seeds"]]
    order = list(pd.read_csv(directory / "soft_aware_table.csv")["policy"])
    return by_policy, cfg, oracle, seeds, order


def write_saved_intervals(
    run_dir: str | Path,
    *,
    out_name: str = "paired_intervals.csv",
    n_draws: int = N_DRAWS,
    rng_seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Write pairwise intervals beside a saved soft-aware run. Does not roll policies."""
    directory = Path(run_dir)
    by_policy, cfg, oracle, seeds, order = load_saved_run(directory)
    rows = all_pair_intervals(
        by_policy,
        order=order,
        seeds=seeds,
        cfg=cfg,
        oracle_revenue_by_seed=oracle,
        n_draws=n_draws,
        rng_seed=rng_seed,
    )
    frame = intervals_frame(rows)
    frame.to_csv(directory / out_name, index=False)
    return frame
