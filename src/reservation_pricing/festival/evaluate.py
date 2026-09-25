"""Score festival policies over held-out seasons, with paired bootstrap intervals."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from reservation_pricing.festival.env import FestivalEnv

PolicyFn = Callable[[np.ndarray, Any, dict], np.ndarray]


def run_season(env: FestivalEnv, policy: PolicyFn, seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    state: dict[str, Any] = {}
    done = False
    sold = np.zeros(env.n_passes)
    price_x_sold = np.zeros(env.n_passes)
    while not done:
        obs, _r, terminated, truncated, info = env.step(policy(obs, env, state))
        sold += info["accepted"]
        price_x_sold += info["accepted"] * info["night_prices"]
        done = terminated or truncated
    row: dict[str, Any] = {
        "seed": seed,
        "score": info["score"],
        "revenue": info["revenue"],
        "unsold": float(info["unsold"].sum()),
        "denied": float(info["denied"].sum()),
        "nights_denied": int((info["denied"] > 0.5).sum()),
    }
    for name, n, px in zip(env.cfg.pass_names(), sold, price_x_sold):
        row[f"sold_{name}"] = float(n)
        row[f"night_price_{name}"] = float(px / n) if n > 0 else float("nan")
    return row


def evaluate(
    make_env: Callable[[], FestivalEnv], policies: Mapping[str, PolicyFn], seeds: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for name, policy in policies.items():
        env = make_env()
        rows += [{"policy": name, **run_season(env, policy, s)} for s in seeds]
    return pd.DataFrame(rows)


def paired_interval(
    per_seed: pd.DataFrame, policy: str, baseline: str, n_draws: int = 10_000, rng_seed: int = 0
) -> dict[str, Any]:
    """95% interval of the mean score difference, resampling seasons with replacement."""
    a = per_seed[per_seed.policy == policy].set_index("seed")["score"]
    b = per_seed[per_seed.policy == baseline].set_index("seed")["score"]
    diff = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(rng_seed)
    draws = diff[rng.integers(0, diff.size, size=(n_draws, diff.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "policy": policy,
        "baseline": baseline,
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_seeds": int(diff.size),
    }


__all__ = ["evaluate", "paired_interval", "run_season"]
