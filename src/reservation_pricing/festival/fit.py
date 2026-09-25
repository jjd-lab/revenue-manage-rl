"""Fit the festival's logit demand from past seasons' booking requests.

History comes from seasons played at random prices: every day, every pass gets
its own per-night price drawn uniformly from the band, and every night's limit
is wide open. That is the best case for estimation -- prices vary a lot and
independently -- so the number of seasons is the only thing held short.

The fit is Poisson maximum likelihood over each (season, day, pass): requests
have mean ``market_size * arrival_weight(days out) * P_k(prices, on sale)``.
It estimates the night appeals, the 2- and 3-day pass appeals (the 1-day appeal
is pinned at 0: shifting every night by ``c`` and every pass by ``-length * c``
leaves every utility unchanged), ``beta_early``, ``beta_late``, the market size
and the arrival decay. Cancellation and no-show rates are taken as known.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from reservation_pricing.festival.env import FestivalEnv
from reservation_pricing.festival.model import FestivalConfig, pass_utility


def random_price_policy(rng: np.random.Generator):
    def _policy(obs: np.ndarray, env: FestivalEnv, state: dict) -> np.ndarray:
        prices = rng.uniform(env.cfg.min_night_price, env.cfg.max_night_price, env.n_passes)
        return env.scale_action(prices, np.full(env.n_nights, env.cfg.max_selling_limit))

    return _policy


def collect_history(
    make_env: Callable[[], FestivalEnv], n_seasons: int, first_seed: int
) -> dict[str, np.ndarray]:
    """Days out, total prices, on-sale flags and requests for every day of ``n_seasons``."""
    env = make_env()
    policy = random_price_policy(np.random.default_rng(first_seed))
    cols: dict[str, list] = {"days_out": [], "prices": [], "on_sale": [], "requests": []}
    for seed in range(first_seed, first_seed + n_seasons):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _r, done, _t, _i = env.step(policy(obs, env, {}))
            cols["days_out"].append(env.days_prior)
            cols["prices"].append(env.prices.copy())
            cols["on_sale"].append(env.on_sale.copy())
            cols["requests"].append(env.requests.copy())
    return {k: np.asarray(v) for k, v in cols.items()}


def _unpack(theta: np.ndarray, n_nights: int) -> dict[str, Any]:
    return {
        "night_appeal": tuple(float(x) for x in theta[:n_nights]),
        "pass_appeal": (0.0, *(float(x) for x in theta[n_nights : n_nights + 2])),
        "beta_early": float(theta[n_nights + 2]),
        "beta_late": float(theta[n_nights + 3]),
        "market_size": float(np.exp(theta[n_nights + 4])),
        "arrival_decay_days": float(np.exp(theta[n_nights + 5])),
    }


def fit_demand(cfg: FestivalConfig, history: dict[str, np.ndarray], n_seasons: int) -> FestivalConfig:
    """``cfg`` with its demand fields replaced by the maximum-likelihood fit."""
    if len(cfg.pass_appeal) != 3 or any(len(p) > 3 for p in cfg.pass_list):
        raise ValueError("fit_demand estimates appeal for passes of 1-3 nights")
    d = history["days_out"].astype(np.float64)
    prices, on_sale, y = history["prices"], history["on_sale"], history["requests"]
    frac = np.clip(d / cfg.horizon, 0.0, 1.0)
    days = np.arange(cfg.horizon)
    n = cfg.n_nights

    def nll(theta: np.ndarray) -> float:
        p = _unpack(theta, n)
        trial = dataclasses.replace(cfg, **p)
        u = pass_utility(trial, p["night_appeal"])
        beta = p["beta_late"] + frac * (p["beta_early"] - p["beta_late"])
        e = np.where(on_sale, np.exp(u[None, :] - beta[:, None] * prices / 100.0), 0.0)
        probs = e / (1.0 + e.sum(axis=1, keepdims=True))
        w = np.exp(-days / p["arrival_decay_days"])
        w /= w.sum()
        mu = p["market_size"] * w[history["days_out"]][:, None] * probs
        mu = np.maximum(mu, 1e-12)
        return float((mu - y * np.log(mu)).sum()) / y.size

    theta0 = np.concatenate(
        [np.zeros(n), np.zeros(2), [2.5, 2.5], [np.log(y.sum() / max(n_seasons, 1) / 0.3)], [np.log(30.0)]]
    )
    res = minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 2000})
    return dataclasses.replace(cfg, **_unpack(res.x, n))


__all__ = ["collect_history", "fit_demand", "random_price_policy"]
