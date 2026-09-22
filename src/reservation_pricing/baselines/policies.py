"""Classical pricing / booking-limit baselines (no learning).

Myopic pricing always uses ``env.demand_model.predict_mean(features, price)`` on a
1D price grid (or closed-form for linear_legacy when available). This keeps the
baseline model-aware but honest for non-linear tree base demand — production RM
would call the same pricing demand API.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from reservation_pricing.envs.reservation import ReservationEnv
from reservation_pricing.metrics import PolicyFn, aggregate, run_episode


def _to_env_action(env: ReservationEnv, price: float, selling_limit: float) -> np.ndarray:
    """Map physical price/SL to env action space (joint 2D or price-only 1D)."""
    low = env.action_space.low
    high = env.action_space.high
    shape = tuple(getattr(env.action_space, "shape", ()) or ())
    price_only = shape == (1,) or int(np.prod(shape)) == 1
    normalized = (
        float(low.reshape(-1)[0]) <= -0.9
        and float(high.reshape(-1)[0]) >= 0.9
        and float(high.reshape(-1)[0]) <= 1.0 + 1e-5
    )
    if normalized:
        p = 2.0 * (price - env.min_price) / max(env.max_price - env.min_price, 1e-6) - 1.0
        if price_only:
            # SL filled by PriceOnlyWrapper / selling-limit controller
            return np.array([p], dtype=np.float32)
        s = (
            2.0
            * (selling_limit - env.min_selling_limit)
            / max(env.max_selling_limit - env.min_selling_limit, 1e-6)
            - 1.0
        )
        return np.array([p, s], dtype=np.float32)
    if price_only:
        return np.array([price], dtype=np.float32)
    return np.array([price, selling_limit], dtype=np.float32)


def fixed_price_policy(price: float = 100.0, selling_limit: float | None = None) -> PolicyFn:
    def _policy(obs: np.ndarray, env: ReservationEnv, state: dict) -> np.ndarray:
        sl = env.max_selling_limit if selling_limit is None else selling_limit
        return _to_env_action(env, price, sl)

    return _policy


def _myopic_price_grid(env: ReservationEnv, n_grid: int = 41) -> float:
    """Maximize price * E[gross | features, price] on a 1D grid via demand API."""
    features = env.demand_features()
    prices = np.linspace(env.min_price, env.max_price, n_grid)
    best_p = float(env.min_price)
    best_rev = -1.0
    for p in prices:
        mean_g = float(env.demand_model.predict_mean(features, float(p)))
        rev = float(p) * max(0.0, mean_g)
        if rev > best_rev:
            best_rev = rev
            best_p = float(p)
    return best_p


def _myopic_price_linear_closed_form(env: ReservationEnv) -> float | None:
    """Closed-form only when demand is linear_legacy with known coefs; else None."""
    dm = env.demand_model
    if getattr(dm, "kind", None) != "linear_legacy":
        return None
    # demand = base + price_coef * price; revenue = p * max(0, base + c p)
    # Unconstrained critical point: base + 2 c p = 0 => p = -base / (2 c) if c < 0
    features = env.demand_features()
    # Reconstruct base (price-free part) by evaluating at price=0 conceptually
    # mean(p) = intercept + c*p + boosts + days term
    c = float(getattr(dm, "price_coef", -0.2))
    if c >= 0:
        return None
    mean_at_ref = dm.predict_mean(features, 0.0)
    # mean(0) = base; p* = -base / (2c)
    p_star = -mean_at_ref / (2.0 * c)
    return float(np.clip(p_star, env.min_price, env.max_price))


# A fixed overbook target over a fixed keep rate, deliberately not the analytic
# controller: a classical baseline should not borrow the control layer it is being
# compared against. These values do **not** move any published table — the limit
# they produce (12,353) sits above the most bookings this policy ever holds
# (11,402), so it never binds, and runs/price_only_long/ scores myopic identically
# under this limit and under the controller's 15,000. What they do fix is the
# behaviour-cloning target: 99% of the BC dataset's limit column is exactly this
# value, so changing them means `rprl-bc-sac` no longer reproduces the shipped
# BC→SAC checkpoint.
MYOPIC_OVERBOOK_FACTOR = 1.05
MYOPIC_KEEP_RATE = 0.85
MYOPIC_LOW_REMAIN_FRAC = 0.05


def myopic_greedy_policy(n_grid: int = 41, prefer_closed_form: bool = True) -> PolicyFn:
    """Myopic: maximize expected immediate revenue under the demand model's mean.

    - ``linear_legacy``: optional closed-form (same as the prototype), else grid
    - ``tree_elastic``: always 1D grid over ``predict_mean`` (no free closed form)

    The selling limit is a fixed overbook target over a fixed keep rate; see the
    module constants for why those are not configurable.
    """

    def _policy(obs: np.ndarray, env: ReservationEnv, state: dict) -> np.ndarray:
        price = None
        if prefer_closed_form:
            price = _myopic_price_linear_closed_form(env)
        if price is None:
            price = _myopic_price_grid(env, n_grid=n_grid)

        target_mat = env.capacity * MYOPIC_OVERBOOK_FACTOR
        sl = float(
            np.clip(target_mat / MYOPIC_KEEP_RATE, env.min_selling_limit, env.max_selling_limit)
        )
        if env.remain_inv < MYOPIC_LOW_REMAIN_FRAC * env.capacity:
            sl = env.min_selling_limit
        return _to_env_action(env, price, sl)

    return _policy


def heuristic_booking_limit_policy() -> PolicyFn:
    def _policy(obs: np.ndarray, env: ReservationEnv, state: dict) -> np.ndarray:
        used_frac = 1.0 - (env.remain_inv / max(env.capacity, 1))
        used_frac = float(np.clip(used_frac, 0.0, 1.5))
        time_frac = 1.0 - (env.days_prior / max(env.booking_horizon, 1))

        peak = env.dow in (5, 6) or env.month in (7, 8, 11, 12)
        base = 105.0 if peak else 90.0

        price = base + 20.0 * used_frac + 10.0 * time_frac
        price = float(np.clip(price, env.min_price, env.max_price))

        sl = env.max_selling_limit - (env.max_selling_limit - env.min_selling_limit) * (
            0.6 * used_frac + 0.3 * time_frac
        )
        if used_frac > 0.95:
            sl = env.min_selling_limit
        sl = float(np.clip(sl, env.min_selling_limit, env.max_selling_limit))
        return _to_env_action(env, price, sl)

    return _policy


BASELINE_FACTORY: dict[str, Callable[[], PolicyFn]] = {
    "fixed_price_100": lambda: fixed_price_policy(100.0, selling_limit=12000.0),
    "fixed_price_80": lambda: fixed_price_policy(80.0, selling_limit=12000.0),
    "fixed_price_120": lambda: fixed_price_policy(120.0, selling_limit=12000.0),
    "myopic_greedy": myopic_greedy_policy,
    "heuristic_booking_limit": heuristic_booking_limit_policy,
}


def evaluate_baselines(
    env_factory: Callable[[], ReservationEnv],
    n_episodes: int = 30,
    seeds: list[int] | None = None,
    names: list[str] | None = None,
    shortfall_weight: float = 50.0,
) -> dict[str, Any]:
    seeds = seeds or list(range(n_episodes))
    names = names or list(BASELINE_FACTORY.keys())
    results = {}
    episode_rows = []

    for name in names:
        policy = BASELINE_FACTORY[name]()
        episodes = []
        for seed in seeds[:n_episodes]:
            env = env_factory()
            ep = run_episode(env, policy, seed=int(seed))
            episodes.append(ep)
            row = ep.to_dict()
            row["policy"] = name
            row["seed"] = int(seed)
            episode_rows.append(row)
        results[name] = aggregate(episodes, shortfall_weight=shortfall_weight)

    return {"aggregates": results, "episodes": episode_rows}
