"""Dynamic-programming baseline: the textbook planner with a demand model.

Backward induction over one night. The state is expected show-ups so far: a
booking taken ``d`` days out shows up with a keep rate that depends only on
``d``, so nothing else about the past moves the terminal outcome. The planner
keeps that count itself, from the bookings it has taken times its own keep
rates. It never reads the env's ``cumulative_mat_boh``, which is built from the
true rates and the night's realized no-show draw, neither of which an operator
sees. Each day the planner picks a price and a cap on today's
bookings, as a fraction of expected demand at that price (1.0 means accept all).
Demand noise is integrated with Gauss-Hermite quadrature.

The terminal charge is the score's own: on peak nights ``lambda_peak`` per
unsold seat and ``lambda_peak * OVERSELL_WEIGHT`` per denied admission; on soft
nights only the denied admissions. Soft/peak is classified from the *forecast*
(``decision_model``), the same model the planner prices with, so a wrong forecast
reaches this baseline exactly as it reaches myopic. Keep rates come from
``estimate_keep_rate``, the disclosed access the cap and the analytic limit use
(EXPERIMENT_LOG §11a); ``keep_overrides`` replaces any of the cancellation or
no-show parameters it reads, to plan with a wrong show-up model.

A cap is carried out as ``selling limit = bookings on hand + cap``, clipped to
the env's limit bounds. Below ``min_selling_limit`` the env cannot refuse, so a
cap the planner wants early in the horizon may not be honoured;
``state["dp_unhonoured"]`` counts those days.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from reservation_pricing.baselines.policies import _to_env_action
from reservation_pricing.controls.selling_limit import estimate_keep_rate
from reservation_pricing.demand.protocol import decision_model, features_from_state
from reservation_pricing.envs.reservation import ReservationEnv
from reservation_pricing.metrics import OVERSELL_WEIGHT, PolicyFn, SoftAwareConfig, classify_soft

CAP_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def solve_dp(
    mean_demand: np.ndarray,
    keep: np.ndarray,
    prices: np.ndarray,
    *,
    capacity: float,
    unsold_cost: float,
    oversold_cost: float,
    noise_std: float,
    m_grid: np.ndarray,
    cap_fractions: tuple[float, ...] = CAP_FRACTIONS,
    n_quad: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward induction for one night.

    ``mean_demand[t, j]`` is expected demand on decision ``t`` (0 = first) at
    ``prices[j]``; ``keep[t]`` its keep rate. Returns ``(value, price_idx, cap_idx)``,
    each indexed ``[t, m]`` over ``m_grid``; ``value[0]`` is the expected dollars
    from each starting state.
    """
    horizon, n_prices = mean_demand.shape
    if noise_std > 0:
        z, w = np.polynomial.hermite_e.hermegauss(n_quad)
        w = w / w.sum()
    else:
        z, w = np.zeros(1), np.ones(1)
    fracs = np.asarray(cap_fractions, dtype=np.float64)

    shortfall = np.maximum(capacity - m_grid, 0.0)
    overshoot = np.maximum(m_grid - capacity, 0.0)
    v_next = -(unsold_cost * shortfall + oversold_cost * overshoot)

    value = np.empty((horizon, m_grid.size))
    price_idx = np.empty((horizon, m_grid.size), dtype=np.int64)
    cap_idx = np.empty((horizon, m_grid.size), dtype=np.int64)
    for t in range(horizon - 1, -1, -1):
        mu = mean_demand[t]  # (P,)
        demand = np.maximum(mu[:, None] + noise_std * z[None, :], 0.0)  # (P, Q)
        cap = np.where(fracs[:, None] >= 1.0, np.inf, fracs[:, None] * mu[None, :])  # (F, P)
        accepted = np.minimum(demand[None, :, :], cap[:, :, None])  # (F, P, Q)
        m_after = m_grid[:, None, None, None] + keep[t] * accepted[None]  # (M, F, P, Q)
        future = np.interp(m_after, m_grid, v_next)
        q = (prices[None, None, :, None] * accepted[None] + future) @ w  # (M, F, P)
        flat = q.reshape(m_grid.size, -1)
        best = flat.argmax(axis=1)
        value[t] = flat[np.arange(m_grid.size), best]
        cap_idx[t], price_idx[t] = np.unravel_index(best, (fracs.size, n_prices))
        v_next = value[t]
    return value, price_idx, cap_idx


def _night_inputs(
    env: ReservationEnv, prices: np.ndarray, keep_overrides: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast mean demand and keep rate for each decision, in env step order.

    ``step`` decrements ``days_prior`` before sampling demand, so decision ``t``
    books ``horizon - 1 - t`` days out.
    """
    model = decision_model(env)
    horizon = int(env.booking_horizon)
    mean = np.empty((horizon, prices.size))
    keep = np.empty(horizon)
    shim = SimpleNamespace(**{k: getattr(env, k) for k in _KEEP_ATTRS})
    for k, v in keep_overrides.items():
        setattr(shim, k, v)
    for t in range(horizon):
        days = horizon - 1 - t
        feats = features_from_state(
            days_prior=days, dow=env.dow, month=env.month, booking_horizon=horizon
        )
        mean[t] = [model.predict_mean(feats, float(p)) for p in prices]
        shim.days_prior = days
        keep[t] = estimate_keep_rate(shim)
    return mean, keep


_KEEP_ATTRS = (
    "dow",
    "month",
    "noshow_base",
    "noshow_dow_coef",
    "noshow_month_coef",
    "cancel_rho_weekday",
    "cancel_rho_weekend",
    "cancel_lambda",
)


def _forecast_is_soft(env: ReservationEnv, soft_cfg: SoftAwareConfig) -> bool:
    horizon = int(env.booking_horizon)
    feats = features_from_state(
        days_prior=max(horizon // 2, 1), dow=env.dow, month=env.month, booking_horizon=horizon
    )
    model = decision_model(env)
    base = float(model.predict_base(feats)) if hasattr(model, "predict_base") else None
    return classify_soft(month=env.month, dow=env.dow, base_demand=base, cfg=soft_cfg)


def dp_policy(
    soft_cfg: Optional[SoftAwareConfig] = None,
    *,
    n_prices: int = 41,
    m_step: float = 20.0,
    n_quad: int = 5,
    keep_overrides: Optional[dict[str, float]] = None,
) -> PolicyFn:
    """Price and selling limit from a per-night dynamic program on the forecast.

    Solved once per night type and forecast, and cached;
    ``state["dp_value"]`` is the planner's expected score contribution for the night.
    """
    soft_cfg = soft_cfg or SoftAwareConfig()
    keep_overrides = dict(keep_overrides or {})
    unknown = set(keep_overrides) - set(_KEEP_ATTRS[2:])
    if unknown:
        raise ValueError(f"unknown keep parameters: {sorted(unknown)}")
    cache: dict[Any, tuple] = {}

    def _plan(env: ReservationEnv) -> tuple:
        model = decision_model(env)
        is_soft = _forecast_is_soft(env, soft_cfg)
        # make_env builds a fresh model per night, so key on what the forecast
        # says rather than on the object: level and price response at mid-horizon.
        mid = features_from_state(
            days_prior=int(env.booking_horizon) // 2,
            dow=env.dow,
            month=env.month,
            booking_horizon=int(env.booking_horizon),
        )
        key = (
            env.dow,
            env.month,
            is_soft,
            round(float(model.predict_mean(mid, env.min_price)), 9),
            round(float(model.predict_mean(mid, env.max_price)), 9),
            float(getattr(model, "demand_noise_std", 0.0)),
        )
        if key not in cache:
            prices = np.linspace(env.min_price, env.max_price, n_prices)
            mean, keep = _night_inputs(env, prices, keep_overrides)
            m_grid = np.arange(0.0, env.max_selling_limit + m_step, m_step)
            noise = float(getattr(model, "demand_noise_std", 0.0))
            value, p_idx, c_idx = solve_dp(
                mean,
                keep,
                prices,
                capacity=float(env.capacity),
                unsold_cost=0.0 if is_soft else soft_cfg.lambda_peak,
                oversold_cost=soft_cfg.lambda_peak * OVERSELL_WEIGHT,
                noise_std=noise,
                m_grid=m_grid,
                n_quad=n_quad,
            )
            cache[key] = (prices, mean, keep, m_grid, value, p_idx, c_idx)
        return cache[key]

    def _policy(obs: np.ndarray, env: Any, state: dict) -> np.ndarray:
        u = getattr(env, "unwrapped", env)
        if "dp_plan" not in state:
            state["dp_plan"] = _plan(u)
        prices, mean, keep, m_grid, value, p_idx, c_idx = state["dp_plan"]
        t = int(u.booking_horizon) - int(u.days_prior)
        if t == 0:
            state["dp_m"] = 0.0
            state["dp_value"] = float(value[0, 0])
            state["dp_unhonoured"] = 0
        else:
            state["dp_m"] += float(u.accepted_booking) * float(keep[t - 1])
        m = state["dp_m"]
        i = int(np.clip(np.rint(m / (m_grid[1] - m_grid[0])), 0, m_grid.size - 1))
        j = int(p_idx[t, i])
        frac = CAP_FRACTIONS[int(c_idx[t, i])]
        price = float(prices[j])
        if frac >= 1.0:
            limit = float(u.max_selling_limit)
        else:
            wanted = u._recompute_boh() + frac * float(mean[t, j])
            limit = float(np.clip(wanted, u.min_selling_limit, u.max_selling_limit))
            if limit > wanted:
                state["dp_unhonoured"] = state.get("dp_unhonoured", 0) + 1
            elif frac == 0.0:
                # Selling nothing, every price ties and the solver returns the
                # first. Hold yesterday's instead, so the quoted price means something.
                price = state.get("dp_price", price)
        state["dp_price"] = price
        return _to_env_action(env, price, limit)

    return _policy


__all__ = ["CAP_FRACTIONS", "dp_policy", "solve_dp"]
