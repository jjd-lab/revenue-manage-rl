"""Forecast-and-optimize planner for the festival: a re-solved fluid program.

Each day the planner splits the days left into ``n_buckets`` blocks and picks,
for each block, the share of arriving buyers who take each pass. Under the
logit model that share fixes the price, and expected revenue is concave in the
shares (price bounds are linear in them), so the program is convex::

    max  sum_b  L_b * sum_k price_k(x_b) * x_bk  -  L_b * sum_k max_price_k * r_bk
         - unsold_cost * sum_i u_i - denied_cost * sum_i o_i
    s.t. show-ups_i = expected so far + sum_b L_b keep_b sum_{k on i} (x_bk - r_bk)
         u_i >= capacity - show-ups_i,  o_i >= show-ups_i - capacity,  u, o >= 0
         0 <= r_bk <= x_bk,  prices within the band

``L_b`` is expected arrivals in block ``b``. ``r`` is demand turned away by the
selling limit, charged at the top of the band: turning a buyer away below it
only pays when the price is already at the top, where the charge is exact.
Customers turned away are assumed lost, not moved to another pass, so the plan
never counts on substitution from a closure.

The first block's prices are charged today. Each night's selling limit stops
sales once expected show-ups, at the usual keep rate, reach capacity. The
planner reads only the configured (usual) market and rates and the operator's
show-up count, never the season's draw. ``pickup=True`` rescales the market
size by booking requests seen so far over those expected at the prices charged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize

from reservation_pricing.festival.env import FestivalEnv
from reservation_pricing.festival.model import choice_probs, pass_utility

PICKUP_FROM_DAY = 10


def _blocks(days_left: int, n_buckets: int) -> list[np.ndarray]:
    """Days out still to be booked, split into up to ``n_buckets`` contiguous blocks."""
    days = np.arange(days_left - 1, -1, -1)
    return [b for b in np.array_split(days, min(n_buckets, days_left)) if b.size]


def solve_fluid(env: FestivalEnv, market_mult: float, n_buckets: int) -> dict[str, np.ndarray]:
    """Solve the program from the env's current state; see the module docstring."""
    cfg = env.cfg
    A, n_len = env.A, env.lengths
    K, N = env.n_passes, env.n_nights
    a = pass_utility(cfg, cfg.night_appeal)
    blocks = _blocks(env.days_prior, n_buckets)
    B = len(blocks)
    w = env.arrival_weights
    lam = np.array([cfg.market_size * market_mult * w[b].sum() for b in blocks])
    beta = np.array([np.average([cfg.beta(d) for d in b], weights=w[b]) for b in blocks])
    keep = np.array(
        [np.average([cfg.keep_rate(d, cfg.cancel_rho, cfg.noshow) for d in b], weights=w[b]) for b in blocks]
    )
    hi_price = n_len * cfg.max_night_price
    log_lo = a[None, :] - beta[:, None] * hi_price[None, :] / 100.0  # share floor, top price
    log_hi = a[None, :] - beta[:, None] * n_len[None, :] * cfg.min_night_price / 100.0
    coef = 100.0 * lam / beta  # (B,)
    m0 = env.showups_usual.copy()
    cap = float(cfg.capacity)
    # SLSQP needs comparable scales: dollars over a full house at top price, and
    # empty or oversold seats (u, o) in units of capacity.
    scale = cap * N * cfg.max_night_price

    nx = B * K
    n_var = 2 * nx + 2 * N
    ix = slice(0, nx)
    ir = slice(nx, 2 * nx)
    iu = slice(2 * nx, 2 * nx + N)
    io = slice(2 * nx + N, n_var)

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        x = z[ix].reshape(B, K)
        r = z[ir].reshape(B, K)
        tot = x.sum(axis=1)
        x0 = np.maximum(1.0 - tot, 1e-12)
        lx = np.log(np.maximum(x, 1e-12))
        rev = coef * ((x * a).sum(axis=1) - (x * lx).sum(axis=1) + tot * np.log(x0))
        lost = lam * (r * hi_price).sum(axis=1)
        f = -(rev.sum() - lost.sum()) + cap * (
            cfg.unsold_cost * z[iu].sum() + cfg.denied_cost * z[io].sum()
        )
        g = np.zeros(n_var)
        g[ix] = (-coef[:, None] * (a - lx - 1.0 + np.log(x0)[:, None] - (tot / x0)[:, None])).ravel()
        g[ir] = (lam[:, None] * hi_price[None, :]).ravel()
        g[iu] = cap * cfg.unsold_cost
        g[io] = cap * cfg.denied_cost
        return float(f) / scale, g / scale

    # Linear inequalities G @ z + h >= 0.
    rows, h = [], []
    for b in range(B):
        for k in range(K):
            lo_e, hi_e = np.exp(log_lo[b, k]), np.exp(log_hi[b, k])
            # x_k >= e^lo * x0  ->  x_k + e^lo * sum(x) - e^lo >= 0
            row = np.zeros(n_var)
            row[b * K : (b + 1) * K] = lo_e
            row[b * K + k] += 1.0
            rows.append(row)
            h.append(-lo_e)
            # x_k <= e^hi * x0  ->  e^hi - e^hi * sum(x) - x_k >= 0
            row = np.zeros(n_var)
            row[b * K : (b + 1) * K] = -hi_e
            row[b * K + k] -= 1.0
            rows.append(row)
            h.append(hi_e)
            # r <= x
            row = np.zeros(n_var)
            row[b * K + k] = 1.0
            row[nx + b * K + k] = -1.0
            rows.append(row)
            h.append(0.0)
    # show-ups S = m0 + M @ (x - r), with M[i, bK+k] = lam_b keep_b A[i, k]
    M = np.concatenate([lam[b] * keep[b] * A for b in range(B)], axis=1)  # (N, nx)
    for i in range(N):
        row = np.zeros(n_var)  # u_i - 1 + S_i / cap >= 0
        row[ix], row[ir] = M[i] / cap, -M[i] / cap
        row[2 * nx + i] = 1.0
        rows.append(row)
        h.append(m0[i] / cap - 1.0)
        row = np.zeros(n_var)  # o_i + 1 - S_i / cap >= 0
        row[ix], row[ir] = -M[i] / cap, M[i] / cap
        row[2 * nx + N + i] = 1.0
        rows.append(row)
        h.append(1.0 - m0[i] / cap)
    G, hv = np.array(rows), np.array(h)

    mid = np.exp(0.5 * (log_lo + log_hi))
    x_start = mid / (1.0 + mid.sum(axis=1, keepdims=True))
    z0 = np.zeros(n_var)
    z0[ix] = x_start.ravel()
    S0 = m0 + M @ z0[ix]
    z0[iu] = np.maximum(1.0 - S0 / cap, 0.0)
    z0[io] = np.maximum(S0 / cap - 1.0, 0.0)
    bounds = [(1e-9, 1.0)] * nx + [(0.0, 1.0)] * nx + [(0.0, None)] * (2 * N)
    res = minimize(
        objective,
        z0,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": lambda z: G @ z + hv, "jac": lambda z: G}],
        options={"maxiter": 300, "ftol": 1e-9},
    )
    x = np.maximum(res.x[ix].reshape(B, K), 1e-12)
    x0 = np.maximum(1.0 - x.sum(axis=1, keepdims=True), 1e-12)
    prices = 100.0 * (a[None, :] - np.log(x / x0)) / beta[:, None]
    night_prices = np.clip(prices / n_len[None, :], cfg.min_night_price, cfg.max_night_price)
    return {
        "night_prices": night_prices,
        "showups": m0 + M @ (x - res.x[ir].reshape(B, K)).ravel(),
        "value": -float(res.fun) * scale + float(env.revenue),
        "success": bool(res.success),
    }


def stop_at_capacity(env: FestivalEnv) -> np.ndarray:
    """Selling limits that stop each night once expected show-ups reach capacity."""
    cfg = env.cfg
    keep = cfg.keep_rate(env.days_prior - 1, cfg.cancel_rho, cfg.noshow)
    gap = np.maximum(cfg.capacity - env.showups_usual, 0.0) / max(keep, 1e-6)
    return np.clip(env.bookings_on_hand() + gap, cfg.min_selling_limit, cfg.max_selling_limit)


def planner_policy(
    *, n_buckets: int = 10, resolve: bool = True, pickup: bool = False
):
    """``resolve=False, n_buckets=1`` is the fixed-price baseline: one price per pass all season."""

    def _policy(obs: np.ndarray, env: Any, state: dict) -> np.ndarray:
        u: FestivalEnv = getattr(env, "unwrapped", env)
        cfg = u.cfg
        t = cfg.horizon - u.days_prior
        if t > 0 and pickup:
            usual = pass_utility(cfg, cfg.night_appeal)
            probs = choice_probs(usual, state["prices"], cfg.beta(u.days_prior), u.on_sale)
            state["seen"] = state.get("seen", 0.0) + float(u.requests.sum())
            state["expected"] = state.get("expected", 0.0) + float(
                cfg.market_size * u.arrival_weights[u.days_prior] * probs.sum()
            )
        mult = 1.0
        if pickup and t >= PICKUP_FROM_DAY and state.get("expected", 0.0) > 0:
            mult = float(np.clip(state["seen"] / state["expected"], 0.5, 2.0))
        if resolve or "plan" not in state:
            state["plan"] = solve_fluid(u, mult, n_buckets)
            if t == 0:
                state["value"] = state["plan"]["value"]
        night_prices = state["plan"]["night_prices"][0]
        state["prices"] = night_prices * u.lengths
        return u.scale_action(night_prices, stop_at_capacity(u))

    return _policy


__all__ = ["planner_policy", "solve_fluid", "stop_at_capacity"]
