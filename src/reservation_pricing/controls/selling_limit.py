"""Selling-limit controllers for price-only RL.

When ``control.price_only: true``, the agent outputs a 1D price; selling limit (SL)
is filled each step by a registered controller before env dynamics.

Kinds
-----
``analytic``
    ``SL = clip(target_mat / keep_rate)`` with
    ``target_mat = capacity * overbook_factor``.
    ``keep_rate`` is either a fixed config value or estimated from the env's
    cancel / no-show structure (deterministic, noise-free). When remaining
    inventory falls below ``low_remain_frac * capacity``, SL is tightened to
    ``min_selling_limit``.

``optimize_1d``
    Grid-search SL in ``[min_selling_limit, max_selling_limit]`` to maximize a
    one-step expected score given today's price and env state::

        mu = demand_model.predict_mean(features, price)
        accepted = min(mu, max(0, SL - current_boh))
        mat_add  = accepted * keep_rate
        overshoot = max(0, cumulative_mat_boh + mat_add - capacity)
        shortfall = max(0, capacity - cumulative_mat_boh - mat_add)
        score = price * accepted
                - oversell_weight * (overshoot / capacity)
                - undersell_weight * (shortfall / capacity)

    Reuses the env demand model when available. Same keep-rate / low-remain
    tightening knobs as ``analytic``.

Registry style mirrors ``demand.registry.get_demand_model``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from reservation_pricing.demand.protocol import decision_model


@runtime_checkable
class SellingLimitController(Protocol):
    """Compute a selling limit given env state and today's price."""

    kind: str

    def compute(self, env: Any, price: float) -> float:
        """Return selling limit in physical units (clipped to env bounds)."""
        ...


def estimate_keep_rate(env: Any) -> float:
    """Deterministic keep-rate from env cancel / no-show structure (no noise)."""
    dow = int(getattr(env, "dow", 0))
    month = int(getattr(env, "month", 1))
    days_prior = float(max(int(getattr(env, "days_prior", 0)), 0))

    dow_effect = 1.5 if dow in (5, 6) else 1.0
    month_effect = 1.5 if month in (7, 8, 11, 12) else 1.0
    noshow = (
        float(getattr(env, "noshow_base", 0.16))
        - float(getattr(env, "noshow_dow_coef", 0.02)) * dow_effect
        - float(getattr(env, "noshow_month_coef", 0.02)) * month_effect
    )
    noshow = float(np.clip(noshow, 0.01, 0.4))

    base_rho = (
        float(getattr(env, "cancel_rho_weekend", 0.5))
        if dow in (5, 6)
        else float(getattr(env, "cancel_rho_weekday", 0.4))
    )
    lam = float(getattr(env, "cancel_lambda", 2000.0))
    if days_prior <= 0:
        cancel_prob = 0.0
    else:
        cancel_prob = 1.0 - float(np.exp(-((days_prior / max(lam, 1e-9)) ** base_rho)))
    cancel_prob = float(np.clip(cancel_prob, 0.0, 0.95))
    keep = (1.0 - cancel_prob) * (1.0 - noshow)
    return float(np.clip(keep, 0.05, 0.99))


def _resolve_keep_rate(env: Any, keep_rate: Optional[float]) -> float:
    if keep_rate is not None:
        return float(np.clip(float(keep_rate), 0.05, 0.99))
    return estimate_keep_rate(env)


def _clip_sl(env: Any, sl: float) -> float:
    lo = float(getattr(env, "min_selling_limit", 0.0))
    hi = float(getattr(env, "max_selling_limit", lo))
    return float(np.clip(sl, lo, hi))


def _maybe_tighten(env: Any, sl: float, low_remain_frac: float) -> float:
    capacity = float(getattr(env, "capacity", 1))
    remain = float(getattr(env, "remain_inv", capacity))
    if remain < float(low_remain_frac) * max(capacity, 1.0):
        return float(getattr(env, "min_selling_limit", sl))
    return sl


class AnalyticSellingLimit:
    """``SL = clip(capacity * overbook_factor / keep_rate)`` with low-remain tighten."""

    kind = "analytic"

    def __init__(
        self,
        overbook_factor: float = 1.05,
        keep_rate: Optional[float] = None,
        low_remain_frac: float = 0.05,
        **_ignored: Any,
    ) -> None:
        self.overbook_factor = float(overbook_factor)
        self.keep_rate = None if keep_rate is None else float(keep_rate)
        self.low_remain_frac = float(low_remain_frac)

    def compute(self, env: Any, price: float) -> float:
        del price  # analytic SL does not depend on today's price
        keep = _resolve_keep_rate(env, self.keep_rate)
        capacity = float(getattr(env, "capacity", 10000))
        target_mat = capacity * self.overbook_factor
        sl = target_mat / max(keep, 1e-6)
        sl = _clip_sl(env, sl)
        return _maybe_tighten(env, sl, self.low_remain_frac)


class Optimize1DSellingLimit:
    """Grid-search SL to maximize one-step expected score (see module docstring)."""

    kind = "optimize_1d"

    def __init__(
        self,
        overbook_factor: float = 1.05,
        keep_rate: Optional[float] = None,
        low_remain_frac: float = 0.05,
        n_grid: int = 41,
        oversell_weight: float = 800.0,
        undersell_weight: float = 200.0,
        **_ignored: Any,
    ) -> None:
        self.overbook_factor = float(overbook_factor)
        self.keep_rate = None if keep_rate is None else float(keep_rate)
        self.low_remain_frac = float(low_remain_frac)
        self.n_grid = int(n_grid)
        self.oversell_weight = float(oversell_weight)
        self.undersell_weight = float(undersell_weight)

    def compute(self, env: Any, price: float) -> float:
        capacity = float(getattr(env, "capacity", 10000))
        remain = float(getattr(env, "remain_inv", capacity))
        if remain < self.low_remain_frac * max(capacity, 1.0):
            return float(getattr(env, "min_selling_limit", 0.0))

        keep = _resolve_keep_rate(env, self.keep_rate)
        boh = float(getattr(env, "current_boh", 0.0))
        cum_mat = float(getattr(env, "cumulative_mat_boh", 0.0))
        lo = float(getattr(env, "min_selling_limit", 0.0))
        hi = float(getattr(env, "max_selling_limit", lo))

        demand = decision_model(env)
        if demand is not None and hasattr(env, "demand_features"):
            mu = float(demand.predict_mean(env.demand_features(), float(price)))
        elif hasattr(env, "expected_gross"):
            mu = float(env.expected_gross(float(price)))
        else:
            mu = 0.0
        mu = max(0.0, mu)

        # Prefer filling toward overbook target; mild bias via undersell vs oversell weights
        best_sl = lo
        best_score = -1e30
        for sl in np.linspace(lo, hi, max(self.n_grid, 2)):
            headroom = max(0.0, float(sl) - boh)
            accepted = min(mu, headroom)
            mat_add = accepted * keep
            # Soft target: treat overbook_factor * capacity as desired total mat
            target = capacity * self.overbook_factor
            overshoot = max(0.0, cum_mat + mat_add - capacity)
            # shortfall vs overbook target (not just capacity) encourages useful buffer
            shortfall = max(0.0, target - (cum_mat + mat_add))
            score = (
                float(price) * accepted
                - self.oversell_weight * (overshoot / max(capacity, 1.0))
                - self.undersell_weight * (shortfall / max(capacity, 1.0))
            )
            if score > best_score:
                best_score = score
                best_sl = float(sl)

        return _clip_sl(env, best_sl)


SLFactory = Callable[..., SellingLimitController]

_REGISTRY: dict[str, SLFactory] = {
    "analytic": AnalyticSellingLimit,
    "optimize_1d": Optimize1DSellingLimit,
}


def register_selling_limit(kind: str, factory: SLFactory) -> None:
    _REGISTRY[kind] = factory


def list_selling_limit_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_selling_limit(
    cfg: dict[str, Any] | None = None, **overrides: Any
) -> SellingLimitController:
    """Build a selling-limit controller from config.

    Accepts either a full ``control`` block (uses ``control.selling_limit`` if
    present), a selling_limit block with ``kind``, or kwargs.
    """
    cfg = dict(cfg or {})
    if "selling_limit" in cfg and isinstance(cfg["selling_limit"], dict):
        sl_cfg = dict(cfg["selling_limit"])
    elif "kind" in cfg:
        sl_cfg = dict(cfg)
    else:
        sl_cfg = {"kind": "analytic"}

    # Drop non-controller keys that may appear on a full control block
    sl_cfg.pop("price_only", None)
    sl_cfg.update(overrides)
    kind = str(sl_cfg.pop("kind", "analytic"))
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown selling_limit kind {kind!r}. Known: {list_selling_limit_kinds()}")
    return _REGISTRY[kind](**sl_cfg)
