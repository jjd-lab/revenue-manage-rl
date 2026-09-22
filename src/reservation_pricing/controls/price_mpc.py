"""Short-horizon price MPC for soft / behind-pace states.

When predicted tree base is soft **or** load is behind a linear booking-curve
pace, replace the RL price with a **grid search** over prices for a short
horizon (3–7 steps) using ``demand.predict_mean`` + simple remain / score
dynamics. Otherwise leave the base policy price unchanged.

Config (under ``control.mpc``)::

    enabled: true
    horizon: 5
    n_grid: 21
    base_demand_threshold: 90.0
    pace_target_final: 0.95
    behind_pace_gap: 0.05      # trigger when fill < target - gap
    oversell_weight: 800.0
    undersell_weight: 200.0
    revenue_scale: 1.0         # physical $ in lookahead (not env shaped)
    keep_rate: null
    weekdays_only: false
    soft_only: true            # if true, ignore behind-pace-only triggers
    prefer_min_when_soft: true # soft days: pick lowest price among near-best scores
    apply_always: false        # if true, MPC every step (debug)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from reservation_pricing.controls.selling_limit import _resolve_keep_rate
from reservation_pricing.demand.protocol import features_from_state


class ShortHorizonPriceMPC:
    """Open-loop price grid MPC; returns first-step price when triggered."""

    kind = "price_mpc"

    def __init__(
        self,
        enabled: bool = True,
        horizon: int = 5,
        n_grid: int = 21,
        base_demand_threshold: float = 90.0,
        pace_target_final: float = 0.95,
        behind_pace_gap: float = 0.05,
        oversell_weight: float = 800.0,
        undersell_weight: float = 200.0,
        revenue_scale: float = 1.0,
        keep_rate: Optional[float] = None,
        weekdays_only: bool = False,
        soft_only: bool = True,
        prefer_min_when_soft: bool = True,
        apply_always: bool = False,
        min_days_prior: float = 0.0,
        **_ignored: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.horizon = int(max(1, horizon))
        self.n_grid = int(max(2, n_grid))
        self.base_demand_threshold = float(base_demand_threshold)
        self.pace_target_final = float(pace_target_final)
        self.behind_pace_gap = float(behind_pace_gap)
        self.oversell_weight = float(oversell_weight)
        self.undersell_weight = float(undersell_weight)
        self.revenue_scale = float(revenue_scale)
        self.keep_rate = None if keep_rate is None else float(keep_rate)
        self.weekdays_only = bool(weekdays_only)
        self.soft_only = bool(soft_only)
        self.prefer_min_when_soft = bool(prefer_min_when_soft)
        self.apply_always = bool(apply_always)
        self.min_days_prior = float(min_days_prior)
        self.last_triggered: bool = False
        self.last_reason: Optional[str] = None
        self.last_base: Optional[float] = None
        self.last_pace_gap: Optional[float] = None
        self.last_best_price: Optional[float] = None

    def predict_base(self, env: Any) -> float:
        demand = getattr(env, "demand_model", None)
        if demand is None or not hasattr(demand, "predict_base"):
            return float("inf")
        if hasattr(env, "demand_features"):
            feats: Mapping[str, Any] = env.demand_features()
        else:
            feats = features_from_state(
                days_prior=int(getattr(env, "days_prior", 0)),
                dow=int(getattr(env, "dow", 0)),
                month=int(getattr(env, "month", 1)),
                booking_horizon=int(getattr(env, "booking_horizon", 100)),
            )
        return float(demand.predict_base(feats))

    def _pace_gap(self, env: Any) -> float:
        capacity = float(getattr(env, "capacity", 10000))
        H = max(int(getattr(env, "booking_horizon", 100)), 1)
        days_prior = int(getattr(env, "days_prior", H))
        elapsed = float(H - days_prior)
        t = float(np.clip(elapsed / float(H), 0.0, 1.0))
        target = self.pace_target_final * t
        fill = float(getattr(env, "cumulative_mat_boh", 0.0)) / max(capacity, 1.0)
        return float(max(0.0, target - fill))

    def should_apply(self, env: Any) -> bool:
        self.last_triggered = False
        self.last_reason = None
        self.last_base = None
        self.last_pace_gap = None
        if not self.enabled:
            return False
        days_prior = float(getattr(env, "days_prior", 0))
        if days_prior < self.min_days_prior:
            return False
        if self.weekdays_only and int(getattr(env, "dow", 0)) in (5, 6):
            return False
        if self.apply_always:
            self.last_triggered = True
            self.last_reason = "always"
            return True

        base = self.predict_base(env)
        self.last_base = base
        soft = base < self.base_demand_threshold

        gap = self._pace_gap(env)
        self.last_pace_gap = gap
        behind = gap > self.behind_pace_gap

        if self.soft_only:
            if not soft:
                return False
            self.last_reason = "soft+behind" if behind else "soft"
            self.last_triggered = True
            return True

        if soft and behind:
            self.last_reason = "soft+behind"
        elif soft:
            self.last_reason = "soft"
        elif behind:
            self.last_reason = "behind"
        else:
            return False
        self.last_triggered = True
        return True

    def _simulate_score(self, env: Any, price: float, keep: float) -> float:
        """Deterministic H-step lookahead with constant price + open SL.

        ``env.days_prior`` is the decision day (the wrapper has already stepped
        it back by one), so the first simulated day is that day itself.
        """
        demand = getattr(env, "demand_model", None)
        if demand is None:
            return -1e30
        capacity = float(getattr(env, "capacity", 10000))
        boh = float(getattr(env, "current_boh", 0.0))
        cum_mat = float(getattr(env, "cumulative_mat_boh", 0.0))
        days_prior = int(getattr(env, "days_prior", 0))
        H = int(getattr(env, "booking_horizon", 100))
        dow = int(getattr(env, "dow", 0))
        month = int(getattr(env, "month", 1))
        sl_hi = float(getattr(env, "max_selling_limit", capacity))

        rev = 0.0
        dp = days_prior
        for _ in range(self.horizon):
            if dp < 0:
                break
            feats = features_from_state(
                days_prior=dp,
                dow=dow,
                month=month,
                booking_horizon=H,
            )
            mu = float(max(0.0, demand.predict_mean(feats, float(price))))
            headroom = max(0.0, sl_hi - boh)
            accepted = min(mu, headroom)
            mat_add = accepted * keep
            boh = boh + accepted  # crude: ignore mid-horizon cancel for speed
            # Soft cancel: shrink boh slightly toward keep of total
            boh = min(boh, cum_mat / max(keep, 1e-6) + accepted)
            cum_mat = cum_mat + mat_add
            rev += float(price) * accepted * self.revenue_scale
            dp -= 1

        overshoot = max(0.0, cum_mat - capacity)
        shortfall = max(0.0, capacity - cum_mat)
        # Terminal-ish score over the days left after the last simulated one
        remaining_frac = max(dp + 1, 0) / max(H, 1)
        # Prefer fill now on soft days: weight shortfall by how much horizon left
        score = (
            rev
            - self.oversell_weight * (overshoot / max(capacity, 1.0))
            - self.undersell_weight * (shortfall / max(capacity, 1.0)) * (0.5 + remaining_frac)
        )
        return float(score)

    def optimize_price(self, env: Any) -> float:
        lo = float(getattr(env, "min_price", 80.0))
        hi = float(getattr(env, "max_price", 120.0))
        keep = _resolve_keep_rate(env, self.keep_rate)
        scores = []
        for p in np.linspace(lo, hi, self.n_grid):
            s = self._simulate_score(env, float(p), keep)
            scores.append((float(s), float(p)))
        best_s = max(s for s, _ in scores)
        # Among near-best, prefer lower price on soft days to maximize fill
        soft = self.last_base is not None and self.last_base < self.base_demand_threshold
        if self.prefer_min_when_soft and soft:
            tol = max(1.0, abs(best_s) * 0.02)
            candidates = [p for s, p in scores if s >= best_s - tol]
            best_p = float(min(candidates)) if candidates else lo
        else:
            best_p = float(max(scores, key=lambda t: t[0])[1])
        self.last_best_price = best_p
        return best_p

    def apply(self, env: Any, price: float) -> float:
        """Possibly replace ``price`` with MPC optimum."""
        lo = float(getattr(env, "min_price", 80.0))
        hi = float(getattr(env, "max_price", 120.0))
        p = float(np.clip(float(price), lo, hi))
        if not self.should_apply(env):
            return p
        return float(np.clip(self.optimize_price(env), lo, hi))


def get_price_mpc(
    cfg: dict[str, Any] | None = None, **overrides: Any
) -> Optional[ShortHorizonPriceMPC]:
    """Build MPC from ``control.mpc``; ``None`` if disabled/absent."""
    cfg = dict(cfg or {})
    if "mpc" in cfg and isinstance(cfg["mpc"], dict):
        block = dict(cfg["mpc"])
    elif "horizon" in cfg or ("enabled" in cfg and "n_grid" in cfg):
        block = dict(cfg)
    else:
        return None
    for drop in ("price_only", "selling_limit", "early_promo", "promo", "safe_sl"):
        block.pop(drop, None)
    block.update(overrides)
    if not bool(block.get("enabled", False)):
        return None
    return ShortHorizonPriceMPC(**block)


__all__ = ["ShortHorizonPriceMPC", "get_price_mpc"]
