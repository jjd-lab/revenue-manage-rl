"""Early promo when tree base demand is low (soft-day undersell fix).

Applied as a **post-process on the RL price** inside ``PriceOnlyWrapper`` (after
the agent acts, before SL + env dynamics). Does **not** reopen joint 2D
``(price, SL)`` — selling limit still comes from the analytic/optimize_1d
controller.

Idea
----
If predicted **price-unaware** base demand is below a threshold, and we are
still early/mid horizon (``days_prior >= apply_when_days_prior_ge``), force or
clip price into a promo/low band so soft days stimulate demand *before*
day-30 when catch-up fails.

Config (under ``control.early_promo`` or ``control.promo``)::

    enabled: true
    base_demand_threshold: 90.0   # absolute tree base
    promo_price: 80.0             # used when mode=set
    max_price_when_soft: 85.0     # used when mode=clip (optional)
    mode: set                     # set | clip
    apply_when_days_prior_ge: 30  # only early/mid; late clearing untouched
    weekdays_only: false
    offpeak_only: false
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from reservation_pricing.demand.protocol import features_from_state


class EarlyPromoController:
    """Decide whether to override RL price toward a promo / low band."""

    kind = "early_promo"

    def __init__(
        self,
        enabled: bool = True,
        base_demand_threshold: float = 90.0,
        promo_price: Optional[float] = 80.0,
        max_price_when_soft: Optional[float] = None,
        mode: str = "set",
        apply_when_days_prior_ge: float = 30.0,
        weekdays_only: bool = False,
        offpeak_only: bool = False,
        **_ignored: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_demand_threshold = float(base_demand_threshold)
        self.promo_price = None if promo_price is None else float(promo_price)
        self.max_price_when_soft = (
            None if max_price_when_soft is None else float(max_price_when_soft)
        )
        self.mode = str(mode).lower().strip()
        if self.mode not in ("set", "clip"):
            raise ValueError(f"early_promo.mode must be 'set' or 'clip', got {mode!r}")
        self.apply_when_days_prior_ge = float(apply_when_days_prior_ge)
        self.weekdays_only = bool(weekdays_only)
        self.offpeak_only = bool(offpeak_only)
        # Diagnostics from last should_apply / apply call
        self.last_base: Optional[float] = None
        self.last_triggered: bool = False

    def _calendar_ok(self, env: Any) -> bool:
        dow = int(getattr(env, "dow", 0))
        month = int(getattr(env, "month", 1))
        is_weekend = dow in (5, 6)
        is_peak = month in (7, 8, 11, 12)
        if self.weekdays_only and is_weekend:
            return False
        if self.offpeak_only and is_peak:
            return False
        return True

    def predict_base(self, env: Any) -> float:
        """Price-unaware base demand at current env calendar / days_prior."""
        demand = getattr(env, "demand_model", None)
        if demand is None or not hasattr(demand, "predict_base"):
            return float("inf")  # never trigger without a base predictor
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

    def should_apply(self, env: Any) -> bool:
        """True when soft (low base) + early/mid horizon (+ optional calendar filters)."""
        self.last_triggered = False
        self.last_base = None
        if not self.enabled:
            return False
        days_prior = float(getattr(env, "days_prior", 0))
        if days_prior < self.apply_when_days_prior_ge:
            return False
        if not self._calendar_ok(env):
            return False
        base = self.predict_base(env)
        self.last_base = base
        if base < self.base_demand_threshold:
            self.last_triggered = True
            return True
        return False

    def apply(self, env: Any, price: float) -> float:
        """Return possibly overridden price (clipped to env min/max)."""
        lo = float(getattr(env, "min_price", 80.0))
        hi = float(getattr(env, "max_price", 120.0))
        p = float(price)
        if not self.should_apply(env):
            return float(np.clip(p, lo, hi))

        if self.mode == "clip":
            cap = self.max_price_when_soft
            if cap is None:
                cap = self.promo_price if self.promo_price is not None else lo
            p = min(p, float(cap))
        else:
            # mode == set
            target = self.promo_price if self.promo_price is not None else lo
            if self.max_price_when_soft is not None:
                # allow set+cap: force down but never above cap (same as set to promo)
                target = min(float(target), float(self.max_price_when_soft))
            p = float(target)
        return float(np.clip(p, lo, hi))


def get_early_promo(
    cfg: dict[str, Any] | None = None, **overrides: Any
) -> Optional[EarlyPromoController]:
    """Build EarlyPromoController from ``control.early_promo`` / ``control.promo``.

    Returns ``None`` when disabled / absent so callers can skip the hook.
    Accepts a full ``control`` block, a promo sub-block, or kwargs.
    """
    cfg = dict(cfg or {})
    # Prefer nested early_promo, then promo; else treat cfg itself as the block
    if "early_promo" in cfg and isinstance(cfg["early_promo"], dict):
        block = dict(cfg["early_promo"])
    elif "promo" in cfg and isinstance(cfg["promo"], dict):
        block = dict(cfg["promo"])
    elif "enabled" in cfg or "base_demand_threshold" in cfg or "mode" in cfg:
        block = dict(cfg)
    else:
        return None

    block.pop("price_only", None)
    block.pop("selling_limit", None)
    block.update(overrides)
    if not bool(block.get("enabled", False)):
        return None
    return EarlyPromoController(**block)
