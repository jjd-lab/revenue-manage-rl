"""One-sided price constraint: later buyers do not pay less than earlier ones.

Config (under ``control.price_monotone``)::

    enabled: true
    direction: up              # up | down
    mode: project              # project | penalty
    tolerance: 0               # dollars of slack that are not a violation
    max_step: null             # if set, also cap the change at last ± max_step
    penalty: 10                # penalty mode only; see the wrapper
    apply_when_days_prior_le: null  # if set, only on decision day <= this

``direction: up`` forbids a decrease. ``mode: project`` clamps the action.
``mode: penalty`` leaves the action alone; ``MonotonePriceEnv`` subtracts
``penalty * (violation_dollars / price_span)`` from the step reward.

The first step of an episode is exempt. ``ReservationEnv.reset()`` sets
``price`` to ``min_price`` as a placeholder, not a decision, and the band floor
is already that placeholder — using it would pin ``direction: down`` at $80
and would leak the previous episode's close into the next opener.
``last_executed_price`` stays ``None`` until the wrapper records ``info["price"]``
after a real step, and the wrapper clears it on every ``reset()``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

KNOWN_MONOTONE_DIRECTIONS = {"up", "down"}
KNOWN_MONOTONE_MODES = {"project", "penalty"}

# Bare-block callers may hand a whole control dict. Drop sibling blocks.
_BARE_EXCLUDE = ("price_only", "selling_limit", "early_promo", "promo", "safe_sl", "mpc")


def _clip_price(env: Any, price: float) -> float:
    lo = float(getattr(env, "min_price", 80.0))
    hi = float(getattr(env, "max_price", lo))
    return float(np.clip(price, lo, hi))


class MonotonePriceControl:
    """Project or measure a one-sided move away from the last charged price."""

    kind = "price_monotone"

    def __init__(
        self,
        enabled: bool = True,
        direction: str = "up",
        mode: str = "project",
        tolerance: float = 0.0,
        max_step: Optional[float] = None,
        penalty: float = 10.0,
        apply_when_days_prior_le: Optional[float] = None,
        **_ignored: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.direction = str(direction).lower().strip()
        if self.direction not in KNOWN_MONOTONE_DIRECTIONS:
            raise ValueError(f"price_monotone.direction must be 'up' or 'down', got {direction!r}")
        self.mode = str(mode).lower().strip()
        if self.mode not in KNOWN_MONOTONE_MODES:
            raise ValueError(f"price_monotone.mode must be 'project' or 'penalty', got {mode!r}")
        self.tolerance = float(tolerance)
        self.max_step = None if max_step is None else float(max_step)
        self.penalty = float(penalty)
        self.apply_when_days_prior_le = (
            None if apply_when_days_prior_le is None else float(apply_when_days_prior_le)
        )
        self.last_executed_price: Optional[float] = None
        self.last_floor: Optional[float] = None
        self.last_projected: bool = False
        self.last_violation: float = 0.0

    def reset_episode(self) -> None:
        """Drop the previous episode's closing price. The wrapper calls this."""
        self.last_executed_price = None
        self.last_floor = None
        self.last_projected = False
        self.last_violation = 0.0

    def note_executed(self, price: float) -> None:
        """Record the price the env actually charged, after ``step``."""
        self.last_executed_price = float(price)

    def _in_window(self, env: Any) -> bool:
        """Judge the optional window on the decision day, ``days_prior - 1``."""
        if self.apply_when_days_prior_le is None:
            return True
        days_prior = float(getattr(env, "days_prior", 0.0))
        decision_day = days_prior - 1.0
        return decision_day <= float(self.apply_when_days_prior_le)

    def _bounds(self, last: float) -> tuple[Optional[float], Optional[float]]:
        """Return ``(floor, ceiling)`` in dollars. ``None`` means that side is open."""
        tol = self.tolerance
        step = self.max_step
        if self.direction == "up":
            floor = last - tol
            ceiling = None if step is None else last + step
            return floor, ceiling
        ceiling = last + tol
        floor = None if step is None else last - step
        return floor, ceiling

    def apply(self, env: Any, price: float) -> float:
        """Return the price to execute. Penalty mode does not clamp."""
        self.last_projected = False
        self.last_violation = 0.0
        self.last_floor = None
        price = float(price)
        if not self.enabled or self.last_executed_price is None or not self._in_window(env):
            return price

        floor, ceiling = self._bounds(float(self.last_executed_price))
        # The named floor in info is the one-sided legal bound for this direction.
        self.last_floor = floor if self.direction == "up" else ceiling

        violation = 0.0
        if floor is not None:
            violation = max(violation, floor - price)
        if ceiling is not None:
            violation = max(violation, price - ceiling)
        self.last_violation = float(violation)

        if self.mode == "penalty":
            return price

        out = price
        if floor is not None and out < floor:
            out = floor
        if ceiling is not None and out > ceiling:
            out = ceiling
        out = _clip_price(env, out)
        self.last_projected = abs(out - price) > 1e-6
        return out


def get_price_monotone(
    cfg: dict[str, Any] | None = None, **overrides: Any
) -> Optional[MonotonePriceControl]:
    """Build from ``control.price_monotone``. ``None`` when absent or disabled."""
    cfg = dict(cfg or {})
    if "price_monotone" in cfg and isinstance(cfg["price_monotone"], dict):
        block = dict(cfg["price_monotone"])
    elif "direction" in cfg:
        block = {k: v for k, v in cfg.items() if k not in _BARE_EXCLUDE}
    else:
        return None
    block.update(overrides)
    if not bool(block.get("enabled", False)):
        return None
    return MonotonePriceControl(**block)


__all__ = [
    "KNOWN_MONOTONE_DIRECTIONS",
    "KNOWN_MONOTONE_MODES",
    "MonotonePriceControl",
    "get_price_monotone",
]
