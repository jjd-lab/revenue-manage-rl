"""One-sided price constraint: later buyers do not pay less than earlier ones.

Config (under ``control.price_monotone``)::

    enabled: true
    direction: up              # up | down
    mode: clamp                # clamp | penalty | ratchet
    reference: last            # last | high_water
    apply_on: always           # always | peak_only
    tolerance: 0               # dollars of slack that are not a violation
    max_step: null             # cap the per-day move; required by ratchet
    penalty: 10                # penalty mode only; see the wrapper
    apply_when_days_prior_le: null  # if set, only on decision day <= this

``direction: up`` forbids a decrease. The three modes differ in *how*:

``clamp``
    Raise an offending action to the floor. Simple, and it works on a frozen
    checkpoint — but every action below the floor executes at the same price,
    so the environment is flat over that whole region and a learner gets no
    gradient there. Fine for constraining a trained policy, poor for training.
``penalty``
    Leave the action alone and charge the violation to the reward. May violate.
``ratchet``
    Reinterpret the action as a *move* from the reference rather than an
    absolute price, so the feasible set is covered bijectively: ``-1`` holds,
    ``+1`` takes the largest legal step, and no two actions collide. Needs
    ``max_step``. This is the mode to train under.

``reference`` picks what the bounds are measured from. ``last`` uses yesterday's
charged price; ``high_water`` uses the episode's running extreme. They coincide
under ``clamp`` with ``tolerance: 0``, because the price then never moves the
wrong way. They differ everywhere else, and ``high_water`` is the correct one:
under ``last`` a ``tolerance`` of *t* permits a *t*-per-day glide without limit
(the floor re-tracks each lowered price), and a penalty charged against
yesterday makes a slow staircase a series of small fines rather than a bar on
leaving the peak. Under ``high_water`` a ``tolerance`` of *t* means what the
word says: total backslide from the peak is at most *t*.

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
KNOWN_MONOTONE_MODES = {"clamp", "penalty", "ratchet"}
KNOWN_MONOTONE_REFERENCES = {"last", "high_water"}
KNOWN_MONOTONE_SCOPES = {"always", "peak_only"}

# Calendar vocabulary for ``apply_on: peak_only``. These mirror SoftAwareConfig,
# which cannot be imported here (controls -> metrics -> envs -> controls), so
# tests/test_price_monotone.py asserts the two stay equal.
PEAK_MONTHS = (7, 8, 11, 12)
WEEKEND_DOW = (5, 6)

# Bare-block callers may hand a whole control dict. Drop sibling blocks.
_BARE_EXCLUDE = ("price_only", "selling_limit", "early_promo", "promo", "safe_sl", "mpc")


def _clip_price(env: Any, price: float) -> float:
    lo = float(getattr(env, "min_price", 80.0))
    hi = float(getattr(env, "max_price", lo))
    return float(np.clip(price, lo, hi))


def _require_non_negative(name: str, value: Optional[float]) -> None:
    if value is not None and float(value) < 0.0:
        raise ValueError(f"price_monotone.{name} must be >= 0, got {value!r}")


class MonotonePriceControl:
    """Constrain a one-sided price move, by clamp, by penalty, or by ratchet."""

    kind = "price_monotone"

    def __init__(
        self,
        enabled: bool = True,
        direction: str = "up",
        mode: str = "clamp",
        reference: str = "last",
        apply_on: str = "always",
        tolerance: float = 0.0,
        max_step: Optional[float] = None,
        penalty: float = 10.0,
        apply_when_days_prior_le: Optional[float] = None,
        peak_months: tuple[int, ...] = PEAK_MONTHS,
        weekend_dow: tuple[int, ...] = WEEKEND_DOW,
        **_ignored: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.direction = str(direction).lower().strip()
        if self.direction not in KNOWN_MONOTONE_DIRECTIONS:
            raise ValueError(f"price_monotone.direction must be 'up' or 'down', got {direction!r}")
        self.mode = str(mode).lower().strip()
        if self.mode not in KNOWN_MONOTONE_MODES:
            raise ValueError(
                f"price_monotone.mode must be one of {sorted(KNOWN_MONOTONE_MODES)}, got {mode!r}"
            )
        self.reference = str(reference).lower().strip()
        if self.reference not in KNOWN_MONOTONE_REFERENCES:
            raise ValueError(
                f"price_monotone.reference must be one of {sorted(KNOWN_MONOTONE_REFERENCES)}, "
                f"got {reference!r}"
            )
        self.apply_on = str(apply_on).lower().strip()
        if self.apply_on not in KNOWN_MONOTONE_SCOPES:
            raise ValueError(
                f"price_monotone.apply_on must be one of {sorted(KNOWN_MONOTONE_SCOPES)}, "
                f"got {apply_on!r}"
            )
        _require_non_negative("tolerance", tolerance)
        _require_non_negative("max_step", max_step)
        _require_non_negative("penalty", penalty)
        self.tolerance = float(tolerance)
        self.max_step = None if max_step is None else float(max_step)
        self.penalty = float(penalty)
        if self.mode == "ratchet" and self.max_step is None:
            raise ValueError("price_monotone.mode='ratchet' requires max_step (dollars per day)")
        self.apply_when_days_prior_le = (
            None if apply_when_days_prior_le is None else float(apply_when_days_prior_le)
        )
        self.peak_months = tuple(int(m) for m in peak_months)
        self.weekend_dow = tuple(int(d) for d in weekend_dow)
        self.last_executed_price: Optional[float] = None
        self.high_water: Optional[float] = None
        self.last_floor: Optional[float] = None
        self.last_clamped: bool = False
        self.last_violation: float = 0.0
        # Recorded by the ratchet mapping so the wrapper can invert it.
        self.last_reference: Optional[float] = None
        self.last_reach: float = 0.0

    def reset_episode(self) -> None:
        """Drop the previous episode's closing price. The wrapper calls this."""
        self.last_executed_price = None
        self.high_water = None
        self.last_floor = None
        self.last_clamped = False
        self.last_violation = 0.0
        self.last_reference = None
        self.last_reach = 0.0

    def note_executed(self, price: float) -> None:
        """Record the price the env actually charged, after ``step``."""
        price = float(price)
        self.last_executed_price = price
        if self.high_water is None:
            self.high_water = price
        elif self.direction == "up":
            self.high_water = max(self.high_water, price)
        else:
            self.high_water = min(self.high_water, price)

    def _reference_price(self) -> Optional[float]:
        """The price the bounds are measured from."""
        if self.reference == "high_water":
            return self.high_water
        return self.last_executed_price

    def _in_window(self, env: Any) -> bool:
        """Judge the optional window on the decision day, ``days_prior - 1``."""
        if self.apply_when_days_prior_le is None:
            return True
        days_prior = float(getattr(env, "days_prior", 0.0))
        decision_day = days_prior - 1.0
        return decision_day <= float(self.apply_when_days_prior_le)

    def _in_scope(self, env: Any) -> bool:
        """``peak_only`` protects every night but a weekday in an off-peak month.

        Calendar only, deliberately: an operator has to know in advance which
        nights carry the guarantee, and realized base demand is not knowable
        then. This is a looser split than the scorer's soft/peak classification,
        which also reads base demand — they are not the same partition.
        """
        if self.apply_on == "always":
            return True
        month = getattr(env, "month", None)
        dow = getattr(env, "dow", None)
        if month is None or dow is None:
            return True
        weekend = int(dow) in self.weekend_dow
        peak_month = int(month) in self.peak_months
        return bool(weekend or peak_month)

    def _active(self, env: Any) -> bool:
        return self.enabled and self._in_window(env) and self._in_scope(env)

    def _bounds(self, ref: float) -> tuple[Optional[float], Optional[float]]:
        """Return ``(floor, ceiling)`` in dollars. ``None`` means that side is open."""
        tol = self.tolerance
        step = self.max_step
        if self.direction == "up":
            floor = ref - tol
            ceiling = None if step is None else ref + step
            return floor, ceiling
        ceiling = ref + tol
        floor = None if step is None else ref - step
        return floor, ceiling

    def reach(self, env: Any, ref: float) -> float:
        """Dollars the ratchet may move in one day, from ``ref``."""
        assert self.max_step is not None  # guaranteed by __init__
        lo = float(getattr(env, "min_price", 80.0))
        hi = float(getattr(env, "max_price", lo))
        headroom = (hi - ref) if self.direction == "up" else (ref - lo)
        return float(max(0.0, min(self.max_step, headroom)))

    def price_from_action(self, env: Any, a0: float) -> float:
        """``ratchet``: read ``a0`` as a move from the reference, not a price.

        ``-1`` holds, ``+1`` takes the largest legal step. Every action maps to
        a distinct price, which is the whole point of this mode.
        """
        self.last_clamped = False
        self.last_violation = 0.0
        self.last_floor = None
        self.last_reference = None
        self.last_reach = 0.0
        a0 = float(np.clip(a0, -1.0, 1.0))
        ref = self._reference_price()
        if not self._active(env) or ref is None:
            # Exempt first step, or out of window/scope: an ordinary price action.
            lo = float(getattr(env, "min_price", 80.0))
            hi = float(getattr(env, "max_price", lo))
            return float(lo + (a0 + 1.0) * 0.5 * (hi - lo))
        reach = self.reach(env, ref)
        self.last_reference = float(ref)
        self.last_reach = reach
        self.last_floor = float(ref)
        g = (a0 + 1.0) * 0.5
        out = ref + g * reach if self.direction == "up" else ref - g * reach
        return _clip_price(env, out)

    def action_from_price(self, env: Any, price: float) -> float:
        """Inverse of :meth:`price_from_action`, for cloning a demonstration."""
        ref = self.last_reference
        reach = self.last_reach
        if ref is None or reach <= 0.0:
            lo = float(getattr(env, "min_price", 80.0))
            hi = float(getattr(env, "max_price", lo))
            span = max(hi - lo, 1e-6)
            return float(np.clip(2.0 * (float(price) - lo) / span - 1.0, -1.0, 1.0))
        moved = (float(price) - ref) if self.direction == "up" else (ref - float(price))
        g = float(np.clip(moved / reach, 0.0, 1.0))
        return float(np.clip(2.0 * g - 1.0, -1.0, 1.0))

    def apply(self, env: Any, price: float) -> float:
        """``clamp`` / ``penalty``: return the price to execute.

        Penalty mode measures the violation and does not clamp.
        """
        self.last_clamped = False
        self.last_violation = 0.0
        self.last_floor = None
        price = float(price)
        ref = self._reference_price()
        if not self._active(env) or ref is None:
            return price

        floor, ceiling = self._bounds(float(ref))
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
        self.last_clamped = abs(out - price) > 1e-6
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
    "KNOWN_MONOTONE_REFERENCES",
    "KNOWN_MONOTONE_SCOPES",
    "MonotonePriceControl",
    "get_price_monotone",
]
