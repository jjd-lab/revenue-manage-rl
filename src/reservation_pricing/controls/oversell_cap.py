"""Oversell cap for joint (price, selling-limit) policies.

After a joint policy outputs ``(price, SL)``, project SL **down** so expected
materialized show-ups stay within ``capacity * overbook_factor`` (+ optional
buffer). This is the control behind the ``oversell_rate`` column in every
``runs/`` table. Reuses keep-rate / low-remain ideas from
``controls/selling_limit.py``.

The config block and the ``info[...]`` keys are still spelled ``safe_sl``:
shipped configs and the reproduce scripts under ``runs/`` build that block by
name, so renaming it would break published experiments for no gain.

Config (under ``control.safe_sl``)::

    enabled: true
    kind: analytic          # analytic | chance
    overbook_factor: 1.02    # 1.0 = no expected oversell buffer
    keep_rate: null         # omit → estimate from cancel/noshow
    low_remain_frac: 0.05
    buffer_mat: 0.0         # extra mat units allowed beyond overbook target
    activate_remain_frac: null  # if set, only project when remain/C below this
    activate_load_frac: null    # if set, only project when cum_mat/C above this
    mix_alpha: 0.0             # 0=full project to cap; 1=keep policy SL

Does **not** retrain — wrap a frozen joint policy via ``OversellGuardEnv``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from reservation_pricing.controls.selling_limit import (
    _clip_sl,
    _maybe_tighten,
    _resolve_keep_rate,
)


def analytic_oversell_cap(
    env: Any,
    *,
    overbook_factor: float = 1.02,
    keep_rate: Optional[float] = None,
    low_remain_frac: float = 0.05,
    buffer_mat: float = 0.0,
) -> float:
    """Absolute analytic cap: ``capacity * overbook_factor / keep`` (low-remain tighten)."""
    keep = _resolve_keep_rate(env, keep_rate)
    capacity = float(getattr(env, "capacity", 10000))
    target_mat = capacity * float(overbook_factor) + float(buffer_mat)
    sl = target_mat / max(keep, 1e-6)
    sl = _clip_sl(env, sl)
    return _maybe_tighten(env, sl, low_remain_frac)


def chance_oversell_cap(
    env: Any,
    *,
    overbook_factor: float = 1.02,
    keep_rate: Optional[float] = None,
    low_remain_frac: float = 0.05,
    buffer_mat: float = 0.0,
) -> float:
    """Remain-aware cap: only accept bookings that can still materialize under target.

    ``mat_headroom = capacity * overbook_factor + buffer - cumulative_mat_boh``
    ``safe_sl = current_boh + mat_headroom / keep``
    Also clipped by the absolute analytic cap and low-remain tighten.
    """
    keep = _resolve_keep_rate(env, keep_rate)
    capacity = float(getattr(env, "capacity", 10000))
    remain = float(getattr(env, "remain_inv", capacity))
    if remain < float(low_remain_frac) * max(capacity, 1.0):
        return float(getattr(env, "min_selling_limit", 0.0))

    cum_mat = float(getattr(env, "cumulative_mat_boh", 0.0))
    boh = float(getattr(env, "current_boh", 0.0))
    target = capacity * float(overbook_factor) + float(buffer_mat)
    mat_headroom = max(0.0, target - cum_mat)
    book_headroom = mat_headroom / max(keep, 1e-6)
    sl_remain = boh + book_headroom
    sl_abs = analytic_oversell_cap(
        env,
        overbook_factor=overbook_factor,
        keep_rate=keep_rate,
        low_remain_frac=low_remain_frac,
        buffer_mat=buffer_mat,
    )
    return _clip_sl(env, min(sl_remain, sl_abs))


class OversellCap:
    """Project a policy SL down to a safe cap (never raises SL)."""

    kind = "safe_sl"

    def __init__(
        self,
        enabled: bool = True,
        kind: str = "analytic",
        overbook_factor: float = 1.02,
        keep_rate: Optional[float] = None,
        low_remain_frac: float = 0.05,
        buffer_mat: float = 0.0,
        activate_remain_frac: Optional[float] = None,
        activate_load_frac: Optional[float] = None,
        mix_alpha: float = 0.0,
        **_ignored: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.cap_kind = str(kind).lower().strip()
        if self.cap_kind not in ("analytic", "chance"):
            raise ValueError(f"safe_sl.kind must be 'analytic' or 'chance', got {kind!r}")
        self.overbook_factor = float(overbook_factor)
        self.keep_rate = None if keep_rate is None else float(keep_rate)
        self.low_remain_frac = float(low_remain_frac)
        self.buffer_mat = float(buffer_mat)
        self.activate_remain_frac = (
            None if activate_remain_frac is None else float(activate_remain_frac)
        )
        self.activate_load_frac = None if activate_load_frac is None else float(activate_load_frac)
        self.mix_alpha = float(np.clip(float(mix_alpha), 0.0, 1.0))
        self.last_cap: Optional[float] = None
        self.last_projected: bool = False
        self.last_policy_sl: Optional[float] = None

    def compute_cap(self, env: Any, price: float = 0.0) -> float:
        del price
        kwargs = dict(
            overbook_factor=self.overbook_factor,
            keep_rate=self.keep_rate,
            low_remain_frac=self.low_remain_frac,
            buffer_mat=self.buffer_mat,
        )
        if self.cap_kind == "chance":
            return chance_oversell_cap(env, **kwargs)
        return analytic_oversell_cap(env, **kwargs)

    def _activation_ok(self, env: Any) -> bool:
        """If activate_* knobs set, only project near capacity / high load."""
        capacity = float(getattr(env, "capacity", 10000))
        remain = float(getattr(env, "remain_inv", capacity))
        cum = float(getattr(env, "cumulative_mat_boh", 0.0))
        if self.activate_remain_frac is not None:
            if remain > float(self.activate_remain_frac) * max(capacity, 1.0):
                return False
        if self.activate_load_frac is not None:
            if cum < float(self.activate_load_frac) * max(capacity, 1.0):
                return False
        return True

    def project(self, env: Any, price: float, policy_sl: float) -> float:
        """Return ``min(policy_sl, safe_cap)`` when enabled; else ``policy_sl``."""
        self.last_policy_sl = float(policy_sl)
        self.last_projected = False
        if not self.enabled:
            self.last_cap = None
            return float(policy_sl)
        if not self._activation_ok(env):
            self.last_cap = None
            return float(policy_sl)
        cap = self.compute_cap(env, price)
        self.last_cap = float(cap)
        # Never raise SL; optionally mix toward policy (allows mild residual oversell)
        hard = float(min(float(policy_sl), cap))
        out = float(self.mix_alpha * float(policy_sl) + (1.0 - self.mix_alpha) * hard)
        out = float(min(float(policy_sl), out))  # still never above policy
        self.last_projected = out + 1e-6 < float(policy_sl)
        return _clip_sl(env, out)


def get_oversell_cap(cfg: dict[str, Any] | None = None, **overrides: Any) -> Optional[OversellCap]:
    """Build OversellCap from ``control.safe_sl``; ``None`` if disabled/absent."""
    cfg = dict(cfg or {})
    if "safe_sl" in cfg and isinstance(cfg["safe_sl"], dict):
        block = dict(cfg["safe_sl"])
    elif "enabled" in cfg or "overbook_factor" in cfg:
        # bare safe_sl block (or kwargs-like)
        block = {
            k: v
            for k, v in cfg.items()
            if k not in ("price_only", "selling_limit", "early_promo", "promo", "mpc")
        }
    else:
        return None
    block.update(overrides)
    if not bool(block.get("enabled", False)):
        return None
    return OversellCap(**block)


__all__ = [
    "OversellCap",
    "analytic_oversell_cap",
    "chance_oversell_cap",
    "get_oversell_cap",
]
