"""DemandModel protocol (single-product) and feature helpers.

Production RM typically separates:
  1. Base (price-unaware) demand from a tree/forest over calendar & booking-curve features
  2. A price-response layer (here: linear elasticity) on top of that base

This protocol is the single-product API. Multi-product / cross-price extensions are
sketched in ``docs/EXTENDING.md`` — not implemented yet.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DemandModel(Protocol):
    """Single-product expected / sampled gross demand given features + price."""

    kind: str

    def predict_mean(self, features: Mapping[str, Any], price: float) -> float:
        """Deterministic expected gross bookings (no residual noise)."""
        ...

    def sample_gross(
        self,
        features: Mapping[str, Any],
        price: float,
        rng: np.random.Generator,
    ) -> float:
        """Stochastic gross demand (mean + residual noise, floored at 0)."""
        ...


def decision_model(env: Any) -> Any:
    """The demand model *decision* code may consult — the forecast, not the truth.

    Baselines, selling-limit controllers, early promo and the price MPC all call
    this instead of ``env.demand_model``. It normally returns that same model, so
    a single-model setup behaves exactly as before.

    When ``demand.forecast`` is configured the env carries a **separate, wrong**
    model here: the environment still generates bookings from its own true model,
    and only the things a real operator would have to forecast with see the wrong
    one. Env dynamics (``gross_fn``) and evaluation machinery (soft/peak
    classification, the soft-day oracle) deliberately keep using the true model —
    misspecifying those would change the world or the scoreboard, not the
    operator's information. See ``docs/DESIGN.md`` § Forecast vs truth.
    """
    return getattr(env, "forecast_model", None) or getattr(env, "demand_model", None)


def booking_curve_bin(days_prior: int, horizon: int = 100, n_bins: int = 5) -> int:
    """Discretize days-prior into booking-curve bins (0 = far out, n_bins-1 = last minute)."""
    if horizon <= 0:
        return 0
    # Invert so near-service days get higher bin indices (late bookers)
    frac = 1.0 - float(days_prior) / float(horizon)
    bin_idx = int(np.clip(np.floor(frac * n_bins), 0, n_bins - 1))
    return bin_idx


def features_from_state(
    *,
    days_prior: int,
    dow: int,
    month: int,
    booking_horizon: int = 100,
) -> dict[str, Any]:
    """Standard single-product feature dict used by demand models and myopic pricing."""
    is_weekend = 1 if int(dow) in (5, 6) else 0
    is_peak_month = 1 if int(month) in (7, 8, 11, 12) else 0
    return {
        "days_prior": int(days_prior),
        "dow": int(dow),
        "month": int(month),
        "is_weekend": is_weekend,
        "is_peak_month": is_peak_month,
        "booking_curve_bin": booking_curve_bin(int(days_prior), int(booking_horizon)),
    }
