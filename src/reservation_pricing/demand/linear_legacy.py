"""Prototype-style linear demand (price-aware closed form).

Preserved for A/B vs the production-like ``tree_elastic`` model.

Mean gross (before noise)::

    mean = 80 - 0.2 * price
         + 60 * dow_effect + 40 * month_effect
         - 0.1 * (days_prior > 30) * days_prior

where dow_effect / month_effect are 1.5 on weekends / peak months else 1.0.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class LinearLegacyDemand:
    """Price-linear demand matching the original ReservationEnv.gross_fn mean."""

    kind = "linear_legacy"

    def __init__(
        self,
        demand_noise_std: float = 8.0,
        intercept: float = 80.0,
        price_coef: float = -0.2,
        dow_boost: float = 60.0,
        month_boost: float = 40.0,
        days_prior_coef: float = -0.1,
        days_prior_threshold: float = 30.0,
        **_extra: Any,
    ) -> None:
        self.demand_noise_std = float(demand_noise_std)
        self.intercept = float(intercept)
        self.price_coef = float(price_coef)
        self.dow_boost = float(dow_boost)
        self.month_boost = float(month_boost)
        self.days_prior_coef = float(days_prior_coef)
        self.days_prior_threshold = float(days_prior_threshold)

    def _effects(self, features: Mapping[str, Any]) -> tuple[float, float]:
        dow = int(features.get("dow", 0))
        month = int(features.get("month", 1))
        dow_effect = 1.5 if dow in (5, 6) else 1.0
        month_effect = 1.5 if month in (7, 8, 11, 12) else 1.0
        return dow_effect, month_effect

    def predict_mean(self, features: Mapping[str, Any], price: float) -> float:
        dow_effect, month_effect = self._effects(features)
        days_prior = float(features.get("days_prior", 0))
        mean = (
            self.intercept
            + self.price_coef * float(price)
            + self.dow_boost * dow_effect
            + self.month_boost * month_effect
            + self.days_prior_coef
            * (1.0 if days_prior > self.days_prior_threshold else 0.0)
            * days_prior
        )
        return float(max(0.0, mean))

    def sample_gross(
        self,
        features: Mapping[str, Any],
        price: float,
        rng: np.random.Generator,
    ) -> float:
        mean = self.predict_mean(features, price)
        eps = float(rng.normal(0.0, self.demand_noise_std))
        return float(max(0.0, mean + eps))
