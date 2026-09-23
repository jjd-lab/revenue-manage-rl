"""True demand that differs from the forecast a little on every night.

Wraps a multiplicative-elasticity model (``tree_elastic``). At each env reset,
``redraw`` picks this night's demand level (lognormal around 1) and price
sensitivity (normal around the model's elasticity). The env generates bookings
from the wrapper; ``make_env`` hands decision code the unwrapped model as the
forecast, so no planner ever sees the night's draw.

``level_sd`` is the standard deviation of log(level): 0.125 puts about 95% of
nights within ±25%. ``elasticity_sd`` 0.15 puts them within −0.9 to −1.5 around
−1.2. ``level_shift`` and ``elasticity_shift`` move every night the same way: a
year that runs above or below the forecast, not a night-by-night error.
``timing_shift`` (days, positive = earlier) and ``timing_sd`` move *when* demand
arrives: the night sees, ``d`` days out, the base the forecast expects
``d - timing`` days out, rescaled so the night's total over days 0-99 matches the
forecast's. Without that rescale, clipping at the ends of the window would also
change how much demand the night gets. Assumes the default 100-day horizon.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from reservation_pricing.demand.protocol import features_from_state


class NightVaryingDemand:
    kind = "night_varying"

    def __init__(
        self,
        inner: Any,
        *,
        level_sd: float = 0.0,
        elasticity_sd: float = 0.0,
        level_shift: float = 1.0,
        elasticity_shift: float = 0.0,
        timing_shift: float = 0.0,
        timing_sd: float = 0.0,
    ) -> None:
        if getattr(inner, "price_mode", "multiplicative") != "multiplicative" or not hasattr(
            inner, "elasticity"
        ):
            raise ValueError("night variation needs a multiplicative-elasticity demand model")
        self.inner = inner
        self.level_sd = float(level_sd)
        self.elasticity_sd = float(elasticity_sd)
        self.level_shift = float(level_shift)
        self.elasticity_shift = float(elasticity_shift)
        self.timing_shift = float(timing_shift)
        self.timing_sd = float(timing_sd)
        self.timing = self.timing_shift
        self._timing_scale: dict[tuple, float] = {}
        self.demand_noise_std = float(inner.demand_noise_std)
        self.level = self.level_shift
        self.elasticity = float(inner.elasticity) + self.elasticity_shift

    def redraw(self, rng: np.random.Generator) -> None:
        self.level = self.level_shift
        if self.level_sd > 0:
            self.level *= float(np.exp(rng.normal(0.0, self.level_sd)))
        self.elasticity = float(self.inner.elasticity) + self.elasticity_shift
        if self.elasticity_sd > 0:
            self.elasticity += float(rng.normal(0.0, self.elasticity_sd))
        self.timing = self.timing_shift
        if self.timing_sd > 0:
            self.timing += float(rng.normal(0.0, self.timing_sd))

    def _shifted(self, days_prior: int, dow: int, month: int) -> float:
        days = int(np.clip(round(days_prior - self.timing), 0, 100))
        return float(
            self.inner.predict_base(features_from_state(days_prior=days, dow=dow, month=month))
        )

    def _conserving_scale(self, dow: int, month: int) -> float:
        key = (dow, month, self.timing)
        if key not in self._timing_scale:
            usual = sum(
                float(
                    self.inner.predict_base(features_from_state(days_prior=d, dow=dow, month=month))
                )
                for d in range(100)
            )
            shifted = sum(self._shifted(d, dow, month) for d in range(100))
            self._timing_scale[key] = usual / shifted if shifted > 0 else 1.0
        return self._timing_scale[key]

    def predict_base(self, features: Mapping[str, Any]) -> float:
        if self.timing:
            dow, month = int(features["dow"]), int(features["month"])
            base = self._shifted(int(features["days_prior"]), dow, month)
            return self.level * base * self._conserving_scale(dow, month)
        return self.level * float(self.inner.predict_base(features))

    def predict_mean(self, features: Mapping[str, Any], price: float) -> float:
        ref = max(float(self.inner.ref_price), 1e-6)
        scale = 1.0 + self.elasticity * (float(price) - ref) / ref
        return float(max(0.0, self.predict_base(features) * scale))

    def sample_gross(
        self, features: Mapping[str, Any], price: float, rng: np.random.Generator
    ) -> float:
        mean = self.predict_mean(features, price)
        return float(max(0.0, mean + rng.normal(0.0, self.demand_noise_std)))


__all__ = ["NightVaryingDemand"]
