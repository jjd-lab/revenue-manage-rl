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
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


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

    def predict_base(self, features: Mapping[str, Any]) -> float:
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
