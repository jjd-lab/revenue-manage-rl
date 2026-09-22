"""Tree / forest base demand + linear price elasticity (production-like default).

**Base (price-unaware):** sklearn GradientBoostingRegressor over calendar &
booking-curve features, loaded from a shipped joblib asset.

**Price effect (multiplicative linear elasticity)**::

    expected_gross = max(0, base * (1 + elasticity * (price - ref_price) / ref_price))

Config knobs: ``elasticity`` (typically negative), ``ref_price``, ``demand_noise_std``.

Additive alternative ``base + beta * (price - ref)`` is available via
``price_mode: additive`` with ``beta`` (usually negative).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from reservation_pricing.demand.synthesize import DEFAULT_ASSET, FEATURE_COLS, load_fitted


class TreeElasticDemand:
    """Production-aligned single-product demand: tree base + linear elasticity."""

    kind = "tree_elastic"

    def __init__(
        self,
        elasticity: float = -1.2,
        ref_price: float = 100.0,
        demand_noise_std: float = 8.0,
        price_mode: str = "multiplicative",
        beta: float = -0.2,
        model_path: Optional[str] = None,
        fitted: Optional[dict] = None,
        **_extra: Any,
    ) -> None:
        self.elasticity = float(elasticity)
        self.ref_price = float(ref_price)
        self.demand_noise_std = float(demand_noise_std)
        self.price_mode = str(price_mode).lower()
        self.beta = float(beta)

        if fitted is not None:
            payload = fitted
        else:
            path = Path(model_path) if model_path else DEFAULT_ASSET
            if not path.is_absolute():
                # Resolve relative to package demand/ or CWD
                cand = Path(__file__).resolve().parent / path
                path = cand if cand.exists() else Path(model_path)
            payload = load_fitted(path)

        self._model = payload["model"]
        self._feature_cols = list(payload.get("feature_cols", FEATURE_COLS))
        self._meta = dict(payload.get("meta") or {})

    def _feature_row(self, features: Mapping[str, Any]) -> np.ndarray:
        row = [float(features.get(c, 0.0)) for c in self._feature_cols]
        return np.asarray(row, dtype=np.float64).reshape(1, -1)

    def predict_base(self, features: Mapping[str, Any]) -> float:
        pred = float(self._model.predict(self._feature_row(features))[0])
        return float(max(0.0, pred))

    def apply_price(self, base: float, price: float) -> float:
        p = float(price)
        ref = max(self.ref_price, 1e-6)
        if self.price_mode == "additive":
            return float(max(0.0, base + self.beta * (p - ref)))
        # default: multiplicative elasticity
        return float(max(0.0, base * (1.0 + self.elasticity * (p - ref) / ref)))

    def predict_mean(self, features: Mapping[str, Any], price: float) -> float:
        return self.apply_price(self.predict_base(features), price)

    def sample_gross(
        self,
        features: Mapping[str, Any],
        price: float,
        rng: np.random.Generator,
    ) -> float:
        mean = self.predict_mean(features, price)
        eps = float(rng.normal(0.0, self.demand_noise_std))
        return float(max(0.0, mean + eps))
