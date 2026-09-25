"""Festival scenario: nights, passes, and multinomial-logit demand.

A pass covers one or more consecutive nights and takes a seat on each. Every
day a Poisson number of buyers arrive; each buys one of the passes on sale, or
nothing, with probability ``exp(u_k) / (1 + sum_j exp(u_j))`` over the passes
on sale, where::

    u_k = pass_appeal[length_k] + sum of night_appeal over k's nights
          - beta(days out) * price_k / 100

``price_k`` is the pass's total price: its per-night price times its length.
``beta`` moves linearly from ``beta_early`` a full horizon out to ``beta_late``
on the last day. Raising one pass's price, or taking it off sale, sends some of
its buyers to the other nights and lengths in proportion to their shares, and
some to nothing: that is the cross-price and the cross-date effect.

Each season draws its own market size and night appeals around the configured
values; decision code knows only the configured values (``Season.usual``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def consecutive_passes(n_nights: int) -> tuple[tuple[int, ...], ...]:
    """Every run of consecutive nights, shortest first: 1-day, 2-day, ..."""
    return tuple(
        tuple(range(start, start + length))
        for length in range(1, n_nights + 1)
        for start in range(n_nights - length + 1)
    )


@dataclass(frozen=True)
class FestivalConfig:
    """Everything that defines the scenario. ``festival:`` in a YAML overrides any field."""

    night_names: tuple[str, ...] = ("fri", "sat", "sun")
    night_appeal: tuple[float, ...] = (0.0, 0.3, -0.3)
    # None means every run of consecutive nights (consecutive_passes).
    passes: Optional[tuple[tuple[int, ...], ...]] = None
    # Extra appeal by pass length, index 0 = 1-day; missing lengths get 0.
    pass_appeal: tuple[float, ...] = (0.0, 0.2, 0.4)
    capacity: int = 10000
    horizon: int = 100
    min_night_price: float = 80.0
    max_night_price: float = 120.0
    min_selling_limit: float = 10000.0
    max_selling_limit: float = 15000.0
    beta_early: float = 3.0
    beta_late: float = 2.0
    market_size: float = 100000.0
    # Arrivals per day are proportional to exp(-days_out / arrival_decay_days).
    arrival_decay_days: float = 40.0
    market_cv: float = 0.15
    night_appeal_std: float = 0.15
    cancel_lambda: float = 2000.0
    cancel_rho: float = 0.45
    cancel_rho_std: float = 0.02
    noshow: float = 0.11
    noshow_std: float = 0.01
    unsold_cost: float = 200.0
    denied_cost: float = 400.0
    revenue_scale: float = 1e-4
    # Potential-based shaping (Ng, Harada & Russell 1999): see FestivalEnv.potential.
    shape_reward: bool = False

    def __post_init__(self) -> None:
        if len(self.night_appeal) != len(self.night_names):
            raise ValueError("festival.night_appeal needs one value per night")
        for p in self.pass_list:
            if (
                not p
                or p[0] < 0
                or list(p) != list(range(p[0], p[0] + len(p)))
                or p[-1] >= self.n_nights
            ):
                raise ValueError(f"festival pass {p!r} must be consecutive nights in range")

    @classmethod
    def from_mapping(cls, raw: Optional[Mapping[str, Any]]) -> "FestivalConfig":
        raw = dict(raw or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown festival keys: {sorted(unknown)}")
        for key in ("night_names", "night_appeal", "pass_appeal"):
            if key in raw:
                raw[key] = tuple(raw[key])
        if raw.get("passes") is not None:
            raw["passes"] = tuple(tuple(int(i) for i in p) for p in raw["passes"])
        return cls(**raw)

    @property
    def n_nights(self) -> int:
        return len(self.night_names)

    @property
    def pass_list(self) -> tuple[tuple[int, ...], ...]:
        return self.passes if self.passes is not None else consecutive_passes(self.n_nights)

    @property
    def n_passes(self) -> int:
        return len(self.pass_list)

    def pass_names(self) -> list[str]:
        return ["_".join(self.night_names[i] for i in p) for p in self.pass_list]

    def incidence(self) -> np.ndarray:
        """``A[i, k] = 1`` when pass ``k`` takes a seat on night ``i``."""
        a = np.zeros((self.n_nights, self.n_passes))
        for k, p in enumerate(self.pass_list):
            a[list(p), k] = 1.0
        return a

    def lengths(self) -> np.ndarray:
        return np.array([len(p) for p in self.pass_list], dtype=np.float64)

    def beta(self, days_out: float) -> float:
        frac = float(np.clip(days_out / max(self.horizon, 1), 0.0, 1.0))
        return self.beta_late + frac * (self.beta_early - self.beta_late)

    def arrival_weights(self) -> np.ndarray:
        """Share of the season's buyers arriving each ``days_out`` (index = days out)."""
        w = np.exp(-np.arange(self.horizon) / self.arrival_decay_days)
        return w / w.sum()

    def keep_rate(self, days_out: float, rho: float, noshow: float) -> float:
        """Chance a booking made ``days_out`` days before the night is kept and used."""
        surv = np.exp(-((max(days_out, 0.0) / self.cancel_lambda) ** rho)) if days_out > 0 else 1.0
        return float(surv * (1.0 - noshow))


@dataclass(frozen=True)
class Season:
    """What differs from season to season. Decision code sees only ``usual``."""

    market_mult: float
    night_appeal: np.ndarray = field(repr=False)
    cancel_rho: float
    noshow: float

    @classmethod
    def usual(cls, cfg: FestivalConfig) -> "Season":
        return cls(1.0, np.asarray(cfg.night_appeal, dtype=np.float64), cfg.cancel_rho, cfg.noshow)

    @classmethod
    def draw(cls, cfg: FestivalConfig, rng: np.random.Generator) -> "Season":
        sigma = float(np.sqrt(np.log1p(cfg.market_cv**2)))
        return cls(
            market_mult=float(rng.lognormal(-0.5 * sigma**2, sigma)),
            night_appeal=np.asarray(cfg.night_appeal)
            + rng.normal(0.0, cfg.night_appeal_std, cfg.n_nights),
            cancel_rho=float(np.clip(rng.normal(cfg.cancel_rho, cfg.cancel_rho_std), 0.15, 1.0)),
            noshow=float(np.clip(rng.normal(cfg.noshow, cfg.noshow_std), 0.01, 0.4)),
        )


def pass_utility(cfg: FestivalConfig, night_appeal: Sequence[float]) -> np.ndarray:
    """Price-free utility of each pass."""
    extra = np.array(
        [cfg.pass_appeal[len(p) - 1] if len(p) <= len(cfg.pass_appeal) else 0.0 for p in cfg.pass_list]
    )
    return cfg.incidence().T @ np.asarray(night_appeal, dtype=np.float64) + extra


def choice_probs(
    utility: np.ndarray, prices: np.ndarray, beta: float, on_sale: Optional[np.ndarray] = None
) -> np.ndarray:
    """Chance an arriving buyer takes each pass; ``1 - sum`` is buying nothing."""
    u = utility - beta * np.asarray(prices, dtype=np.float64) / 100.0
    e = np.exp(u)
    if on_sale is not None:
        e = np.where(on_sale, e, 0.0)
    return e / (1.0 + e.sum())
