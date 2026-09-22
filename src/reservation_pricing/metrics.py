"""Business metrics and episode rollouts (separate from shaped RL reward)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from reservation_pricing.demand.protocol import features_from_state
from reservation_pricing.envs.reservation import ReservationEnv

# Published `score` charges every unsold seat once and every oversold seat twice;
# `remain > UNDERSELL_THRESHOLD` is the diagnostic undersell rate in every table.
OVERSELL_WEIGHT = 2.0
UNDERSELL_THRESHOLD = 1500.0


@dataclass
class EpisodeMetrics:
    true_revenue: float
    shaped_return: float
    load_factor: float
    remain_inv: float
    sellout_day: Optional[int]
    capacity: int
    service_date: Optional[str] = None
    month: Optional[int] = None
    dow: Optional[int] = None
    mean_price: float = 0.0
    mean_selling_limit: float = 0.0
    n_steps: int = 0
    # Soft-day-aware extras (filled by run_episode when available)
    base_demand: Optional[float] = None
    frac_at_floor: float = 0.0
    is_soft: Optional[bool] = None
    is_peak: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetrics:
    n_episodes: int
    mean_true_revenue: float
    std_true_revenue: float
    mean_shaped_return: float
    mean_load_factor: float
    mean_remain_inv: float
    mean_sellout_day: float
    sellout_rate: float
    score: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SoftAwareConfig:
    """Thresholds / formulas for soft vs peak stratification (eval only).

    Soft episodes are those where structural demand is too weak to fill even at
    floor price; undersell>1500 there is diagnostic, not a selection KPI.
    """

    peak_months: tuple[int, ...] = (7, 8, 11, 12)
    weekend_dow: tuple[int, ...] = (5, 6)
    base_demand_threshold: Optional[float] = 90.0
    # structural: low mid-horizon base OR (weekday AND month in soft_focus_months)
    # any: weekday OR off-peak month OR low base
    rule: str = "structural"
    soft_focus_months: tuple[int, ...] = (6,)
    soft_oracle: str = "min_price"  # min_price | myopic
    floor_price_tol: float = 1.0
    undersell_threshold: float = UNDERSELL_THRESHOLD
    lambda_peak: float = 200.0
    mu_soft: float = 200.0
    # gap_to_oracle: score_soft = mean_rev_soft - gap  (gap = mean max(0, oracle-rev))
    # oversell_only: score_soft = mean_rev_soft - mu * mean_oversell_amount
    soft_score_mode: str = "gap_to_oracle"

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]] = None) -> "SoftAwareConfig":
        d = dict(d or {})
        peak = d.get("peak_months", (7, 8, 11, 12))
        weekend = d.get("weekend_dow", (5, 6))
        soft_focus = d.get("soft_focus_months", (6,))
        thr = d.get("base_demand_threshold", 90.0)
        return cls(
            peak_months=tuple(int(x) for x in peak),
            weekend_dow=tuple(int(x) for x in weekend),
            base_demand_threshold=None if thr is None else float(thr),
            rule=str(d.get("rule", "structural")),
            soft_focus_months=tuple(int(x) for x in soft_focus),
            soft_oracle=str(d.get("soft_oracle", "min_price")),
            floor_price_tol=float(d.get("floor_price_tol", 1.0)),
            undersell_threshold=float(d.get("undersell_threshold", UNDERSELL_THRESHOLD)),
            lambda_peak=float(d.get("lambda_peak", d.get("shortfall_weight", 200.0))),
            mu_soft=float(d.get("mu_soft", 200.0)),
            soft_score_mode=str(d.get("soft_score_mode", "gap_to_oracle")),
        )


PolicyFn = Callable[[np.ndarray, ReservationEnv, dict], np.ndarray]


def _mid_horizon_base(env: ReservationEnv) -> Optional[float]:
    u = getattr(env, "unwrapped", env)
    dm = getattr(u, "demand_model", None)
    if dm is None or not hasattr(dm, "predict_base"):
        return None
    horizon = int(getattr(u, "booking_horizon", 100))
    feats = features_from_state(
        days_prior=max(horizon // 2, 1),
        dow=int(u.dow),
        month=int(u.month),
        booking_horizon=horizon,
    )
    return float(dm.predict_base(feats))


def classify_soft(
    *,
    month: Optional[int],
    dow: Optional[int],
    base_demand: Optional[float] = None,
    cfg: Optional[SoftAwareConfig] = None,
) -> bool:
    """Return True if episode is soft (structurally weak demand / calendar)."""
    cfg = cfg or SoftAwareConfig()
    if month is None or dow is None:
        return False
    month_i = int(month)
    dow_i = int(dow)
    weekend = dow_i in cfg.weekend_dow
    weekday = not weekend
    peak_month = month_i in cfg.peak_months
    offpeak = not peak_month
    low_base = (
        cfg.base_demand_threshold is not None
        and base_demand is not None
        and float(base_demand) < float(cfg.base_demand_threshold)
    )
    rule = (cfg.rule or "structural").lower()
    if rule in ("any", "or", "broad"):
        return bool(weekday or offpeak or low_base)
    # structural (default): same signals as oracle soft-day ceiling notes
    soft_month = month_i in cfg.soft_focus_months
    return bool(low_base or (weekday and soft_month))


def run_episode(
    env: ReservationEnv,
    policy: PolicyFn,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
    soft_cfg: Optional[SoftAwareConfig] = None,
) -> EpisodeMetrics:
    obs, info = env.reset(seed=seed, options=options)
    terminated = truncated = False
    shaped_return = 0.0
    prices: list[float] = []
    limits: list[float] = []
    steps = 0
    state: dict[str, Any] = {}

    u = getattr(env, "unwrapped", env)
    base_demand = _mid_horizon_base(env)
    month0 = int(getattr(u, "month", info.get("month") or 0) or 0)
    dow0 = int(getattr(u, "dow", info.get("dow") or 0) or 0)
    min_price = float(getattr(u, "min_price", 80.0))
    floor_tol = float(soft_cfg.floor_price_tol) if soft_cfg is not None else 1.0

    while not (terminated or truncated):
        action = policy(obs, env, state)
        obs, reward, terminated, truncated, info = env.step(action)
        shaped_return += float(reward)
        prices.append(float(info["price"]))
        limits.append(float(info["selling_limit"]))
        steps += 1

    sellout = info.get("sellout_day")
    near_floor = [1.0 if float(p) <= min_price + floor_tol else 0.0 for p in prices]
    frac_at_floor = float(np.mean(near_floor)) if near_floor else 0.0
    is_soft = classify_soft(
        month=int(info.get("month", month0)),
        dow=int(info.get("dow", dow0)),
        base_demand=base_demand,
        cfg=soft_cfg,
    )

    return EpisodeMetrics(
        true_revenue=float(info["true_revenue"]),
        shaped_return=float(shaped_return),
        load_factor=float(info["load_factor"]),
        remain_inv=float(info["remain_inv"]),
        sellout_day=int(sellout) if sellout is not None else None,
        capacity=int(info["capacity"]),
        service_date=info.get("service_date"),
        month=info.get("month", month0),
        dow=info.get("dow", dow0),
        mean_price=float(np.mean(prices)) if prices else 0.0,
        mean_selling_limit=float(np.mean(limits)) if limits else 0.0,
        n_steps=steps,
        base_demand=base_demand,
        frac_at_floor=frac_at_floor,
        is_soft=bool(is_soft),
        is_peak=not bool(is_soft),
    )


def aggregate(
    episodes: Sequence[EpisodeMetrics],
    shortfall_weight: float = 50.0,
    oversell_weight: float = OVERSELL_WEIGHT,
    undersell_threshold: float = UNDERSELL_THRESHOLD,
) -> AggregateMetrics:
    revs = np.array([e.true_revenue for e in episodes], dtype=np.float64)
    shaped = np.array([e.shaped_return for e in episodes], dtype=np.float64)
    loads = np.array([e.load_factor for e in episodes], dtype=np.float64)
    remain = np.array([e.remain_inv for e in episodes], dtype=np.float64)
    sellouts = [e.sellout_day for e in episodes]
    sellout_vals = np.array([s for s in sellouts if s is not None], dtype=np.float64)

    shortfalls = np.maximum(remain, 0.0) + oversell_weight * np.maximum(-remain, 0.0)
    mean_shortfall = float(np.mean(shortfalls)) if len(shortfalls) else 0.0
    mean_rev = float(np.mean(revs)) if len(revs) else 0.0
    score = mean_rev - shortfall_weight * mean_shortfall

    extras: dict[str, float] = {"mean_capacity_shortfall": mean_shortfall}
    if len(remain):
        extras["oversell_rate"] = float(np.mean(remain < 0))
        extras["undersell_gt1500_rate"] = float(np.mean(remain > undersell_threshold))

    return AggregateMetrics(
        n_episodes=len(episodes),
        mean_true_revenue=mean_rev,
        std_true_revenue=float(np.std(revs)) if len(revs) else 0.0,
        mean_shaped_return=float(np.mean(shaped)) if len(shaped) else 0.0,
        mean_load_factor=float(np.mean(loads)) if len(loads) else 0.0,
        mean_remain_inv=float(np.mean(remain)) if len(remain) else 0.0,
        mean_sellout_day=float(np.mean(sellout_vals)) if len(sellout_vals) else float("nan"),
        sellout_rate=float(np.mean([s is not None for s in sellouts])) if sellouts else 0.0,
        score=score,
        extras=extras,
    )


def metrics_table(rows: dict[str, AggregateMetrics]) -> pd.DataFrame:
    records = []
    for name, agg in rows.items():
        records.append(
            {
                "policy": name,
                "n_episodes": agg.n_episodes,
                "mean_true_revenue": round(agg.mean_true_revenue, 2),
                "std_true_revenue": round(agg.std_true_revenue, 2),
                "mean_load_factor": round(agg.mean_load_factor, 4),
                "mean_remain_inv": round(agg.mean_remain_inv, 2),
                "mean_sellout_day": (
                    None if np.isnan(agg.mean_sellout_day) else round(agg.mean_sellout_day, 2)
                ),
                "sellout_rate": round(agg.sellout_rate, 3),
                "mean_shaped_return": round(agg.mean_shaped_return, 4),
                "score": round(agg.score, 2),
            }
        )
    return pd.DataFrame(records)
