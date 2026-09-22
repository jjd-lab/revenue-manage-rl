"""Gymnasium ReservationEnv for amphitheater reservation dynamic pricing.

Preserves the original problem (capacity, horizon, price + selling-limit actions,
Weibull cancellations, DOW/month effects) with:
1. Pluggable DemandModel (linear_legacy | tree_elastic)
2. Stochastic demand / no-show / cancel noise
3. Gymnasium API; obs optionally scaled to ~[-1, 1]
4. Shaped training reward separate from true business metrics (in info)
5. Config-driven cancel / no-show parameters
6. Optional pace (booking-curve) penalty + soft-day reward upweight (training only)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from reservation_pricing.demand.protocol import features_from_state
from reservation_pricing.demand.registry import get_demand_model


class ReservationEnv(gym.Env):
    """Single-venue, single-product reservation pricing MDP.

    Inventory vocabulary used throughout the state and the controllers:
    ``boh`` is bookings on hand (accepted bookings not yet cancelled) and
    ``mat`` is materialized demand — bookings expected to survive cancellation
    and no-show and actually occupy a seat on the performance date.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        capacity: int = 10000,
        booking_horizon: int = 100,
        min_price: float = 80.0,
        max_price: float = 120.0,
        min_selling_limit: float = 10000.0,
        max_selling_limit: float = 15000.0,
        demand_noise_std: float = 8.0,
        noshow_noise_std: float = 0.01,
        cancel_rho_noise_std: float = 0.02,
        # Cancel / no-show structure (config-driven)
        cancel_lambda: float = 2000.0,
        cancel_rho_weekday: float = 0.4,
        cancel_rho_weekend: float = 0.5,
        noshow_base: float = 0.16,
        noshow_dow_coef: float = 0.02,
        noshow_month_coef: float = 0.02,
        revenue_scale: float = 1e-4,
        undersell_penalty: float = 400.0,
        oversell_penalty: float = 800.0,
        utilization_bonus: float = 80.0,
        # Pace / booking-curve shaped reward (dense undersell signal)
        pace_reward: bool = False,
        pace_schedule: str = "linear",  # linear | concave
        pace_penalty: float = 5.0,
        pace_target_final: float = 0.95,
        # Soft-day upweight (weekday / off-peak / low base demand)
        soft_day_upweight: bool = False,
        soft_day_weekday_factor: float = 1.5,
        soft_day_offpeak_factor: float = 1.25,
        soft_day_base_threshold: Optional[float] = None,
        soft_day_base_factor: float = 1.25,
        soft_day_sample_boost: float = 2.0,
        normalize_obs: bool = True,
        held_out_months: Optional[Sequence[int]] = None,
        use_held_out: bool = False,
        service_date: Optional[pd.Timestamp] = None,
        demand_model: Any = None,
        demand_cfg: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.capacity = int(capacity)
        self.booking_horizon = int(booking_horizon)
        self.min_price = float(min_price)
        self.max_price = float(max_price)
        self.min_selling_limit = float(min_selling_limit)
        self.max_selling_limit = float(max_selling_limit)
        self.demand_noise_std = float(demand_noise_std)
        self.noshow_noise_std = float(noshow_noise_std)
        self.cancel_rho_noise_std = float(cancel_rho_noise_std)
        self.cancel_lambda = float(cancel_lambda)
        self.cancel_rho_weekday = float(cancel_rho_weekday)
        self.cancel_rho_weekend = float(cancel_rho_weekend)
        self.noshow_base = float(noshow_base)
        self.noshow_dow_coef = float(noshow_dow_coef)
        self.noshow_month_coef = float(noshow_month_coef)
        self.revenue_scale = float(revenue_scale)
        self.undersell_penalty = float(undersell_penalty)
        self.oversell_penalty = float(oversell_penalty)
        self.utilization_bonus = float(utilization_bonus)
        self.pace_reward = bool(pace_reward)
        self.pace_schedule = str(pace_schedule).lower()
        self.pace_penalty = float(pace_penalty)
        self.pace_target_final = float(pace_target_final)
        self.soft_day_upweight = bool(soft_day_upweight)
        self.soft_day_weekday_factor = float(soft_day_weekday_factor)
        self.soft_day_offpeak_factor = float(soft_day_offpeak_factor)
        self.soft_day_base_threshold = (
            None if soft_day_base_threshold is None else float(soft_day_base_threshold)
        )
        self.soft_day_base_factor = float(soft_day_base_factor)
        self.soft_day_sample_boost = float(soft_day_sample_boost)
        self.normalize_obs = bool(normalize_obs)
        self.held_out_months = list(held_out_months) if held_out_months else []
        self.use_held_out = bool(use_held_out)
        if self.use_held_out and not self.held_out_months:
            raise ValueError(
                "use_held_out=True requires a non-empty held_out_months; "
                "otherwise service-date sampling cannot honour the split and "
                "silently falls back to the full year."
            )
        self._fixed_service_date = pd.Timestamp(service_date) if service_date is not None else None
        self.render_mode = render_mode

        if demand_model is not None:
            self.demand_model = demand_model
        elif demand_cfg is not None:
            self.demand_model = get_demand_model(demand_cfg)
        else:
            self.demand_model = get_demand_model(
                {"kind": "linear_legacy", "demand_noise_std": self.demand_noise_std}
            )

        self.action_space = spaces.Box(
            low=-np.ones(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        self._obs_low = np.concatenate(
            [
                np.array(
                    [
                        0.0,
                        0.0,
                        self.min_price,
                        self.min_selling_limit,
                        0.0,
                        0.0,
                        0.0,
                        self.capacity - self.max_selling_limit,
                    ],
                    dtype=np.float32,
                ),
                np.zeros(19, dtype=np.float32),
            ]
        )
        self._obs_high = np.concatenate(
            [
                np.array(
                    [
                        float(self.booking_horizon),
                        self.max_selling_limit,
                        self.max_price,
                        self.max_selling_limit,
                        self.max_selling_limit,
                        self.max_selling_limit,
                        self.max_selling_limit,
                        float(self.capacity),
                    ],
                    dtype=np.float32,
                ),
                np.ones(19, dtype=np.float32),
            ]
        )

        if self.normalize_obs:
            self.observation_space = spaces.Box(
                low=-1.5 * np.ones(27, dtype=np.float32),
                high=1.5 * np.ones(27, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=self._obs_low, high=self._obs_high, dtype=np.float32
            )

        self.np_random: np.random.Generator
        self.days_prior = self.booking_horizon
        self.service_date: pd.Timestamp
        self.dow = 0
        self.month = 1
        self.price = self.min_price
        self.selling_limit = self.max_selling_limit
        self.current_boh = 0.0
        self.gross_pickup = 0.0
        self.accepted_booking = 0.0
        self.no_show_rate = 0.0
        self.cumulative_mat_boh = 0.0
        self.remain_inv = float(self.capacity)
        self.cumulative_income = 0.0
        self.booking_trace: list[tuple[int, float]] = []
        self.sellout_day: Optional[int] = None
        self._episode_step = 0
        self._is_soft_day = False
        self._soft_day_mult = 1.0
        self._last_pace_target = 0.0
        self._last_pace_gap = 0.0

    def demand_features(self) -> dict[str, Any]:
        return features_from_state(
            days_prior=self.days_prior,
            dow=self.dow,
            month=self.month,
            booking_horizon=self.booking_horizon,
        )

    def _dow_month_effects(self) -> Tuple[float, float]:
        dow_effect = 1.5 if self.dow in (5, 6) else 1.0
        month_effect = 1.5 if self.month in (7, 8, 11, 12) else 1.0
        return dow_effect, month_effect

    def gross_fn(self) -> float:
        """Sample stochastic gross demand via the configured DemandModel."""
        return float(
            self.demand_model.sample_gross(self.demand_features(), self.price, self.np_random)
        )

    def expected_gross(self, price: Optional[float] = None) -> float:
        """Deterministic mean demand at current features (for myopic / diagnostics)."""
        p = self.price if price is None else float(price)
        return float(self.demand_model.predict_mean(self.demand_features(), p))

    def no_show_fn(self) -> float:
        dow_effect, month_effect = self._dow_month_effects()
        eps = float(self.np_random.normal(0.0, self.noshow_noise_std))
        rate = (
            self.noshow_base
            - self.noshow_dow_coef * dow_effect
            - self.noshow_month_coef * month_effect
            + eps
        )
        return float(np.clip(rate, 0.01, 0.4))

    @staticmethod
    def weibull_surv(time: float, lambda_: float, rho_: float) -> float:
        if time <= 0:
            return 1.0
        return float(np.exp(-((time / lambda_) ** rho_)))

    def cancel_fn(
        self,
        today_to_target_date: float,
        days_book_to_today: Optional[float] = None,
        conditional: bool = False,
    ) -> float:
        base_rho = self.cancel_rho_weekend if self.dow in (5, 6) else self.cancel_rho_weekday
        cancel_rho = base_rho + float(self.np_random.normal(0.0, self.cancel_rho_noise_std))
        cancel_rho = float(np.clip(cancel_rho, 0.15, 1.0))
        if not conditional:
            return 1.0 - self.weibull_surv(today_to_target_date, self.cancel_lambda, cancel_rho)
        assert days_book_to_today is not None
        surv_full = self.weibull_surv(
            days_book_to_today + today_to_target_date, self.cancel_lambda, cancel_rho
        )
        surv_past = self.weibull_surv(days_book_to_today, self.cancel_lambda, cancel_rho)
        return 1.0 - surv_full / max(surv_past, 1e-12)

    @staticmethod
    def one_hot_dow(dow: int) -> np.ndarray:
        vec = np.zeros(7, dtype=np.float32)
        vec[int(dow) % 7] = 1.0
        return vec

    @staticmethod
    def one_hot_month(month: int) -> np.ndarray:
        vec = np.zeros(12, dtype=np.float32)
        vec[int(month) - 1] = 1.0
        return vec

    def _raw_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array(
                    [
                        float(self.days_prior),
                        float(self.current_boh),
                        float(self.price),
                        float(self.selling_limit),
                        float(self.gross_pickup),
                        float(self.accepted_booking),
                        float(self.cumulative_mat_boh),
                        float(self.remain_inv),
                    ],
                    dtype=np.float32,
                ),
                self.one_hot_dow(self.dow),
                self.one_hot_month(self.month),
            ]
        ).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        raw = self._raw_obs()
        if not self.normalize_obs:
            return raw
        span = np.maximum(self._obs_high - self._obs_low, 1e-6)
        scaled = 2.0 * (raw - self._obs_low) / span - 1.0
        return np.clip(scaled, -1.5, 1.5).astype(np.float32)

    @staticmethod
    def _is_weekend(dow: int) -> bool:
        return int(dow) in (5, 6)

    @staticmethod
    def _is_peak_month(month: int) -> bool:
        return int(month) in (7, 8, 11, 12)

    def _pace_target_load(self) -> float:
        """Target load factor given elapsed booking progress in [0, 1]."""
        H = max(self.booking_horizon, 1)
        # After days_prior decrement in step: progress = elapsed / H
        elapsed = float(H - self.days_prior)
        t = float(np.clip(elapsed / float(H), 0.0, 1.0))
        final = float(np.clip(self.pace_target_final, 0.0, 1.5))
        if self.pace_schedule == "concave":
            # Front-loaded target: faster early fill expectation (1-(1-t)^2)
            return final * (1.0 - (1.0 - t) ** 2)
        # default linear
        return final * t

    def _compute_soft_day_mult(self) -> tuple[bool, float]:
        """Return (is_soft, reward_multiplier). Business metrics stay unscaled."""
        if not self.soft_day_upweight:
            return False, 1.0
        weekday = not self._is_weekend(self.dow)
        offpeak = not self._is_peak_month(self.month)
        soft = weekday or offpeak
        mult = 1.0
        if weekday:
            mult *= self.soft_day_weekday_factor
        if offpeak:
            mult *= self.soft_day_offpeak_factor
        if self.soft_day_base_threshold is not None and hasattr(self.demand_model, "predict_base"):
            # Mid-horizon base demand as a soft-day proxy
            feats = features_from_state(
                days_prior=max(self.booking_horizon // 2, 1),
                dow=self.dow,
                month=self.month,
                booking_horizon=self.booking_horizon,
            )
            base = float(self.demand_model.predict_base(feats))
            if base < self.soft_day_base_threshold:
                soft = True
                mult *= self.soft_day_base_factor
        return soft, float(mult)

    def _info(self, step_revenue: float = 0.0, shaped_reward: float = 0.0) -> dict[str, Any]:
        load_factor = float(self.cumulative_mat_boh / max(self.capacity, 1))
        return {
            "true_revenue_step": float(step_revenue),
            "true_revenue": float(self.cumulative_income),
            "remain_inv": float(self.remain_inv),
            "load_factor": load_factor,
            "days_prior": int(self.days_prior),
            "accepted_booking": float(self.accepted_booking),
            "price": float(self.price),
            "selling_limit": float(self.selling_limit),
            "sellout_day": self.sellout_day,
            "shaped_reward": float(shaped_reward),
            "service_date": str(self.service_date.date())
            if hasattr(self, "service_date")
            else None,
            "month": int(self.month),
            "dow": int(self.dow),
            "capacity": int(self.capacity),
            "demand_kind": getattr(self.demand_model, "kind", None),
            "pace_target": float(self._last_pace_target),
            "pace_gap": float(self._last_pace_gap),
            "soft_day": bool(self._is_soft_day),
            "soft_day_mult": float(self._soft_day_mult),
        }

    def _recompute_boh(self) -> float:
        if not self.booking_trace:
            return 0.0
        total = 0.0
        n = len(self.booking_trace)
        cancel_rho = self.cancel_rho_weekend if self.dow in (5, 6) else self.cancel_rho_weekday
        for idx, (_dp, bk) in enumerate(self.booking_trace):
            day_past = n - idx
            cancel_prob = 1.0 - self.weibull_surv(float(day_past), self.cancel_lambda, cancel_rho)
            total += bk * (1.0 - cancel_prob)
        return float(total)

    def _sample_service_date(self) -> pd.Timestamp:
        if self._fixed_service_date is not None:
            return self._fixed_service_date

        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2022-12-31")
        boost = (
            self.soft_day_sample_boost
            if (self.soft_day_upweight and not self.use_held_out)
            else 1.0
        )
        boost = max(float(boost), 1.0)
        for _ in range(500):
            delta_days = int(self.np_random.integers(0, (end - start).days + 1))
            dt = start + pd.Timedelta(days=delta_days)
            in_held = dt.month in self.held_out_months
            if self.use_held_out and in_held:
                return dt
            if (not self.use_held_out) and (not in_held or not self.held_out_months):
                # Upweight soft calendar days in training by rejecting hard days
                if boost > 1.0:
                    soft_cal = (int(dt.dayofweek) not in (5, 6)) or (
                        int(dt.month) not in (7, 8, 11, 12)
                    )
                    if (not soft_cal) and float(self.np_random.random()) > (1.0 / boost):
                        continue
                return dt
        return start + pd.Timedelta(days=int(self.np_random.integers(0, 365)))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        options = options or {}

        self.days_prior = self.booking_horizon
        if "service_date" in options:
            self.service_date = pd.Timestamp(options["service_date"])
        else:
            self.service_date = self._sample_service_date()
        self.dow = int(self.service_date.dayofweek)
        self.month = int(self.service_date.month)

        self.price = self.min_price
        self.selling_limit = self.max_selling_limit
        self.current_boh = 0.0
        self.gross_pickup = 0.0
        self.accepted_booking = 0.0
        self.no_show_rate = self.no_show_fn()
        self.cumulative_mat_boh = 0.0
        self.remain_inv = float(self.capacity)
        self.cumulative_income = 0.0
        self.booking_trace = []
        self.sellout_day = None
        self._episode_step = 0
        self._last_pace_target = 0.0
        self._last_pace_gap = 0.0
        self._is_soft_day, self._soft_day_mult = self._compute_soft_day_mult()

        return self._get_obs(), self._info()

    def _unscale_action(self, action: np.ndarray) -> tuple[float, float]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        price = self.min_price + (a[0] + 1.0) * 0.5 * (self.max_price - self.min_price)
        sl = self.min_selling_limit + (a[1] + 1.0) * 0.5 * (
            self.max_selling_limit - self.min_selling_limit
        )
        return float(price), float(sl)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        self.price, self.selling_limit = self._unscale_action(action)

        self.days_prior -= 1
        self._episode_step += 1

        self.current_boh = self._recompute_boh()
        self.gross_pickup = self.gross_fn()
        headroom = max(0.0, self.selling_limit - self.current_boh)
        self.accepted_booking = float(min(self.gross_pickup, headroom))

        cancel_to_service = self.cancel_fn(float(max(self.days_prior, 0)))
        mat_today = self.accepted_booking * (1.0 - cancel_to_service) * (1.0 - self.no_show_rate)
        self.cumulative_mat_boh += mat_today
        self.remain_inv = float(self.capacity - self.cumulative_mat_boh)

        step_revenue = self.accepted_booking * self.price
        self.cumulative_income += step_revenue

        if self.sellout_day is None and self.remain_inv <= 0:
            self.sellout_day = int(self.days_prior)

        self.booking_trace.append((self.days_prior, self.accepted_booking))

        shaped = step_revenue * self.revenue_scale
        terminated = self.days_prior <= 0
        truncated = False

        # Dense pace / booking-curve penalty when behind target fill path
        self._last_pace_target = 0.0
        self._last_pace_gap = 0.0
        if self.pace_reward:
            target = self._pace_target_load()
            fill = float(self.cumulative_mat_boh / max(self.capacity, 1))
            gap = max(0.0, target - fill)
            self._last_pace_target = float(target)
            self._last_pace_gap = float(gap)
            shaped -= self.pace_penalty * gap

        if terminated:
            shortfall = max(0.0, self.remain_inv)
            overshoot = max(0.0, -self.remain_inv)
            shaped -= self.undersell_penalty * (shortfall / self.capacity)
            shaped -= self.oversell_penalty * (overshoot / self.capacity)
            util = float(np.clip(self.cumulative_mat_boh / self.capacity, 0.0, 1.0))
            if self.remain_inv >= 0:
                shaped += self.utilization_bonus * util

        # Soft-day upweight: scale shaped training return only (metrics unshaped)
        if self.soft_day_upweight and self._soft_day_mult != 1.0:
            shaped *= self._soft_day_mult

        info = self._info(step_revenue=step_revenue, shaped_reward=shaped)
        return self._get_obs(), float(shaped), terminated, truncated, info

    def render(self) -> None:
        print(
            f"days_prior={self.days_prior} date={self.service_date.date()} "
            f"price={self.price:.1f} SL={self.selling_limit:.0f} "
            f"accepted={self.accepted_booking:.1f} remain={self.remain_inv:.1f} "
            f"revenue={self.cumulative_income:.1f} demand={getattr(self.demand_model, 'kind', '?')}"
        )
