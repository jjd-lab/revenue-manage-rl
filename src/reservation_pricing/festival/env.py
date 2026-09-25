"""Gymnasium env for a multi-night festival sold as passes over one booking window.

Each day the agent sets a per-night price for every pass and a selling limit for
every night: ``n_passes + n_nights`` actions in ``[-1, 1]``. A pass is on sale
only while every night it covers has bookings on hand below that night's limit.
When today's requests would overrun a night's headroom, every pass using that
night is cut back in proportion, and the buyers cut are lost.

Bookings follow the single-night env: revenue is taken at booking and kept, a
booking made ``d`` days out survives cancellation with a Weibull curve, and the
survivors no-show at the season's rate. The reward is the score itself:
revenue, minus ``unsold_cost`` per empty seat and ``denied_cost`` per denied
admission on each night at the end, times ``revenue_scale``.

``shape_reward`` adds ``potential(after) - potential(before)`` each day, with
the potential zero once the season ends. With gamma 1 the shaping sums to a
constant over a season, so the best policy is unchanged; only when the charge
for an empty seat arrives changes, from the last day to the day it is filled.

The observation never carries the season's own rates: its show-up count is
bookings times the usual keep rate, what an operator can compute.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reservation_pricing.festival.model import (
    FestivalConfig,
    Season,
    choice_probs,
    pass_utility,
)


class FestivalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[FestivalConfig] = None, seed_offset: int = 0) -> None:
        super().__init__()
        self.cfg = cfg or FestivalConfig()
        # Added to every reset seed. The season is drawn from the seed, so a
        # training env needs an offset to keep clear of the held-out seasons.
        self.seed_offset = int(seed_offset)
        self.A = self.cfg.incidence()
        self.lengths = self.cfg.lengths()
        self.n_passes = self.cfg.n_passes
        self.n_nights = self.cfg.n_nights
        self.arrival_weights = self.cfg.arrival_weights()
        self._peak_arrivals = float(self.cfg.market_size * self.arrival_weights.max())
        self.usual = Season.usual(self.cfg)

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_passes + self.n_nights,), dtype=np.float32
        )
        obs_dim = 1 + 2 * self.n_passes + 3 * self.n_nights
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(obs_dim,), dtype=np.float32)
        self.season = self.usual
        self._reset_state()

    def _reset_state(self) -> None:
        cfg = self.cfg
        self.days_prior = cfg.horizon
        self.night_prices = np.full(self.n_passes, cfg.min_night_price)
        self.limits = np.full(self.n_nights, cfg.max_selling_limit)
        # bookings[d] = passes booked d days out, per pass.
        self.bookings = np.zeros((cfg.horizon, self.n_passes))
        self.requests = np.zeros(self.n_passes)
        self.accepted = np.zeros(self.n_passes)
        self.on_sale = np.ones(self.n_passes, dtype=bool)
        self.showups = np.zeros(self.n_nights)
        self.showups_usual = np.zeros(self.n_nights)
        self.revenue = 0.0

    @property
    def prices(self) -> np.ndarray:
        """Total price of each pass."""
        return self.night_prices * self.lengths

    def bookings_on_hand(self) -> np.ndarray:
        """Bookings not yet cancelled, per night."""
        rho = self.season.cancel_rho
        lam = self.cfg.cancel_lambda
        d = np.arange(self.cfg.horizon, dtype=np.float64)
        age = np.maximum(d - self.days_prior, 0.0)
        surv = np.exp(-((age / lam) ** rho))
        return self.A @ (surv @ self.bookings)

    def unscale_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        frac = (a + 1.0) / 2.0
        prices = cfg.min_night_price + frac[: self.n_passes] * (
            cfg.max_night_price - cfg.min_night_price
        )
        limits = cfg.min_selling_limit + frac[self.n_passes :] * (
            cfg.max_selling_limit - cfg.min_selling_limit
        )
        return prices, limits

    def scale_action(self, night_prices: np.ndarray, limits: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        p = (np.asarray(night_prices) - cfg.min_night_price) / (
            cfg.max_night_price - cfg.min_night_price
        )
        s = (np.asarray(limits) - cfg.min_selling_limit) / (
            cfg.max_selling_limit - cfg.min_selling_limit
        )
        return np.clip(np.concatenate([p, s]) * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        cfg = self.cfg
        raw = np.concatenate(
            [
                [self.days_prior / cfg.horizon],
                (self.night_prices - cfg.min_night_price)
                / (cfg.max_night_price - cfg.min_night_price),
                (self.limits - cfg.min_selling_limit)
                / (cfg.max_selling_limit - cfg.min_selling_limit),
                self.bookings_on_hand() / cfg.max_selling_limit,
                self.showups_usual / cfg.capacity,
                self.requests / self._peak_arrivals,
            ]
        )
        return np.clip(2.0 * raw - 1.0, -1.5, 1.5).astype(np.float32)

    def potential(self) -> float:
        """Minus the end-of-season charge if selling stopped now, at the usual keep rate."""
        cfg = self.cfg
        m = self.showups_usual
        return -float(
            cfg.unsold_cost * np.maximum(cfg.capacity - m, 0.0).sum()
            + cfg.denied_cost * np.maximum(m - cfg.capacity, 0.0).sum()
        )

    def end_costs(self) -> tuple[np.ndarray, np.ndarray]:
        """Empty seats and denied admissions per night, from expected show-ups."""
        cap = float(self.cfg.capacity)
        return np.maximum(cap - self.showups, 0.0), np.maximum(self.showups - cap, 0.0)

    def _info(self) -> dict[str, Any]:
        unsold, denied = self.end_costs()
        cfg = self.cfg
        return {
            "days_prior": int(self.days_prior),
            "revenue": float(self.revenue),
            "night_prices": self.night_prices.copy(),
            "limits": self.limits.copy(),
            "requests": self.requests.copy(),
            "accepted": self.accepted.copy(),
            "showups": self.showups.copy(),
            "unsold": unsold,
            "denied": denied,
            "score": float(
                self.revenue - cfg.unsold_cost * unsold.sum() - cfg.denied_cost * denied.sum()
            ),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=None if seed is None else seed + self.seed_offset)
        self._reset_state()
        self.season = Season.draw(self.cfg, self.np_random)
        self.utility = pass_utility(self.cfg, self.season.night_appeal)
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray):
        cfg = self.cfg
        self.night_prices, self.limits = self.unscale_action(action)
        potential_before = self.potential()
        headroom = np.maximum(self.limits - self.bookings_on_hand(), 0.0)
        self.days_prior -= 1
        d = self.days_prior

        self.on_sale = (self.A.T @ (headroom < 1.0)) == 0
        probs = choice_probs(self.utility, self.prices, cfg.beta(d), self.on_sale)
        mean_arrivals = cfg.market_size * self.season.market_mult * self.arrival_weights[d]
        n = int(self.np_random.poisson(mean_arrivals))
        self.requests = self.np_random.multinomial(n, np.append(probs, 1.0 - probs.sum()))[
            :-1
        ].astype(np.float64)

        per_night = self.A @ self.requests
        fill = np.where(per_night > headroom, headroom / np.maximum(per_night, 1e-9), 1.0)
        scale = np.array([fill[list(p)].min() for p in cfg.pass_list])
        self.accepted = np.floor(self.requests * scale)
        self.bookings[d] += self.accepted

        step_revenue = float(self.accepted @ self.prices)
        self.revenue += step_revenue
        seats = self.A @ self.accepted
        s = self.season
        self.showups += seats * cfg.keep_rate(d, s.cancel_rho, s.noshow)
        self.showups_usual += seats * cfg.keep_rate(d, cfg.cancel_rho, cfg.noshow)

        reward = step_revenue
        terminated = self.days_prior <= 0
        if terminated:
            unsold, denied = self.end_costs()
            reward -= cfg.unsold_cost * unsold.sum() + cfg.denied_cost * denied.sum()
        if cfg.shape_reward:
            reward += (0.0 if terminated else self.potential()) - potential_before
        return self._get_obs(), float(reward * cfg.revenue_scale), terminated, False, self._info()
