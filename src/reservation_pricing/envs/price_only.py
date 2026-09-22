"""Gymnasium wrapper: RL chooses price only; selling limit from a controller."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reservation_pricing.controls.selling_limit import SellingLimitController


class PriceOnlyWrapper(gym.Wrapper):
    """Expose a 1D price action; fill selling limit via ``sl_controller`` each step.

    Underlying ``ReservationEnv`` still expects a joint ``(price, SL)`` action in
    ``[-1, 1]^2``. This wrapper maps ``action[0]`` → price, asks the controller
    for SL given the unscaled price + env state, then forwards a joint action.

    Optional ``early_promo`` (``EarlyPromoController``)
    -------------------------------------------------
    After the RL price is unscaled and **before** SL + dynamics, if predicted
    price-unaware base demand is soft (below threshold) and
    ``days_prior >= apply_when_days_prior_ge``, the price is forced/clipped to
    the promo band. This is a deterministic post-process — the agent still
    outputs a 1D price; the override only affects the executed price (and thus
    the SL query). Info keys: ``early_promo_triggered``, ``early_promo_base``,
    ``rl_price``, ``price`` (executed).

    Optional ``mpc`` (``ShortHorizonPriceMPC``)
    ------------------------------------------
    After early_promo (if any), when soft or behind pace, replace price with a
    short-horizon grid-search optimum using ``predict_mean``. Info keys:
    ``mpc_triggered``, ``mpc_reason``, ``mpc_base``, ``mpc_price``.
    """

    def __init__(
        self,
        env: gym.Env,
        sl_controller: SellingLimitController,
        early_promo: Any = None,
        mpc: Any = None,
    ) -> None:
        super().__init__(env)
        self.sl_controller = sl_controller
        self.early_promo = early_promo
        self.mpc = mpc
        self.action_space = spaces.Box(
            low=-np.ones(1, dtype=np.float32),
            high=np.ones(1, dtype=np.float32),
            dtype=np.float32,
        )
        self._last_controller_sl: Optional[float] = None

    def __getattr__(self, name: str) -> Any:
        """Forward env attributes (Gymnasium 1.x wrappers no longer do this)."""
        # Avoid recursion on Wrapper internals
        if name.startswith("_") and name not in ("_get_obs", "_raw_obs", "_info"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _unscale_price(self, price_norm: float) -> float:
        env = self.unwrapped
        a = float(np.clip(price_norm, -1.0, 1.0))
        return float(env.min_price + (a + 1.0) * 0.5 * (env.max_price - env.min_price))

    def _scale_price(self, price: float) -> float:
        env = self.unwrapped
        span = max(env.max_price - env.min_price, 1e-6)
        return float(np.clip(2.0 * (price - env.min_price) / span - 1.0, -1.0, 1.0))

    def _scale_sl(self, sl: float) -> float:
        env = self.unwrapped
        span = max(env.max_selling_limit - env.min_selling_limit, 1e-6)
        return float(np.clip(2.0 * (sl - env.min_selling_limit) / span - 1.0, -1.0, 1.0))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        price_norm = float(np.clip(a[0], -1.0, 1.0))
        rl_price = self._unscale_price(price_norm)
        price = rl_price
        promo_triggered = False
        promo_base: Optional[float] = None
        # ReservationEnv.step() decrements days_prior *before* applying the limit
        # and discounts the day's bookings with cancel_fn(days_prior) afterwards, so
        # the day this action governs is days_prior - 1. Every controller is shown
        # that day, matching what the env will actually charge it against.
        if self.early_promo is not None:
            self.env.days_prior -= 1
            try:
                price = float(self.early_promo.apply(self.env, rl_price))
            finally:
                self.env.days_prior += 1
            promo_triggered = bool(getattr(self.early_promo, "last_triggered", False))
            promo_base = getattr(self.early_promo, "last_base", None)
            price_norm = self._scale_price(price)

        mpc_triggered = False
        mpc_reason: Optional[str] = None
        mpc_base: Optional[float] = None
        mpc_price: Optional[float] = None
        if self.mpc is not None:
            self.env.days_prior -= 1
            try:
                price = float(self.mpc.apply(self.env, price))
            finally:
                self.env.days_prior += 1
            mpc_triggered = bool(getattr(self.mpc, "last_triggered", False))
            mpc_reason = getattr(self.mpc, "last_reason", None)
            mpc_base = getattr(self.mpc, "last_base", None)
            if mpc_triggered:
                mpc_price = float(price)
            price_norm = self._scale_price(price)

        self.env.days_prior -= 1
        try:
            sl = float(self.sl_controller.compute(self.env, price))
        finally:
            self.env.days_prior += 1
        self._last_controller_sl = sl
        joint = np.array([price_norm, self._scale_sl(sl)], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(joint)
        info = dict(info)
        info["price_only"] = True
        info["controller_sl"] = sl
        info["sl_kind"] = getattr(self.sl_controller, "kind", None)
        info["rl_price"] = float(rl_price)
        info["early_promo_triggered"] = bool(promo_triggered)
        if promo_base is not None:
            info["early_promo_base"] = float(promo_base)
        info["mpc_triggered"] = bool(mpc_triggered)
        if mpc_reason is not None:
            info["mpc_reason"] = str(mpc_reason)
        if mpc_base is not None:
            info["mpc_base"] = float(mpc_base)
        if mpc_price is not None:
            info["mpc_price"] = float(mpc_price)
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["price_only"] = True
        info["sl_kind"] = getattr(self.sl_controller, "kind", None)
        info["early_promo_triggered"] = False
        info["mpc_triggered"] = False
        self._last_controller_sl = None
        return obs, info
