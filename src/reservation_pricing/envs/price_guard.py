"""Gymnasium wrapper: keep a joint or price-only action, then constrain price."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from reservation_pricing.controls.price_monotone import MonotonePriceControl


class MonotonePriceEnv(gym.Wrapper):
    """Outermost wrapper. Reads ``action[0]`` so 1D and 2D actions both work.

    Unscales the price, lets ``MonotonePriceControl.apply`` clamp it (project
    mode) or report the violation (penalty mode), re-scales, and forwards the
    rest of the action unchanged. Penalty mode subtracts
    ``penalty * (violation_dollars / price_span)`` from the returned reward
    and does not clamp, so the frozen ``env.reward`` block stays untouched.

    ``reset`` clears the controller's last charged price. ``OversellGuardEnv``
    does not, and copying that would pin the next episode to the previous close.
    The charged price is taken from ``info["price"]`` after the inner step.
    """

    def __init__(self, env: gym.Env, control: MonotonePriceControl) -> None:
        super().__init__(env)
        self.control = control

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name not in ("_get_obs", "_raw_obs", "_info"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _unscale_price(self, action: np.ndarray) -> tuple[float, np.ndarray]:
        env = self.unwrapped
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        span = float(env.max_price - env.min_price)
        price = float(env.min_price + (a[0] + 1.0) * 0.5 * span)
        return price, a

    def _scale_price(self, price: float) -> float:
        env = self.unwrapped
        span = max(float(env.max_price - env.min_price), 1e-6)
        scaled = 2.0 * (float(price) - float(env.min_price)) / span - 1.0
        return float(np.clip(scaled, -1.0, 1.0))

    def _forward_action(self, action: np.ndarray, price: float) -> np.ndarray:
        _policy_price, scaled = self._unscale_price(action)
        scaled = scaled.copy()
        scaled[0] = self._scale_price(price)
        width = int(np.asarray(action, dtype=np.float64).reshape(-1).size)
        out = scaled[:1] if width == 1 else scaled
        return np.asarray(out, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        policy_price, _scaled = self._unscale_price(action)
        executed = float(self.control.apply(self.env, policy_price))
        obs, reward, terminated, truncated, info = self.env.step(
            self._forward_action(action, executed)
        )
        info = dict(info)
        charged = float(info["price"])
        violation = float(self.control.last_violation)
        if self.control.mode == "penalty" and violation > 0.0:
            span = max(float(self.unwrapped.max_price - self.unwrapped.min_price), 1e-6)
            reward = float(reward) - self.control.penalty * (violation / span)
        self.control.note_executed(charged)
        info["price_monotone"] = True
        info["price_monotone_floor"] = self.control.last_floor
        info["price_monotone_projected"] = bool(self.control.last_projected)
        info["price_monotone_violation"] = violation
        info["policy_price"] = float(policy_price)
        return obs, float(reward), terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        self.control.reset_episode()
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["price_monotone"] = True
        info["price_monotone_floor"] = None
        info["price_monotone_projected"] = False
        info["price_monotone_violation"] = 0.0
        return obs, info
