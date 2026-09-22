"""Gymnasium wrapper: cap a joint policy's selling limit to prevent oversell."""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np

from reservation_pricing.controls.oversell_cap import OversellCap


class OversellGuardEnv(gym.Wrapper):
    """Keep the joint 2D action space; after the agent acts, project SL down.

    The counterpart to ``envs/price_only.py``: that one adapts the env for a
    policy that outputs price alone, this one leaves the policy's action shape
    untouched and constrains what it asked for.

    Underlying ``ReservationEnv`` still receives a joint ``(price, SL)`` action
    in ``[-1, 1]^2``. This wrapper unscales, applies ``OversellCap.project``,
    re-scales, and forwards. Info keys: ``safe_sl_cap``, ``safe_sl_projected``,
    ``policy_sl``, ``selling_limit`` (executed, from env info).
    """

    def __init__(self, env: gym.Env, projector: OversellCap) -> None:
        super().__init__(env)
        self.projector = projector

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") and name not in ("_get_obs", "_raw_obs", "_info"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _unscale(self, action: np.ndarray) -> tuple[float, float]:
        env = self.unwrapped
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        price = env.min_price + (a[0] + 1.0) * 0.5 * (env.max_price - env.min_price)
        sl = env.min_selling_limit + (a[1] + 1.0) * 0.5 * (
            env.max_selling_limit - env.min_selling_limit
        )
        return float(price), float(sl)

    def _scale(self, price: float, sl: float) -> np.ndarray:
        env = self.unwrapped
        p_span = max(env.max_price - env.min_price, 1e-6)
        s_span = max(env.max_selling_limit - env.min_selling_limit, 1e-6)
        p = 2.0 * (price - env.min_price) / p_span - 1.0
        s = 2.0 * (sl - env.min_selling_limit) / s_span - 1.0
        return np.array(
            [float(np.clip(p, -1.0, 1.0)), float(np.clip(s, -1.0, 1.0))],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        price, policy_sl = self._unscale(action)
        safe_sl = float(self.projector.project(self.env, price, policy_sl))
        joint = self._scale(price, safe_sl)
        obs, reward, terminated, truncated, info = self.env.step(joint)
        info = dict(info)
        info["safe_sl"] = True
        info["safe_sl_kind"] = getattr(self.projector, "cap_kind", None)
        info["safe_sl_cap"] = self.projector.last_cap
        info["safe_sl_projected"] = bool(self.projector.last_projected)
        info["policy_sl"] = float(policy_sl)
        info["rl_price"] = float(price)
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["safe_sl"] = True
        info["safe_sl_projected"] = False
        return obs, info
