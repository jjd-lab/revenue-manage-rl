"""Gymnasium wrapper: the DP planner proposes, the agent corrects.

Each day the planner (``baselines.dp.dp_policy``) picks a price and selling
limit from the forecast. The agent's action in ``[-1, 1]^2`` is a correction
added to the planner's scaled action, bounded by ``price_scale`` and
``limit_scale``: 0.5 and 0.2 are ±$10 and ±500 seats on the default bounds. An
agent that outputs zero is the planner.

The observation appends three slots to the inner one: the planner's proposed
price and limit (scaled) and the pickup ratio, booking requests so far over
what the forecast expected at the prices charged, mapped from [0, 2] to [-1, 1].
Everything here reads the forecast (``decision_model``), never the night's draw.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reservation_pricing.baselines.dp import dp_policy
from reservation_pricing.demand.protocol import decision_model


class ResidualPlannerEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, price_scale: float = 0.5, limit_scale: float = 0.2) -> None:
        super().__init__(env)
        self.scales = np.array([price_scale, limit_scale], dtype=np.float32)
        self._planner = dp_policy()
        inner = env.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([inner.low, -np.ones(3, dtype=np.float32)]),
            high=np.concatenate([inner.high, np.ones(3, dtype=np.float32)]),
            dtype=np.float32,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _propose(self, obs: np.ndarray) -> np.ndarray:
        self._base = np.asarray(self._planner(obs, self.env, self._state), dtype=np.float32)
        ratio = self._seen / self._expected if self._expected > 0 else 1.0
        extra = np.array([*self._base, np.clip(ratio / 2.0, 0.0, 1.0) * 2.0 - 1.0])
        return np.concatenate([obs, extra.astype(np.float32)])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._state: dict = {}
        self._seen = self._expected = 0.0
        return self._propose(obs), info

    def step(self, action: np.ndarray):
        delta = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        executed = np.clip(self._base + self.scales * delta, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        u = self.env.unwrapped
        self._seen += float(u.gross_pickup)
        self._expected += float(decision_model(u).predict_mean(u.demand_features(), u.price))
        info = dict(info)
        info["planner_action"] = self._base.tolist()
        if terminated or truncated:
            return (
                np.concatenate([obs, np.zeros(3, dtype=np.float32)]),
                reward,
                terminated,
                truncated,
                info,
            )
        return self._propose(obs), reward, terminated, truncated, info


__all__ = ["ResidualPlannerEnv"]
