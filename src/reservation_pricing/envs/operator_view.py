"""Gymnasium wrapper: show decision code the operator's show-up count, not the env's.

``ReservationEnv`` keeps ``cumulative_mat_boh`` (expected show-ups so far) and
``remain_inv`` from the *true* cancellation and no-show process, including the
night's realized no-show draw. Both reach the learned policies through the
observation and the oversell cap through ``getattr``. An operator sees neither:
it knows the bookings it took and has to estimate how many will show up.

This wrapper sits directly on ``ReservationEnv`` and replaces both with that
estimate, ``bookings taken x estimate_keep_rate``, everywhere decision code
looks: observation slots 6 and 7, and the two attributes. ``keep_overrides``
sets the cancellation or no-show parameters the estimate uses, so the operator
can be wrong. The env still generates the night from the truth, and ``info``
is untouched, so scoring is unchanged. Retrains that must not see the true
count (EXPERIMENT_LOG §16, §17) switch it on with ``env.operator_view``.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np

from reservation_pricing.controls.selling_limit import estimate_keep_rate

KEEP_PARAMS = (
    "noshow_base",
    "noshow_dow_coef",
    "noshow_month_coef",
    "cancel_rho_weekday",
    "cancel_rho_weekend",
    "cancel_lambda",
)
_SHOW_UPS, _REMAIN = 6, 7


class OperatorViewEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, keep_overrides: Optional[dict[str, float]] = None) -> None:
        super().__init__(env)
        overrides = dict(keep_overrides or {})
        unknown = set(overrides) - set(KEEP_PARAMS)
        if unknown:
            raise ValueError(f"unknown keep parameters: {sorted(unknown)}")
        for name in KEEP_PARAMS:
            setattr(self, name, overrides.get(name, getattr(env.unwrapped, name)))
        self._estimate = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    @property
    def cumulative_mat_boh(self) -> float:
        return self._estimate

    @property
    def remain_inv(self) -> float:
        return float(self.env.unwrapped.capacity) - self._estimate

    def _view(self, obs: np.ndarray) -> np.ndarray:
        u = self.env.unwrapped
        raw = u._raw_obs()
        raw[_SHOW_UPS] = self.cumulative_mat_boh
        raw[_REMAIN] = self.remain_inv
        if not u.normalize_obs:
            return raw
        span = np.maximum(u._obs_high - u._obs_low, 1e-6)
        scaled = 2.0 * (raw - u._obs_low) / span - 1.0
        return np.clip(scaled, -1.5, 1.5).astype(obs.dtype)

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._estimate = 0.0
        return self._view(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # ``step`` has already moved ``days_prior`` to the day this booking was
        # made, which is the day the env's own keep rate is taken at.
        self._estimate += float(info["accepted_booking"]) * estimate_keep_rate(self)
        return self._view(obs), reward, terminated, truncated, info


__all__ = ["KEEP_PARAMS", "OperatorViewEnv"]
