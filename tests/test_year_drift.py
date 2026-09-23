"""A year shifted off the forecast, and the planner's pickup adjustment (EXPERIMENT_LOG §18)."""

from __future__ import annotations

import numpy as np

from reservation_pricing.baselines import dp_policy
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env

ACTION = np.array([0.2, 1.0], dtype=np.float32)


def _world(shift):
    cfg = load_config()
    cfg["demand"]["night_variation"] = {"level_shift": shift}
    return cfg


def _run(cfg, policy=None, seed=4):
    env = make_env(cfg, use_held_out=True)
    obs, _ = env.reset(seed=seed)
    state, trace, done = {}, [], False
    while not done:
        action = ACTION if policy is None else policy(obs, env, state)
        obs, _r, terminated, truncated, info = env.step(action)
        trace.append((info["accepted_booking"], info["true_revenue"]))
        done = terminated or truncated
    return np.array(trace), state, env.unwrapped


def test_an_unshifted_year_is_the_default_night():
    default, _, _ = _run(load_config())
    unshifted, _, _ = _run(_world(1.0))
    np.testing.assert_array_equal(default, unshifted)


def test_a_shift_moves_every_night_and_not_the_forecast():
    for seed in (0, 1, 4):
        _, _, u = _run(_world(1.2), seed=seed)
        assert u.demand_model.level == 1.2
        assert u.forecast_model is u.demand_model.inner


def test_pickup_reads_the_shift_from_booking_requests():
    _, state, _ = _run(_world(1.2), dp_policy(pickup_days=(70, 50, 30)))
    assert abs(state["dp_seen"] / state["dp_expected"] - 1.2) < 0.05
    _, plain, _ = _run(_world(1.2), dp_policy())
    assert "dp_seen" not in plain
