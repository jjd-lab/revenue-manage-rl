"""The score-aligned reward and the night-type observation (EXPERIMENT_LOG §17)."""

from __future__ import annotations

import numpy as np
import pytest

from reservation_pricing.config import load_config
from reservation_pricing.demand.protocol import features_from_state
from reservation_pricing.envs import make_env
from reservation_pricing.envs.reservation import NIGHT_BASE_MAX

CONFIG = "configs/experiment_score_sac.yaml"
ACTION = np.array([0.2, 1.0], dtype=np.float32)


def _rollout(cfg, seed):
    env = make_env(cfg, use_held_out=True)
    obs, _ = env.reset(seed=seed)
    observations, ret, done = [obs], 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(ACTION)
        observations.append(obs)
        ret += reward
        done = terminated or truncated
    return np.stack(observations), ret, info, env.unwrapped


def test_switches_change_no_dynamics_and_only_append_two_slots():
    off = load_config(CONFIG)
    off["env"].update(night_features=False, undersell_on_soft=True)
    on = load_config(CONFIG)
    obs_off, _, info_off, _ = _rollout(off, seed=4)
    obs_on, _, info_on, _ = _rollout(on, seed=4)
    assert obs_off.shape[1] == 27 and obs_on.shape[1] == 29
    np.testing.assert_array_equal(obs_on[:, :27], obs_off)
    assert info_on["true_revenue"] == info_off["true_revenue"]


def test_night_features_come_from_the_forecast_not_the_night_draw():
    cfg = load_config("configs/experiment_uncertain_cu200_sac.yaml")
    cfg["env"]["night_features"] = True
    env = make_env(cfg, use_held_out=True)
    obs, _ = env.reset(seed=0)
    u = env.unwrapped
    assert u.demand_model.level != 1.0
    base, soft = u._forecast_night()
    feats = features_from_state(days_prior=50, dow=u.dow, month=u.month)
    assert base == u.forecast_model.predict_base(feats)
    assert base != u.demand_model.predict_mean(feats, 100.0)
    assert obs[-2] == pytest.approx(2.0 * base / NIGHT_BASE_MAX - 1.0, abs=1e-6)
    assert obs[-1] == (1.0 if soft else -1.0)


def test_unsold_seats_are_charged_on_peak_nights_only():
    cfg = load_config(CONFIG)
    env_cfg = cfg["env"]
    # Held-out seed 1 is a soft night and seed 4 a peak night (see test_scripts).
    for seed, soft in ((1, True), (4, False)):
        _, ret, info, u = _rollout(cfg, seed)
        assert u._forecast_soft is soft
        remain = info["remain_inv"]
        unsold = 0.0 if soft else max(remain, 0.0)
        expected = (
            info["true_revenue"] * env_cfg.get("revenue_scale", 1e-4)
            - env_cfg["undersell_penalty"] * unsold / 10_000
            - env_cfg["oversell_penalty"] * max(-remain, 0.0) / 10_000
        )
        assert abs(ret - expected) < 1e-6
