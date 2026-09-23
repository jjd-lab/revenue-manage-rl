"""Night-to-night variation and the operator-view switch, wired through make_env."""

from __future__ import annotations

import numpy as np

from reservation_pricing.config import load_config
from reservation_pricing.demand.night_varying import NightVaryingDemand
from reservation_pricing.demand.protocol import decision_model, features_from_state
from reservation_pricing.envs import make_env
from reservation_pricing.envs.operator_view import OperatorViewEnv

CONFIG = "configs/experiment_uncertain_cu200_sac.yaml"
ACTION = np.array([0.2, 1.0], dtype=np.float32)


def _rollout(cfg, seed=3):
    env = make_env(cfg, use_held_out=True)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(ACTION)
        trace.append(np.append(obs, [reward, info["true_revenue"], info["remain_inv"]]))
        done = terminated or truncated
    return np.concatenate(trace)


def test_switched_off_variation_leaves_the_default_night_unchanged():
    base = load_config()
    off = load_config()
    off["demand"]["night_variation"] = {"level_sd": 0.0, "elasticity_sd": 0.0}
    off["env"]["cancel_rho_night_std"] = 0.0
    np.testing.assert_array_equal(_rollout(base), _rollout(off))


def test_draws_repeat_for_a_seed_and_differ_between_seeds():
    cfg = load_config(CONFIG)
    draws = []
    for seed in (0, 0, 1):
        u = make_env(cfg, use_held_out=True)
        u.reset(seed=seed)
        u = u.unwrapped
        draws.append((u.demand_model.level, u.demand_model.elasticity, u.cancel_rho_night_shift))
    assert draws[0] == draws[1]
    assert draws[0] != draws[2]


def test_decision_code_keeps_the_usual_model():
    env = make_env(load_config(CONFIG), use_held_out=True)
    env.reset(seed=0)
    u = env.unwrapped
    assert isinstance(u.demand_model, NightVaryingDemand)
    forecast = decision_model(u)
    assert forecast is u.demand_model.inner
    feats = features_from_state(days_prior=50, dow=u.dow, month=u.month)
    assert u.demand_model.level != 1.0
    assert u.demand_model.predict_mean(feats, 100.0) != forecast.predict_mean(feats, 100.0)


def test_operator_view_switch_wraps_the_bare_env():
    env = make_env(load_config(CONFIG), use_held_out=True)
    assert isinstance(env, OperatorViewEnv)
    assert env.env is env.unwrapped
