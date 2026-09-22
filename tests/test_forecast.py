"""Decision code consults the forecast; the environment always generates from truth."""

from __future__ import annotations

import numpy as np

from reservation_pricing.config import deep_merge, load_config
from reservation_pricing.demand.protocol import decision_model
from reservation_pricing.demand.registry import get_demand_model
from reservation_pricing.envs import make_env


def _linear_stub():
    return get_demand_model({"kind": "linear_legacy", "demand_noise_std": 0.0})


def test_forecast_defaults_to_the_true_model():
    """Every published table assumes decisions see the generating model."""
    env = make_env(load_config(), use_held_out=True).unwrapped
    assert decision_model(env) is env.demand_model


def test_injected_forecast_separates_decisions_from_truth():
    stub = _linear_stub()
    env = make_env(load_config(), use_held_out=True, forecast_model=stub).unwrapped
    assert decision_model(env) is stub
    assert env.demand_model is not stub
    assert env.demand_model.kind == "tree_elastic"


def test_forecast_block_builds_a_separate_model_and_can_be_disabled():
    cfg = load_config()
    on = deep_merge(cfg, {"demand": {"forecast": {"kind": "linear_legacy"}}})
    env_on = make_env(on, use_held_out=True).unwrapped
    assert decision_model(env_on).kind == "linear_legacy"
    assert env_on.demand_model.kind == "tree_elastic"

    off = deep_merge(cfg, {"demand": {"forecast": {"enabled": False, "kind": "linear_legacy"}}})
    env_off = make_env(off, use_held_out=True).unwrapped
    assert decision_model(env_off) is env_off.demand_model


def test_a_wrong_forecast_does_not_change_the_world():
    """The env's own dynamics must be identical whatever the forecast says."""
    cfg = load_config()
    truth, wrong = [], []
    for forecast in (None, _linear_stub()):
        env = make_env(cfg, use_held_out=True, forecast_model=forecast)
        env.reset(seed=3)
        out = []
        for _ in range(12):
            obs, r, term, trunc, info = env.step(np.array([0.2, 0.4], dtype=np.float32))
            out.append((info["accepted_booking"], info["true_revenue"], info["remain_inv"]))
            if term or trunc:
                break
        (truth if forecast is None else wrong).append(out)
    assert truth[0] == wrong[0]


def test_the_myopic_baseline_prices_off_the_forecast():
    """Its price comes from predict_mean, so a wrong forecast must move it."""
    from reservation_pricing.baselines import myopic_greedy_policy

    cfg = load_config()
    policy = myopic_greedy_policy(n_grid=21)
    prices = []
    for forecast in (None, _linear_stub()):
        env = make_env(cfg, use_held_out=True, forecast_model=forecast)
        obs, _ = env.reset(seed=4)
        prices.append(float(policy(obs, env, {})[0]))
    assert prices[0] != prices[1]
