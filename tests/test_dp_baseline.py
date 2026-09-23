"""The dynamic-programming baseline: solver correctness and env wiring."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from reservation_pricing.baselines import dp_policy
from reservation_pricing.baselines.dp import solve_dp
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.metrics import SoftAwareConfig, _mid_horizon_base, classify_soft


def _brute_force(mean, keep, prices, capacity, unsold, oversold, fracs):
    best = -np.inf
    horizon, n_prices = mean.shape
    choices = list(itertools.product(range(n_prices), range(len(fracs))))
    for plan in itertools.product(choices, repeat=horizon):
        m = revenue = 0.0
        for t, (j, f) in enumerate(plan):
            cap = np.inf if fracs[f] >= 1.0 else fracs[f] * mean[t, j]
            accepted = min(mean[t, j], cap)
            revenue += prices[j] * accepted
            m += keep[t] * accepted
        total = revenue - unsold * max(capacity - m, 0.0) - oversold * max(m - capacity, 0.0)
        best = max(best, total)
    return best


def test_solver_matches_brute_force_without_noise():
    prices = np.array([80.0, 100.0, 120.0])
    mean = np.array([[50.0, 40.0, 30.0], [60.0, 50.0, 40.0], [30.0, 20.0, 10.0]])
    keep = np.array([0.5, 0.8, 1.0])
    fracs = (0.0, 0.5, 1.0)
    for unsold, oversold in ((200.0, 400.0), (0.0, 400.0), (200.0, 5.0)):
        value, _p, _c = solve_dp(
            mean,
            keep,
            prices,
            capacity=90.0,
            unsold_cost=unsold,
            oversold_cost=oversold,
            noise_std=0.0,
            m_grid=np.arange(0.0, 200.0, 0.5),
            cap_fractions=fracs,
        )
        expected = _brute_force(mean, keep, prices, 90.0, unsold, oversold, fracs)
        assert abs(value[0, 0] - expected) < 1e-6


def _quiet_config():
    cfg = load_config()
    cfg["demand"]["demand_noise_std"] = 0.0
    cfg["env"]["demand_noise_std"] = 0.0
    cfg["env"]["noshow_noise_std"] = 0.0
    cfg["env"]["cancel_rho_noise_std"] = 0.0
    return cfg


def _rollout(cfg, seed, policy):
    env = make_env(cfg, use_held_out=True)
    obs, _info = env.reset(seed=seed)
    state: dict = {}
    done = False
    while not done:
        obs, _r, terminated, truncated, info = env.step(policy(obs, env, state))
        done = terminated or truncated
    return env, state, info


def test_prediction_matches_rollout_when_noise_is_off():
    cfg = _quiet_config()
    soft_cfg = SoftAwareConfig.from_dict(cfg["eval"]["soft_aware"])
    policy = dp_policy(soft_cfg)
    for seed in (0, 1, 4):  # a December weekday, a soft June weekday, a December weekend
        env, state, info = _rollout(cfg, seed, policy)
        u = env.unwrapped
        soft = classify_soft(
            month=u.month, dow=u.dow, base_demand=_mid_horizon_base(env), cfg=soft_cfg
        )
        remain = float(info["remain_inv"])
        realized = float(info["true_revenue"]) - 400.0 * max(-remain, 0.0)
        if not soft:
            realized -= 200.0 * max(remain, 0.0)
        assert state["dp_unhonoured"] == 0
        assert abs(realized - state["dp_value"]) / abs(state["dp_value"]) < 0.005


def test_a_prohibitive_oversell_charge_leaves_no_denied_admission():
    cfg = _quiet_config()
    soft_cfg = SoftAwareConfig.from_dict(cfg["eval"]["soft_aware"])
    soft_cfg.lambda_peak = 1e6
    _env, _state, info = _rollout(cfg, 4, dp_policy(soft_cfg))
    assert float(info["remain_inv"]) >= -1.0


def test_planner_never_reads_the_env_show_up_count():
    cfg = load_config()
    soft_cfg = SoftAwareConfig.from_dict(cfg["eval"]["soft_aware"])
    runs = []
    for tamper in (False, True):
        env = make_env(cfg, use_held_out=True)
        obs, _info = env.reset(seed=4)
        policy, state, actions, done = dp_policy(soft_cfg), {}, [], False
        while not done:
            u = env.unwrapped
            true_count = u.cumulative_mat_boh
            if tamper:
                u.cumulative_mat_boh = 1e9
            action = policy(obs, env, state)
            u.cumulative_mat_boh = true_count
            actions.append(action)
            obs, _r, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
        runs.append(np.array(actions))
    np.testing.assert_array_equal(runs[0], runs[1])


def test_unknown_keep_parameter_is_rejected():
    with pytest.raises(ValueError):
        dp_policy(keep_overrides={"noshow_rate": 0.2})
