"""OperatorViewEnv: decision code sees the operator's show-up estimate, not the env's."""

from __future__ import annotations

import numpy as np
import pytest

from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.envs.operator_view import OperatorViewEnv

ACTION = np.array([0.0, 1.0], dtype=np.float32)


def _pair(cfg, overrides=None, seed=4):
    truth = make_env(cfg, use_held_out=True)
    view = OperatorViewEnv(make_env(cfg, use_held_out=True), overrides)
    return truth, view, truth.reset(seed=seed)[0], view.reset(seed=seed)[0]


def test_with_no_noise_and_true_rates_the_view_is_the_truth():
    cfg = load_config()
    cfg["env"]["noshow_noise_std"] = 0.0
    cfg["env"]["cancel_rho_noise_std"] = 0.0
    truth, view, a, b = _pair(cfg)
    np.testing.assert_allclose(a, b, atol=1e-6)
    done = False
    while not done:
        a, _r, terminated, truncated, info_a = truth.step(ACTION)
        b, _r, _t, _tr, info_b = view.step(ACTION)
        np.testing.assert_allclose(a, b, atol=1e-5)
        assert info_a == info_b
        done = terminated or truncated
    assert view.cumulative_mat_boh == pytest.approx(truth.cumulative_mat_boh, rel=1e-9)


def test_only_the_show_up_slots_differ_and_info_is_untouched():
    truth, view, _a, _b = _pair(load_config())
    for _ in range(40):
        a, _r, _t, _tr, info_a = truth.step(ACTION)
        b, _r, _t, _tr, info_b = view.step(ACTION)
    assert info_a == info_b
    assert np.flatnonzero(~np.isclose(a, b, atol=1e-6)).tolist() == [6, 7]
    assert view.remain_inv == pytest.approx(10_000 - view.cumulative_mat_boh)


def test_a_higher_assumed_no_show_rate_expects_fewer_show_ups():
    cfg = load_config()
    _t, low, _a, _b = _pair(cfg, {"noshow_base": 0.12})
    _t, high, _a, _b = _pair(cfg, {"noshow_base": 0.20})
    for _ in range(30):
        low.step(ACTION)
        high.step(ACTION)
    assert high.cumulative_mat_boh < low.cumulative_mat_boh


def test_unknown_keep_parameter_is_rejected():
    with pytest.raises(ValueError):
        OperatorViewEnv(make_env(load_config()), {"noshow_rate": 0.2})
