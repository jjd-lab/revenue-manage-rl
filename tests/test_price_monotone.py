"""Monotone price constraint: projection, the reset exemption, and the promo/MPC ban."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from reservation_pricing.config import load_config, validate_config
from reservation_pricing.controls.price_monotone import MonotonePriceControl, get_price_monotone
from reservation_pricing.envs import MonotonePriceEnv, make_env

ROOT = Path(__file__).resolve().parents[1]


def _env(**overrides):
    state = dict(min_price=80.0, max_price=120.0, days_prior=50)
    state.update(overrides)
    return SimpleNamespace(**state)


def test_direction_up_projects_a_markdown_and_leaves_a_rise_alone():
    control = MonotonePriceControl(enabled=True, direction="up", mode="project", tolerance=0)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 90.0) == pytest.approx(110.0)
    assert control.last_projected
    assert control.last_violation == pytest.approx(20.0)
    assert control.last_floor == pytest.approx(110.0)

    assert control.apply(env, 115.0) == pytest.approx(115.0)
    assert not control.last_projected
    assert control.last_violation == 0.0


def test_first_step_is_exempt_for_both_directions():
    env = _env()
    for direction in ("up", "down"):
        control = MonotonePriceControl(enabled=True, direction=direction, mode="project")
        assert control.last_executed_price is None
        assert control.apply(env, 100.0) == pytest.approx(100.0)
        assert not control.last_projected
        assert control.last_floor is None


def test_direction_down_ceilings_at_the_last_charged_price():
    """A placeholder of $80 would pin the opener; a real last price ceilings it."""
    control = MonotonePriceControl(enabled=True, direction="down", mode="project")
    env = _env()
    control.last_executed_price = 80.0
    assert control.apply(env, 100.0) == pytest.approx(80.0)
    assert control.last_projected


def test_tolerance_absorbs_a_small_markdown():
    control = MonotonePriceControl(enabled=True, direction="up", mode="project", tolerance=2.0)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 109.0) == pytest.approx(109.0)
    assert not control.last_projected
    assert control.last_violation == 0.0
    assert control.apply(env, 100.0) == pytest.approx(108.0)


def test_max_step_caps_the_rise():
    control = MonotonePriceControl(enabled=True, direction="up", mode="project", max_step=5.0)
    env = _env()
    control.last_executed_price = 100.0
    assert control.apply(env, 110.0) == pytest.approx(105.0)
    assert control.last_projected
    assert control.last_violation == pytest.approx(5.0)


def test_window_uses_the_decision_day_not_the_raw_counter():
    """days_prior 11 is decision day 10, which is inside a window of 10.
    days_prior 12 is decision day 11, which is outside it.
    """
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="project", apply_when_days_prior_le=10
    )
    control.last_executed_price = 110.0
    assert control.apply(_env(days_prior=12), 90.0) == pytest.approx(90.0)
    assert not control.last_projected
    assert control.apply(_env(days_prior=11), 90.0) == pytest.approx(110.0)
    assert control.last_projected


def test_penalty_mode_reports_the_violation_and_does_not_clamp():
    control = MonotonePriceControl(enabled=True, direction="up", mode="penalty", penalty=10)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 90.0) == pytest.approx(90.0)
    assert not control.last_projected
    assert control.last_violation == pytest.approx(20.0)


def test_disabled_and_missing_blocks_build_nothing():
    assert get_price_monotone(load_config()["control"]) is None
    assert get_price_monotone({"enabled": False, "direction": "up"}) is None
    assert get_price_monotone({}) is None


class _PriceBox(gym.Env):
    """Joint box whose charged price is exactly the unscaled action, reward 1."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.min_price = 80.0
        self.max_price = 120.0
        self.price = self.min_price

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.min_price
        return np.zeros(1, dtype=np.float32), {"price": float(self.price)}

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        self.price = self.min_price + (a + 1.0) * 0.5 * (self.max_price - self.min_price)
        return np.zeros(1, dtype=np.float32), 1.0, False, False, {"price": float(self.price)}


def test_wrapper_penalty_is_subtracted_from_the_returned_reward():
    control = MonotonePriceControl(enabled=True, direction="up", mode="penalty", penalty=10)
    env = MonotonePriceEnv(_PriceBox(), control)
    env.reset()
    _, reward, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert reward == pytest.approx(1.0)
    assert info["price"] == pytest.approx(120.0)
    _, reward, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    # $40 markdown / $40 span * penalty 10.
    assert info["price"] == pytest.approx(80.0)
    assert info["price_monotone_violation"] == pytest.approx(40.0)
    assert reward == pytest.approx(1.0 - 10.0)
    assert info["policy_price"] == pytest.approx(80.0)


def test_default_config_does_not_wrap():
    env = make_env(load_config())
    assert not isinstance(env, MonotonePriceEnv)


def test_project_mode_on_a_shipped_config_holds_the_charged_price():
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_sac.yaml")
    env = make_env(cfg)
    assert isinstance(env, MonotonePriceEnv)
    env.reset(seed=0)
    _, _, _, _, first = env.step(np.array([1.0, 0.0], dtype=np.float32))
    _, _, _, _, second = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert first["price"] > env.unwrapped.min_price + 1.0
    assert second["price"] + 1e-4 >= first["price"]
    assert second["price_monotone_projected"]
    assert second["policy_price"] < first["price"] - 1.0
    assert "price_monotone_floor" in second


def test_reset_clears_the_previous_episodes_close():
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_sac.yaml")
    env = make_env(cfg)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["price"] == pytest.approx(env.unwrapped.min_price, abs=1e-3)
    assert not info["price_monotone_projected"]


def test_project_mode_accepts_a_price_only_action():
    cfg = load_config(ROOT / "configs" / "experiment_price_only_ppo.yaml")
    cfg["control"]["price_monotone"]["enabled"] = True
    cfg["control"]["price_monotone"]["mode"] = "project"
    env = make_env(cfg)
    assert env.action_space.shape == (1,)
    env.reset(seed=0)
    _, _, _, _, first = env.step(np.array([1.0], dtype=np.float32))
    _, _, _, _, second = env.step(np.array([-1.0], dtype=np.float32))
    assert second["price"] + 1e-4 >= first["price"]


def test_validate_rejects_monotone_combined_with_promo_or_mpc():
    promo = load_config(ROOT / "configs" / "experiment_price_only_promo_ppo.yaml")
    promo["control"]["price_monotone"]["enabled"] = True
    with pytest.raises(ValueError, match="price_monotone"):
        validate_config(promo)

    alias = load_config()
    alias["control"]["promo"] = {"enabled": True, "mode": "set", "promo_price": 80.0}
    alias["control"]["price_monotone"]["enabled"] = True
    with pytest.raises(ValueError, match="promo"):
        validate_config(alias)

    mpc = load_config(ROOT / "configs" / "experiment_pace_mpc.yaml")
    mpc["control"]["price_monotone"]["enabled"] = True
    with pytest.raises(ValueError, match="price_monotone"):
        validate_config(mpc)

    # Disabled blocks coexist. The default config is that case.
    validate_config(load_config())
