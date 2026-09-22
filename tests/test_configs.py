"""Every shipped config loads, builds an env, and keeps observations in the space."""

from __future__ import annotations

from pathlib import Path

import pytest

from reservation_pricing.config import load_config
from reservation_pricing.config.load import PACKAGED_DEFAULT
from reservation_pricing.envs import make_env

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yaml")) + [
    ROOT / "runs" / "soft_aware_eval" / "config_soft_aware.yaml"
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads_and_builds_an_env(path):
    env = make_env(load_config(path))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_packaged_default_matches_the_checkout():
    """A non-editable install reads the packaged copy; it must not drift."""
    assert PACKAGED_DEFAULT.read_bytes() == (ROOT / "configs" / "default.yaml").read_bytes()


def test_default_config_falls_back_to_the_packaged_copy(monkeypatch):
    """Outside a checkout (pip install .) the packaged copy is the base config."""
    from reservation_pricing.config import load as load_mod

    monkeypatch.setattr(load_mod, "package_root", lambda: None)
    assert load_mod.default_config_path() == PACKAGED_DEFAULT
    assert load_mod.load_config()["env"]["capacity"] == 10000


def test_default_config_carries_the_full_schema_with_everything_off():
    cfg = load_config()
    assert set(cfg["control"]) >= {"price_only", "selling_limit", "early_promo", "safe_sl", "mpc"}
    assert cfg["control"]["price_only"] is False
    for block in ("early_promo", "safe_sl", "mpc"):
        assert cfg["control"][block]["enabled"] is False
    assert cfg["env"]["reward"]["pace_reward"] is False
    assert cfg["env"]["reward"]["soft_day_upweight"] is False


def test_observation_stays_inside_the_space_through_an_episode():
    """The normalisation in ReservationEnv._get_obs is what the checkpoints were trained on."""
    env = make_env(load_config())
    obs, _ = env.reset(seed=3)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        done = terminated or truncated
