"""Monotone price constraint: the clamp, the ratchet, the reset exemption, the promo/MPC ban."""

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


def test_direction_up_clamps_a_markdown_and_leaves_a_rise_alone():
    control = MonotonePriceControl(enabled=True, direction="up", mode="clamp", tolerance=0)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 90.0) == pytest.approx(110.0)
    assert control.last_clamped
    assert control.last_violation == pytest.approx(20.0)
    assert control.last_floor == pytest.approx(110.0)

    assert control.apply(env, 115.0) == pytest.approx(115.0)
    assert not control.last_clamped
    assert control.last_violation == 0.0


def test_first_step_is_exempt_for_both_directions():
    env = _env()
    for direction in ("up", "down"):
        control = MonotonePriceControl(enabled=True, direction=direction, mode="clamp")
        assert control.last_executed_price is None
        assert control.apply(env, 100.0) == pytest.approx(100.0)
        assert not control.last_clamped
        assert control.last_floor is None


def test_direction_down_ceilings_at_the_last_charged_price():
    """A placeholder of $80 would pin the opener; a real last price ceilings it."""
    control = MonotonePriceControl(enabled=True, direction="down", mode="clamp")
    env = _env()
    control.last_executed_price = 80.0
    assert control.apply(env, 100.0) == pytest.approx(80.0)
    assert control.last_clamped


def test_tolerance_absorbs_a_small_markdown():
    control = MonotonePriceControl(enabled=True, direction="up", mode="clamp", tolerance=2.0)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 109.0) == pytest.approx(109.0)
    assert not control.last_clamped
    assert control.last_violation == 0.0
    assert control.apply(env, 100.0) == pytest.approx(108.0)


def test_max_step_caps_the_rise():
    control = MonotonePriceControl(enabled=True, direction="up", mode="clamp", max_step=5.0)
    env = _env()
    control.last_executed_price = 100.0
    assert control.apply(env, 110.0) == pytest.approx(105.0)
    assert control.last_clamped
    assert control.last_violation == pytest.approx(5.0)


def test_window_uses_the_decision_day_not_the_raw_counter():
    """days_prior 11 is decision day 10, which is inside a window of 10.
    days_prior 12 is decision day 11, which is outside it.
    """
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="clamp", apply_when_days_prior_le=10
    )
    control.last_executed_price = 110.0
    assert control.apply(_env(days_prior=12), 90.0) == pytest.approx(90.0)
    assert not control.last_clamped
    assert control.apply(_env(days_prior=11), 90.0) == pytest.approx(110.0)
    assert control.last_clamped


def test_penalty_mode_reports_the_violation_and_does_not_clamp():
    control = MonotonePriceControl(enabled=True, direction="up", mode="penalty", penalty=10)
    env = _env()
    control.last_executed_price = 110.0
    assert control.apply(env, 90.0) == pytest.approx(90.0)
    assert not control.last_clamped
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


def test_clamp_mode_on_a_shipped_config_holds_the_charged_price():
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    env = make_env(cfg)
    assert isinstance(env, MonotonePriceEnv)
    env.reset(seed=0)
    _, _, _, _, first = env.step(np.array([1.0, 0.0], dtype=np.float32))
    _, _, _, _, second = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert first["price"] > env.unwrapped.min_price + 1.0
    assert second["price"] + 1e-4 >= first["price"]
    assert second["price_monotone_clamped"]
    assert second["policy_price"] < first["price"] - 1.0
    assert "price_monotone_floor" in second


def test_reset_clears_the_previous_episodes_close():
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    env = make_env(cfg)
    env.reset(seed=0)
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
    assert info["price"] == pytest.approx(env.unwrapped.min_price, abs=1e-3)
    assert not info["price_monotone_clamped"]


def test_clamp_mode_accepts_a_price_only_action():
    cfg = load_config(ROOT / "configs" / "experiment_price_only_ppo.yaml")
    cfg["control"]["price_monotone"]["enabled"] = True
    cfg["control"]["price_monotone"]["mode"] = "clamp"
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


# --- reference: the difference between bounding a step and bounding a level ---


def _glide(control, env, asked=80.0, days=12):
    """Charge ``asked`` every day and return the executed path."""
    path = []
    for _ in range(days):
        out = control.apply(env, asked)
        control.note_executed(out)
        path.append(round(out, 2))
    return path


def test_reference_last_lets_tolerance_glide_without_limit():
    """The documented failure mode of ``reference: last``, pinned so it cannot return.

    A tolerance of $2 means "$2 below *yesterday*", and yesterday re-tracks each
    lowered price, so a "non-decreasing" price walks down $2/day forever.
    """
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="clamp", reference="last", tolerance=2.0
    )
    control.note_executed(110.0)
    path = _glide(control, _env())
    assert path == [108.0, 106.0, 104.0, 102.0, 100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0]


def test_reference_high_water_bounds_total_backslide_by_tolerance():
    """``high_water`` makes ``tolerance`` mean what the word says: a total, not a rate."""
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="clamp", reference="high_water", tolerance=2.0
    )
    control.note_executed(110.0)
    path = _glide(control, _env())
    assert path == [108.0] * 12
    assert min(path) >= 110.0 - 2.0


def test_high_water_remembers_the_peak_after_a_penalty_markdown():
    """Penalty mode may move down; the reference must not follow it down."""
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="penalty", reference="high_water", penalty=10
    )
    env = _env()
    control.note_executed(100.0)
    control.note_executed(118.0)
    control.note_executed(90.0)  # a markdown the penalty allowed
    control.apply(env, 90.0)
    assert control.high_water == pytest.approx(118.0)
    # Charged against the peak, not against yesterday's $90.
    assert control.last_violation == pytest.approx(28.0)


def test_reference_last_charges_only_the_step_not_the_level():
    """The same state under ``last`` barely registers — this is why weight 10 failed."""
    control = MonotonePriceControl(
        enabled=True, direction="up", mode="penalty", reference="last", penalty=10
    )
    control.note_executed(118.0)
    control.note_executed(90.0)
    control.apply(_env(), 89.7)
    assert control.last_violation == pytest.approx(0.3)


# --- ratchet: the action is a move, and no two moves collide ---


def test_ratchet_requires_max_step():
    with pytest.raises(ValueError, match="requires max_step"):
        MonotonePriceControl(enabled=True, direction="up", mode="ratchet")


def test_ratchet_maps_the_action_onto_the_legal_move():
    control = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=1.0)
    env = _env()
    control.note_executed(100.0)
    assert control.price_from_action(env, -1.0) == pytest.approx(100.0)  # hold
    assert control.price_from_action(env, 0.0) == pytest.approx(100.5)
    assert control.price_from_action(env, 1.0) == pytest.approx(101.0)  # largest legal step


def test_ratchet_never_lowers_the_price():
    control = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=1.0)
    env = _env()
    control.note_executed(100.0)
    for a in (-1.0, -0.9, -0.5, 0.0, 0.5, 1.0):
        assert control.price_from_action(env, a) >= 100.0 - 1e-9


def test_ratchet_distinguishes_actions_that_the_clamp_would_alias():
    """The point of the mode.

    Under ``clamp`` every action below the floor charges the floor, so the
    environment is flat there and a learner sees no gradient. Under ``ratchet``
    the same two actions are two different prices.
    """
    env = _env()
    clamp = MonotonePriceControl(enabled=True, direction="up", mode="clamp")
    clamp.note_executed(100.0)
    assert clamp.apply(env, 85.0) == clamp.apply(env, 95.0)  # aliased

    ratchet = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=1.0)
    ratchet.note_executed(100.0)
    assert ratchet.price_from_action(env, -0.5) != ratchet.price_from_action(env, 0.5)


def test_ratchet_first_step_is_an_ordinary_absolute_price():
    control = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=1.0)
    env = _env()
    assert control.price_from_action(env, -1.0) == pytest.approx(80.0)
    assert control.price_from_action(env, 1.0) == pytest.approx(120.0)


def test_ratchet_reach_shrinks_as_the_price_approaches_the_band_top():
    control = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=5.0)
    env = _env()
    control.note_executed(118.0)
    assert control.reach(env, 118.0) == pytest.approx(2.0)  # headroom, not max_step
    assert control.price_from_action(env, 1.0) == pytest.approx(120.0)


def test_ratchet_action_round_trips_through_the_charged_price():
    """What the BC seam relies on: charged price -> the action that reproduces it."""
    control = MonotonePriceControl(enabled=True, direction="up", mode="ratchet", max_step=1.0)
    env = _env()
    control.note_executed(100.0)
    for a in (-1.0, -0.6, 0.0, 0.4, 1.0):
        price = control.price_from_action(env, a)
        assert control.action_from_price(env, price) == pytest.approx(a, abs=1e-9)


# --- apply_on: peak_only ---


def test_peak_only_frees_a_weekday_in_an_offpeak_month():
    control = MonotonePriceControl(enabled=True, direction="up", mode="clamp", apply_on="peak_only")
    control.note_executed(110.0)
    # June (off-peak) Tuesday: unprotected, the markdown stands.
    assert control.apply(_env(month=6, dow=1), 90.0) == pytest.approx(90.0)


def test_peak_only_still_protects_weekends_and_peak_months():
    control = MonotonePriceControl(enabled=True, direction="up", mode="clamp", apply_on="peak_only")
    control.note_executed(110.0)
    assert control.apply(_env(month=6, dow=5), 90.0) == pytest.approx(110.0)  # weekend
    control.note_executed(110.0)
    assert control.apply(_env(month=12, dow=1), 90.0) == pytest.approx(110.0)  # peak month


def test_peak_only_calendar_matches_the_scorers_vocabulary():
    """The gate is calendar-only by design, but must not invent its own calendar."""
    from reservation_pricing.controls.price_monotone import PEAK_MONTHS, WEEKEND_DOW
    from reservation_pricing.metrics import SoftAwareConfig

    cfg = SoftAwareConfig()
    assert PEAK_MONTHS == cfg.peak_months
    assert WEEKEND_DOW == cfg.weekend_dow


def test_negative_knobs_are_rejected():
    for knob in ("tolerance", "max_step", "penalty"):
        with pytest.raises(ValueError, match=f"{knob} must be >= 0"):
            MonotonePriceControl(enabled=True, **{knob: -1.0})


def test_ratchet_holds_the_guarantee_through_a_whole_episode():
    """End to end, on the charged price the env reports — not the wrapper's own value."""
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    cfg["control"]["price_monotone"].update(mode="ratchet", max_step=1.0)
    env = make_env(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    prices = []
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        _obs, _r, terminated, truncated, info = env.step(action)
        prices.append(float(info["price"]))
        assert -1.0 <= float(info["price_monotone_executed_action"]) <= 1.0
        if terminated or truncated:
            break
    assert len(prices) == 100
    assert min(np.diff(prices)) >= -1e-9


def test_ratchet_config_is_rejected_without_max_step():
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    cfg["control"]["price_monotone"]["mode"] = "ratchet"
    cfg["control"]["price_monotone"]["max_step"] = None
    with pytest.raises(ValueError, match="requires max_step"):
        validate_config(cfg)


def test_clamp_executed_action_reproduces_the_charged_price():
    """Under clamp the executed action differs from the request exactly when it bound."""
    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    env = make_env(cfg)
    env.reset(seed=0)
    up = np.ones(env.action_space.shape, dtype=np.float32)
    down = -np.ones(env.action_space.shape, dtype=np.float32)
    _obs, _r, _t, _tr, first = env.step(up)
    _obs, _r, _t, _tr, second = env.step(down)
    assert second["price_monotone_clamped"]
    unwrapped = env.unwrapped
    span = float(unwrapped.max_price - unwrapped.min_price)
    recovered = unwrapped.min_price + (second["price_monotone_executed_action"] + 1.0) * 0.5 * span
    assert recovered == pytest.approx(float(second["price"]), abs=1e-3)
    assert float(second["price"]) >= float(first["price"]) - 1e-9


# --- what behaviour cloning has to be told ---


def _implied_prices(dataset, lo=80.0, hi=120.0):
    return lo + (dataset.actions[:, 0] + 1.0) * 0.5 * (hi - lo)


def _markdown_count(dataset, prices):
    """Steps whose stored action implies charging less than the episode's peak."""
    bad = 0
    run = -1e9
    for i, price in enumerate(prices):
        if dataset.episode_starts[i]:
            run = -1e9
        if price < run - 1e-6:
            bad += 1
        run = max(run, price)
    return bad


def test_cloning_the_request_under_a_clamp_teaches_the_forbidden_move():
    """Why collect_expert_dataset needs executed_action_key.

    An unconstrained expert keeps *asking* for markdowns that the clamp
    overrides. Storing the request records a path the env never charged, and
    every one of those actions sits in the dead region below the floor.
    """
    from reservation_pricing.algorithms.bc import collect_expert_dataset, resolve_expert_policy

    model = ROOT / "artifacts" / "tree_long" / "best" / "rl_best.zip"
    if not model.exists():
        pytest.skip("needs artifacts/tree_long/best/rl_best.zip; see README Model checkpoints")

    cfg = load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml")
    policy = resolve_expert_policy(model_path=str(model), algo="sac")

    requested = collect_expert_dataset(
        lambda: make_env(cfg, use_held_out=False), policy, n_episodes=2, seed=0
    )
    executed = collect_expert_dataset(
        lambda: make_env(cfg, use_held_out=False),
        policy,
        n_episodes=2,
        seed=0,
        executed_action_key="price_monotone_executed_action",
    )

    assert _markdown_count(requested, _implied_prices(requested)) > 100
    assert _markdown_count(executed, _implied_prices(executed)) == 0


def test_resolve_expert_policy_accepts_a_checkpoint():
    from reservation_pricing.algorithms.bc import resolve_expert_policy

    model = ROOT / "artifacts" / "tree_long" / "best" / "rl_best.zip"
    if not model.exists():
        pytest.skip("needs artifacts/tree_long/best/rl_best.zip; see README Model checkpoints")
    policy = resolve_expert_policy(model_path=str(model), algo="sac")
    env = make_env(load_config(ROOT / "configs" / "experiment_monotone_up_clamp_sac.yaml"))
    obs, _info = env.reset(seed=0)
    action = policy(obs, env, {})
    assert np.asarray(action).shape == env.action_space.shape
