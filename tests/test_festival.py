"""Festival passes: logit demand across nights and lengths, shared nightly seats, planner."""

from __future__ import annotations

import numpy as np
import pytest

from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.festival import FestivalConfig, FestivalEnv
from reservation_pricing.festival.evaluate import run_season
from reservation_pricing.festival.model import choice_probs, consecutive_passes, pass_utility
from reservation_pricing.festival.planner import planner_policy


def test_default_passes_are_every_consecutive_run():
    cfg = FestivalConfig()
    assert cfg.pass_names() == ["fri", "sat", "sun", "fri_sat", "sat_sun", "fri_sat_sun"]
    assert len(consecutive_passes(4)) == 10
    four = FestivalConfig.from_mapping(
        {"night_names": ["thu", "fri", "sat", "sun"], "night_appeal": [0, 0, 0.3, -0.3]}
    )
    assert FestivalEnv(four).action_space.shape == (10 + 4,)


def test_passes_must_be_consecutive_nights():
    with pytest.raises(ValueError):
        FestivalConfig.from_mapping({"passes": [[0, 2]]})
    with pytest.raises(ValueError):
        FestivalConfig.from_mapping({"passes": [[-1, 0]]})
    with pytest.raises(ValueError):
        FestivalConfig.from_mapping({"no_such_key": 1})


def test_a_dearer_saturday_sends_buyers_to_other_nights_and_lengths():
    cfg = FestivalConfig()
    u = pass_utility(cfg, cfg.night_appeal)
    prices = 100.0 * cfg.lengths()
    base = choice_probs(u, prices, beta=2.5)
    dearer = prices.copy()
    dearer[1] += 20.0
    moved = choice_probs(u, dearer, beta=2.5)
    assert base.sum() < 1.0
    assert moved[1] < base[1]
    assert np.all(np.delete(moved, 1) > np.delete(base, 1))


def test_taking_a_pass_off_sale_moves_its_buyers():
    cfg = FestivalConfig()
    u = pass_utility(cfg, cfg.night_appeal)
    prices = 100.0 * cfg.lengths()
    base = choice_probs(u, prices, beta=2.5)
    on_sale = np.ones(cfg.n_passes, dtype=bool)
    on_sale[1] = False
    closed = choice_probs(u, prices, beta=2.5, on_sale=on_sale)
    assert closed[1] == 0.0
    assert np.all(np.delete(closed, 1) > np.delete(base, 1))


def test_a_night_at_its_limit_stops_every_pass_that_uses_it():
    env = FestivalEnv()
    env.reset(seed=0)
    env.bookings[env.cfg.horizon - 1, 1] = 10_000.0  # a Saturday pass sold 100 days out
    limits = np.array([15_000.0, 10_000.0, 15_000.0])
    env.step(env.scale_action(np.full(env.n_passes, 100.0), limits))
    sat = env.A[1] > 0
    assert not env.on_sale[sat].any()
    assert env.on_sale[~sat].all()
    assert env.accepted[sat].sum() == 0


def test_accepted_bookings_never_overrun_a_nights_headroom():
    env = FestivalEnv(FestivalConfig(market_size=400_000.0))
    env.reset(seed=1)
    limits = np.array([10_000.0, 10_000.0, 10_000.0])
    for _ in range(env.cfg.horizon):
        before = env.bookings_on_hand()
        env.step(env.scale_action(np.full(env.n_passes, 80.0), limits))
        assert np.all(env.A @ env.accepted <= np.maximum(limits - before, 0.0) + 1e-9)


def test_same_seed_same_season():
    env = FestivalEnv()
    action = env.scale_action(np.full(env.n_passes, 100.0), np.full(env.n_nights, 12_000.0))
    runs = []
    for _ in range(2):
        env.reset(seed=5)
        runs.append([env.step(action)[4]["accepted"] for _ in range(10)])
    assert np.array_equal(np.array(runs[0]), np.array(runs[1]))


def test_a_festival_block_selects_the_festival_env():
    cfg = load_config()
    cfg["festival"] = {"horizon": 20}
    env = make_env(cfg, use_held_out=True)
    assert isinstance(env, FestivalEnv)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_training_envs_never_draw_a_held_out_season():
    cfg = load_config()
    cfg["festival"] = {}
    train = make_env(cfg, use_held_out=False)
    test = make_env(cfg, use_held_out=True)
    for seed in range(30):
        train.reset(seed=seed)
        test.reset(seed=seed)
        assert train.season.market_mult != test.season.market_mult


def test_the_planner_beats_fixed_prices():
    cfg = FestivalConfig(horizon=30)
    env = FestivalEnv(cfg)
    fixed = planner_policy(n_buckets=1, resolve=False)
    planner = planner_policy(n_buckets=5)
    gain = [
        run_season(env, planner, s)["score"] - run_season(env, fixed, s)["score"]
        for s in range(3)
    ]
    assert np.mean(gain) > 0


def test_shaping_adds_only_a_constant_over_a_season():
    """With gamma 1 the potential telescopes: shaped minus plain is -potential(start)."""
    totals = []
    for shaped in (False, True):
        env = FestivalEnv(FestivalConfig(shape_reward=shaped))
        env.reset(seed=4)
        start = env.potential()
        rng = np.random.default_rng(0)
        total, done = 0.0, False
        while not done:
            _, r, done, _, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
            total += r
        totals.append(total)
    assert totals[1] - totals[0] == pytest.approx(-start * env.cfg.revenue_scale)
