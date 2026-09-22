"""Controller paths the shipped experiments do not exercise, plus the MPC decision day."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reservation_pricing.config import load_config
from reservation_pricing.controls import OversellCap
from reservation_pricing.envs import make_env

ROOT = Path(__file__).resolve().parents[1]


def _env(**overrides):
    state = dict(
        capacity=10000.0,
        remain_inv=6000.0,
        cumulative_mat_boh=4000.0,
        current_boh=5000.0,
        min_selling_limit=10000.0,
        max_selling_limit=15000.0,
        dow=2,
        month=3,
        days_prior=50,
    )
    state.update(overrides)
    return SimpleNamespace(**state)


def test_oversell_cap_analytic_kind_never_raises_the_limit():
    cap = OversellCap(enabled=True, kind="analytic", overbook_factor=1.02)
    env = _env()
    projected = cap.project(env, 100.0, policy_sl=15000.0)
    assert projected < 15000.0 and cap.last_projected
    assert cap.last_cap == pytest.approx(projected)
    assert cap.project(env, 100.0, policy_sl=10000.0) == 10000.0
    assert not cap.last_projected


def test_oversell_cap_activate_load_frac_gates_projection():
    cap = OversellCap(enabled=True, kind="chance", activate_load_frac=0.9)
    assert cap.project(_env(cumulative_mat_boh=4000.0), 100.0, 15000.0) == 15000.0
    assert cap.last_cap is None
    assert cap.project(_env(cumulative_mat_boh=9500.0, remain_inv=500.0), 100.0, 15000.0) < 15000.0


def test_oversell_cap_buffer_mat_loosens_the_chance_cap():
    tight = OversellCap(enabled=True, kind="chance", buffer_mat=0.0)
    loose = OversellCap(enabled=True, kind="chance", buffer_mat=800.0)
    env = _env(cumulative_mat_boh=9000.0, remain_inv=1000.0, current_boh=9000.0)
    assert loose.compute_cap(env) > tight.compute_cap(env)


def test_safe_sl_analytic_kind_from_config():
    cfg = load_config(ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml")
    cfg["control"]["safe_sl"]["kind"] = "analytic"
    env = make_env(cfg)
    env.reset(seed=0)
    assert env.projector.cap_kind == "analytic"
    u = env.unwrapped
    u.cumulative_mat_boh = 8500.0
    u.remain_inv = 1500.0
    u.current_boh = 12000.0
    _, _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert info["safe_sl_cap"] is not None
    assert float(info["selling_limit"]) <= float(info["policy_sl"]) + 1e-3


def test_mpc_sees_the_decision_day():
    """Like the SL controller and early promo, the MPC's trigger conditions must
    be evaluated on days_prior - 1, the day ReservationEnv.step() settles."""
    cfg = load_config(ROOT / "configs" / "experiment_pace_mpc.yaml")
    env = make_env(cfg)
    env.reset(seed=0)
    base = env.unwrapped
    seen: list[int] = []
    orig = env.mpc.apply

    def spy(e, price, _orig=orig):
        seen.append(int(e.days_prior))
        return _orig(e, price)

    env.mpc.apply = spy
    before = int(base.days_prior)
    env.step(env.action_space.sample())
    assert seen == [before - 1]
    assert int(base.days_prior) == before - 1
