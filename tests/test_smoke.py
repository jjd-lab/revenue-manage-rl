"""Smoke tests: legacy + tree_elastic env, baselines, tiny PPO from config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reservation_pricing.baselines import evaluate_baselines, myopic_greedy_policy
from reservation_pricing.config import load_config
from reservation_pricing.demand import get_demand_model
from reservation_pricing.envs import ReservationEnv, make_env
from reservation_pricing.evaluate import evaluate_policy, sb3_policy
from reservation_pricing.metrics import run_episode
from reservation_pricing.train import load_sb3_model, train, train_from_config

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_env_steps():
    cfg = load_config(ROOT / "configs" / "demand_linear_legacy.yaml")
    env = make_env(cfg)
    assert getattr(env.demand_model, "kind") == "linear_legacy"
    obs, info = env.reset(seed=0)
    assert obs.shape == (27,)
    for _ in range(env.booking_horizon + 5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated
    assert info["true_revenue"] >= 0
    assert info["demand_kind"] == "linear_legacy"


def test_tree_elastic_env_steps():
    cfg = load_config()  # default.yaml -> tree_elastic
    env = make_env(cfg)
    assert getattr(env.demand_model, "kind") == "tree_elastic"
    obs, info = env.reset(seed=1)
    assert obs.shape == (27,)
    assert "true_revenue" in info
    for _ in range(env.booking_horizon + 5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated
    assert info["demand_kind"] == "tree_elastic"


def test_demand_registry_and_elasticity_direction():
    dm = get_demand_model(
        {"kind": "tree_elastic", "elasticity": -1.2, "ref_price": 100.0, "demand_noise_std": 0.0}
    )
    feats = {
        "days_prior": 20,
        "dow": 5,
        "month": 7,
        "is_weekend": 1,
        "is_peak_month": 1,
        "booking_curve_bin": 3,
    }
    low = dm.predict_mean(feats, 80.0)
    high = dm.predict_mean(feats, 120.0)
    assert low > high  # negative elasticity => higher price, lower demand
    assert low > 0 and high >= 0


def test_baselines_on_tree_demand_including_myopic():
    cfg = load_config()

    def factory():
        return make_env(cfg, use_held_out=True)

    out = evaluate_baselines(
        factory,
        n_episodes=3,
        seeds=[0, 1, 2],
        names=["fixed_price_100", "myopic_greedy", "heuristic_booking_limit"],
    )
    assert out["aggregates"]["myopic_greedy"].mean_true_revenue > 0
    assert out["aggregates"]["fixed_price_100"].mean_true_revenue > 0


def test_myopic_uses_grid_on_tree(tmp_path=None):
    """Myopic should return a finite price action on tree demand (grid path)."""
    cfg = load_config()
    env = make_env(cfg)
    env.reset(seed=0)
    policy = myopic_greedy_policy(n_grid=21)
    action = policy(env._get_obs(), env, {})
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


@pytest.mark.slow
def test_tiny_ppo_train_from_config(tmp_path):
    cfg_path = ROOT / "configs" / "experiment_tree_ppo.yaml"
    meta = train(
        config_path=str(cfg_path),
        total_timesteps=2048,
        seed=0,
        algo="ppo",
        run_name="smoke_tree_ppo",
        out_dir=str(tmp_path / "runs"),
    )
    assert Path(meta["model_path"]).exists()
    assert meta.get("demand_kind") == "tree_elastic"

    model = load_sb3_model(meta["model_path"], algo="ppo")
    policy = sb3_policy(model)
    full_cfg = load_config(cfg_path)

    def factory():
        return make_env(full_cfg, use_held_out=True)

    agg, rows = evaluate_policy(factory, policy, n_episodes=2, seeds=[0, 1])
    assert agg.n_episodes == 2
    assert np.isfinite(agg.mean_true_revenue)


@pytest.mark.slow
def test_run_name_from_config_controls_output_paths(tmp_path):
    cfg = load_config(ROOT / "configs" / "experiment_price_only_pace_ppo.yaml")
    cfg["train"]["model_dir"] = str(tmp_path / "artifacts")
    cfg["train"]["log_dir"] = str(tmp_path / "runs")

    meta = train_from_config(cfg, total_timesteps=256)
    assert meta["run_name"] == "ppo_pace"
    assert Path(meta["model_path"]) == tmp_path / "artifacts" / "ppo_pace" / "final_model.zip"
    assert Path(meta["model_path"]).exists()

    override = train_from_config(cfg, total_timesteps=256, run_name="explicit")
    assert override["run_name"] == "explicit"


def test_held_out_month_sampling():
    env = ReservationEnv(
        held_out_months=[6, 12],
        use_held_out=True,
        demand_model=get_demand_model({"kind": "linear_legacy", "demand_noise_std": 5.0}),
    )
    months = set()
    for s in range(40):
        env.reset(seed=s)
        months.add(env.month)
    assert months.issubset({6, 12})

    env2 = ReservationEnv(
        held_out_months=[6, 12],
        use_held_out=False,
        demand_model=get_demand_model({"kind": "linear_legacy", "demand_noise_std": 5.0}),
    )
    months2 = set()
    for s in range(80):
        env2.reset(seed=s)
        months2.add(env2.month)
    assert months2.isdisjoint({6, 12})


def test_linear_legacy_still_via_direct_env():
    env = ReservationEnv(demand_noise_std=5.0)  # default demand = linear_legacy when no cfg
    obs, _ = env.reset(seed=0)
    assert obs.shape == (27,)
    ep = run_episode(env, lambda o, e, s: e.action_space.sample(), seed=1)
    assert ep.true_revenue >= 0


def test_bc_pretrain_actor_smoke(tmp_path):
    """Tiny BC dataset → pretrain SAC actor → MSE finite; does not break predict."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from reservation_pricing.algorithms.bc import (
        collect_expert_dataset,
        pretrain_actor_mse,
        resolve_expert_policy,
        seed_replay_buffer,
    )
    from reservation_pricing.algorithms.registry import build_model
    from reservation_pricing.config import load_config
    from reservation_pricing.envs import make_env

    cfg = load_config()
    cfg.setdefault("algorithm", {})["name"] = "sac"
    cfg["algorithm"]["net_arch"] = [64, 64]
    cfg["algorithm"]["buffer_size"] = 5000
    cfg["algorithm"]["device"] = "cpu"

    def factory():
        return make_env(cfg, use_held_out=False)

    policy = resolve_expert_policy("myopic_greedy", n_grid=11)
    ds = collect_expert_dataset(factory, policy, n_episodes=2, seed=0)
    assert len(ds) >= 100
    ds.save(tmp_path / "expert.npz")

    vec = DummyVecEnv([lambda: make_env(cfg, use_held_out=False)])
    train_cfg = {
        "learning_rate": 3e-4,
        "gamma": 1.0,
        "buffer_size": 5000,
        "batch_size": 64,
        "learning_starts": 10,
        "net_arch": [64, 64],
        "device": "cpu",
        "ent_coef": "auto",
    }
    model = build_model("sac", vec, train_cfg, seed=0)
    metrics = pretrain_actor_mse(model, ds, epochs=2, batch_size=64, lr=1e-3)
    assert np.isfinite(metrics["bc_mse"])
    n_add = seed_replay_buffer(model, ds)
    assert n_add == len(ds)
    obs = vec.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape[-1] == 2
    assert np.all(np.isfinite(action))
    vec.close()


def test_bc_pretrain_actor_td3(tmp_path):
    """td3 is an advertised BC target: its actor has no get_action_dist_params."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from reservation_pricing.algorithms.bc import (
        collect_expert_dataset,
        pretrain_actor_mse,
        resolve_expert_policy,
    )
    from reservation_pricing.algorithms.registry import build_model

    cfg = load_config()
    cfg.setdefault("algorithm", {})["name"] = "td3"
    cfg["algorithm"]["net_arch"] = [64, 64]
    cfg["algorithm"]["device"] = "cpu"

    def factory():
        return make_env(cfg, use_held_out=False)

    ds = collect_expert_dataset(
        factory, resolve_expert_policy("myopic_greedy", n_grid=11), n_episodes=1, seed=0
    )
    vec = DummyVecEnv([factory])
    model = build_model(
        "td3",
        vec,
        {
            "learning_rate": 3e-4,
            "gamma": 1.0,
            "buffer_size": 5000,
            "batch_size": 64,
            "learning_starts": 10,
            "net_arch": [64, 64],
            "device": "cpu",
        },
        seed=0,
    )
    metrics = pretrain_actor_mse(model, ds, epochs=1, batch_size=64, lr=1e-3)
    assert np.isfinite(metrics["bc_mse"])

    action, _ = model.predict(vec.reset(), deterministic=True)
    assert action.shape[-1] == 2 and np.all(np.isfinite(action))
    vec.close()


def test_soft_oracle_prices_at_the_env_floor():
    """gap_to_oracle is only a ceiling if the oracle sits at this env's min_price.

    Floor is 60 here, so the old hard-coded 80.0 lands inside the band and
    produces a genuinely different (non-clipped) revenue.
    """
    from reservation_pricing.baselines import fixed_price_policy
    from reservation_pricing.evaluate.soft_aware import collect_soft_oracle_revenues
    from reservation_pricing.metrics import SoftAwareConfig

    def factory():
        return ReservationEnv(min_price=60.0, max_price=130.0, held_out_months=[6, 12])

    at_floor = run_episode(
        factory(), fixed_price_policy(price=60.0, selling_limit=None), seed=0
    ).true_revenue
    at_eighty = run_episode(
        factory(), fixed_price_policy(price=80.0, selling_limit=None), seed=0
    ).true_revenue
    assert at_floor != pytest.approx(at_eighty), "floor and 80 must differ for this to bite"

    got = collect_soft_oracle_revenues(factory, [0], SoftAwareConfig())
    assert got[0] == pytest.approx(at_floor)


def test_missing_base_config_raises_instead_of_returning_empty(tmp_path):
    """An unreadable base used to yield {}, which validates and trains the wrong model."""
    with pytest.raises(FileNotFoundError):
        load_config(base=tmp_path / "nope.yaml")

    # An explicit experiment config must not paper over the missing schema either.
    frag = tmp_path / "frag.yaml"
    frag.write_text("algorithm:\n  name: ppo\n")
    with pytest.raises(FileNotFoundError):
        load_config(frag, base=tmp_path / "nope.yaml")


def test_package_root_found_by_walking_up():
    """Resolution must not depend on counting directory levels from __file__."""
    from reservation_pricing.config.load import default_config_path, package_root

    assert package_root() == ROOT
    assert default_config_path() == ROOT / "configs" / "default.yaml"


def test_held_out_without_months_is_rejected():
    """Empty held_out_months + use_held_out silently sampled the full year."""
    with pytest.raises(ValueError, match="held_out_months"):
        ReservationEnv(use_held_out=True, held_out_months=[])


def test_price_only_controller_sees_the_decision_day():
    """The SL/early-promo controller must see days_prior - 1, the day
    ReservationEnv.step() actually settles the booking against (it decrements
    days_prior before applying the limit and before calling cancel_fn). A
    controller that sees the pre-decrement day is planning for the wrong day.
    """
    from reservation_pricing.config import load_config
    from reservation_pricing.envs import make_env

    cfg = load_config(ROOT / "configs" / "experiment_price_only_ppo.yaml")
    env = make_env(cfg)
    env.reset(seed=0)

    base = env.unwrapped
    seen_days = []
    orig_compute = env.sl_controller.compute

    def spy(e, price, _orig=orig_compute):
        seen_days.append(int(e.days_prior))
        return _orig(e, price)

    env.sl_controller.compute = spy
    days_prior_before_step = int(base.days_prior)
    env.step(env.action_space.sample())

    assert seen_days == [days_prior_before_step - 1]
    # And the controller's mutation must be transient: env.step() itself decrements
    # days_prior exactly once more, from days_prior_before_step to -2 total.
    assert int(base.days_prior) == days_prior_before_step - 1


def test_price_only_env_analytic_sl():
    """Price-only wrapper: 1D action, analytic SL in bounds, joint mode still 2D."""
    from reservation_pricing.controls import get_selling_limit

    cfg = load_config(ROOT / "configs" / "experiment_price_only_ppo.yaml")
    env = make_env(cfg)
    assert env.action_space.shape == (1,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (27,)
    assert info.get("price_only") is True

    for _ in range(5):
        action = env.action_space.sample()
        assert action.shape == (1,)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        sl = float(info["selling_limit"])
        lo = float(env.unwrapped.min_selling_limit)
        hi = float(env.unwrapped.max_selling_limit)
        assert lo - 1e-3 <= sl <= hi + 1e-3
        assert "controller_sl" in info
        if terminated or truncated:
            break

    # Joint mode unbroken
    joint = make_env(load_config())
    assert joint.action_space.shape == (2,)
    joint.reset(seed=0)
    a = joint.action_space.sample()
    assert a.shape == (2,)
    joint.step(a)

    # Controllers produce finite SL; optimize_1d often differs from analytic
    base = make_env(load_config())
    base.reset(seed=0)
    base.step(base.action_space.sample())
    analytic = get_selling_limit({"kind": "analytic", "overbook_factor": 1.05})
    opt = get_selling_limit({"kind": "optimize_1d", "n_grid": 21, "overbook_factor": 1.05})
    price = 100.0
    sl_a = analytic.compute(base, price)
    sl_o = opt.compute(base, price)
    assert base.min_selling_limit <= sl_a <= base.max_selling_limit
    assert base.min_selling_limit <= sl_o <= base.max_selling_limit
    assert np.isfinite(sl_a) and np.isfinite(sl_o)


@pytest.mark.slow
def test_price_only_baselines_and_tiny_ppo(tmp_path):
    """Baselines emit 1D actions under price_only; tiny PPO trains."""
    cfg = load_config(ROOT / "configs" / "experiment_price_only_ppo.yaml")

    def factory():
        return make_env(cfg, use_held_out=True)

    env = factory()
    obs, _ = env.reset(seed=1)
    policy = myopic_greedy_policy(n_grid=11)
    action = policy(obs, env, {})
    assert action.shape == (1,), action.shape
    obs, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(reward)
    assert info.get("price_only") is True

    out = evaluate_baselines(
        factory,
        n_episodes=2,
        seeds=[0, 1],
        names=["fixed_price_100", "myopic_greedy"],
    )
    assert out["aggregates"]["myopic_greedy"].mean_true_revenue > 0

    meta = train(
        config_path=str(ROOT / "configs" / "experiment_price_only_ppo.yaml"),
        total_timesteps=2048,
        seed=0,
        algo="ppo",
        run_name="smoke_price_only_ppo",
        out_dir=str(tmp_path / "runs"),
    )
    assert Path(meta["model_path"]).exists()
    model = load_sb3_model(meta["model_path"], algo="ppo")
    pred, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    assert np.asarray(pred).reshape(-1).shape[0] == 1


def test_pace_penalty_when_behind():
    """Pace shaping fires when far behind target fill; business metrics unshaped."""
    cfg = load_config(ROOT / "configs" / "experiment_price_only_pace_ppo.yaml")
    env = make_env(cfg)
    assert env.unwrapped.pace_reward is True
    assert env.unwrapped.soft_day_upweight is True
    # Soft weekday off-peak so soft-day mult > 1
    obs, info = env.reset(seed=0, options={"service_date": "2022-03-09"})
    assert info.get("soft_day") is True
    assert info.get("soft_day_mult", 1.0) > 1.0

    # Force near-zero fill by pricing high with tiny SL via raw joint on unwrapped
    # Use wrapper step with max price; analytic SL still allows bookings — so instead
    # step with high price many times and check mid-horizon gap, OR zero out inventory
    # by manually setting state after a few steps.
    base = env.unwrapped
    # Simulate being far behind: jump to mid-horizon with almost no fill
    base.days_prior = base.booking_horizon // 2
    base.cumulative_mat_boh = 0.0
    base.remain_inv = float(base.capacity)
    target = base._pace_target_load()
    assert target > 0.3, target
    # One step at max price (price-only action +1)
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["pace_gap"] > 0.2, info
    # Shaped reward should include a negative pace term (gap * penalty * soft_mult)
    # Revenue may be small at max price; gap penalty should dominate or at least be recorded
    assert info["pace_target"] > 0
    assert np.isfinite(reward)
    # true_revenue metric path stays the raw cumulative income (not multiplied by soft_day)
    assert "true_revenue" in info
    assert info["true_revenue"] == base.cumulative_income


def test_pace_config_loads_into_env():
    cfg = load_config(ROOT / "configs" / "experiment_price_only_pace_ppo.yaml")
    env = make_env(cfg)
    u = env.unwrapped
    assert u.pace_penalty == 5.0
    assert u.pace_schedule == "linear"
    assert u.soft_day_weekday_factor == 1.5


def test_early_promo_triggers_on_soft_state():
    """Early promo forces promo_price when tree base is low early/mid horizon."""
    import numpy as np

    from reservation_pricing.controls import EarlyPromoController, get_early_promo

    cfg = load_config(ROOT / "configs" / "experiment_price_only_promo_ppo.yaml")
    env = make_env(cfg)
    assert env.action_space.shape == (1,)
    assert env.early_promo is not None
    assert env.early_promo.enabled is True
    assert env.unwrapped.pace_reward is True

    # Soft weekday off-peak (Mar weekday): base mid-horizon typically < 90
    obs, info = env.reset(seed=0, options={"service_date": "2022-03-09"})
    u = env.unwrapped
    assert u.days_prior >= 30
    base = env.early_promo.predict_base(u)
    assert base < env.early_promo.base_demand_threshold, base

    # Agent asks for max price (+1 → 120); promo should force to promo_price=80
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info.get("early_promo_triggered") is True, info
    assert abs(float(info["price"]) - 80.0) < 1e-3, info
    assert float(info["rl_price"]) > 100.0
    assert np.isfinite(reward)

    # Late horizon: days_prior < 30 → promo must NOT trigger (late clearing)
    u.days_prior = 20
    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert info.get("early_promo_triggered") is False, info
    assert abs(float(info["price"]) - float(info["rl_price"])) < 1e-3

    # Controller unit: high-base peak weekend should not fire
    ctrl = EarlyPromoController(
        enabled=True,
        base_demand_threshold=90.0,
        promo_price=80.0,
        mode="set",
        apply_when_days_prior_ge=30,
    )
    env2 = make_env(cfg)
    env2.reset(seed=0, options={"service_date": "2022-07-09"})  # Sat peak
    # Ensure early horizon
    assert env2.unwrapped.days_prior >= 30
    assert ctrl.should_apply(env2.unwrapped) is False
    # get_early_promo disabled → None
    assert get_early_promo({"early_promo": {"enabled": False}}) is None


def test_early_promo_config_loads():
    cfg = load_config(ROOT / "configs" / "experiment_price_only_promo_ppo.yaml")
    ep = cfg["control"]["early_promo"]
    assert ep["enabled"] is True
    assert ep["base_demand_threshold"] == 90.0
    assert ep["promo_price"] == 80.0
    assert ep["apply_when_days_prior_ge"] == 30


def test_safe_sl_projects_joint_action_down():
    """Safe SL wrapper keeps joint 2D action and never raises SL above cap."""
    from reservation_pricing.controls import get_oversell_cap

    cfg = load_config(ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml")
    env = make_env(cfg)
    assert env.action_space.shape == (2,)
    assert hasattr(env, "projector")
    obs, info = env.reset(seed=0)
    assert info.get("safe_sl") is True

    # Near capacity so activate_remain_frac triggers projection
    u = env.unwrapped
    u.cumulative_mat_boh = 8500.0
    u.remain_inv = 1500.0
    u.current_boh = 12000.0

    # Force max SL action (+1, +1) → price=120, SL=max; projection should pull SL down
    action = np.array([1.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(reward)
    assert info.get("safe_sl_cap") is not None
    # mix_alpha may leave executed SL between cap and policy_sl; never above policy
    assert float(info["selling_limit"]) <= float(info["policy_sl"]) + 1e-3
    assert float(info["selling_limit"]) < float(info["policy_sl"]) - 1.0  # pulled down
    assert info.get("safe_sl_projected") is True
    # hard projection bound: mix*policy + (1-mix)*cap
    mix = float(env.projector.mix_alpha)
    blended = mix * float(info["policy_sl"]) + (1.0 - mix) * float(info["safe_sl_cap"])
    assert float(info["selling_limit"]) <= blended + 1e-2
    assert u.min_selling_limit - 1e-3 <= float(info["safe_sl_cap"]) <= u.max_selling_limit + 1e-3

    # Disabled → None
    assert get_oversell_cap({"safe_sl": {"enabled": False}}) is None


def test_safe_sl_config_loads():
    cfg = load_config(ROOT / "configs" / "experiment_bc_sac_safe_sl.yaml")
    ss = cfg["control"]["safe_sl"]
    assert ss["enabled"] is True
    assert ss["kind"] == "chance"
    assert ss["overbook_factor"] == 1.02
    assert ss.get("activate_remain_frac") == 0.20
    assert ss.get("mix_alpha") == 0.25


def test_mpc_triggers_on_soft_or_behind():
    """MPC overrides price on soft weekday when enabled."""
    from reservation_pricing.controls import get_price_mpc

    cfg = load_config(ROOT / "configs" / "experiment_pace_mpc.yaml")
    env = make_env(cfg)
    assert env.action_space.shape == (1,)
    assert env.mpc is not None
    obs, info = env.reset(seed=0, options={"service_date": "2022-06-08"})  # June Wed
    u = env.unwrapped
    base = env.mpc.predict_base(u)
    # Soft or we can force behind pace
    if base >= env.mpc.base_demand_threshold:
        u.cumulative_mat_boh = 0.0
        u.remain_inv = float(u.capacity)
        u.days_prior = u.booking_horizon // 2

    obs, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert np.isfinite(reward)
    # On soft June weekday should trigger; if not, behind-pace force should
    assert info.get("mpc_triggered") is True, info
    assert "mpc_price" in info
    assert abs(float(info["price"]) - float(info["mpc_price"])) < 1e-3
    assert get_price_mpc({"mpc": {"enabled": False}}) is None


def test_mpc_config_loads():
    cfg = load_config(ROOT / "configs" / "experiment_pace_mpc.yaml")
    m = cfg["control"]["mpc"]
    assert m["enabled"] is True
    assert m["horizon"] == 5
    assert m["n_grid"] == 21


def test_soft_aware_classify_and_stratify():
    """Soft vs peak split runs on a few held-out episodes; soft KPIs present."""
    from reservation_pricing.baselines import fixed_price_policy, myopic_greedy_policy
    from reservation_pricing.evaluate.soft_aware import (
        collect_soft_oracle_revenues,
        evaluate_policy_soft_aware,
    )
    from reservation_pricing.metrics import SoftAwareConfig, classify_soft

    cfg = load_config()
    soft_cfg = SoftAwareConfig.from_dict(
        {
            "rule": "structural",
            "base_demand_threshold": 90.0,
            "soft_focus_months": [6],
            "soft_oracle": "min_price",
            "lambda_peak": 200.0,
            "soft_score_mode": "gap_to_oracle",
        }
    )

    # Known soft (June weekday, low base) vs peak (Dec weekend) from oracle notes
    assert classify_soft(month=6, dow=2, base_demand=89.3, cfg=soft_cfg) is True
    assert classify_soft(month=12, dow=6, base_demand=192.5, cfg=soft_cfg) is False
    # Broad rule marks Dec weekday soft
    broad = SoftAwareConfig.from_dict({"rule": "any", "base_demand_threshold": 90.0})
    assert classify_soft(month=12, dow=2, base_demand=127.0, cfg=broad) is True

    def factory():
        return make_env(cfg, use_held_out=True)

    seeds = [1, 4, 5]  # soft, peak, soft under structural
    oracle = collect_soft_oracle_revenues(factory, seeds, soft_cfg)
    assert set(oracle.keys()) == set(seeds)

    sa, rows = evaluate_policy_soft_aware(
        factory,
        myopic_greedy_policy(),
        n_episodes=len(seeds),
        seeds=seeds,
        soft_cfg=soft_cfg,
        oracle_revenue_by_seed=oracle,
    )
    assert sa["n_soft"] + sa["n_peak"] == len(seeds)
    assert sa["n_soft"] >= 1 and sa["n_peak"] >= 1
    assert "gap_to_oracle" in sa["soft"]
    assert "mean_frac_at_floor" in sa["soft"]
    assert "score_aware" in sa
    assert "undersell_gt_thresh_overall" in sa["diagnostic"]
    # Soft primary path should not require undersell as the lead metric
    assert rows[0]["is_soft"] is not None
    assert "frac_at_floor" in rows[0]

    # Floor oracle should show high frac_at_floor on soft when we actually try
    sa80, rows80 = evaluate_policy_soft_aware(
        factory,
        fixed_price_policy(80.0),
        n_episodes=len(seeds),
        seeds=seeds,
        soft_cfg=soft_cfg,
        oracle_revenue_by_seed=oracle,
    )
    soft_rows = [r for r in rows80 if r.get("is_soft")]
    assert soft_rows
    assert all(float(r["frac_at_floor"]) > 0.9 for r in soft_rows)
