"""Env factory from experiment config."""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym

from reservation_pricing.demand.registry import get_demand_model
from reservation_pricing.envs.reservation import ReservationEnv

_ENV_KEYS = {
    "capacity",
    "booking_horizon",
    "min_price",
    "max_price",
    "min_selling_limit",
    "max_selling_limit",
    "demand_noise_std",
    "noshow_noise_std",
    "cancel_rho_noise_std",
    "cancel_rho_night_std",
    "cancel_lambda",
    "cancel_rho_weekday",
    "cancel_rho_weekend",
    "noshow_base",
    "noshow_dow_coef",
    "noshow_month_coef",
    "revenue_scale",
    "undersell_penalty",
    "oversell_penalty",
    "utilization_bonus",
    "pace_reward",
    "pace_schedule",
    "pace_penalty",
    "pace_target_final",
    "soft_day_upweight",
    "soft_day_weekday_factor",
    "soft_day_offpeak_factor",
    "soft_day_base_threshold",
    "soft_day_base_factor",
    "soft_day_sample_boost",
    "undersell_on_soft",
    "night_features",
    "normalize_obs",
    "held_out_months",
    "use_held_out",
    "service_date",
    "render_mode",
}

_CONTROL_OVERRIDE_KEYS = (
    "price_only",
    "selling_limit",
    "early_promo",
    "promo",
    "safe_sl",
    "mpc",
    "price_monotone",
    "residual",
)


def _enabled(block: Any) -> bool:
    if isinstance(block, dict):
        return bool(block.get("enabled", False))
    return bool(block)


def make_env(cfg: dict, **overrides: Any) -> Union[ReservationEnv, gym.Env]:
    """Build ReservationEnv from a full experiment config (or nested env dict).

    When ``control.price_only: true``, wraps with ``PriceOnlyWrapper`` so the
    action space is 1D price and selling limit comes from ``control.selling_limit``.
    Legacy joint ``(price, SL)`` mode is unchanged when ``price_only`` is false/absent.

    Optional post-process wrappers (config-driven):
    - ``control.safe_sl`` → ``OversellGuardEnv`` on joint actions
    - ``control.mpc`` → short-horizon price MPC inside ``PriceOnlyWrapper``
    - ``control.residual`` → ``ResidualPlannerEnv``: DP planner proposes, agent corrects
    - ``control.price_monotone`` → ``MonotonePriceEnv``, outermost, either action space
    """
    env_cfg = dict(cfg.get("env", cfg))
    # Flatten optional env.reward / control.reward nests into env kwargs
    for nest_src in (
        env_cfg.pop("reward", None),
        (cfg.get("control") or {}).get("reward") if isinstance(cfg.get("control"), dict) else None,
    ):
        if isinstance(nest_src, dict):
            for rk, rv in nest_src.items():
                env_cfg.setdefault(rk, rv)
    # Env ctor overrides (exclude control-only keys)
    control_overrides = {}
    for key in _CONTROL_OVERRIDE_KEYS:
        if key in overrides:
            control_overrides[key] = overrides.pop(key)
    # Allow overrides["reward"] dict too
    if isinstance(overrides.get("reward"), dict):
        reward_over = overrides.pop("reward")
        overrides = {**reward_over, **overrides}
    env_cfg.update(overrides)

    demand_model = env_cfg.pop("demand_model", None)
    night_cfg: Optional[dict] = None
    operator_view = bool(env_cfg.pop("operator_view", False))
    forecast_model = env_cfg.pop("forecast_model", None)
    # Prefer top-level demand block; allow env.demand override
    demand_cfg: Optional[dict] = None
    forecast_cfg: Optional[dict] = None
    if demand_model is None:
        if "demand" in cfg and isinstance(cfg["demand"], dict):
            demand_cfg = dict(cfg["demand"])
        elif "demand" in env_cfg and isinstance(env_cfg["demand"], dict):
            demand_cfg = dict(env_cfg.pop("demand"))
        if demand_cfg is not None:
            forecast_cfg = demand_cfg.pop("forecast", None)
            night_cfg = demand_cfg.pop("night_variation", None)
        # Propagate env noise into demand if not set
        if demand_cfg is not None and "demand_noise_std" not in demand_cfg:
            if "demand_noise_std" in env_cfg:
                demand_cfg["demand_noise_std"] = env_cfg["demand_noise_std"]
        demand_model = (
            get_demand_model(demand_cfg)
            if demand_cfg is not None
            else get_demand_model(
                {"kind": "linear_legacy", "demand_noise_std": env_cfg.get("demand_noise_std", 8.0)}
            )
        )
    else:
        env_cfg.pop("demand", None)

    # demand.forecast: a second, deliberately wrong model that only decision code
    # sees. Inherits the true block's knobs, so a forecast usually names just a
    # model_path. See demand.protocol.decision_model.
    if forecast_model is None and isinstance(forecast_cfg, dict):
        if forecast_cfg.get("enabled", True):
            overrides = {k: v for k, v in forecast_cfg.items() if k != "enabled"}
            forecast_model = get_demand_model({**(demand_cfg or {}), **overrides})

    # demand.night_variation: the world draws its own demand each night; decision
    # code keeps the unvaried model as its forecast unless one is configured.
    if isinstance(night_cfg, dict) and night_cfg.get("enabled", True):
        from reservation_pricing.demand.night_varying import NightVaryingDemand

        knobs = {k: v for k, v in night_cfg.items() if k != "enabled"}
        forecast_model = forecast_model or demand_model
        demand_model = NightVaryingDemand(demand_model, **knobs)

    kwargs = {k: v for k, v in env_cfg.items() if k in _ENV_KEYS}
    env: gym.Env = ReservationEnv(
        demand_model=demand_model, forecast_model=forecast_model, **kwargs
    )
    # env.operator_view: decision code sees bookings x usual keep rate instead of
    # the true show-up count. Innermost, so every wrapper and policy reads it.
    if operator_view:
        from reservation_pricing.envs.operator_view import OperatorViewEnv

        env = OperatorViewEnv(env)

    control = dict(cfg.get("control") or {}) if isinstance(cfg.get("control"), dict) else {}
    control.update(control_overrides)
    price_only = bool(control.get("price_only", False))
    if price_only:
        from reservation_pricing.controls import get_early_promo, get_selling_limit
        from reservation_pricing.controls.price_mpc import get_price_mpc
        from reservation_pricing.envs.price_only import PriceOnlyWrapper

        sl_cfg = control.get("selling_limit", control)
        if not isinstance(sl_cfg, dict):
            sl_cfg = {"kind": "analytic"}
        sl_controller = get_selling_limit(sl_cfg)
        early_promo = get_early_promo(control)
        mpc = get_price_mpc(control)
        env = PriceOnlyWrapper(env, sl_controller, early_promo=early_promo, mpc=mpc)

    # control.residual: the DP planner proposes, the agent corrects. Joint only.
    residual = control.get("residual")
    if isinstance(residual, dict) and residual.get("enabled", True):
        # The outer guards would read the agent's correction as a full action.
        clashes = [
            k for k in ("price_only", "safe_sl", "price_monotone") if _enabled(control.get(k))
        ]
        if clashes:
            raise ValueError(f"control.residual cannot be combined with {clashes}")
        from reservation_pricing.envs.residual import ResidualPlannerEnv

        env = ResidualPlannerEnv(
            env,
            price_scale=float(residual.get("price_scale", 0.5)),
            limit_scale=float(residual.get("limit_scale", 0.2)),
        )

    # Safe SL projection for joint (price, SL) policies only.
    # Price-only already fills SL via analytic/optimize_1d controllers.
    if not price_only:
        from reservation_pricing.controls.oversell_cap import get_oversell_cap
        from reservation_pricing.envs.oversell_guard import OversellGuardEnv

        safe = get_oversell_cap(control)
        if safe is not None:
            env = OversellGuardEnv(env, safe)

    # Outermost, and only when enabled. A disabled block returns None and
    # leaves the env unwrapped, so the default config is a true no-op.
    from reservation_pricing.controls.price_monotone import get_price_monotone
    from reservation_pricing.envs.price_guard import MonotonePriceEnv

    monotone = get_price_monotone(control)
    if monotone is not None:
        env = MonotonePriceEnv(env, monotone)

    return env
