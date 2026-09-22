"""Demand model registry: config ``demand.kind`` -> instance."""

from __future__ import annotations

from typing import Any, Callable

from reservation_pricing.demand.linear_legacy import LinearLegacyDemand
from reservation_pricing.demand.tree_elastic import TreeElasticDemand

DemandFactory = Callable[..., Any]

_REGISTRY: dict[str, DemandFactory] = {
    "linear_legacy": LinearLegacyDemand,
    "tree_elastic": TreeElasticDemand,
}


def register_demand(kind: str, factory: DemandFactory) -> None:
    _REGISTRY[kind] = factory


def list_demand_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_demand_model(cfg: dict[str, Any] | None = None, **overrides: Any) -> Any:
    """Build a DemandModel from config.

    Accepts either full experiment cfg (uses ``cfg['demand']``) or a demand block.
    Legacy: if no demand block, falls back to linear_legacy using env noise knobs.
    """
    cfg = cfg or {}
    if "demand" in cfg and isinstance(cfg["demand"], dict):
        demand_cfg = dict(cfg["demand"])
    elif "kind" in cfg:
        demand_cfg = dict(cfg)
    else:
        # Backward-compatible default when only env block existed historically
        env = cfg.get("env", {})
        demand_cfg = {
            "kind": "linear_legacy",
            "demand_noise_std": env.get("demand_noise_std", 8.0),
        }

    demand_cfg.update(overrides)
    kind = str(demand_cfg.pop("kind", "tree_elastic"))
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown demand kind {kind!r}. Known: {list_demand_kinds()}")
    # Pull noise from env if not set on demand
    if "demand_noise_std" not in demand_cfg and "env" in cfg:
        if "demand_noise_std" in cfg["env"]:
            demand_cfg["demand_noise_std"] = cfg["env"]["demand_noise_std"]
    return _REGISTRY[kind](**demand_cfg)
