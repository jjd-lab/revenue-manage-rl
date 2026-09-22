"""Demand models: protocol, registry, linear_legacy, tree_elastic."""

from reservation_pricing.demand.linear_legacy import LinearLegacyDemand
from reservation_pricing.demand.protocol import DemandModel, features_from_state
from reservation_pricing.demand.registry import get_demand_model, list_demand_kinds, register_demand
from reservation_pricing.demand.tree_elastic import TreeElasticDemand

__all__ = [
    "DemandModel",
    "LinearLegacyDemand",
    "TreeElasticDemand",
    "features_from_state",
    "get_demand_model",
    "list_demand_kinds",
    "register_demand",
]
