"""Multi-night festival passes: several products sharing nightly seats.

Separate from the single-night ``ReservationEnv``; a config selects it with a
top-level ``festival:`` block (any ``FestivalConfig`` field, all optional).
"""

from __future__ import annotations

from typing import Any

from reservation_pricing.festival.env import FestivalEnv
from reservation_pricing.festival.model import FestivalConfig

TRAIN_SEED_OFFSET = 1_000_000


def make_festival_env(cfg: dict, **overrides: Any) -> FestivalEnv:
    """Build the env from an experiment config.

    The season is drawn from the reset seed, so held-out means held-out seeds.
    ``use_held_out=False``, which both trainers pass for their training and
    expert-rollout envs, shifts every seed by ``TRAIN_SEED_OFFSET`` so training
    never draws a season that evaluation on seeds 0-29 scores.
    """
    held_out = bool(overrides.pop("use_held_out", True))
    block = {**(cfg.get("festival") or {}), **overrides}
    return FestivalEnv(
        FestivalConfig.from_mapping(block), seed_offset=0 if held_out else TRAIN_SEED_OFFSET
    )


__all__ = ["FestivalConfig", "FestivalEnv", "make_festival_env"]
