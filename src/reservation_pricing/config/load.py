"""YAML config loading, merging, and light validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

KNOWN_DEMAND_KINDS = {"linear_legacy", "tree_elastic"}
KNOWN_ALGOS = {"ppo", "sac", "td3", "baseline"}
KNOWN_SL_KINDS = {"analytic", "optimize_1d"}
KNOWN_SAFE_SL_KINDS = {"analytic", "chance"}
KNOWN_MONOTONE_DIRECTIONS = {"up", "down"}
KNOWN_MONOTONE_MODES = {"project", "penalty"}


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


PACKAGED_DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def package_root() -> Optional[Path]:
    """Repo root: the nearest ancestor holding ``configs/default.yaml``, or ``None``.

    Walks up from this file rather than counting directory levels, so it holds
    for a source checkout and an editable install alike. A non-editable install
    has no repo root; callers fall back to :data:`PACKAGED_DEFAULT`.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        # pyproject.toml distinguishes the checkout from the packaged configs/ copy.
        if (parent / "configs" / "default.yaml").is_file() and (
            parent / "pyproject.toml"
        ).is_file():
            return parent
    return None


def default_config_path() -> Path:
    """The schema-bearing ``default.yaml``: the checkout's copy, else the packaged one.

    The two files are kept byte-identical by a test, so which one is found only
    changes where experiment configs are resolved from, never what they mean.
    """
    root = package_root()
    if root is not None:
        return root / "configs" / "default.yaml"
    if PACKAGED_DEFAULT.is_file():
        return PACKAGED_DEFAULT
    raise FileNotFoundError(
        f"No default.yaml found: neither a checkout above {Path(__file__).resolve()} "
        f"nor the packaged copy at {PACKAGED_DEFAULT}. Reinstall the package."
    )


def load_config(path: str | Path | None = None, base: str | Path | None = None) -> dict[str, Any]:
    """Load config, optionally merging `path` onto `base` (default.yaml).

    Raises if the base config cannot be read. It carries the whole schema, so
    silently continuing without it yields a config that looks valid and trains
    the wrong demand model with the wrong reward weights.
    """
    default_path = Path(base) if base else default_config_path()
    if not default_path.is_file():
        raise FileNotFoundError(f"Base config not found: {default_path}")
    cfg = load_yaml(default_path)
    if path is not None:
        cfg = deep_merge(cfg, load_yaml(path))
    validate_config(cfg)
    return cfg


def _block_enabled(control: dict[str, Any], key: str) -> bool:
    block = control.get(key)
    return isinstance(block, dict) and bool(block.get("enabled", False))


def _reject_monotone_with_price_overrides(control: dict[str, Any]) -> None:
    """Promo and MPC rewrite price inside PriceOnlyWrapper, after any outer clamp.

    Reject the combination when the monotone constraint is enabled. Presence of
    a disabled block is fine: the default config carries both.
    """
    mono = control.get("price_monotone")
    if not isinstance(mono, dict) or not bool(mono.get("enabled", False)):
        return
    direction = str(mono.get("direction", "up")).lower().strip()
    if direction not in KNOWN_MONOTONE_DIRECTIONS:
        raise ValueError(
            f"Unknown control.price_monotone.direction={direction!r}; "
            f"expected one of {sorted(KNOWN_MONOTONE_DIRECTIONS)}"
        )
    mode = str(mono.get("mode", "project")).lower().strip()
    if mode not in KNOWN_MONOTONE_MODES:
        raise ValueError(
            f"Unknown control.price_monotone.mode={mode!r}; "
            f"expected one of {sorted(KNOWN_MONOTONE_MODES)}"
        )
    if (
        _block_enabled(control, "early_promo")
        or _block_enabled(control, "promo")
        or _block_enabled(control, "mpc")
    ):
        raise ValueError(
            "control.price_monotone cannot be combined with an enabled "
            "early_promo, promo, or mpc block: those override price inside "
            "PriceOnlyWrapper after the outer projection, so the guarantee "
            "would be silently violated"
        )


def validate_config(cfg: dict[str, Any]) -> None:
    """Light structural checks; raises ValueError on obvious mistakes."""
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict")

    demand = cfg.get("demand", {})
    if demand:
        kind = str(demand.get("kind", "tree_elastic"))
        if kind not in KNOWN_DEMAND_KINDS:
            raise ValueError(
                f"Unknown demand.kind={kind!r}; expected one of {sorted(KNOWN_DEMAND_KINDS)}"
            )

    # Support both algorithm: {name: ...} and legacy train.algo
    algo_block = cfg.get("algorithm") or {}
    name = None
    if isinstance(algo_block, dict) and algo_block.get("name"):
        name = str(algo_block["name"]).lower()
    elif cfg.get("train", {}).get("algo"):
        name = str(cfg["train"]["algo"]).lower()
    if name is not None and name not in KNOWN_ALGOS:
        raise ValueError(f"Unknown algorithm.name={name!r}; expected one of {sorted(KNOWN_ALGOS)}")

    control = cfg.get("control") or {}
    if isinstance(control, dict) and control:
        sl = control.get("selling_limit", control)
        if isinstance(sl, dict) and sl.get("kind") is not None:
            sl_kind = str(sl["kind"])
            if sl_kind not in KNOWN_SL_KINDS:
                raise ValueError(
                    f"Unknown control.selling_limit.kind={sl_kind!r}; "
                    f"expected one of {sorted(KNOWN_SL_KINDS)}"
                )
        safe = control.get("safe_sl")
        if isinstance(safe, dict) and safe.get("kind") is not None and safe.get("enabled", False):
            sk = str(safe["kind"])
            if sk not in KNOWN_SAFE_SL_KINDS:
                raise ValueError(
                    f"Unknown control.safe_sl.kind={sk!r}; "
                    f"expected one of {sorted(KNOWN_SAFE_SL_KINDS)}"
                )
        _reject_monotone_with_price_overrides(control)
