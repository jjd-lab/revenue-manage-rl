"""Train any experiment config at extra seeds.

Two training modes, chosen from the config:

- a ``bc:`` block uses the behaviour-clone warm start (``bc_finetune_from_config``)
- anything else uses the standard trainer (``train_from_config``)

Both write ``artifacts/training_seeds/<label>_s<seed>/final_model.zip``.
The oversell cap is not a training mode; it is an eval wrapper around a
finished checkpoint.

Seed 42 is the shipped pair and is not retrained. Held-out nights 0–29 and
the hyperparameter block 100–129 are refused. The driver also refuses to
write ``artifacts/bc_sac/`` or ``artifacts/pace_ppo/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reservation_pricing.config import load_config
from reservation_pricing.config.load import package_root
from reservation_pricing.train.runner import output_dirs

NEW_SEEDS = (43, 44, 46)
# (config, directory label). The label is the run_name stem, so pace stays
# pace_s43 — the same folder the standard trainer already filled.
DEFAULT_CONFIGS = (
    ("configs/experiment_price_only_pace_ppo.yaml", "pace"),
    ("configs/experiment_bc_sac.yaml", "bc_sac"),
)
PUBLISHED_DIRS = ("artifacts/bc_sac", "artifacts/pace_ppo")
RESERVED_SEEDS = frozenset([42, *range(30), *range(100, 130)])


def repo_root() -> Path:
    root = package_root()
    if root is None:
        raise RuntimeError("training seeds must be launched from a source checkout")
    return root


def training_mode(cfg: dict) -> str:
    """``bc`` when the config warm-starts from an expert, else ``standard``."""
    return "bc" if cfg.get("bc") else "standard"


def assert_outside_published(path: Path) -> None:
    """Raise if ``path`` is a published checkpoint directory or anything inside one."""
    resolved = path.resolve()
    root = repo_root()
    for relative in PUBLISHED_DIRS:
        banned = (root / relative).resolve()
        if resolved == banned or banned in resolved.parents:
            raise ValueError(
                f"refusing to write {resolved}: that tree holds a published checkpoint ({relative})"
            )


def require_seed(seed: int) -> None:
    if seed in RESERVED_SEEDS:
        raise ValueError(
            f"training seed {seed} is reserved: held-out nights are 0–29, "
            "seed 42 is the shipped checkpoint, and 100–129 is the hyperparameter block"
        )


def destination(label: str, seed: int) -> tuple[Path, Path]:
    """``(model_dir, log_dir)`` for one seed. Both trainers append nothing further."""
    root = repo_root()
    run_name = f"{label}_s{seed}"
    model_dir = root / "artifacts" / "training_seeds" / run_name
    log_dir = root / "runs" / "training_seeds" / run_name
    assert_outside_published(model_dir)
    assert_outside_published(log_dir)
    return model_dir, log_dir


def prepare(
    config_path: str | Path, seed: int, *, label: str | None = None
) -> tuple[str, dict, str, Path]:
    """Load ``config_path`` and point its outputs at this seed. Does not train.

    Returns ``(mode, cfg, run_name, log_root)``. ``cfg['train']['model_dir']``
    is the parent both trainers join with ``run_name``.
    """
    require_seed(seed)
    path = Path(config_path)
    cfg = load_config(path if path.is_absolute() else repo_root() / path)
    mode = training_mode(cfg)
    stem = label or str(cfg.get("train", {}).get("run_name") or path.stem)
    model_dir, log_dir = destination(stem, seed)
    run_name = model_dir.name
    cfg.setdefault("train", {})["model_dir"] = str(model_dir.parent)
    return mode, cfg, run_name, log_dir.parent


def train_config(config_path: str | Path, seed: int, *, label: str | None = None) -> dict:
    """Train one config at one seed, or return the existing checkpoint."""
    mode, cfg, run_name, log_root = prepare(config_path, seed, label=label)
    model_dir, _run_dir = output_dirs(cfg["train"], run_name, str(log_root))
    final = model_dir / "final_model.zip"
    if final.is_file():
        return {
            "seed": seed,
            "algo": mode,
            "model_path": str(final),
            "skipped": True,
            "run_name": run_name,
        }
    if mode == "bc":
        from reservation_pricing.train.bc_finetune import bc_finetune_from_config

        return bc_finetune_from_config(cfg, seed=seed, run_name=run_name, out_dir=str(log_root))
    from reservation_pricing.train.runner import train_from_config

    return train_from_config(cfg, seed=seed, run_name=run_name, out_dir=str(log_root))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train experiment configs at extra seeds")
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        dest="configs",
        help="Experiment YAML. Repeat for several. Default: pace PPO and BC→SAC.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Directory stem for the matching --config. Default: that config's run_name.",
    )
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    args = parser.parse_args(argv)
    seeds = args.seeds or list(NEW_SEEDS)
    if args.configs:
        labels = args.label or [None] * len(args.configs)
        if args.label and len(args.label) != len(args.configs):
            parser.error("--label must be given once per --config, or not at all")
        pairs = list(zip(args.configs, labels))
    else:
        pairs = list(DEFAULT_CONFIGS)
    # Standard runs are short. Finish them before a behaviour-clone warm start
    # so a long SAC fine-tune is not what you wait on to see the first zip.
    prepared = []
    for config_path, label in pairs:
        for seed in seeds:
            mode, _cfg, _run_name, _log_root = prepare(config_path, seed, label=label)
            prepared.append((mode, config_path, seed, label))
    prepared.sort(key=lambda item: item[0] == "bc")
    for _mode, config_path, seed, label in prepared:
        meta = train_config(config_path, seed, label=label)
        skipped = " (existing)" if meta.get("skipped") else ""
        print(
            f"seed {meta.get('seed', seed)} {meta.get('algo')} -> {meta['model_path']}{skipped}",
            flush=True,
        )


if __name__ == "__main__":
    main()
