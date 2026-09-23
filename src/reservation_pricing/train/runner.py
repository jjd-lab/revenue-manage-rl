"""Train SB3 agents from experiment config (algorithm + env + demand)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from reservation_pricing.algorithms.registry import (
    build_model,
    load_sb3_model,
    merge_algo_train_cfg,
    resolve_algo_name,
)
from reservation_pricing.config import load_config
from reservation_pricing.train.common import eval_settings, limit_torch_threads, make_monitored


def output_dirs(
    train_cfg: dict[str, Any],
    run_name: str,
    out_dir: Optional[str] = None,
) -> tuple[Path, Path]:
    """``(model_dir, run_dir)`` for one named run.

    Both trainers use this, so a behaviour-clone warm start and a standard
    run land in the same shape: ``<model_dir>/<run_name>/`` and
    ``<log_dir or out_dir>/<run_name>/``.
    """
    model_root = Path(train_cfg.get("model_dir", "artifacts"))
    log_root = Path(out_dir) if out_dir else Path(train_cfg.get("log_dir", "runs"))
    return model_root / run_name, log_root / run_name


def resolve_run_name(
    run_name: Optional[str],
    train_cfg: dict[str, Any],
    algo_name: str,
    seed: int,
    prefix: str = "",
) -> str:
    """Explicit argument wins, then ``train.run_name``, then a timestamp.

    Shared by both trainers so a config-named experiment writes the same
    directory whichever entry point produced it (see ``docs/EXPERIMENTS.md``).
    """
    if run_name:
        return run_name
    from_cfg = train_cfg.get("run_name")
    if from_cfg:
        return str(from_cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{algo_name}_{stamp}_s{seed}"


def train_from_config(
    cfg: dict[str, Any],
    *,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    algo: Optional[str] = None,
    run_name: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Train using an already-loaded config dict."""
    limit_torch_threads()
    train_cfg = merge_algo_train_cfg(cfg)
    if algo:
        train_cfg["algo"] = algo
    if total_timesteps is not None:
        train_cfg["total_timesteps"] = int(total_timesteps)
    if seed is not None:
        train_cfg["seed"] = int(seed)

    seed_i = int(train_cfg.get("seed", 42))
    algo_name = str(train_cfg.get("algo", resolve_algo_name(cfg))).lower()
    if algo_name == "baseline":
        raise ValueError("Cannot train algorithm.name=baseline; use rprl-baselines")
    steps = int(train_cfg.get("total_timesteps", 100_000))

    run_name = resolve_run_name(run_name, train_cfg, algo_name, seed_i)
    model_dir, run_dir = output_dirs(train_cfg, run_name, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(train_cfg.get("n_envs", 1))
    vec = DummyVecEnv([make_monitored(cfg, seed_i, i, use_held_out=False) for i in range(n_envs)])

    model = build_model(algo_name, vec, train_cfg, seed_i)

    ev = eval_settings(train_cfg, steps, n_envs)
    eval_env = DummyVecEnv(
        [make_monitored(cfg, seed_i + 10_000, 0, use_held_out=ev["use_held_out"])]
    )
    (model_dir / "best").mkdir(parents=True, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=ev["eval_freq"],
        n_eval_episodes=ev["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=steps, callback=eval_cb, progress_bar=False)
    final_path = model_dir / "final_model"
    model.save(str(final_path))

    meta = {
        "run_name": run_name,
        "algo": algo_name,
        "seed": seed_i,
        "total_timesteps": steps,
        "model_path": str(final_path) + ".zip",
        "best_model_dir": str(model_dir / "best"),
        "run_dir": str(run_dir),
        "demand_kind": (cfg.get("demand") or {}).get("kind"),
        "config": cfg,
    }
    with (run_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    vec.close()
    eval_env.close()
    return meta


def train(
    config_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    algo: Optional[str] = None,
    run_name: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    return train_from_config(
        cfg,
        total_timesteps=total_timesteps,
        seed=seed,
        algo=algo,
        run_name=run_name,
        out_dir=out_dir,
    )


# Re-export for callers
__all__ = ["train", "train_from_config", "load_sb3_model", "build_model"]
