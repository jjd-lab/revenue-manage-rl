"""BC warm-start from an expert baseline → SAC (or TD3) fine-tune.

Config-driven entrypoint. Typical flow::

    rprl-bc-sac -c configs/experiment_bc_sac.yaml

Steps
-----
1. Collect expert trajectories on **train** months (not held-out).
2. Build SAC, behavioral-clone the actor (MSE on normalized actions).
3. Optionally seed the replay buffer + warm the critic.
4. Fine-tune with ``model.learn``.
5. Save under ``<model_dir>/<run_name>/``, the same place ``train_from_config``
   writes. The curated ``artifacts/bc_sac/rl_bc_sac_final.zip`` is a copy of that
   file, not the trainer's own path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from reservation_pricing.algorithms.bc import (
    ExpertDataset,
    collect_expert_dataset,
    pretrain_actor_mse,
    resolve_expert_policy,
    seed_replay_buffer,
    warm_critic,
)
from reservation_pricing.algorithms.registry import (
    build_model,
    merge_algo_train_cfg,
    resolve_algo_name,
)
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.train.common import eval_settings, limit_torch_threads, make_monitored
from reservation_pricing.train.runner import output_dirs, resolve_run_name


def bc_finetune_from_config(
    cfg: dict[str, Any],
    *,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    run_name: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Run BC → off-policy fine-tune using an already-loaded config dict."""
    limit_torch_threads()
    train_cfg = merge_algo_train_cfg(cfg)
    bc_cfg = dict(cfg.get("bc") or {})

    if total_timesteps is not None:
        train_cfg["total_timesteps"] = int(total_timesteps)
    if seed is not None:
        train_cfg["seed"] = int(seed)

    seed_i = int(train_cfg.get("seed", 42))
    algo_name = str(train_cfg.get("algo", resolve_algo_name(cfg))).lower()
    if algo_name not in ("sac", "td3"):
        raise ValueError(
            f"bc_finetune requires algorithm.name sac|td3 (got {algo_name!r}); "
            "on-policy algos destroy the clone without an aux BC loss"
        )
    steps = int(train_cfg.get("total_timesteps", 150_000))

    run_name = resolve_run_name(run_name, train_cfg, algo_name, seed_i, prefix="bc_")

    # Same layout as train_from_config: <model_dir>/<run_name>/, not the family
    # directory itself. The curated zip in artifacts/bc_sac/ is a copy.
    model_dir, run_dir = output_dirs(train_cfg, run_name, out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect expert data (train months) ---
    n_expert_eps = int(bc_cfg.get("n_episodes", 300))
    expert_name = str(bc_cfg.get("expert_policy", "myopic_greedy"))
    dataset_path = run_dir / "expert_dataset.npz"
    if bc_cfg.get("dataset_path") and Path(bc_cfg["dataset_path"]).exists():
        dataset = ExpertDataset.load(bc_cfg["dataset_path"])
        collect_meta = {"loaded_from": str(bc_cfg["dataset_path"]), "n_transitions": len(dataset)}
    else:

        def train_env_factory():
            return make_env(cfg, use_held_out=False)

        expert_model_path = bc_cfg.get("expert_model_path")
        policy = resolve_expert_policy(
            expert_name,
            model_path=expert_model_path,
            algo=bc_cfg.get("expert_algo"),
            n_grid=int(bc_cfg.get("n_grid", 41)),
            prefer_closed_form=bool(bc_cfg.get("prefer_closed_form", True)),
        )
        dataset = collect_expert_dataset(
            train_env_factory,
            policy,
            n_episodes=n_expert_eps,
            seed=seed_i,
            executed_action_key=bc_cfg.get("executed_action_key"),
        )
        dataset.save(dataset_path)
        collect_meta = {
            "expert_policy": expert_name,
            "expert_model_path": None if expert_model_path is None else str(expert_model_path),
            "expert_algo": bc_cfg.get("expert_algo"),
            "executed_action_key": bc_cfg.get("executed_action_key"),
            "n_episodes": n_expert_eps,
            "n_transitions": len(dataset),
            "dataset_path": str(dataset_path),
        }

    # --- 2. Build model ---
    train_cfg["n_envs"] = 1  # single-env for clean buffer seeding
    vec = DummyVecEnv([make_monitored(cfg, seed_i, 0, use_held_out=False)])
    model = build_model(algo_name, vec, train_cfg, seed_i)

    # --- 3. BC pretrain actor ---
    bc_metrics = pretrain_actor_mse(
        model,
        dataset,
        epochs=int(bc_cfg.get("epochs", 50)),
        batch_size=int(bc_cfg.get("batch_size", 256)),
        lr=float(bc_cfg.get("lr", 1e-3)),
        log_std_target=float(bc_cfg.get("log_std_target", -1.0)),
        device=str(train_cfg.get("device", "cpu")),
    )

    bc_only_path = model_dir / "bc_only_model"
    model.save(str(bc_only_path))

    # --- 4. Seed buffer + optional critic warm ---
    n_seeded = 0
    if bool(bc_cfg.get("seed_replay_buffer", True)):
        n_seeded = seed_replay_buffer(model, dataset)
    critic_steps = int(bc_cfg.get("critic_warm_steps", 1000))
    if n_seeded > 0 and critic_steps > 0:
        warm_critic(
            model,
            gradient_steps=critic_steps,
            batch_size=int(train_cfg.get("batch_size", 256)),
        )
    if bc_cfg.get("learning_starts") is not None:
        model.learning_starts = int(bc_cfg["learning_starts"])

    # --- 5. Fine-tune ---
    ev = eval_settings(train_cfg, steps)
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

    best_src = model_dir / "best" / "best_model.zip"
    if best_src.exists():
        shutil.copy2(best_src, model_dir / "rl_bc_sac_best.zip")
    shutil.copy2(str(final_path) + ".zip", model_dir / "rl_bc_sac_final.zip")
    shutil.copy2(str(bc_only_path) + ".zip", model_dir / "bc_only.zip")

    meta = {
        "run_name": run_name,
        "algo": algo_name,
        "seed": seed_i,
        "total_timesteps": steps,
        "model_path": str(final_path) + ".zip",
        "bc_only_path": str(bc_only_path) + ".zip",
        "best_model_dir": str(model_dir / "best"),
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "demand_kind": (cfg.get("demand") or {}).get("kind"),
        "bc": {
            **collect_meta,
            **bc_metrics,
            "n_seeded": n_seeded,
            "critic_warm_steps": critic_steps,
            "epochs": int(bc_cfg.get("epochs", 50)),
        },
        "config": cfg,
    }
    with (run_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    with (model_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)

    vec.close()
    eval_env.close()
    return meta


def bc_finetune(
    config_path: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = None,
    run_name: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    return bc_finetune_from_config(
        cfg,
        total_timesteps=total_timesteps,
        seed=seed,
        run_name=run_name,
        out_dir=out_dir,
    )


__all__ = ["bc_finetune", "bc_finetune_from_config"]
