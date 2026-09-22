"""Hyperparameter / config search loop (grid or Optuna)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import deep_merge, load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate import evaluate_policy, sb3_policy
from reservation_pricing.train import train

PPO_GRID = [
    # HPs under algorithm (and mirrored in train by run_grid) so they survive merge.
    {
        "algorithm": {
            "learning_rate": 1e-4,
            "ent_coef": 0.01,
            "n_steps": 1024,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 3e-4,
            "ent_coef": 0.02,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 3e-4,
            "ent_coef": 0.05,
            "n_steps": 2048,
            "batch_size": 128,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 1e-3,
            "ent_coef": 0.02,
            "n_steps": 1024,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.1,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 1e-4,
            "ent_coef": 0.02,
            "n_steps": 2048,
            "batch_size": 128,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 8,
        }
    },
    {
        "algorithm": {
            "learning_rate": 5e-4,
            "ent_coef": 0.03,
            "n_steps": 1024,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
    # Extra cells for tree_long screen
    {
        "algorithm": {
            "learning_rate": 5e-4,
            "ent_coef": 0.01,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 1.0,
            "clip_range": 0.1,
            "n_envs": 4,
        }
    },
    {
        "algorithm": {
            "learning_rate": 1e-3,
            "ent_coef": 0.05,
            "n_steps": 2048,
            "batch_size": 128,
            "gamma": 1.0,
            "clip_range": 0.2,
            "n_envs": 4,
        }
    },
]


def _write_trial_config(base_cfg: dict, override: dict, path: Path) -> Path:
    import yaml

    cfg = deep_merge(base_cfg, override)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _eval_trained(meta: dict, cfg: dict, n_episodes: int, shortfall_weight: float) -> dict:
    model = load_sb3_model(meta["model_path"], algo=meta["algo"])
    seeds = list(range(n_episodes))

    def env_factory():
        return make_env(cfg, use_held_out=True)

    agg, _ = evaluate_policy(
        env_factory,
        sb3_policy(model, deterministic=True),
        n_episodes=n_episodes,
        seeds=seeds,
        shortfall_weight=shortfall_weight,
    )
    return agg.to_dict()


def run_grid(
    config_path: Optional[str] = None,
    timesteps_per_trial: int = 20_000,
    eval_episodes: int = 15,
    out_dir: Optional[str] = None,
    grid: Optional[list[dict]] = None,
) -> dict[str, Any]:
    base_cfg = load_config(config_path)
    tune_cfg = base_cfg.get("tune", {})
    timesteps = int(timesteps_per_trial or tune_cfg.get("timesteps_per_trial", 20_000))
    eval_episodes = int(eval_episodes or tune_cfg.get("eval_episodes", 15))
    shortfall_weight = float(tune_cfg.get("shortfall_weight", 50.0))
    grid = grid or PPO_GRID

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir else Path("runs") / f"tune_grid_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    # Trial weights stay under train.model_dir (artifacts/), never under runs/.
    model_root = Path(base_cfg.get("train", {}).get("model_dir", "artifacts")) / root.name

    rows: list[dict] = []
    best: Optional[dict] = None

    for i, override in enumerate(grid):
        trial_name = f"trial_{i:03d}"
        trial_dir = root / trial_name
        algo_hp = dict(override.get("algorithm") or override.get("train") or {})
        # Drop non-algo keys if present
        for drop in ("algo", "total_timesteps", "seed", "model_dir", "log_dir"):
            algo_hp.pop(drop, None)
        override_train = {
            "algorithm": {"name": "ppo", **algo_hp},
            "train": {
                "algo": "ppo",
                "total_timesteps": timesteps,
                "seed": 42 + i,
                "model_dir": str(model_root),
                "log_dir": str(root),
                **algo_hp,
            },
        }
        cfg_path = _write_trial_config(base_cfg, override_train, trial_dir / "config.yaml")

        meta = train(
            config_path=str(cfg_path),
            total_timesteps=timesteps,
            seed=42 + i,
            run_name=trial_name,
            out_dir=str(root),
        )
        trial_cfg = load_config(cfg_path)
        metrics = _eval_trained(meta, trial_cfg, eval_episodes, shortfall_weight)
        row = {
            "trial": trial_name,
            "timesteps": timesteps,
            "override": json.dumps(override),
            "model_path": meta["model_path"],
            **{
                k: metrics[k]
                for k in (
                    "mean_true_revenue",
                    "mean_load_factor",
                    "mean_remain_inv",
                    "mean_shaped_return",
                    "score",
                    "sellout_rate",
                )
            },
        }
        rows.append(row)
        with (trial_dir / "metrics.json").open("w") as f:
            json.dump({"meta": meta, "metrics": metrics}, f, indent=2, default=str)

        if best is None or row["score"] > best["score"]:
            best = row

    csv_path = root / "tuning_results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = root / "tuning_summary.md"
    lines = [
        "# Tuning summary",
        "",
        f"- Method: grid ({len(rows)} trials)",
        f"- Timesteps / trial: {timesteps}",
        f"- Selection metric: `score = mean_true_revenue - {shortfall_weight} * mean_capacity_shortfall`",
        f"- Output dir: `{root}`",
        "",
        "## Results",
        "",
    ]
    if rows:
        headers = [
            "trial",
            "mean_true_revenue",
            "mean_load_factor",
            "mean_remain_inv",
            "score",
            "sellout_rate",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in sorted(rows, key=lambda x: -x["score"]):
            lines.append(
                "| "
                + " | ".join(
                    str(round(r[h], 4) if isinstance(r[h], float) else r[h]) for h in headers
                )
                + " |"
            )
        lines += ["", f"**Winner:** `{best['trial']}` with score={best['score']:.2f}", ""]
        lines += [f"- Model: `{best['model_path']}`", f"- Override: `{best['override']}`", ""]
    summary_path.write_text("\n".join(lines))

    return {"root": str(root), "rows": rows, "best": best, "summary": str(summary_path)}


def run_optuna(
    config_path: Optional[str] = None,
    n_trials: int = 8,
    timesteps_per_trial: int = 20_000,
    eval_episodes: int = 15,
    out_dir: Optional[str] = None,
) -> dict[str, Any]:
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "Optuna is required for method=optuna. Install with: pip install 'reservation-pricing[tune]'"
        ) from e

    base_cfg = load_config(config_path)
    tune_cfg = base_cfg.get("tune", {})
    timesteps = int(timesteps_per_trial or tune_cfg.get("timesteps_per_trial", 20_000))
    eval_episodes = int(eval_episodes or tune_cfg.get("eval_episodes", 15))
    shortfall_weight = float(tune_cfg.get("shortfall_weight", 50.0))
    n_trials = int(n_trials or tune_cfg.get("n_trials", 8))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(out_dir) if out_dir else Path("runs") / f"tune_optuna_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    model_root = Path(base_cfg.get("train", {}).get("model_dir", "artifacts")) / root.name

    rows: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        override = {
            "algorithm": {"name": "ppo"},
            "train": {
                "algo": "ppo",
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [1024, 2048]),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "total_timesteps": timesteps,
                "seed": 100 + trial.number,
                "model_dir": str(model_root),
                "log_dir": str(root),
            },
        }
        trial_name = f"trial_{trial.number:03d}"
        cfg_path = _write_trial_config(base_cfg, override, root / trial_name / "config.yaml")
        meta = train(
            config_path=str(cfg_path),
            total_timesteps=timesteps,
            seed=100 + trial.number,
            run_name=trial_name,
            out_dir=str(root),
        )
        trial_cfg = load_config(cfg_path)
        metrics = _eval_trained(meta, trial_cfg, eval_episodes, shortfall_weight)
        row = {
            "trial": trial_name,
            "timesteps": timesteps,
            "override": json.dumps(override),
            "model_path": meta["model_path"],
            **{
                k: metrics[k]
                for k in (
                    "mean_true_revenue",
                    "mean_load_factor",
                    "mean_remain_inv",
                    "mean_shaped_return",
                    "score",
                    "sellout_rate",
                )
            },
        }
        rows.append(row)
        with (root / trial_name / "metrics.json").open("w") as f:
            json.dump(
                {"meta": meta, "metrics": metrics, "params": trial.params}, f, indent=2, default=str
            )
        return float(metrics["score"])

    sampler = optuna.samplers.TPESampler(seed=int((base_cfg.get("train") or {}).get("seed", 42)))
    study = optuna.create_study(direction="maximize", study_name="reservation_ppo", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_row = max(rows, key=lambda r: r["score"]) if rows else None
    csv_path = root / "tuning_results.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = root / "tuning_summary.md"
    lines = [
        "# Tuning summary",
        "",
        f"- Method: optuna ({n_trials} trials)",
        f"- Timesteps / trial: {timesteps}",
        f"- Selection metric: `score = mean_true_revenue - {shortfall_weight} * mean_capacity_shortfall`",
        f"- Best params: `{study.best_params}`",
        f"- Best value: {study.best_value:.2f}",
        "",
    ]
    if best_row:
        lines += [
            f"**Winner trial:** `{best_row['trial']}`",
            f"- Model: `{best_row['model_path']}`",
            "",
        ]
    summary_path.write_text("\n".join(lines))

    return {
        "root": str(root),
        "rows": rows,
        "best": best_row,
        "best_params": study.best_params,
        "summary": str(summary_path),
    }


def run_tune(
    config_path: Optional[str] = None,
    method: Optional[str] = None,
    **kwargs,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    method = (method or cfg.get("tune", {}).get("method", "grid")).lower()
    if method == "optuna":
        return run_optuna(config_path=config_path, **kwargs)
    kwargs.pop("n_trials", None)  # grid runs every cell; the trial count is the grid size
    return run_grid(config_path=config_path, **kwargs)
