"""Algorithm registry: config ``algorithm.name`` / ``train.algo`` -> SB3 builder."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

AlgoBuilder = Callable[..., Any]

_REGISTRY: dict[str, str] = {
    "ppo": "ppo",
    "sac": "sac",
    "td3": "td3",
}


def register_algorithm(name: str, key: str | None = None) -> None:
    _REGISTRY[name.lower()] = (key or name).lower()


def list_algorithms() -> list[str]:
    return sorted(set(_REGISTRY) | {"baseline"})


def resolve_algo_name(cfg: dict[str, Any]) -> str:
    """Prefer algorithm.name, fall back to train.algo, default ppo."""
    algo_block = cfg.get("algorithm") or {}
    if isinstance(algo_block, dict) and algo_block.get("name"):
        return str(algo_block["name"]).lower()
    return str(cfg.get("train", {}).get("algo", "ppo")).lower()


def merge_algo_train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge algorithm + train hyperparams.

    Defaults live under ``algorithm``; run-specific / grid overrides under
    ``train`` win on conflict so tune grids and long_*.yaml actually apply.
    """
    algo_block = dict(cfg.get("algorithm") or {})
    train_block = dict(cfg.get("train") or {})
    name = resolve_algo_name(cfg)
    merged: dict[str, Any] = {}
    for k, v in algo_block.items():
        if k == "name":
            continue
        merged[k] = v
    merged.update(train_block)
    merged["algo"] = name
    return merged


def build_model(algo: str, env: Any, train_cfg: dict, seed: int) -> Any:
    """Instantiate an SB3 model from algo name + hyperparam dict."""
    policy = train_cfg.get("policy", "MlpPolicy")
    net_arch = train_cfg.get("net_arch", [128, 128])
    device = train_cfg.get("device", "cpu")
    lr = float(train_cfg.get("learning_rate", 3e-4))
    gamma = float(train_cfg.get("gamma", 0.99))

    common = dict(
        policy=policy,
        env=env,
        learning_rate=lr,
        gamma=gamma,
        seed=seed,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": net_arch},
    )

    algo = algo.lower()
    if algo == "baseline":
        raise ValueError("algorithm.name=baseline is for eval only; use rprl-baselines / evaluate")

    if algo == "ppo":
        return PPO(
            **common,
            n_steps=int(train_cfg.get("n_steps", 2048)),
            batch_size=int(train_cfg.get("batch_size", 64)),
            n_epochs=int(train_cfg.get("n_epochs", 10)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            clip_range=float(train_cfg.get("clip_range", 0.2)),
            ent_coef=float(train_cfg.get("ent_coef", 0.01)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        )
    if algo == "sac":
        return SAC(
            **common,
            buffer_size=int(train_cfg.get("buffer_size", 100_000)),
            learning_starts=int(train_cfg.get("learning_starts", 1000)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            tau=float(train_cfg.get("tau", 0.005)),
            train_freq=int(train_cfg.get("train_freq", 1)),
            gradient_steps=int(train_cfg.get("gradient_steps", 1)),
            ent_coef=train_cfg.get("ent_coef", "auto"),
        )
    if algo == "td3":
        n_actions = int(np.prod(env.action_space.shape))
        noise_std = float(train_cfg.get("action_noise_std", 0.1))
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )
        return TD3(
            **common,
            buffer_size=int(train_cfg.get("buffer_size", 100_000)),
            learning_starts=int(train_cfg.get("learning_starts", 1000)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            tau=float(train_cfg.get("tau", 0.005)),
            train_freq=int(train_cfg.get("train_freq", 1)),
            gradient_steps=int(train_cfg.get("gradient_steps", 1)),
            action_noise=action_noise,
            policy_delay=int(train_cfg.get("policy_delay", 2)),
        )
    raise ValueError(f"Unsupported algo: {algo}. Known: {list_algorithms()}")


def load_sb3_model(path: str, algo: str = "ppo") -> Any:
    algo = algo.lower()
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}.get(algo)
    if cls is None:
        raise ValueError(f"Cannot load algo={algo}")
    return cls.load(path)


def get_algorithm(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return resolved algorithm metadata (name + merged train hyperparams)."""
    name = resolve_algo_name(cfg)
    train_cfg = merge_algo_train_cfg(cfg)
    return {"name": name, "train": train_cfg}
