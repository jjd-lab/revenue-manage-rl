"""Pieces shared by the two trainers (``rprl-train`` and ``rprl-bc-sac``)."""

from __future__ import annotations

import os

from stable_baselines3.common.monitor import Monitor

from reservation_pricing.envs import make_env


def limit_torch_threads() -> None:
    """Avoid OpenMP/Torch oversubscription on small CPU boxes."""
    n = int(os.environ.get("RPRL_TORCH_THREADS", "1"))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(max(1, min(n, 2)))
    except Exception:
        pass


def make_monitored(cfg: dict, seed: int, rank: int = 0, use_held_out: bool = False):
    def _thunk():
        env = make_env(cfg, use_held_out=use_held_out)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _thunk


def eval_freq(total_timesteps: int, n_envs: int = 1) -> int:
    """Five held-out evaluations per run, never more often than every 2000 calls."""
    return max(int(total_timesteps) // (5 * max(int(n_envs), 1)), 2000)
