"""Behavioral cloning utilities for continuous-control SB3 policies (SAC/TD3).

Collect expert (obs, action[, transition]) data from a classical baseline, fit the
actor with MSE on normalized actions, optionally seed the replay buffer, then
hand off to RL fine-tuning (see ``train.bc_finetune``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from reservation_pricing.baselines.policies import BASELINE_FACTORY
from reservation_pricing.envs.reservation import ReservationEnv
from reservation_pricing.metrics import PolicyFn


@dataclass
class ExpertDataset:
    """Transitions collected from an expert policy (single-env rollouts)."""

    observations: np.ndarray  # (N, obs_dim)
    actions: np.ndarray  # (N, act_dim) in env action space
    rewards: np.ndarray  # (N,)
    next_observations: np.ndarray  # (N, obs_dim)
    dones: np.ndarray  # (N,) float 0/1
    episode_starts: np.ndarray  # (N,) bool — True if step starts an episode

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            dones=self.dones,
            episode_starts=self.episode_starts,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExpertDataset":
        data = np.load(path)
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_observations=data["next_observations"],
            dones=data["dones"],
            episode_starts=data["episode_starts"],
        )


def resolve_expert_policy(name: str = "myopic_greedy", **kwargs: Any) -> PolicyFn:
    key = str(name).lower()
    if key not in BASELINE_FACTORY:
        raise ValueError(f"Unknown expert policy {name!r}; known={sorted(BASELINE_FACTORY)}")
    factory = BASELINE_FACTORY[key]
    # Factories are zero-arg callables; myopic accepts optional kwargs via partial-like call
    if key == "myopic_greedy":
        from reservation_pricing.baselines.policies import myopic_greedy_policy

        return myopic_greedy_policy(
            n_grid=int(kwargs.get("n_grid", 41)),
            prefer_closed_form=bool(kwargs.get("prefer_closed_form", True)),
        )
    return factory()


def collect_expert_dataset(
    env_factory: Callable[[], ReservationEnv],
    policy: PolicyFn,
    *,
    n_episodes: int = 300,
    seed: int = 0,
) -> ExpertDataset:
    """Roll out ``policy`` for ``n_episodes`` and stack transitions."""
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    rew_list: list[float] = []
    next_list: list[np.ndarray] = []
    done_list: list[float] = []
    start_list: list[bool] = []

    for ep in range(int(n_episodes)):
        env = env_factory()
        obs, _info = env.reset(seed=int(seed) + ep)
        terminated = truncated = False
        state: dict[str, Any] = {}
        ep_start = True
        while not (terminated or truncated):
            action = np.asarray(policy(obs, env, state), dtype=np.float32)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = float(terminated or truncated)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(action)
            rew_list.append(float(reward))
            next_list.append(np.asarray(next_obs, dtype=np.float32))
            done_list.append(done)
            start_list.append(ep_start)
            obs = next_obs
            ep_start = False

    return ExpertDataset(
        observations=np.stack(obs_list, axis=0),
        actions=np.stack(act_list, axis=0),
        rewards=np.asarray(rew_list, dtype=np.float32),
        next_observations=np.stack(next_list, axis=0),
        dones=np.asarray(done_list, dtype=np.float32),
        episode_starts=np.asarray(start_list, dtype=bool),
    )


def pretrain_actor_mse(
    model: Any,
    dataset: ExpertDataset,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    log_std_target: float = -1.0,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Fit ``model.actor`` so deterministic tanh(mu(obs)) ≈ expert actions (MSE).

    For SAC, also pulls ``log_std`` toward ``log_std_target`` so the warm-started
    policy is not overly stochastic; TD3's actor is deterministic and has no
    ``log_std``. Syncs ``actor_target`` when present (SAC/TD3).
    """
    device = device or str(getattr(model, "device", "cpu"))
    actor = model.actor
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=float(lr))

    obs_t = torch.as_tensor(dataset.observations, device=device, dtype=torch.float32)
    act_t = torch.as_tensor(dataset.actions, device=device, dtype=torch.float32)
    n = len(dataset)
    batch_size = max(1, min(int(batch_size), n))
    n_batches = max(1, (n + batch_size - 1) // batch_size)

    last_mse = float("nan")
    last_std_loss = float("nan")
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        epoch_mse = 0.0
        epoch_std = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            if idx.numel() == 0:
                continue
            o = obs_t[idx]
            a = act_t[idx]
            if hasattr(actor, "get_action_dist_params"):
                # SAC: squashed Gaussian, deterministic action = tanh(mean).
                mean_actions, log_std, _kwargs = actor.get_action_dist_params(o)
                pred = torch.tanh(mean_actions)
                std_loss = F.mse_loss(log_std, torch.full_like(log_std, float(log_std_target)))
            else:
                # TD3: deterministic actor already returns the squashed action,
                # and has no log_std to pull toward a target.
                pred = actor(o)
                std_loss = torch.zeros((), device=pred.device)
            mse = F.mse_loss(pred, a)
            loss = mse + 0.01 * std_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_mse += float(mse.item())
            epoch_std += float(std_loss.item())
        last_mse = epoch_mse / n_batches
        last_std_loss = epoch_std / n_batches

    actor.eval()
    if hasattr(model, "actor_target"):
        model.actor_target.load_state_dict(actor.state_dict())

    return {"bc_mse": last_mse, "bc_log_std_loss": last_std_loss, "n_samples": float(n)}


def seed_replay_buffer(model: Any, dataset: ExpertDataset) -> int:
    """Add expert transitions into an off-policy SB3 replay buffer.

    Returns the number of transitions added (capped by buffer capacity).
    """
    buf = getattr(model, "replay_buffer", None)
    if buf is None:
        raise ValueError("Model has no replay_buffer; seed_replay_buffer is for SAC/TD3")

    n = len(dataset)
    capacity = int(buf.buffer_size)
    n_add = min(n, capacity)
    # SB3 VecEnv layout: batch dim = n_envs (we use 1)
    for i in range(n_add):
        obs = dataset.observations[i : i + 1]
        next_obs = dataset.next_observations[i : i + 1]
        action = dataset.actions[i : i + 1]
        reward = np.asarray([dataset.rewards[i]], dtype=np.float32)
        done = np.asarray([dataset.dones[i]], dtype=np.float32)
        infos = [{"TimeLimit.truncated": False}]
        buf.add(obs, next_obs, action, reward, done, infos)
    return n_add


def warm_critic(
    model: Any,
    *,
    gradient_steps: int = 1000,
    batch_size: Optional[int] = None,
) -> None:
    """Run off-policy gradient updates (no env interaction) to warm critic/actor."""
    if gradient_steps <= 0:
        return
    bs = int(batch_size or getattr(model, "batch_size", 256))
    # ``learn()`` normally installs the logger; warm-start may run before that.
    if not hasattr(model, "_logger") or model._logger is None:
        from stable_baselines3.common.utils import configure_logger

        model._logger = configure_logger(verbose=0, tensorboard_log=None, tb_log_name="")
    if not hasattr(model, "_current_progress_remaining"):
        model._current_progress_remaining = 1.0
    # SAC/TD3 expose .train(gradient_steps, batch_size)
    model.train(gradient_steps=int(gradient_steps), batch_size=bs)
