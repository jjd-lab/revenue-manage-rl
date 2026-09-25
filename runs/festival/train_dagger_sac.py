#!/usr/bin/env python3
"""SAC fine-tuned from the round-5 DAgger clone: can trial and error beat the planner?

Starts from ``artifacts/festival/festival_dagger/dagger_round5.zip`` (the last
round, fixed in advance), fills the replay buffer with 100 of the clone's own
training seasons, warms the critic with the actor frozen, then runs the same
200k SAC steps as ``configs/festival_bc_sac.yaml``. Freezing the actor matters:
``bc.warm_critic`` also steps the actor, and against a critic that has not yet
learned anything that can undo the clone before training starts. So does the
entropy coefficient: SB3's ``auto`` starts it at 1.0, large next to this reward
scale, so it starts at ``INIT_ENT_COEF`` and is tuned from there.

``--shaped`` trains on the shaped reward (``FestivalEnv.potential``) and writes
``festival_dagger_sac_shaped/`` instead.

Writes ``artifacts/festival/festival_dagger_sac/``; ``run_festival.py`` scores it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from reservation_pricing.algorithms.bc import (
    collect_expert_dataset,
    seed_replay_buffer,
    warm_critic,
)
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.train.common import limit_torch_threads, make_monitored

ROOT = Path(__file__).resolve().parents[2]
CLONE = ROOT / "artifacts" / "festival" / "festival_dagger" / "dagger_round5.zip"
SEED = 7
ROLLOUT_SEED = 7000
N_ROLLOUTS = 100
CRITIC_STEPS = 5000
TIMESTEPS = 200_000
INIT_ENT_COEF = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaped", action="store_true")
    shaped = parser.parse_args().shaped
    limit_torch_threads()
    cfg = load_config(ROOT / "configs" / "festival_bc_sac.yaml")
    cfg["festival"] = {**(cfg.get("festival") or {}), "shape_reward": shaped}
    name = "festival_dagger_sac_shaped" if shaped else "festival_dagger_sac"
    MODEL_DIR = ROOT / "artifacts" / "festival" / name
    LOG_DIR = Path(__file__).resolve().parent / name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    vec = DummyVecEnv([make_monitored(cfg, SEED, 0, use_held_out=False)])
    model = SAC.load(str(CLONE), env=vec, seed=SEED)
    with torch.no_grad():
        model.log_ent_coef.fill_(float(np.log(INIT_ENT_COEF)))

    rollouts = collect_expert_dataset(
        lambda: make_env(cfg, use_held_out=False),
        sb3_policy(model),
        n_episodes=N_ROLLOUTS,
        seed=ROLLOUT_SEED,
    )
    n_seeded = seed_replay_buffer(model, rollouts)

    actor_lr = [g["lr"] for g in model.actor.optimizer.param_groups]
    for g in model.actor.optimizer.param_groups:
        g["lr"] = 0.0
    warm_critic(model, gradient_steps=CRITIC_STEPS)
    for g, lr in zip(model.actor.optimizer.param_groups, actor_lr):
        g["lr"] = lr
    model.learning_starts = 0

    eval_env = DummyVecEnv([make_monitored(cfg, SEED + 10_000, 0, use_held_out=True)])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, reset_num_timesteps=True)
    model.save(str(MODEL_DIR / "final_model"))
    (LOG_DIR / "train_meta.json").write_text(
        json.dumps(
            {
                "clone": str(CLONE.relative_to(ROOT)),
                "seed": SEED,
                "n_seeded": n_seeded,
                "critic_warm_steps": CRITIC_STEPS,
                "init_ent_coef": INIT_ENT_COEF,
                "shape_reward": shaped,
                "total_timesteps": TIMESTEPS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
