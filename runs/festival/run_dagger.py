#!/usr/bin/env python3
"""Can a policy network reproduce the festival planner, once it sees its own mistakes?

The clone in ``configs/festival_bc_sac.yaml`` matches the planner on the
planner's own seasons but drifts once it runs alone: it only ever saw states the
planner reaches. DAgger fixes that. Each round the current clone plays
``SEASONS_PER_ROUND`` training seasons, the planner labels every state the clone
visits, and the clone is refit on everything labelled so far.

Round 0 is the clone as shipped by ``rprl-bc-sac``. After each round the clone
is scored on held-out seeds 0-29 against the planner rows of ``per_season.csv``.

Needs ``artifacts/festival/festival_bc_sac/`` (``rprl-bc-sac -c
configs/festival_bc_sac.yaml``) and ``per_season.csv`` (``run_festival.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.bc import ExpertDataset, pretrain_actor_mse
from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.festival.evaluate import evaluate, paired_interval
from reservation_pricing.festival.planner import planner_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BC = ROOT / "artifacts" / "festival" / "festival_bc_sac"
MODEL_DIR = ROOT / "artifacts" / "festival" / "festival_dagger"
SEEDS = list(range(30))
ROUNDS = 5
SEASONS_PER_ROUND = 30
EPOCHS = 30
FIRST_TRAIN_SEED = 5000


def label_clone_seasons(model, cfg: dict, seeds: list[int]) -> ExpertDataset:
    """Play the clone; record the planner's action at every state it reaches."""
    env = make_env(cfg, use_held_out=False)
    planner = planner_policy()
    cols: dict[str, list] = {k: [] for k in ("obs", "act", "rew", "next", "done", "start")}
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, start = False, True
        while not done:
            label = planner(obs, env, {})
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            for k, v in zip(cols, (obs, label, reward, next_obs, float(done), start)):
                cols[k].append(v)
            obs, start = next_obs, False
    return ExpertDataset(
        observations=np.stack(cols["obs"]).astype(np.float32),
        actions=np.stack(cols["act"]).astype(np.float32),
        rewards=np.asarray(cols["rew"], dtype=np.float32),
        next_observations=np.stack(cols["next"]).astype(np.float32),
        dones=np.asarray(cols["done"], dtype=np.float32),
        episode_starts=np.asarray(cols["start"], dtype=bool),
    )


def merge(a: ExpertDataset, b: ExpertDataset) -> ExpertDataset:
    return ExpertDataset(
        *(
            np.concatenate([getattr(a, f), getattr(b, f)])
            for f in (
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "dones",
                "episode_starts",
            )
        )
    )


def main() -> None:
    cfg = load_config(ROOT / "configs" / "festival_bc_sac.yaml")
    model = load_sb3_model(str(BC / "bc_only.zip"), algo="sac")
    data = ExpertDataset.load(ROOT / "runs" / "festival" / "festival_bc_sac" / "expert_dataset.npz")
    planner_rows = pd.read_csv(OUT / "per_season.csv").query("policy == 'planner'")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows, per_season = [], []
    for rnd in range(ROUNDS + 1):
        if rnd > 0:
            first = FIRST_TRAIN_SEED + (rnd - 1) * SEASONS_PER_ROUND
            seeds = list(range(first, first + SEASONS_PER_ROUND))
            data = merge(data, label_clone_seasons(model, cfg, seeds))
            fit = pretrain_actor_mse(model, data, epochs=EPOCHS)
            model.save(str(MODEL_DIR / f"dagger_round{rnd}"))
        else:
            fit = {}
        name = f"dagger round {rnd}"
        scored = evaluate(lambda: make_env(cfg), {name: sb3_policy(model)}, SEEDS)
        both = pd.concat([planner_rows, scored])
        rows.append(
            {
                "round": rnd,
                "n_labels": len(data),
                "bc_mse": fit.get("bc_mse"),
                "score": scored.score.mean(),
                "unsold": scored.unsold.mean(),
                "denied": scored.denied.mean(),
                **paired_interval(both, name, "planner"),
            }
        )
        print(pd.DataFrame(rows).round(1).to_string(index=False), flush=True)
        per_season.append(scored.assign(round=rnd))

    pd.DataFrame(rows).round(1).to_csv(OUT / "dagger.csv", index=False)
    pd.concat(per_season).to_csv(OUT / "dagger_per_season.csv", index=False)


if __name__ == "__main__":
    main()
