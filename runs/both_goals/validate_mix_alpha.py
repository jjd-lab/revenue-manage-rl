#!/usr/bin/env python3
"""Re-select the cap's mix_alpha on validation seeds, not the reported ones.

`final_eval.py` picks a "recommended safe-SL variant" from held-out seeds 0..29 —
the same thirty `docs/EXPERIMENT_LOG.md` §7 reports on. That is selection on the
evaluation set: the reported score for `bc_sac+safe_sl` is a best-of-three.

This applies the identical rule (prefer peak oversell < 0.05, then highest
`score_aware`) on a **disjoint** seed block, 100..129. If the two agree, the
shipped `mix_alpha` is not an artefact of looking at the test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import deep_merge, load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.evaluate.soft_aware import (
    collect_soft_oracle_revenues,
    evaluate_policy_soft_aware,
)
from reservation_pricing.metrics import SoftAwareConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GRID = (0.0, 0.25, 0.4)
OVERSELL_TOL = 0.05
SEED_BLOCKS = {"validation": list(range(100, 130)), "test": list(range(30))}


def main() -> None:
    base = load_config()
    soft_cfg = SoftAwareConfig.from_dict(base["eval"]["soft_aware"])
    capped = load_config(str(ROOT / "configs/experiment_bc_sac_safe_sl.yaml"))
    ckpt = ROOT / "artifacts/bc_sac/rl_bc_sac_final.zip"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}; see README 'Model checkpoints'")
    policy = sb3_policy(load_sb3_model(str(ckpt), algo="sac"), deterministic=True)

    rows: list[dict] = []
    for block, seeds in SEED_BLOCKS.items():
        oracle = collect_soft_oracle_revenues(
            lambda: make_env(base, use_held_out=True), seeds, soft_cfg
        )
        scored = []
        for mix in GRID:
            cfg = deep_merge(capped, {"control": {"safe_sl": {"mix_alpha": mix}}})
            sa, _ = evaluate_policy_soft_aware(
                lambda cfg=cfg: make_env(cfg, use_held_out=True),
                policy,
                n_episodes=len(seeds),
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle,
            )
            score = float(sa["score_aware"])
            oversell = float(sa["peak"]["oversell_rate"])
            scored.append((mix, score, oversell))
            rows.append(
                {
                    "seed_block": block,
                    "seeds": f"{seeds[0]}-{seeds[-1]}",
                    "mix_alpha": mix,
                    "score_aware": round(score, 2),
                    "peak_oversell": round(oversell, 4),
                }
            )
            print(
                f"  {block:<11} mix={mix:<5} {score:>13,.0f}  oversell {oversell:.4f}", flush=True
            )
        pick = sorted(scored, key=lambda r: (r[2] > OVERSELL_TOL, r[2], -r[1]))[0][0]
        for r in rows:
            if r["seed_block"] == block:
                r["rule_picks"] = pick
        print(f"  {block:<11} -> rule picks mix_alpha={pick}\n", flush=True)

    table = pd.DataFrame(rows)
    picks = set(table["rule_picks"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "mix_alpha_selection.csv", index=False)
    print(f"Wrote {OUT / 'mix_alpha_selection.csv'}")
    print(table.to_string(index=False))
    print(
        f"\nValidation and test {'AGREE' if len(picks) == 1 else 'DISAGREE'} "
        f"on mix_alpha={picks if len(picks) > 1 else picks.pop()}"
    )


if __name__ == "__main__":
    main()
