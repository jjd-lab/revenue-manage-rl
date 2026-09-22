"""Shared pieces of the held-out comparison scripts under ``runs/``.

Each ``runs/<experiment>/final_eval.py`` keeps only its candidate list; the
per-policy rollout, the extra rate columns, and the four output files
(``comparison_table.csv``, ``episode_metrics.csv``, ``comparison_summary.json``,
``comparison.md``) come from here so every table has the same columns, the
same rounding, and repo-relative checkpoint paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.config import load_config
from reservation_pricing.config.load import package_root
from reservation_pricing.envs import ReservationEnv
from reservation_pricing.evaluate.compare import evaluate_policy, sb3_policy
from reservation_pricing.metrics import UNDERSELL_THRESHOLD, PolicyFn

TABLE_COLUMNS = [
    "policy",
    "n_episodes",
    "mean_true_revenue",
    "mean_load_factor",
    "mean_remain_inv",
    "oversell_rate",
    "undersell_gt1500_rate",
    "sellout_rate",
    "score",
]


def published_shortfall_weight() -> float:
    """The ``score`` weight every table in ``runs/`` uses: ``tune.shortfall_weight``."""
    return float(load_config().get("tune", {}).get("shortfall_weight", 200.0))


def repo_relative(path: str | Path) -> str:
    """Checkpoint path as written into the tables: relative to the repo root."""
    root = package_root()
    p = Path(path).resolve()
    if root is not None:
        try:
            return str(p.relative_to(root))
        except ValueError:
            pass
    return str(path)


def extra_rates(
    rows: Sequence[dict], undersell_threshold: float = UNDERSELL_THRESHOLD
) -> dict[str, float]:
    remain = np.array([r["remain_inv"] for r in rows], dtype=np.float64)
    return {
        "oversell_rate": float(np.mean(remain < 0)),
        "undersell_gt1500_rate": float(np.mean(remain > undersell_threshold)),
    }


def run_labelled(
    label: str,
    env_factory: Callable[[], ReservationEnv],
    policy: PolicyFn,
    *,
    episodes: int = 30,
    seeds: Optional[Sequence[int]] = None,
    shortfall_weight: Optional[float] = None,
    model_path: Optional[str | Path] = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Roll ``policy`` out on held-out seeds; return (aggregate row, episode rows)."""
    weight = published_shortfall_weight() if shortfall_weight is None else shortfall_weight
    agg, rows = evaluate_policy(
        env_factory,
        policy,
        n_episodes=episodes,
        seeds=list(seeds) if seeds is not None else list(range(episodes)),
        shortfall_weight=weight,
    )
    for r in rows:
        r["policy"] = label
    d = agg.to_dict()
    d["policy"] = label
    if model_path is not None:
        d["model_path"] = repo_relative(model_path)
    d.update(extra_rates(rows))
    print(
        label,
        "score",
        round(d["score"], 1),
        "rev",
        round(d["mean_true_revenue"], 1),
        "remain",
        round(d["mean_remain_inv"], 1),
        "load",
        round(d["mean_load_factor"], 3),
        "oversell",
        round(d["oversell_rate"], 3),
        "undersell>1500",
        round(d["undersell_gt1500_rate"], 3),
        flush=True,
    )
    return d, rows


def run_checkpoint(
    label: str,
    env_factory: Callable[[], ReservationEnv],
    model_path: str | Path,
    algo: str,
    **kwargs: Any,
) -> Optional[tuple[dict[str, Any], list[dict]]]:
    """``run_labelled`` for an SB3 checkpoint; ``None`` (with a SKIP line) if it is missing."""
    path = Path(model_path)
    if not path.exists():
        print(f"SKIP missing {label}: {path}", flush=True)
        return None
    model = load_sb3_model(str(path), algo=algo)
    return run_labelled(
        label, env_factory, sb3_policy(model, deterministic=True), model_path=path, **kwargs
    )


def comparison_table(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(results))
    cols = [c for c in TABLE_COLUMNS if c in df.columns]
    table = df[cols].sort_values("score", ascending=False).reset_index(drop=True)
    for c in ("mean_true_revenue", "mean_remain_inv", "score"):
        if c in table.columns:
            table[c] = table[c].round(2)
    for c in ("mean_load_factor", "oversell_rate", "undersell_gt1500_rate", "sellout_rate"):
        if c in table.columns:
            table[c] = table[c].round(4)
    return table


def write_comparison(
    out_dir: str | Path,
    results: Sequence[dict[str, Any]],
    all_episodes: Sequence[dict],
    *,
    title: str,
    episodes: int,
    shortfall_weight: Optional[float] = None,
    preamble: Sequence[str] = (),
    footer: Sequence[str] = (),
) -> pd.DataFrame:
    """Write the four comparison files and return the table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weight = published_shortfall_weight() if shortfall_weight is None else shortfall_weight
    table = comparison_table(results)

    table.to_csv(out / "comparison_table.csv", index=False)
    pd.DataFrame(list(all_episodes)).to_csv(out / "episode_metrics.csv", index=False)
    with (out / "comparison_summary.json").open("w") as f:
        json.dump(list(results), f, indent=2, default=str)

    lines = [
        f"# {title}",
        "",
        f"demand.kind: `tree_elastic` | held-out months 6 & 12 | n={episodes} | "
        f"`score = mean_true_revenue - {weight:g} * mean_capacity_shortfall`",
        "",
    ]
    if preamble:
        lines += [*preamble, ""]
    lines += [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in table.columns) + " |")
    lines += ["", "## Artifact paths", ""]
    for r in results:
        if "model_path" in r:
            lines.append(f"- `{r['policy']}`: `{r['model_path']}`")
    if footer:
        lines += ["", *footer]
    lines.append("")
    (out / "comparison.md").write_text("\n".join(lines))
    print("Wrote", out / "comparison.md")
    print(table.to_string(index=False))
    return table
