"""Soft-day-aware evaluation: stratify soft vs peak and report oracle-gap KPIs.

Avoid leading with overall undersell>1500 when soft demand cannot fill even at
floor price (structural ceiling). See docs/EVALUATING_POLICIES.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model, resolve_algo_name
from reservation_pricing.baselines.policies import (
    BASELINE_FACTORY,
    fixed_price_policy,
    myopic_greedy_policy,
)
from reservation_pricing.config import load_config
from reservation_pricing.envs import ReservationEnv, make_env
from reservation_pricing.evaluate.compare import _df_to_markdown, sb3_policy
from reservation_pricing.metrics import (
    OVERSELL_WEIGHT,
    EpisodeMetrics,
    PolicyFn,
    SoftAwareConfig,
    classify_soft,
    run_episode,
)


def _slice_stats(
    episodes: Sequence[EpisodeMetrics],
    *,
    shortfall_weight: float,
    undersell_threshold: float,
    oversell_weight: float = OVERSELL_WEIGHT,
) -> dict[str, float]:
    if not episodes:
        return {
            "n_episodes": 0,
            "mean_true_revenue": float("nan"),
            "std_true_revenue": float("nan"),
            "mean_load_factor": float("nan"),
            "mean_remain_inv": float("nan"),
            "oversell_rate": float("nan"),
            "undersell_gt_thresh_rate": float("nan"),
            "mean_frac_at_floor": float("nan"),
            "mean_shortfall": float("nan"),
            "mean_oversell_amount": float("nan"),
            "score": float("nan"),
            "sellout_rate": float("nan"),
        }
    revs = np.array([e.true_revenue for e in episodes], dtype=np.float64)
    loads = np.array([e.load_factor for e in episodes], dtype=np.float64)
    remain = np.array([e.remain_inv for e in episodes], dtype=np.float64)
    floors = np.array([e.frac_at_floor for e in episodes], dtype=np.float64)
    shortfalls = np.maximum(remain, 0.0) + oversell_weight * np.maximum(-remain, 0.0)
    oversell_amt = np.maximum(-remain, 0.0)
    sellouts = [e.sellout_day is not None for e in episodes]
    mean_rev = float(np.mean(revs))
    mean_sf = float(np.mean(shortfalls))
    return {
        "n_episodes": int(len(episodes)),
        "mean_true_revenue": mean_rev,
        "std_true_revenue": float(np.std(revs)),
        "mean_load_factor": float(np.mean(loads)),
        "mean_remain_inv": float(np.mean(remain)),
        "oversell_rate": float(np.mean(remain < 0)),
        "undersell_gt_thresh_rate": float(np.mean(remain > undersell_threshold)),
        "mean_frac_at_floor": float(np.mean(floors)),
        "mean_shortfall": mean_sf,
        "mean_oversell_amount": float(np.mean(oversell_amt)),
        "score": mean_rev - shortfall_weight * mean_sf,
        "sellout_rate": float(np.mean(sellouts)),
    }


def aggregate_soft_aware(
    episodes: Sequence[EpisodeMetrics],
    cfg: Optional[SoftAwareConfig] = None,
    *,
    oracle_revenue_by_seed: Optional[dict[int, float]] = None,
    seed_by_index: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Stratified aggregates: overall | soft | peak + soft-aware selection scores.

    Formulas (documented in docs/EVALUATING_POLICIES.md)::

        score_peak  = mean_rev_peak - λ * mean_shortfall_peak
        score_soft  = mean_rev_soft - gap_to_oracle     (mode=gap_to_oracle)
                   or mean_rev_soft - μ * mean_oversell_amount  (mode=oversell_only)
        score_aware = score_peak + score_soft

    ``undersell_gt_thresh`` overall remains available as a **secondary / diagnostic**
    metric (do not lead with it when soft demand cannot fill at floor).
    """
    cfg = cfg or SoftAwareConfig()
    # Episodes from run_episode are already tagged; this only fills in a
    # hand-built EpisodeMetrics whose is_soft was left at its None default.
    # It does NOT re-classify: passing a different cfg here will not relabel a
    # rollout, because run_episode always records a bool.
    tagged: list[EpisodeMetrics] = []
    for e in episodes:
        soft = e.is_soft
        if soft is None:
            soft = classify_soft(month=e.month, dow=e.dow, base_demand=e.base_demand, cfg=cfg)
        tagged.append(
            EpisodeMetrics(
                **{
                    **e.to_dict(),
                    "is_soft": bool(soft),
                    "is_peak": not bool(soft),
                }
            )
        )

    soft_eps = [e for e in tagged if e.is_soft]
    peak_eps = [e for e in tagged if not e.is_soft]

    overall = _slice_stats(
        tagged,
        shortfall_weight=cfg.lambda_peak,
        undersell_threshold=cfg.undersell_threshold,
    )
    soft_stats = _slice_stats(
        soft_eps,
        shortfall_weight=cfg.lambda_peak,
        undersell_threshold=cfg.undersell_threshold,
    )
    peak_stats = _slice_stats(
        peak_eps,
        shortfall_weight=cfg.lambda_peak,
        undersell_threshold=cfg.undersell_threshold,
    )

    # Oracle gap on soft episodes (primary soft KPI)
    gap_to_oracle = float("nan")
    mean_oracle_rev = float("nan")
    if soft_eps and oracle_revenue_by_seed is not None and seed_by_index is not None:
        gaps = []
        oracle_revs = []
        for i, e in enumerate(tagged):
            if not e.is_soft:
                continue
            seed = int(seed_by_index[i]) if i < len(seed_by_index) else i
            if seed not in oracle_revenue_by_seed:
                continue
            orev = float(oracle_revenue_by_seed[seed])
            oracle_revs.append(orev)
            gaps.append(max(0.0, orev - float(e.true_revenue)))
        if gaps:
            gap_to_oracle = float(np.mean(gaps))
            mean_oracle_rev = float(np.mean(oracle_revs))

    soft_stats["gap_to_oracle"] = gap_to_oracle
    soft_stats["mean_oracle_revenue"] = mean_oracle_rev
    # Signed gap (oracle - policy); negative means beat oracle on soft revenue
    if soft_eps and oracle_revenue_by_seed is not None and seed_by_index is not None:
        signed = []
        for i, e in enumerate(tagged):
            if not e.is_soft:
                continue
            seed = int(seed_by_index[i]) if i < len(seed_by_index) else i
            if seed in oracle_revenue_by_seed:
                signed.append(float(oracle_revenue_by_seed[seed]) - float(e.true_revenue))
        soft_stats["signed_gap_to_oracle"] = float(np.mean(signed)) if signed else float("nan")
    else:
        soft_stats["signed_gap_to_oracle"] = float("nan")

    score_peak = float(peak_stats["score"]) if peak_eps else 0.0
    if not soft_eps:
        score_soft = 0.0
    elif (cfg.soft_score_mode or "").lower() in ("oversell_only", "oversell"):
        score_soft = float(soft_stats["mean_true_revenue"]) - cfg.mu_soft * float(
            soft_stats["mean_oversell_amount"]
        )
    else:
        # gap_to_oracle (default): revenue − gap; if no oracle, fall back to oversell-only
        if np.isnan(gap_to_oracle):
            score_soft = float(soft_stats["mean_true_revenue"]) - cfg.mu_soft * float(
                soft_stats["mean_oversell_amount"]
            )
        else:
            score_soft = float(soft_stats["mean_true_revenue"]) - gap_to_oracle

    score_aware = score_peak + score_soft

    return {
        "overall": overall,
        "soft": soft_stats,
        "peak": peak_stats,
        "n_soft": int(len(soft_eps)),
        "n_peak": int(len(peak_eps)),
        "score_peak": score_peak,
        "score_soft": score_soft,
        "score_aware": score_aware,
        "cfg": asdict(cfg),
        # Secondary / diagnostic (legacy continuity)
        "diagnostic": {
            "undersell_gt_thresh_overall": overall["undersell_gt_thresh_rate"],
            "undersell_threshold": cfg.undersell_threshold,
            "note": "undersell>thresh overall is secondary when soft demand cannot fill at floor",
        },
    }


def soft_aware_table(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Wide summary table for soft-aware comparison markdown."""
    records = []
    for name, sa in rows.items():
        soft = sa.get("soft", {})
        peak = sa.get("peak", {})
        overall = sa.get("overall", {})
        records.append(
            {
                "policy": name,
                "n_soft": sa.get("n_soft", 0),
                "n_peak": sa.get("n_peak", 0),
                "score_aware": round(float(sa.get("score_aware", float("nan"))), 2),
                "score_peak": round(float(sa.get("score_peak", float("nan"))), 2),
                "score_soft": round(float(sa.get("score_soft", float("nan"))), 2),
                # Soft primary KPIs
                "soft_rev": _round_or_nan(soft.get("mean_true_revenue")),
                "soft_gap_to_oracle": _round_or_nan(soft.get("gap_to_oracle")),
                "soft_frac_at_floor": _round_or_nan(soft.get("mean_frac_at_floor"), 4),
                "soft_oversell": _round_or_nan(soft.get("oversell_rate"), 4),
                # Peak primary KPIs
                "peak_rev": _round_or_nan(peak.get("mean_true_revenue")),
                "peak_remain": _round_or_nan(peak.get("mean_remain_inv")),
                "peak_load": _round_or_nan(peak.get("mean_load_factor"), 4),
                "peak_oversell": _round_or_nan(peak.get("oversell_rate"), 4),
                "peak_score": _round_or_nan(peak.get("score")),
                # Overall (continuity) — undersell is diagnostic/secondary
                "overall_rev": _round_or_nan(overall.get("mean_true_revenue")),
                "overall_score": _round_or_nan(overall.get("score")),
                "overall_oversell": _round_or_nan(overall.get("oversell_rate"), 4),
                "undersell_gt1500_diag": _round_or_nan(overall.get("undersell_gt_thresh_rate"), 4),
            }
        )
    return pd.DataFrame(records)


def _round_or_nan(v: Any, nd: int = 2) -> Any:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(fv):
        return None
    return round(fv, nd)


def evaluate_policy_soft_aware(
    env_factory: Callable[[], ReservationEnv],
    policy: PolicyFn,
    n_episodes: int = 30,
    seeds: Optional[list[int]] = None,
    soft_cfg: Optional[SoftAwareConfig] = None,
    oracle_revenue_by_seed: Optional[dict[int, float]] = None,
) -> tuple[dict[str, Any], list[dict]]:
    soft_cfg = soft_cfg or SoftAwareConfig()
    seeds = seeds or list(range(n_episodes))
    episodes: list[EpisodeMetrics] = []
    rows: list[dict] = []
    for seed in seeds[:n_episodes]:
        env = env_factory()
        ep = run_episode(env, policy, seed=int(seed), soft_cfg=soft_cfg)
        episodes.append(ep)
        row = ep.to_dict()
        row["seed"] = int(seed)
        rows.append(row)
    sa = aggregate_soft_aware(
        episodes,
        soft_cfg,
        oracle_revenue_by_seed=oracle_revenue_by_seed,
        seed_by_index=seeds[:n_episodes],
    )
    return sa, rows


def soft_oracle_policy(cfg: SoftAwareConfig, min_price: float) -> PolicyFn:
    """Soft oracle: always ``min_price`` or myopic (max SL via baseline helpers).

    ``min_price`` is the env's own floor: an oracle priced anywhere else is not a
    ceiling, and ``gap_to_oracle`` silently stops meaning what it says.
    """
    mode = (cfg.soft_oracle or "min_price").lower()
    if mode in ("myopic", "myopic_greedy"):
        return myopic_greedy_policy(n_grid=41)
    return fixed_price_policy(price=float(min_price), selling_limit=None)


def collect_soft_oracle_revenues(
    env_factory: Callable[[], ReservationEnv],
    seeds: list[int],
    soft_cfg: SoftAwareConfig,
) -> dict[int, float]:
    """Roll soft oracle on all seeds; return {seed: revenue} (used for soft gap)."""
    policy = soft_oracle_policy(soft_cfg, min_price=float(env_factory().min_price))
    out: dict[int, float] = {}
    for seed in seeds:
        env = env_factory()
        ep = run_episode(env, policy, seed=int(seed), soft_cfg=soft_cfg)
        out[int(seed)] = float(ep.true_revenue)
    return out


def run_soft_aware_comparison(
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    algo: Optional[str] = None,
    n_episodes: int = 30,
    seeds: Optional[list[int]] = None,
    held_out: bool = True,
    out_dir: Optional[str] = None,
    include_baselines: bool = True,
    baseline_names: Optional[list[str]] = None,
    extra_policies: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compare policies with soft/peak stratification + soft-aware scores.

    ``extra_policies`` entries: ``{name, model_path, algo, config_path?}`` so
    demo scripts can mix bc_sac+safe_sl (joint) with pace_ppo (price-only).
    """
    cfg = load_config(config_path)
    eval_cfg = cfg.get("eval", {}) or {}
    soft_raw = dict(eval_cfg.get("soft_aware") or {})
    soft_cfg = SoftAwareConfig.from_dict(soft_raw)
    if "lambda_peak" not in soft_raw:
        soft_cfg.lambda_peak = float(
            cfg.get("tune", {}).get("shortfall_weight", soft_cfg.lambda_peak)
        )

    n_episodes = int(n_episodes or eval_cfg.get("n_episodes", 30))
    seeds = seeds or list(eval_cfg.get("seeds", list(range(n_episodes))))
    if len(seeds) < n_episodes:
        seeds = list(seeds) + list(range(1000, 1000 + n_episodes - len(seeds)))
    seeds = list(seeds)[:n_episodes]

    def env_factory():
        return make_env(cfg, use_held_out=held_out)

    # Soft oracle revenues once (shared seeds / env factory from primary config)
    oracle_by_seed = collect_soft_oracle_revenues(env_factory, seeds, soft_cfg)

    soft_results: dict[str, dict[str, Any]] = {}
    all_episodes: list[dict] = []

    if include_baselines:
        names = baseline_names or ["myopic_greedy", "fixed_price_80", "fixed_price_100"]
        for name in names:
            if name not in BASELINE_FACTORY:
                continue
            policy = BASELINE_FACTORY[name]()
            sa, rows = evaluate_policy_soft_aware(
                env_factory,
                policy,
                n_episodes=n_episodes,
                seeds=seeds,
                soft_cfg=soft_cfg,
                oracle_revenue_by_seed=oracle_by_seed,
            )
            soft_results[name] = sa
            for r in rows:
                r["policy"] = name
            all_episodes.extend(rows)

    if model_path:
        algo_name = (algo or resolve_algo_name(cfg)).lower()
        model = load_sb3_model(model_path, algo=algo_name)
        policy = sb3_policy(model, deterministic=True)
        name = f"rl_{algo_name}"
        sa, rows = evaluate_policy_soft_aware(
            env_factory,
            policy,
            n_episodes=n_episodes,
            seeds=seeds,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle_by_seed,
        )
        soft_results[name] = sa
        for r in rows:
            r["policy"] = name
        all_episodes.extend(rows)

    for entry in extra_policies or []:
        label = str(entry["name"])
        mpath = entry["model_path"]
        malgo = str(entry.get("algo") or "sac").lower()
        ecfg_path = entry.get("config_path")
        if ecfg_path:
            ecfg = load_config(str(ecfg_path))
        else:
            ecfg = cfg

        def _factory(c=ecfg):
            return make_env(c, use_held_out=held_out)

        # Soft oracle is demand/seed based; reuse primary oracle (controls do not change demand)
        model = load_sb3_model(str(mpath), algo=malgo)
        policy = sb3_policy(model, deterministic=True)
        sa, rows = evaluate_policy_soft_aware(
            _factory,
            policy,
            n_episodes=n_episodes,
            seeds=seeds,
            soft_cfg=soft_cfg,
            oracle_revenue_by_seed=oracle_by_seed,
        )
        soft_results[label] = sa
        for r in rows:
            r["policy"] = label
        all_episodes.extend(rows)

    table = soft_aware_table(soft_results)
    out: dict[str, Any] = {
        "table": table,
        "soft_aware": soft_results,
        "episodes": all_episodes,
        "soft_cfg": soft_cfg.__dict__,
        "oracle_revenue_by_seed": oracle_by_seed,
        "demand_kind": (cfg.get("demand") or {}).get("kind"),
        "n_episodes": n_episodes,
        "seeds": seeds,
    }

    if out_dir:
        write_soft_aware_report(out, out_dir)

    return out


def write_soft_aware_report(out: dict[str, Any], out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    table: pd.DataFrame = out["table"]
    table.to_csv(out_path / "soft_aware_table.csv", index=False)
    pd.DataFrame(out["episodes"]).to_csv(out_path / "episode_metrics.csv", index=False)
    serializable = {k: v for k, v in out.items() if k not in ("table", "episodes")}
    # soft_aware values are plain dicts already
    with (out_path / "soft_aware_summary.json").open("w") as f:
        json.dump(serializable, f, indent=2, default=str)

    sc = out.get("soft_cfg") or {}
    md = [
        "# Soft-day-aware evaluation comparison",
        "",
        "Stratifies episodes into **soft** (structurally weak demand) vs **peak/rich** "
        "so we do not over-penalize undersell when demand cannot fill even at floor price.",
        "",
        f"demand.kind: `{out.get('demand_kind')}` | n={out.get('n_episodes')}",
        "",
        "## Classification",
        "",
        f"- rule: `{sc.get('rule')}`",
        f"- peak_months: `{sc.get('peak_months')}`",
        f"- weekend_dow: `{sc.get('weekend_dow')}`",
        f"- base_demand_threshold: `{sc.get('base_demand_threshold')}`",
        f"- soft_focus_months: `{sc.get('soft_focus_months')}`",
        f"- soft_oracle: `{sc.get('soft_oracle')}`",
        "",
        "Soft if (`structural`): mid-horizon tree base < threshold **OR** "
        "(weekday AND month in soft_focus_months). "
        "Alternative `rule=any`: weekday OR off-peak month OR low base.",
        "",
        "## Selection scores",
        "",
        "```",
        "score_peak  = mean_rev_peak  - λ * mean_shortfall_peak",
        "score_soft  = mean_rev_soft  - gap_to_oracle   # or - μ * mean_oversell_amount",
        "score_aware = score_peak + score_soft",
        "```",
        "",
        f"λ (lambda_peak)={sc.get('lambda_peak')}, μ (mu_soft)={sc.get('mu_soft')}, "
        f"soft_score_mode=`{sc.get('soft_score_mode')}`",
        "",
        "**Primary soft KPIs:** revenue, gap to soft oracle, frac days at/near floor, oversell.",
        "",
        "**Primary peak KPIs:** revenue, load/remain, oversell, score (undersell matters here).",
        "",
        "**Secondary / diagnostic:** overall `undersell>1500` (do not lead with this).",
        "",
        "## Summary table",
        "",
        _df_to_markdown(table),
        "",
        "## Soft slice (primary)",
        "",
        _df_to_markdown(
            table[
                [
                    c
                    for c in (
                        "policy",
                        "soft_rev",
                        "soft_gap_to_oracle",
                        "soft_frac_at_floor",
                        "soft_oversell",
                        "score_soft",
                    )
                    if c in table.columns
                ]
            ]
        ),
        "",
        "## Peak slice (primary)",
        "",
        _df_to_markdown(
            table[
                [
                    c
                    for c in (
                        "policy",
                        "peak_rev",
                        "peak_remain",
                        "peak_load",
                        "peak_oversell",
                        "peak_score",
                        "score_peak",
                    )
                    if c in table.columns
                ]
            ]
        ),
        "",
        "## Diagnostic (secondary)",
        "",
        "Overall `undersell_gt1500_diag` is kept for continuity — do **not** lead with it "
        "when soft demand cannot fill at floor.",
        "",
        _df_to_markdown(
            table[
                [
                    c
                    for c in (
                        "policy",
                        "overall_rev",
                        "overall_score",
                        "overall_oversell",
                        "undersell_gt1500_diag",
                        "score_aware",
                    )
                    if c in table.columns
                ]
            ]
        ),
        "",
    ]
    md_path = out_path / "soft_aware_comparison.md"
    # Also write comparison.md for the demo path requested
    body = "\n".join(md)
    (out_path / "comparison.md").write_text(body)
    md_path.write_text(body)
    return md_path
