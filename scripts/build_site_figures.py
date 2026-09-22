#!/usr/bin/env python3
"""Charts for site/index.html, drawn from committed run tables.

No checkpoints and no training. Rebuild with::

    python scripts/build_site_figures.py

Sources
-------
F1, F2  runs/joint_vs_price_only_soft_aware/soft_aware_table.csv
F3      runs/oracle_ceiling/oracle_soft_episodes.csv
F4      runs/explain_rl_best/rollouts.csv
F5      runs/explain_rl_best/episode_summary.csv

The script also prints the numbers the page's captions quote, so a caption can
be checked against the figure it sits under.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "figures"

INK = "#1c1917"
RULE = "#d9d3c9"
PAPER = "#f7f4ef"
ACCENT = "#8c3a2f"
JOINT = "#1c1917"
PRICE = "#8a6a4a"
BASE = "#8a8178"
FIGSIZE = (7.2, 4.4)
DPI = 160

# Display names and the family used to distinguish points.
POLICIES = {
    "joint_sac_rl_best": ("Joint SAC", "joint"),
    "joint_bc_sac_raw": ("BC to SAC", "joint"),
    "rl_sac": ("BC to SAC + cap", "joint"),
    "joint_ppo_long_007": ("Joint PPO", "joint"),
    "price_only_pace_ppo": ("Pace PPO", "price"),
    "price_only_ppo": ("Price-only PPO", "price"),
    "myopic_greedy": ("Myopic", "base"),
    "fixed_price_80": ("Floor price", "base"),
}
FAMILY_COLOR = {"joint": JOINT, "price": PRICE, "base": BASE}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",  # ships with matplotlib, so the PNGs rebuild anywhere
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)


def _color(policy: str, winner: str) -> str:
    return ACCENT if policy == winner else FAMILY_COLOR[POLICIES[policy][1]]


def _find_key(obj: Any, key: str) -> Any:
    """First value of ``key`` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _assert_inside(ax, xs, ys) -> None:
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for x, y in zip(xs, ys):
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            raise SystemExit(f"point ({x:.3f}, {y:.3f}) falls outside the axes")


def score_dots(table: pd.DataFrame, winner: str, out: Path) -> None:
    """F1: soft-aware score per policy as dots on a non-zero axis, values printed."""
    rows = table[table["policy"].isin(POLICIES)].copy()
    rows["label"] = rows["policy"].map(lambda p: POLICIES[p][0])
    rows["score_m"] = rows["score_aware"] / 1e6
    rows = rows.sort_values("score_m", ascending=True).reset_index(drop=True)
    lo, hi = rows["score_m"].min(), rows["score_m"].max()
    pad = (hi - lo) * 0.12
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hlines(rows["label"], lo - pad, rows["score_m"], color=RULE, linewidth=1.0, zorder=1)
    ax.scatter(
        rows["score_m"],
        rows["label"],
        s=64,
        color=[_color(p, winner) for p in rows["policy"]],
        zorder=3,
    )
    for _, r in rows.iterrows():
        ax.annotate(
            f"{r['score_m']:.2f}",
            (r["score_m"], r["label"]),
            xytext=(9, -3),
            textcoords="offset points",
            fontsize=9.5,
            color=INK,
        )
    ax.set_xlim(lo - pad, hi + pad * 1.6)
    ax.set_xlabel("Soft-aware score, millions of dollars (axis does not start at zero)")
    _despine(ax)
    _assert_inside(ax, rows["score_m"], range(len(rows)))
    fig.tight_layout()
    fig.savefig(out / "f1_score.png", dpi=DPI)
    plt.close(fig)


def risk_frontier(table: pd.DataFrame, winner: str, out: Path) -> None:
    """F2: score against the share of peak nights with denied admission."""
    pts = []
    for policy, (label, _family) in POLICIES.items():
        row = table.loc[table["policy"] == policy].iloc[0]
        pts.append((policy, label, float(row["peak_oversell"]), float(row["score_aware"]) / 1e6))
    xs = [p[2] for p in pts]
    ys = [p[3] for p in pts]
    xpad = max(max(xs) - min(xs), 0.1) * 0.12
    ypad = max(max(ys) - min(ys), 0.05) * 0.15

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Label placement: points that share an x column alternate above / below by rank.
    columns: dict[float, list[tuple[str, str, float, float]]] = {}
    for p in pts:
        columns.setdefault(round(p[2], 2), []).append(p)
    for col in columns.values():
        col.sort(key=lambda p: -p[3])
        for i, (policy, label, x, y) in enumerate(col):
            dy = 6 if i % 2 == 0 else -11
            ax.scatter([x], [y], s=46, color=_color(policy, winner), zorder=3)
            ax.annotate(
                label, (x, y), textcoords="offset points", xytext=(8, dy), fontsize=10, color=INK
            )
    ax.set_xlabel("Share of peak nights with denied admission")
    ax.set_ylabel("Soft-aware score, millions of dollars")
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad * 3)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    _despine(ax)
    _assert_inside(ax, xs, ys)
    fig.tight_layout()
    fig.savefig(out / "f2_frontier.png", dpi=DPI)
    plt.close(fig)


def oracle_ceiling(episodes: pd.DataFrame, threshold: float, out: Path) -> pd.DataFrame:
    """F3: best remaining seats any oracle price reaches on each soft night."""
    soft = episodes.loc[episodes["soft"].astype(str).isin(["True", "true", "1"])]
    best = soft.groupby("seed", as_index=False)["remain_inv"].min().sort_values("seed")
    if best.empty:
        raise SystemExit("oracle ceiling: no soft episodes")
    if (best["remain_inv"] <= threshold).any():
        raise SystemExit("oracle ceiling: a soft date fills, caption would be wrong")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [str(int(s)) for s in best["seed"]]
    ax.bar(labels, best["remain_inv"], color=INK, width=0.72)
    ax.axhline(threshold, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.annotate(
        f"{threshold:,.0f} unsold",
        (len(labels) - 0.5, threshold),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
    )
    ax.set_xlabel("Held-out soft night (seed number)")
    ax.set_ylabel("Unsold seats, best any price can do")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out / "f3_ceiling.png", dpi=DPI)
    plt.close(fig)
    return best


def booking_paths(rollouts: pd.DataFrame, out: Path) -> dict[str, float]:
    """F4: mean price against days before the performance, joint SAC vs myopic."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    facts: dict[str, float] = {}
    for policy, label, color in [("myopic", "Myopic", BASE), ("rl_best", "Joint SAC", ACCENT)]:
        g = (
            rollouts[rollouts["policy"] == policy]
            .groupby("days_prior")["price"]
            .mean()
            .sort_index(ascending=False)
        )
        ax.plot(g.index, g.values, color=color, linewidth=2.2)
        ax.annotate(
            label,
            (g.index[-1], g.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=10,
        )
        facts[f"{policy}_mean_price"] = float(
            rollouts.loc[rollouts["policy"] == policy, "price"].mean()
        )
    rl = rollouts[rollouts["policy"] == "rl_best"]
    facts["rl_weekend_price"] = float(rl.loc[rl["weekend"] == 1, "price"].mean())
    facts["rl_weekday_price"] = float(rl.loc[rl["weekend"] == 0, "price"].mean())
    ax.set_xlim(rollouts["days_prior"].max() + 1, -8)
    ax.set_xlabel("Days before the performance")
    ax.set_ylabel("Mean price, dollars")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out / "f4_booking_curve.png", dpi=DPI)
    plt.close(fig)
    return facts


def soft_peak_lift(episodes: pd.DataFrame, out: Path) -> dict[str, float]:
    """F5: mean revenue per night on soft and peak nights, joint SAC vs myopic."""
    means = episodes.groupby(["policy", "is_soft"])["revenue"].mean() / 1e3
    counts = episodes[episodes["policy"] == "rl_best"]["is_soft"].value_counts()
    groups = [("Soft nights", 1), ("Peak nights", 0)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.34
    series = [(-width / 2, "myopic", "Myopic", BASE), (width / 2, "rl_best", "Joint SAC", ACCENT)]
    for offset, policy, label, color in series:
        vals = [means[(policy, soft)] for _, soft in groups]
        bars = ax.bar(
            [i + offset for i in range(len(groups))], vals, width, color=color, label=label
        )
        for b, v in zip(bars, vals):
            ax.annotate(
                f"${v:,.0f}k",
                (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{name} (n={int(counts[soft])})" for name, soft in groups])
    ax.set_ylabel("Mean revenue per night, thousands of dollars")
    ax.legend(frameon=False, loc="upper left")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out / "f5_lift.png", dpi=DPI)
    plt.close(fig)
    return {
        "soft_lift": float(means[("rl_best", 1)] - means[("myopic", 1)]) * 1e3,
        "peak_lift": float(means[("rl_best", 0)] - means[("myopic", 0)]) * 1e3,
        "n_soft": int(counts[1]),
        "n_peak": int(counts[0]),
    }


def caption_facts(table: pd.DataFrame, ceiling: pd.DataFrame, paths: dict, lift: dict) -> None:
    """Print what the page's captions quote."""
    by = table.set_index("policy")
    best_price_only = max(
        float(by.loc[p, "score_aware"]) for p in ("price_only_pace_ppo", "price_only_ppo")
    )
    raw, capped = by.loc["joint_bc_sac_raw"], by.loc["rl_sac"]
    ranked = by["score_aware"].sort_values(ascending=False)
    print("F1  ranking:", ", ".join(f"{POLICIES[p][0]} {v / 1e6:.3f}M" for p, v in ranked.items()))
    for p in ("rl_sac", "joint_sac_rl_best"):
        v = float(by.loc[p, "score_aware"])
        print(
            f"    {POLICIES[p][0]} vs best price-only: {100 * (v / best_price_only - 1):+.2f}%"
            f"  ({v:,.0f} vs {best_price_only:,.0f})"
        )
    print(
        "F2  cap: peak denied-admission share "
        f"{float(raw['peak_oversell']):.2f} -> {float(capped['peak_oversell']):.2f}, "
        f"score {float(raw['score_aware']):,.0f} -> {float(capped['score_aware']):,.0f} "
        f"({100 * (1 - float(capped['score_aware']) / float(raw['score_aware'])):.2f}% given up)"
    )
    print(
        f"F3  {len(ceiling)} soft nights; best remaining "
        f"{ceiling['remain_inv'].min():,.0f} to {ceiling['remain_inv'].max():,.0f}"
    )
    print(
        f"F4  myopic mean ${paths['myopic_mean_price']:.2f}; joint SAC weekend "
        f"${paths['rl_weekend_price']:.2f}, weekday ${paths['rl_weekday_price']:.2f}"
    )
    print(
        f"F5  soft {lift['n_soft']} / peak {lift['n_peak']}; lift per night: "
        f"soft {lift['soft_lift']:+,.0f}, peak {lift['peak_lift']:+,.0f}"
    )
    print(
        "    soft gap to oracle, all policies:",
        sorted(float(v) for v in table["soft_gap_to_oracle"]),
    )


def main(out: Path = OUT) -> list[Path]:
    _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    headline = ROOT / "runs" / "joint_vs_price_only_soft_aware"
    table = pd.read_csv(headline / "soft_aware_table.csv")
    missing = set(POLICIES) - set(table["policy"])
    if missing:
        raise SystemExit(f"soft-aware table missing {sorted(missing)}")
    summary = json.loads((headline / "soft_aware_summary.json").read_text())
    threshold = float(_find_key(summary, "undersell_threshold"))

    winner = str(table.loc[table["score_aware"].idxmax(), "policy"])

    score_dots(table, winner, out)
    risk_frontier(table, winner, out)
    oracle = pd.read_csv(ROOT / "runs" / "oracle_ceiling" / "oracle_soft_episodes.csv")
    ceiling = oracle_ceiling(oracle, threshold, out)
    paths = booking_paths(pd.read_csv(ROOT / "runs" / "explain_rl_best" / "rollouts.csv"), out)
    lift = soft_peak_lift(
        pd.read_csv(ROOT / "runs" / "explain_rl_best" / "episode_summary.csv"), out
    )
    caption_facts(table, ceiling, paths, lift)
    written = sorted(out.glob("f*.png"))
    print(f"Wrote {len(written)} figures to {out}")
    return written


if __name__ == "__main__":
    main()
