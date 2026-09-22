#!/usr/bin/env python3
"""Explainability plots for joint SAC ``artifacts/tree_long/best/rl_best.zip``.

Rolls out rl_best vs myopic on held-out seeds and writes PNGs + CSVs under
``runs/explain_rl_best/`` (override with --out).

Usage (from repo root, venv active)::

    python scripts/explain_rl_best.py
    python scripts/explain_rl_best.py --seeds 30 --model artifacts/tree_long/best/rl_best.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reservation_pricing.algorithms.registry import load_sb3_model
from reservation_pricing.baselines.policies import myopic_greedy_policy
from reservation_pricing.config import load_config
from reservation_pricing.demand.protocol import features_from_state
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate.compare import sb3_policy
from reservation_pricing.metrics import _mid_horizon_base, classify_soft

# Series key for the RL policy in every frame and plot. ``main`` sets it from
# --label, so a run against a different checkpoint does not record itself as
# "rl_best" beside the genuine runs/explain_rl_best/.
RL_LABEL = "rl_best"


def rollout_rows(policy, name: str, seeds, cfg) -> pd.DataFrame:
    rows: list[dict] = []
    for seed in seeds:
        env = make_env(cfg, use_held_out=True)
        obs, info = env.reset(seed=int(seed))
        core = env.unwrapped
        dm = core.demand_model
        # Same soft/peak rule as every table in runs/ (metrics.classify_soft).
        is_soft = classify_soft(month=core.month, dow=core.dow, base_demand=_mid_horizon_base(env))
        state: dict = {}
        done = False
        seed_rows: list[dict] = []
        while not done:
            days_prior = env.days_prior
            remain = env.remain_inv
            dow, month = env.dow, env.month
            action = policy(obs, env, state)
            a = np.asarray(action, dtype=float).reshape(-1)
            if a.size >= 2 and a.min() >= -1.05 and a.max() <= 1.05:
                price = env.min_price + (a[0] + 1) * 0.5 * (env.max_price - env.min_price)
                sl = env.min_selling_limit + (a[1] + 1) * 0.5 * (
                    env.max_selling_limit - env.min_selling_limit
                )
            else:
                price = float(a[0])
                sl = float(a[1]) if a.size > 1 else np.nan
            obs, r, term, trunc, info = env.step(action)
            # Charged price, not the action. A project-mode wrapper can raise a
            # markdown after this unscale, and the path has to show that.
            if info.get("price") is not None:
                price = float(info["price"])
            if info.get("selling_limit") is not None:
                sl = float(info["selling_limit"])
            done = term or trunc
            base = np.nan
            if hasattr(dm, "predict_base"):
                feats = features_from_state(
                    days_prior=int(days_prior),
                    dow=int(dow),
                    month=int(month),
                    booking_horizon=int(core.booking_horizon),
                )
                base = float(dm.predict_base(feats))
            seed_rows.append(
                {
                    "policy": name,
                    "seed": seed,
                    "days_prior": days_prior,
                    "dow": dow,
                    "month": month,
                    "weekend": int(dow in (5, 6)),
                    "peak_month": int(month in (7, 8, 11, 12)),
                    "price": price,
                    "selling_limit": sl,
                    "remain": remain,
                    "accepted": info.get("accepted_booking", np.nan),
                    "cum_revenue": info.get(
                        "true_revenue", getattr(env, "cumulative_income", np.nan)
                    ),
                    "base_demand": base,
                    "is_soft": int(is_soft),
                }
            )
        rows.extend(seed_rows)
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, ep: pd.DataFrame, out: Path) -> None:
    # 01 paths
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, color in [(RL_LABEL, "#2563eb"), ("myopic", "#64748b")]:
        g = (
            df[df.policy == name]
            .groupby("days_prior")[["price", "remain"]]
            .mean()
            .sort_index(ascending=False)
        )
        axes[0].plot(g.index, g["price"], label=name, color=color, lw=2)
        axes[1].plot(g.index, g["remain"], label=name, color=color, lw=2)
    axes[0].invert_xaxis()
    axes[1].invert_xaxis()
    axes[0].set_xlabel("Days prior")
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Average price path")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Days prior")
    axes[1].set_ylabel("Remaining inventory")
    axes[1].set_title("Average inventory path")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Joint SAC vs myopic: dynamic price path", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_price_inventory_paths.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 02 weekend
    fig, ax = plt.subplots(figsize=(7, 4))
    rl = df[df.policy == RL_LABEL]
    for label, mask, color in [
        ("Weekend", rl.weekend == 1, "#dc2626"),
        ("Weekday", rl.weekend == 0, "#2563eb"),
    ]:
        g = rl[mask].groupby("days_prior")["price"].mean().sort_index(ascending=False)
        if len(g):
            ax.plot(g.index, g.values, label=label, color=color, lw=2)
    ax.invert_xaxis()
    ax.set_xlabel("Days prior")
    ax.set_ylabel("Price ($)")
    ax.set_title("Joint SAC: weekend price premium")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02_weekend_vs_weekday_price.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 03 soft vs peak boxes
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sub = ep[ep.policy == RL_LABEL]
    for ax, col, title in [
        (axes[0], "mean_price", "Mean price by regime"),
        (axes[1], "revenue", "Revenue by regime"),
    ]:
        data = [
            sub.loc[sub.is_soft == 1, col].dropna(),
            sub.loc[sub.is_soft == 0, col].dropna(),
        ]
        bp = ax.boxplot(data, tick_labels=["Soft", "Peak"], patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#93c5fd", "#fca5a5"]):
            patch.set_facecolor(c)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Joint SAC adapts to soft vs peak", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "03_soft_vs_peak_boxplots.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 04 price vs base
    fig, ax = plt.subplots(figsize=(6, 4))
    m = rl[(rl.days_prior >= 40) & (rl.days_prior <= 70) & rl.base_demand.notna()]
    sc = ax.scatter(m["base_demand"], m["price"], c=m["weekend"], cmap="coolwarm", alpha=0.35, s=12)
    ax.set_xlabel("Tree base demand (price-unaware)")
    ax.set_ylabel("Chosen price ($)")
    ax.set_title("Higher base demand → higher price")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Weekend")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04_price_vs_base_demand.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 05 SL vs remain
    fig, ax = plt.subplots(figsize=(6, 4))
    m = rl[rl.selling_limit.notna()].copy()
    ax.scatter(m["remain"], m["selling_limit"], alpha=0.12, s=8, c="#2563eb")
    bins = np.linspace(0, 10000, 12)
    m["rb"] = pd.cut(m["remain"], bins)
    bm = m.groupby("rb", observed=True).agg(remain=("remain", "mean"), sl=("selling_limit", "mean"))
    ax.plot(bm["remain"], bm["sl"], "o-", color="#dc2626", lw=2, label="Binned mean")
    ax.set_xlabel("Remaining inventory")
    ax.set_ylabel("Selling limit")
    ax.set_title("Selling limit vs remaining seats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "05_selling_limit_vs_remain.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 06 example trajectories
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    peak_seeds = ep[(ep.policy == RL_LABEL) & (ep.is_soft == 0) & (ep.weekend == 1)][
        "seed"
    ].tolist()
    soft_seeds = ep[(ep.policy == RL_LABEL) & (ep.is_soft == 1)]["seed"].tolist()
    if not peak_seeds:
        peak_seeds = ep[(ep.policy == RL_LABEL) & (ep.is_soft == 0)]["seed"].tolist()
    ps, ss = int(peak_seeds[0]), int(soft_seeds[0])
    for ax, seed, title in [
        (axes[0], ps, f"Peak-ish episode (seed {ps})"),
        (axes[1], ss, f"Soft episode (seed {ss})"),
    ]:
        for name, color, ls in [(RL_LABEL, "#2563eb", "-"), ("myopic", "#64748b", "--")]:
            g = df[(df.policy == name) & (df.seed == seed)].sort_values(
                "days_prior", ascending=False
            )
            ax.plot(
                g["days_prior"],
                g["price"],
                color=color,
                ls=ls,
                lw=2,
                label=f"{name} price",
            )
        ax2 = ax.twinx()
        g = df[(df.policy == RL_LABEL) & (df.seed == seed)].sort_values(
            "days_prior", ascending=False
        )
        ax2.plot(
            g["days_prior"],
            g["remain"],
            color="#16a34a",
            alpha=0.55,
            label="remain (RL)",
        )
        ax.invert_xaxis()
        ax.set_ylabel("Price")
        ax2.set_ylabel("Remain")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        lines, labels = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines + l2, labels + lab2, loc="best", fontsize=8)
    axes[1].set_xlabel("Days prior")
    fig.suptitle("Example paths: peak harvest vs soft pricing", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "06_example_peak_vs_soft_trajectories.png", dpi=140, bbox_inches="tight")
    plt.close()

    # 07 revenue soft vs peak
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    w = 0.35
    ax.bar(
        x - w / 2,
        [
            ep[(ep.policy == RL_LABEL) & (ep.is_soft == 1)]["revenue"].mean(),
            ep[(ep.policy == RL_LABEL) & (ep.is_soft == 0)]["revenue"].mean(),
        ],
        w,
        label=RL_LABEL,
        color="#2563eb",
    )
    ax.bar(
        x + w / 2,
        [
            ep[(ep.policy == "myopic") & (ep.is_soft == 1)]["revenue"].mean(),
            ep[(ep.policy == "myopic") & (ep.is_soft == 0)]["revenue"].mean(),
        ],
        w,
        label="myopic",
        color="#94a3b8",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Soft", "Peak"])
    ax.set_ylabel("Mean revenue")
    ax.set_title("Lift vs myopic is concentrated on peak days")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "07_revenue_soft_vs_peak.png", dpi=140, bbox_inches="tight")
    plt.close()


def main(argv: list[str] | None = None) -> None:
    global RL_LABEL

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="artifacts/tree_long/best/rl_best.zip")
    p.add_argument("--algo", default="sac")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default="runs/explain_rl_best")
    p.add_argument(
        "--label",
        default="rl_best",
        help="series name for the RL policy in the CSVs and plots",
    )
    args = p.parse_args(argv)

    RL_LABEL = str(args.label)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    model = load_sb3_model(args.model, algo=args.algo)
    rl_pol = sb3_policy(model, deterministic=True)
    myopic = myopic_greedy_policy()
    seeds = list(range(int(args.seeds)))

    df_rl = rollout_rows(rl_pol, RL_LABEL, seeds, cfg)
    df_my = rollout_rows(myopic, "myopic", seeds, cfg)
    df = pd.concat([df_rl, df_my], ignore_index=True)
    df.to_csv(out / "rollouts.csv", index=False)

    ep = (
        df.groupby(["policy", "seed"])
        .agg(
            revenue=("cum_revenue", "last"),
            remain=("remain", "last"),
            mean_price=("price", "mean"),
            mean_sl=("selling_limit", "mean"),
            weekend=("weekend", "first"),
            peak_month=("peak_month", "first"),
            is_soft=("is_soft", "first"),
            month=("month", "first"),
        )
        .reset_index()
    )
    ep["load"] = 1 - ep["remain"] / float(cfg["env"]["capacity"])
    ep.to_csv(out / "episode_summary.csv", index=False)

    make_plots(df, ep, out)
    print(f"Wrote plots + CSVs to {out.resolve()}")
    print(
        "soft",
        int(((ep.policy == RL_LABEL) & (ep.is_soft == 1)).sum()),
        "peak",
        int(((ep.policy == RL_LABEL) & (ep.is_soft == 0)).sum()),
    )
    print(
        "weekend px",
        float(df_rl[df_rl.weekend == 1]["price"].mean()),
        "weekday",
        float(df_rl[df_rl.weekend == 0]["price"].mean()),
    )


if __name__ == "__main__":
    main()
