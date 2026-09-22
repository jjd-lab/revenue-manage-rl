#!/usr/bin/env python3
"""Oracle soft-day fill ceiling under tree_elastic (diagnostic).

For held-out soft episodes (June-like / weekday / low tree base), compute best
achievable end remain/load under:
  1) always min_price ($80) + max SL
  2) myopic 1D price grid each day + max SL
  3) open-loop best constant price + max SL

Writes runs/oracle_ceiling/NOTES.md and CSV artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reservation_pricing.baselines.policies import _myopic_price_grid, _to_env_action
from reservation_pricing.config import load_config
from reservation_pricing.demand.protocol import features_from_state
from reservation_pricing.envs import make_env
from reservation_pricing.metrics import UNDERSELL_THRESHOLD, run_episode

OUT = Path(__file__).resolve().parent
EPISODES = 30
UNDERSELL = UNDERSELL_THRESHOLD


def _is_soft(env) -> tuple[bool, float]:
    u = env.unwrapped
    feats = features_from_state(
        days_prior=max(u.booking_horizon // 2, 1),
        dow=u.dow,
        month=u.month,
        booking_horizon=u.booking_horizon,
    )
    base = float(u.demand_model.predict_base(feats))
    weekday = u.dow not in (5, 6)
    soft = (base < 90.0) or (weekday and u.month == 6)
    return soft, base


def _cfg_noise0():
    cfg = load_config()
    cfg = dict(cfg)
    cfg["demand"] = dict(cfg.get("demand") or {})
    cfg["demand"]["demand_noise_std"] = 0.0
    cfg["env"] = dict(cfg.get("env") or {})
    cfg["env"]["demand_noise_std"] = 0.0
    cfg["env"]["noshow_noise_std"] = 0.0
    cfg["env"]["cancel_rho_noise_std"] = 0.0
    return cfg


def fixed_price_max_sl(price: float):
    def _policy(obs, env, state):
        return _to_env_action(env, price, env.max_selling_limit)

    return _policy


def myopic_max_sl(n_grid: int = 41):
    def _policy(obs, env, state):
        p = _myopic_price_grid(env, n_grid=n_grid)
        return _to_env_action(env, p, env.max_selling_limit)

    return _policy


def best_constant_price(env_factory, seed: int, n_grid: int = 21) -> tuple[float, dict]:
    """Open-loop: try constant prices, pick lowest remain (best fill) then revenue."""
    prices = np.linspace(80.0, 120.0, n_grid)
    best = None
    best_p = 80.0
    for p in prices:
        env = env_factory()
        ep = run_episode(env, fixed_price_max_sl(float(p)), seed=seed)
        d = ep.to_dict()
        # Prefer fill (lower remain if remain>=0), then revenue; penalize oversell
        remain = float(d["remain_inv"])
        score_key = (remain < 0, abs(remain) if remain < 0 else remain, -float(d["true_revenue"]))
        if best is None or score_key < best[0]:
            best = (score_key, d)
            best_p = float(p)
    assert best is not None
    d = dict(best[1])
    d["const_price"] = best_p
    return best_p, d


def main() -> None:
    cfg = _cfg_noise0()
    seeds = list(range(EPISODES))

    def env_factory():
        return make_env(cfg, use_held_out=True)

    soft_rows = []
    all_rows = []

    for seed in seeds:
        env = env_factory()
        obs, info = env.reset(seed=seed)
        soft, base = _is_soft(env)
        meta = {
            "seed": seed,
            "soft": soft,
            "base50": base,
            "month": int(env.unwrapped.month),
            "dow": int(env.unwrapped.dow),
            "date": str(env.unwrapped.service_date.date()),
        }

        # always min price
        env = env_factory()
        ep80 = run_episode(env, fixed_price_max_sl(80.0), seed=seed)
        d80 = ep80.to_dict()
        d80.update(meta)
        d80["policy"] = "always_min_price_80"

        # myopic
        env = env_factory()
        epm = run_episode(env, myopic_max_sl(41), seed=seed)
        dm = epm.to_dict()
        dm.update(meta)
        dm["policy"] = "myopic_grid_max_sl"

        # best constant
        bp, dc = best_constant_price(env_factory, seed, n_grid=21)
        dc.update(meta)
        dc["policy"] = "best_constant_price"
        dc["const_price"] = bp

        for d in (d80, dm, dc):
            d["undersell_gt1500"] = float(d["remain_inv"] > UNDERSELL)
            d["oversell"] = float(d["remain_inv"] < 0)
            all_rows.append(d)
            if soft:
                soft_rows.append(d)

        print(
            f"seed={seed} soft={soft} base={base:.1f} "
            f"min80_remain={d80['remain_inv']:.0f} "
            f"myopic_remain={dm['remain_inv']:.0f} "
            f"bestC_p={bp:.0f} remain={dc['remain_inv']:.0f}",
            flush=True,
        )

    df = pd.DataFrame(all_rows)
    soft_df = pd.DataFrame(soft_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "oracle_all_episodes.csv", index=False)
    soft_df.to_csv(OUT / "oracle_soft_episodes.csv", index=False)

    def summarize(sub: pd.DataFrame, name: str) -> dict:
        out = {"slice": name, "n": int(len(sub) // 3) if len(sub) else 0}
        for pol in ["always_min_price_80", "myopic_grid_max_sl", "best_constant_price"]:
            s = sub[sub["policy"] == pol]
            if s.empty:
                continue
            out[pol] = {
                "mean_remain": float(s["remain_inv"].mean()),
                "mean_load": float(s["load_factor"].mean()),
                "mean_revenue": float(s["true_revenue"].mean()),
                "undersell_gt1500_rate": float(s["undersell_gt1500"].mean()),
                "oversell_rate": float(s["oversell"].mean()),
                "min_remain": float(s["remain_inv"].min()),
                "max_remain": float(s["remain_inv"].max()),
            }
        return out

    soft_summary = summarize(soft_df, "soft")
    all_summary = summarize(df, "all_heldout")
    with (OUT / "oracle_summary.json").open("w") as f:
        json.dump({"soft": soft_summary, "all": all_summary}, f, indent=2)

    # Per soft episode: can any oracle beat undersell>1500?
    soft_seeds = sorted(soft_df["seed"].unique()) if len(soft_df) else []
    unavoidable = []
    avoidable = []
    for seed in soft_seeds:
        s = soft_df[soft_df["seed"] == seed]
        best_remain = float(s["remain_inv"].min())
        row = {
            "seed": int(seed),
            "best_remain": best_remain,
            "unavoidable_undersell_gt1500": best_remain > UNDERSELL,
            "date": str(s["date"].iloc[0]),
            "base50": float(s["base50"].iloc[0]),
        }
        if best_remain > UNDERSELL:
            unavoidable.append(row)
        else:
            avoidable.append(row)

    n_soft = len(soft_seeds)
    n_unav = len(unavoidable)
    rate_unav = n_unav / max(n_soft, 1)

    # Best oracle undersell rate on soft
    best_pol = "always_min_price_80"
    best_u = 1.0
    for pol, stats in soft_summary.items():
        if not isinstance(stats, dict):
            continue
        u = stats.get("undersell_gt1500_rate", 1.0)
        if u < best_u:
            best_u = u
            best_pol = pol

    lines = [
        "# Oracle soft-day fill ceiling",
        "",
        "**Demand:** `tree_elastic` with **noise std = 0** (deterministic mean ceiling)",
        "**Held-out:** months 6 & 12, seeds 0..29",
        "**Soft definition:** mid-horizon tree base < 90 **or** June weekday",
        "**SL:** max selling limit (15000) — inventory not the binding constraint",
        "",
        "## Soft-episode counts",
        "",
        f"- Soft episodes: **{n_soft}** / {EPISODES}",
        f"- Soft episodes where **all** oracles still end with remain > {UNDERSELL:g}: "
        f"**{n_unav}** ({rate_unav:.3f})",
        f"- Soft episodes where some oracle can get remain ≤ {UNDERSELL:g}: **{len(avoidable)}**",
        "",
        "## Soft-slice aggregate (noise-free)",
        "",
        "| policy | mean remain | mean load | undersell>1500 | oversell | mean revenue |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in ["always_min_price_80", "myopic_grid_max_sl", "best_constant_price"]:
        st = soft_summary.get(pol, {})
        if not st:
            continue
        lines.append(
            f"| {pol} | {st['mean_remain']:.1f} | {st['mean_load']:.3f} | "
            f"{st['undersell_gt1500_rate']:.3f} | {st['oversell_rate']:.3f} | "
            f"{st['mean_revenue']:.0f} |"
        )

    lines += [
        "",
        "## Full held-out aggregate (noise-free)",
        "",
        "| policy | mean remain | mean load | undersell>1500 | oversell | mean revenue |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pol in ["always_min_price_80", "myopic_grid_max_sl", "best_constant_price"]:
        st = all_summary.get(pol, {})
        if not st:
            continue
        lines.append(
            f"| {pol} | {st['mean_remain']:.1f} | {st['mean_load']:.3f} | "
            f"{st['undersell_gt1500_rate']:.3f} | {st['oversell_rate']:.3f} | "
            f"{st['mean_revenue']:.0f} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
    ]
    # Compare to the 0.433 floor (~13/30)
    floor_note = (
        f"Prior RL / myopic / bc_sac all tied at **undersell>1500 ≈ 0.433** on noisy held-out "
        f"({int(round(0.433 * EPISODES))}/{EPISODES} episodes). Soft count here is **{n_soft}**."
    )
    if rate_unav >= 0.95 or (n_soft > 0 and best_u >= 0.95):
        verdict = (
            f"**YES — structural ceiling:** even always-min-price / best-constant / myopic "
            f"with max SL cannot clear remain≤{UNDERSELL:g} on essentially all soft days "
            f"(best soft undersell>1500 rate = {best_u:.3f} via `{best_pol}`). "
            f"Undersell>1500 on soft days is **unavoidable** given current tree base + "
            f"elasticity={cfg['demand'].get('elasticity', -1.2)} and price≥$80."
        )
    elif best_u < 0.433:
        verdict = (
            f"**PARTIAL ceiling:** some soft undersell is avoidable under oracle pricing "
            f"(best soft rate {best_u:.3f} via `{best_pol}`), but {n_unav}/{n_soft} soft "
            f"episodes remain stuck above remain {UNDERSELL:g}."
        )
    else:
        verdict = (
            f"**Structural soft-day floor remains:** best oracle soft undersell>1500 = "
            f"{best_u:.3f} (`{best_pol}`), not below the 0.433 RL floor. "
            f"{n_unav}/{n_soft} soft episodes unavoidable."
        )

    lines += [verdict, "", floor_note, ""]
    if unavoidable:
        lines += ["### Unavoidable soft seeds (best oracle remain > 1500)", ""]
        for r in unavoidable:
            lines.append(
                f"- seed {r['seed']} ({r['date']}, base50={r['base50']:.1f}): "
                f"best_remain={r['best_remain']:.0f}"
            )
        lines.append("")

    lines += [
        "## Implication for goals",
        "",
        "- **(B) better soft-day undersell:** if structural ceiling binds, MPC / promo / "
        "pace shaping **cannot** beat the 0.433 floor on this demand; treat undersell "
        "improvements as only possible on the avoidable subset (if any).",
        "- **(A) score + low oversell:** still addressable via safe-SL projection on "
        "bc_sac (cut oversell without needing more soft fill).",
        "",
        "## Artifacts",
        "",
        "- `oracle_all_episodes.csv`, `oracle_soft_episodes.csv`, `oracle_summary.json`",
        "",
    ]
    (OUT / "NOTES.md").write_text("\n".join(lines))
    print("Wrote", OUT / "NOTES.md")
    print("soft", n_soft, "unavoidable", n_unav, "best_u", best_u, best_pol)


if __name__ == "__main__":
    main()
