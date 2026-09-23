"""The figure scripts under scripts/, imported by path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from reservation_pricing.baselines import myopic_greedy_policy
from reservation_pricing.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_site_figures_writes_five_charts_from_committed_tables(tmp_path):
    build = _load("build_site_figures")
    written = build.main(out=tmp_path)
    assert [p.name for p in written] == [
        "f1_score.png",
        "f3_ceiling.png",
        "f4_booking_curve.png",
        "f5_lift.png",
        "f6_night_controls.png",
    ]
    assert all(p.stat().st_size > 10_000 for p in written)


def test_explain_rollouts_use_the_repo_soft_rule_and_canonical_features():
    explain = _load("explain_rl_best")
    df = explain.rollout_rows(myopic_greedy_policy(), "myopic", [1, 4], load_config())
    assert df["base_demand"].notna().all()
    # Seeds 1 and 4 are soft and peak under the structural rule (see test_smoke).
    assert int(df.loc[df.seed == 1, "is_soft"].iloc[0]) == 1
    assert int(df.loc[df.seed == 4, "is_soft"].iloc[0]) == 0
    # Canonical 5-bin inverted booking-curve encoding: far-out days are low base.
    far = df[(df.policy == "myopic") & (df.seed == 4) & (df.days_prior >= 90)]["base_demand"].mean()
    near = df[(df.policy == "myopic") & (df.seed == 4) & (df.days_prior.between(7, 14))][
        "base_demand"
    ].mean()
    assert far < near
