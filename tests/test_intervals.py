"""Paired bootstrap of score_aware. The self-comparison must have width zero."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reservation_pricing.cli import eval_main
from reservation_pricing.evaluate.intervals import (
    load_saved_run,
    paired_difference,
)
from reservation_pricing.evaluate.soft_aware import aggregate_soft_aware, run_soft_aware_comparison
from reservation_pricing.metrics import EpisodeMetrics, SoftAwareConfig

ROOT = Path(__file__).resolve().parents[1]


def _episode(revenue: float, remain: float, soft: bool) -> EpisodeMetrics:
    return EpisodeMetrics(
        true_revenue=revenue,
        shaped_return=0.0,
        load_factor=0.5,
        remain_inv=remain,
        sellout_day=None,
        capacity=10_000,
        month=6 if soft else 12,
        dow=2,
        base_demand=80.0 if soft else 120.0,
        is_soft=soft,
        is_peak=not soft,
    )


def _by_seed(revenues: dict[int, tuple[float, float, bool]]) -> dict[int, EpisodeMetrics]:
    return {
        seed: _episode(revenue, remain, soft) for seed, (revenue, remain, soft) in revenues.items()
    }


def test_self_comparison_is_a_zero_width_interval():
    nights = {
        0: (1_000_000.0, 400.0, False),
        1: (800_000.0, 3_000.0, True),
        2: (1_200_000.0, -50.0, False),
        3: (850_000.0, 2_800.0, True),
    }
    same = _by_seed(nights)
    interval = paired_difference(
        same,
        same,
        policy="a",
        baseline="a",
        n_draws=40,
    )
    assert interval.point_diff == 0.0
    assert interval.mean_diff == 0.0
    assert interval.ci_low == 0.0
    assert interval.ci_high == 0.0
    assert interval.covers_zero


def test_pairing_follows_the_seed_not_the_row_order():
    left = _by_seed(
        {
            0: (1_000_000.0, 100.0, False),
            1: (900_000.0, 2_000.0, True),
        }
    )
    right = _by_seed(
        {
            1: (900_000.0, 2_000.0, True),
            0: (1_010_000.0, 100.0, False),
        }
    )
    forward = paired_difference(left, right, seeds=[0, 1], n_draws=20)
    backward = paired_difference(left, right, seeds=[1, 0], n_draws=20)
    assert forward.point_diff == backward.point_diff
    assert forward.point_diff != 0.0


def test_saved_headline_scores_match_the_published_table():
    run = ROOT / "runs" / "joint_vs_price_only_soft_aware"
    by_policy, cfg, oracle, seeds, order = load_saved_run(run)
    published = {
        row.policy: row.score_aware
        for row in pd.read_csv(run / "soft_aware_table.csv").itertuples()
    }
    for name in order:
        episodes = [by_policy[name][seed] for seed in seeds]
        scored = aggregate_soft_aware(
            episodes,
            cfg,
            oracle_revenue_by_seed=oracle,
            seed_by_index=seeds,
        )
        assert scored["score_aware"] == pytest.approx(published[name], abs=0.02)


def test_soft_aware_interval_columns_are_relative_to_the_baseline(tmp_path):
    out = run_soft_aware_comparison(
        n_episodes=2,
        seeds=[0, 1],
        out_dir=str(tmp_path),
        include_baselines=True,
        baseline_names=["myopic_greedy", "fixed_price_80"],
        interval=True,
        baseline_policy="myopic_greedy",
        n_bootstrap=15,
    )
    table = out["table"].set_index("policy")
    assert {"paired_diff", "paired_ci_low", "paired_ci_high"} <= set(table.columns)
    assert table.loc["myopic_greedy", "paired_ci_low"] == 0.0
    assert table.loc["myopic_greedy", "paired_ci_high"] == 0.0
    text = (tmp_path / "soft_aware_comparison.md").read_text()
    assert "myopic_greedy" in text and "covers zero is a tie" in text
    assert (tmp_path / "soft_aware_summary.json").read_text().count("oracle_revenue_by_seed") == 1


def test_interval_flag_requires_soft_aware_and_a_baseline():
    with pytest.raises(SystemExit):
        eval_main(["--interval", "--baseline-policy", "myopic_greedy"])
    with pytest.raises(SystemExit):
        eval_main(["--soft-aware", "--interval"])


def _pair(frame, left: str, right: str):
    forward = frame[(frame["policy"] == left) & (frame["baseline"] == right)]
    backward = frame[(frame["policy"] == right) & (frame["baseline"] == left)]
    found = forward if len(forward) else backward
    assert len(found) == 1
    return found.iloc[0]


def test_published_intervals_match_the_section_7_calls():
    """The experiment log calls these pairs ties or leads. Lock that to the CSV."""
    frame = pd.read_csv(ROOT / "runs" / "joint_vs_price_only_soft_aware" / "paired_intervals.csv")
    assert bool(_pair(frame, "rl_sac", "joint_bc_sac_raw")["tie"])
    assert bool(_pair(frame, "joint_sac_rl_best", "price_only_pace_ppo")["tie"])
    assert bool(_pair(frame, "joint_ppo_long_007", "price_only_pace_ppo")["tie"])
    assert not bool(_pair(frame, "rl_sac", "price_only_pace_ppo")["tie"])
    assert not bool(_pair(frame, "joint_bc_sac_raw", "price_only_pace_ppo")["tie"])
    pace = _pair(frame, "rl_sac", "price_only_pace_ppo")
    assert pace["paired_ci_low"] > 0


def test_config_round_trip_uses_the_published_lambda():
    """Guard the loader the headline intervals depend on. Not a bootstrap."""
    _by_policy, cfg, _oracle, _seeds, _order = load_saved_run(
        ROOT / "runs" / "joint_vs_price_only_soft_aware"
    )
    assert cfg == SoftAwareConfig.from_dict(
        {
            "lambda_peak": 200.0,
            "mu_soft": 200.0,
            "soft_score_mode": "gap_to_oracle",
        }
    )
