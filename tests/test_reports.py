"""The writers behind every table under runs/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reservation_pricing.baselines import fixed_price_policy
from reservation_pricing.config import load_config
from reservation_pricing.envs import make_env
from reservation_pricing.evaluate import run_soft_aware_comparison
from reservation_pricing.evaluate.report import (
    TABLE_COLUMNS,
    repo_relative,
    run_labelled,
    write_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def test_run_labelled_and_write_comparison(tmp_path):
    cfg = load_config()

    def factory():
        return make_env(cfg, use_held_out=True)

    row, episodes = run_labelled(
        "floor",
        factory,
        fixed_price_policy(80.0),
        episodes=2,
        seeds=[0, 1],
        model_path=ROOT / "artifacts" / "nope.zip",
    )
    assert row["n_episodes"] == 2 and len(episodes) == 2
    assert {"oversell_rate", "undersell_gt1500_rate", "score"} <= set(row)
    assert row["model_path"] == "artifacts/nope.zip"
    assert all(e["policy"] == "floor" for e in episodes)

    table = write_comparison(tmp_path, [row], episodes, title="Test", episodes=2)
    assert list(table.columns) == TABLE_COLUMNS
    for name in (
        "comparison_table.csv",
        "episode_metrics.csv",
        "comparison_summary.json",
        "comparison.md",
    ):
        assert (tmp_path / name).exists(), name
    md = (tmp_path / "comparison.md").read_text()
    assert md.startswith("# Test") and "## Artifact paths" in md and "`artifacts/nope.zip`" in md


def test_repo_relative_never_writes_a_machine_path(tmp_path):
    assert repo_relative(ROOT / "artifacts" / "x.zip") == "artifacts/x.zip"
    outside = tmp_path / "m.zip"
    assert repo_relative(outside) == str(outside)


def test_soft_aware_report_files_and_columns(tmp_path):
    out = run_soft_aware_comparison(
        n_episodes=3,
        seeds=[1, 4, 5],
        out_dir=str(tmp_path),
        include_baselines=True,
        baseline_names=["myopic_greedy", "fixed_price_80"],
    )
    for name in (
        "soft_aware_table.csv",
        "soft_aware_summary.json",
        "soft_aware_comparison.md",
        "episode_metrics.csv",
    ):
        assert (tmp_path / name).exists(), name
    table = pd.read_csv(tmp_path / "soft_aware_table.csv")
    assert {"policy", "score_aware", "peak_oversell"} <= set(table.columns)
    assert set(table["policy"]) == {"myopic_greedy", "fixed_price_80"}
    assert out["table"].shape[0] == 2
