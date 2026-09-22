"""The console entry points, driven with argv lists."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reservation_pricing.cli import baselines_main, eval_main, fit_demand_main, tune_main

ROOT = Path(__file__).resolve().parents[1]


def test_baselines_cli_writes_the_tables(tmp_path):
    baselines_main(
        [
            "-c",
            str(ROOT / "configs" / "default.yaml"),
            "--episodes",
            "1",
            "--out-dir",
            str(tmp_path),
        ]
    )
    table = pd.read_csv(tmp_path / "comparison_table.csv")
    assert (table["n_episodes"] == 1).all()
    assert (tmp_path / "comparison.md").exists()


def test_eval_cli_soft_aware_without_a_model(tmp_path):
    eval_main(["--episodes", "2", "--out-dir", str(tmp_path), "--soft-aware"])
    assert (tmp_path / "soft_aware_comparison.md").exists()
    assert (tmp_path / "soft_aware_table.csv").exists()


def test_episode_count_comes_from_the_config_when_the_flag_is_absent(tmp_path):
    from reservation_pricing.evaluate import run_comparison

    cfg = tmp_path / "two.yaml"
    cfg.write_text("eval:\n  n_episodes: 2\n  seeds: [0, 1]\n")
    out = run_comparison(config_path=str(cfg), n_episodes=None, include_baselines=True)
    assert (out["table"]["n_episodes"] == 2).all()


def test_fit_demand_cli_writes_a_loadable_asset(tmp_path):
    from reservation_pricing.demand.synthesize import load_fitted

    out = tmp_path / "tree.joblib"
    fit_demand_main(["--out", str(out), "--n-samples", "300", "--seed", "1"])
    payload = load_fitted(out)
    assert payload["meta"]["n_samples"] == 300
    assert payload["meta"]["source"] == "synthetic"


def test_tune_cli_help_names_the_trials_flag(capsys):
    with pytest.raises(SystemExit):
        tune_main(["--help"])
    assert "--trials" in capsys.readouterr().out


@pytest.mark.slow
def test_run_tune_grid_ignores_the_trial_count_and_stays_in_out_dir(tmp_path):
    """`rprl-tune --trials N` used to TypeError under the default grid method."""
    from reservation_pricing.tune import run_tune

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(f"train:\n  model_dir: {tmp_path / 'artifacts'}\n")
    cell = {"algorithm": {"learning_rate": 3e-4, "n_steps": 64, "batch_size": 32, "n_envs": 1}}
    out = run_tune(
        config_path=str(cfg),
        method="grid",
        n_trials=2,
        timesteps_per_trial=64,
        eval_episodes=1,
        out_dir=str(tmp_path / "grid"),
        grid=[cell],
    )
    assert len(out["rows"]) == 1
    assert Path(out["summary"]) == tmp_path / "grid" / "tuning_summary.md"
    model = Path(out["rows"][0]["model_path"])
    assert model.is_relative_to(tmp_path / "artifacts" / "grid")  # train.model_dir/<out_dir name>
    assert not model.is_relative_to(tmp_path / "grid")
