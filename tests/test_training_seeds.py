"""The training-seed driver must not be able to replace a published checkpoint.

It also has to send both training modes — a behaviour-clone warm start and a
standard run — to the same ``<root>/<label>_s<seed>/`` layout.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from reservation_pricing.train.runner import output_dirs

ROOT = Path(__file__).resolve().parents[1]


def _driver():
    path = ROOT / "runs" / "training_seeds" / "train.py"
    spec = importlib.util.spec_from_file_location("training_seeds_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_published_checkpoint_directories_are_refused():
    driver = _driver()
    with pytest.raises(ValueError, match="published checkpoint"):
        driver.assert_outside_published(ROOT / "artifacts" / "bc_sac")
    with pytest.raises(ValueError, match="published checkpoint"):
        driver.assert_outside_published(ROOT / "artifacts" / "bc_sac" / "rl_bc_sac_final.zip")
    with pytest.raises(ValueError, match="published checkpoint"):
        driver.assert_outside_published(ROOT / "artifacts" / "pace_ppo" / "somewhere")


def test_both_training_modes_share_one_directory_layout():
    driver = _driver()
    pace_cfg = ROOT / "configs" / "experiment_price_only_pace_ppo.yaml"
    bc_cfg = ROOT / "configs" / "experiment_bc_sac.yaml"
    pace_mode, pace, pace_run, pace_logs = driver.prepare(pace_cfg, 43, label="pace")
    bc_mode, bc, bc_run, bc_logs = driver.prepare(bc_cfg, 43, label="bc_sac")
    assert pace_mode == "standard"
    assert bc_mode == "bc"
    pace_model, pace_log = output_dirs(pace["train"], pace_run, str(pace_logs))
    bc_model, bc_log = output_dirs(bc["train"], bc_run, str(bc_logs))
    assert pace_model.parts[-2:] == ("training_seeds", "pace_s43")
    assert bc_model.parts[-2:] == ("training_seeds", "bc_sac_s43")
    assert pace_log.parts[-2:] == ("training_seeds", "pace_s43")
    assert bc_log.parts[-2:] == ("training_seeds", "bc_sac_s43")


def test_published_lead_is_training_seed_sensitive():
    """Seed 46's matched interval covers zero; the shipped pair does not.

    Locked to ``paired_intervals.csv``, which ``eval.py`` writes. A re-run that
    flips either flag has to update ``NOTES.md`` and §7 together.
    """
    frame = pd.read_csv(ROOT / "runs" / "training_seeds" / "paired_intervals.csv")
    matched = frame[frame["comparison"] == "matched"].set_index("policy")
    published = frame[frame["comparison"] == "published_pace"].set_index("policy")
    assert bool(matched.loc["bc_cap_s42", "tie"]) is False
    assert float(matched.loc["bc_cap_s42", "paired_ci_low"]) > 0
    assert bool(matched.loc["bc_cap_s46", "tie"]) is True
    assert float(published.loc["bc_cap_s46", "paired_ci_high"]) < 0


def test_reserved_seeds_are_not_retrained():
    driver = _driver()
    for seed in (42, 0, 100):
        with pytest.raises(ValueError, match="reserved"):
            driver.require_seed(seed)
