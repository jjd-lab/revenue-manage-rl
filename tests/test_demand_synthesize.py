"""The synthetic corpus and the refit path for a user's own CSV."""

from __future__ import annotations

import numpy as np
import pytest

from reservation_pricing.demand.synthesize import (
    DEFAULT_ASSET,
    FEATURE_COLS,
    build_and_ship_asset,
    fit_base_demand_from_csv,
    load_fitted,
    synthesize_corpus,
)


def test_fit_from_csv_roundtrip_and_missing_column_error(tmp_path):
    df = synthesize_corpus(n_samples=200, seed=3)
    csv = tmp_path / "corpus.csv"
    df.to_csv(csv, index=False)
    payload = load_fitted(fit_base_demand_from_csv(csv, tmp_path / "fit.joblib"))
    assert payload["feature_cols"] == FEATURE_COLS
    assert payload["meta"] == {"source": "csv", "csv_path": str(csv), "n_rows": 200}

    df.drop(columns=["booking_curve_bin"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="booking_curve_bin"):
        fit_base_demand_from_csv(csv, tmp_path / "fit2.joblib")


def test_shipped_asset_is_reproducible_from_the_documented_seed(tmp_path):
    """README: the asset is `rprl-fit-demand --n-samples 8000` at seed 7."""
    refit = load_fitted(build_and_ship_asset(tmp_path / "refit.joblib", n_samples=8000, seed=7))
    shipped = load_fitted(DEFAULT_ASSET)
    X = synthesize_corpus(n_samples=200, seed=11)[FEATURE_COLS].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(refit["model"].predict(X), shipped["model"].predict(X), rtol=1e-6)
