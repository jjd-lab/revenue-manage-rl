"""Synthesize a plausible amphitheater-reservation base-demand corpus and fit a tree model.

Used offline to ship ``assets/tree_base_demand.joblib``. Also exposed as
``rprl-fit-demand`` / ``fit_base_demand_from_csv`` for refitting on real data later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

FEATURE_COLS = [
    "days_prior",
    "dow",
    "month",
    "is_weekend",
    "is_peak_month",
    "booking_curve_bin",
]

DEFAULT_ASSET = Path(__file__).resolve().parent / "assets" / "tree_base_demand.joblib"


def _booking_curve_bin(days_prior: int, horizon: int = 100, n_bins: int = 5) -> int:
    frac = 1.0 - float(days_prior) / float(max(horizon, 1))
    return int(np.clip(np.floor(frac * n_bins), 0, n_bins - 1))


def synthetic_base_demand_mean(
    days_prior: int,
    dow: int,
    month: int,
    horizon: int = 100,
) -> float:
    """Plausible price-unaware mean bookings for a performance date.

    Shape notes (intentional for synthetic corpus):
    - Strong weekends and peak months (summer + holidays)
    - Booking-curve: low far out, rises toward ~14–21 days prior, mild late spike
    - No price in this function — price response is applied separately
    """
    is_weekend = 1 if dow in (5, 6) else 0
    is_peak = 1 if month in (7, 8, 11, 12) else 0
    # Soft shoulder peaks (spring / early fall)
    is_shoulder = 1 if month in (4, 5, 9, 10) else 0

    # Base level
    level = 95.0 + 55.0 * is_weekend + 45.0 * is_peak + 15.0 * is_shoulder

    # Booking curve multiplier vs days_prior
    # Far out (>60d): ~0.55; mid (21–45): ~1.0; near (7–14): ~1.25; last-minute (<5): ~0.9
    if days_prior > 60:
        curve = 0.55 + 0.15 * (100 - min(days_prior, 100)) / 40.0
    elif days_prior > 30:
        curve = 0.85 + 0.15 * (60 - days_prior) / 30.0
    elif days_prior > 14:
        curve = 1.05 + 0.15 * (30 - days_prior) / 16.0
    elif days_prior > 5:
        curve = 1.20 + 0.05 * (14 - days_prior) / 9.0
    else:
        curve = 0.85 + 0.05 * days_prior

    # Mild DOW fine structure (Fri slightly higher than Sat for live venues)
    dow_fine = {4: 1.05, 5: 1.12, 6: 1.08}.get(dow, 1.0)

    return float(max(0.0, level * curve * dow_fine))


def synthesize_corpus(
    n_samples: int = 8000,
    seed: int = 7,
    horizon: int = 100,
    noise_std: float = 12.0,
) -> pd.DataFrame:
    """Draw (features, base_demand) rows for fitting the price-unaware tree."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(n_samples):
        days_prior = int(rng.integers(0, horizon + 1))
        dow = int(rng.integers(0, 7))
        month = int(rng.integers(1, 13))
        mean = synthetic_base_demand_mean(days_prior, dow, month, horizon=horizon)
        y = max(0.0, mean + float(rng.normal(0.0, noise_std)))
        rows.append(
            {
                "days_prior": days_prior,
                "dow": dow,
                "month": month,
                "is_weekend": 1 if dow in (5, 6) else 0,
                "is_peak_month": 1 if month in (7, 8, 11, 12) else 0,
                "booking_curve_bin": _booking_curve_bin(days_prior, horizon),
                "base_demand": y,
            }
        )
    return pd.DataFrame(rows)


def fit_gradient_boosted(
    df: pd.DataFrame,
    *,
    n_estimators: int = 80,
    max_depth: int = 3,
    learning_rate: float = 0.08,
    random_state: int = 7,
) -> GradientBoostingRegressor:
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df["base_demand"].to_numpy(dtype=np.float64)
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def save_fitted(model: Any, path: str | Path, meta: Optional[dict] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_cols": FEATURE_COLS, "meta": meta or {}}
    joblib.dump(payload, path)
    return path


def load_fitted(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_ASSET
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted base-demand asset missing: {path}. "
            "Run `rprl-fit-demand` or reservation_pricing.demand.synthesize.build_and_ship_asset()."
        )
    return joblib.load(path)


def build_and_ship_asset(
    out_path: str | Path | None = None,
    n_samples: int = 8000,
    seed: int = 7,
) -> Path:
    """Fit on synthetic corpus and write the shipped joblib asset."""
    out = Path(out_path) if out_path else DEFAULT_ASSET
    df = synthesize_corpus(n_samples=n_samples, seed=seed)
    model = fit_gradient_boosted(df, random_state=seed)
    # Quick in-sample R^2 for meta
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df["base_demand"].to_numpy(dtype=np.float64)
    r2 = float(model.score(X, y))
    return save_fitted(
        model,
        out,
        meta={
            "source": "synthetic",
            "n_samples": n_samples,
            "seed": seed,
            "in_sample_r2": r2,
            "formula_note": "price-unaware base; elasticity applied at inference",
        },
    )


def fit_base_demand_from_csv(
    csv_path: str | Path,
    out_path: str | Path,
    target_col: str = "base_demand",
    feature_cols: Optional[Sequence[str]] = None,
) -> Path:
    """Refit tree from a CSV (real data path). Requires columns in FEATURE_COLS + target."""
    cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
    df = pd.read_csv(csv_path)
    missing = [c for c in cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    work = df[cols].copy()
    work["base_demand"] = df[target_col]
    model = fit_gradient_boosted(work)
    return save_fitted(
        model,
        out_path,
        meta={"source": "csv", "csv_path": str(csv_path), "n_rows": len(work)},
    )


def fit_demand_main(argv: list[str] | None = None) -> None:
    """CLI: rprl-fit-demand — synthesize+ship or fit from CSV."""
    import argparse

    p = argparse.ArgumentParser(description="Fit / ship tree base-demand model")
    p.add_argument("--csv", default=None, help="Optional CSV with FEATURE_COLS + base_demand")
    p.add_argument("--out", default=str(DEFAULT_ASSET), help="Output joblib path")
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    if args.csv:
        path = fit_base_demand_from_csv(args.csv, args.out)
    else:
        path = build_and_ship_asset(out_path=args.out, n_samples=args.n_samples, seed=args.seed)
    payload = load_fitted(path)
    print(f"Wrote {path}")
    print(f"meta={payload.get('meta')}")
