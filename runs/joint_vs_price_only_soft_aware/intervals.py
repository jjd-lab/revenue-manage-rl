"""Paired intervals for the published headline table.

Reads ``episode_metrics.csv`` and ``soft_aware_summary.json``. Does not roll
policies and does not rewrite the point-estimate table.
"""

from __future__ import annotations

from pathlib import Path

from reservation_pricing.evaluate.intervals import write_saved_intervals

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "paired_intervals.csv"
    frame = write_saved_intervals(Path(__file__).resolve().parent)
    print(f"Wrote {len(frame)} pairs to {out}")
