# Explainability pack — joint SAC `rl_best`

**Model:** `artifacts/tree_long/best/rl_best.zip` (joint continuous price + selling limit, SAC)  
**Baseline:** myopic greedy  
**Eval:** 30 held-out seeds, `configs/default.yaml`, tree_elastic demand  
**Regenerate:** from repo root with venv active:

```bash
python scripts/explain_rl_best.py
# optional: --seeds 30 --model artifacts/tree_long/best/rl_best.zip --out runs/explain_rl_best
```

## What the policy sees (important for interpretation)

The policy observation is **model-free**: inventory/booking state + calendar one-hots only.  
Tree `predict_base` is used **only for post-hoc analysis** in figures 03/04/07 (soft labels and scatter), not as an RL input.

## Headline numbers from this run

| Slice | Metric | rl_best | myopic |
| --- | --- | --- | --- |
| Overall steps | Mean weekend price | ~$109 | — |
| Overall steps | Mean weekday price | ~$84 | — |
| Soft episodes | Mean revenue | ~$863k | ~$871k |
| Peak episodes | Mean revenue | ~$1.26M | ~$1.19M |

**Takeaway:** lift vs myopic comes from **peak harvest**, not soft-day fill.

---

## Figure-by-figure guide

### `01_price_inventory_paths.png` — Dynamic booking-curve strategy

**What it shows:** Average price (left) and remaining inventory (right) vs days prior, averaged over 30 episodes, for SAC vs myopic.

**How to read it:**
- Myopic price is essentially **flat** (~$92).
- SAC **varies** price over the horizon: higher mid-curve, then a **sharp cut** in the last ~15 days.
- Inventory: SAC holds a bit more mid-horizon, then sells down harder late; ends with slightly less remain than myopic.

**Why it matters:** The agent learned a classic RM booking-curve pattern (protect early / clear late) rather than a constant price heuristic.

---

### `02_weekend_vs_weekday_price.png` — Calendar-aware weekend premium

**What it shows:** SAC’s average price path split by weekend vs weekday service dates.

**How to read it:**
- Weekend prices sit ~$25–35 above weekday for most of the curve (peak near ~$117 around 37 days prior).
- Weekdays stay low (~$80–90, mean ~$84).
- Both drop near departure; weekend drop is steeper from a higher base.

**Why it matters:** Even without an explicit demand forecast in the obs, SAC recovers a strong **calendar premium** from day-of-week / month one-hots + learning.

---

### `03_soft_vs_peak_boxplots.png` — Regime adaptation

**What it shows:** Episode-level mean price and total revenue for SAC, split into soft vs peak episodes.

Soft label (analysis only): June weekday **or** mid-horizon tree base demand &lt; 90 — the same `metrics.classify_soft` rule every table in `runs/` uses.

**How to read it:**
- Soft: lower mean prices and lower revenue.
- Peak: higher prices and much higher revenue.

**Why it matters:** The policy does not treat all days the same; it **prices down on soft** and **prices up on peak**. Soft undersell still remains structural (demand floor), which is why soft revenue does not beat myopic.

---

### `04_price_vs_base_demand.png` — Implicit demand awareness

**What it shows:** Scatter of chosen price vs tree base demand (price-unaware) at mid-horizon steps (days prior 40–70). Color = weekend.

**How to read it:**
- Higher base demand tends to line up with higher prices.
- Weekend points cluster toward higher price/base.

**Why it matters:** Post-hoc check that SAC’s actions **correlate** with the same demand signal production uses, even though that signal is **not** fed into the policy. Correlation ≠ causal input; it shows the policy’s calendar+pace behavior aligns with true demand richness.

---

### `05_selling_limit_vs_remain.png` — Joint action is used

**What it shows:** Chosen selling limit vs remaining inventory (scatter + binned mean).

**How to read it:**
- Points are noisy (continuous joint control).
- Binned mean shows SL **tracks** remaining seats (higher remain → higher average SL).

**Why it matters:** Confirms the second action dimension is not ignored; SAC co-controls availability with price. This is the “joint” part of joint SAC.

---

### `06_example_peak_vs_soft_trajectories.png` — Single-episode stories

**What it shows:** One peak-ish and one soft episode: SAC vs myopic price paths, plus SAC remain (green, right axis).

**How to read it:**
- **Peak panel:** SAC prices above myopic and manages inventory more aggressively for harvest.
- **Soft panel:** Both policies sit low; remain stays high — soft days do not clear even at low prices.

**Why it matters:** Concrete illustration of the aggregate story: win on peak, soft is a demand ceiling problem.

---

### `07_revenue_soft_vs_peak.png` — Where the $ comes from

**What it shows:** Mean episode revenue for SAC vs myopic on soft vs peak.

**How to read it:**
- Soft: essentially **tied** (SAC slightly below myopic in this run).
- Peak: SAC **ahead** by roughly ~$70k mean revenue.

**Why it matters:** Direct answer to “why does the best joint SAC work?” — **peak harvest + weekend premium + late clearance**, not soft fill.

---

## Files in this folder

| File | Role |
| --- | --- |
| `01_…`–`07_….png` | Figures |
| `rollouts.csv` | Step-level traces (policy, seed, price, SL, remain, base_demand, …) |
| `episode_summary.csv` | One row per (policy, seed) |
| `README.md` | This guide |

Generator: `scripts/explain_rl_best.py` at the repo root.

## Reproduce checklist

1. From repo root, with `artifacts/tree_long/best/rl_best.zip` present.
2. `source .venv/bin/activate && pip install -e ".[dev]"` (includes matplotlib).
3. `python scripts/explain_rl_best.py`
