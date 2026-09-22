# How to tell if an RL pricing policy is actually good

Aggregate score (revenue − shortfall penalty) is necessary but not sufficient.
A policy can win the table by accident, by overselling on lucky seeds, or by
collapsing to a blunt rule. Use this checklist.

## 1. Beat honest baselines on held-out business metrics

Compare on the **same** seeds / held-out months:

- Fixed prices (low / mid / high)
- Myopic (optimize today’s expected revenue via the demand API)
- Heuristic booking-limit

Report **true revenue**, **load / remaining inventory**, **sellout / oversell rate**,
and the selection **score**. Winning only on shaped training reward does not count.

### Know what your baseline is actually doing

`myopic_greedy` emits a selling limit as well as a price, but **that limit never
binds**. It is a fixed `capacity * 1.05 / 0.85 = 12,353`, and the most bookings the
policy ever holds is **11,402**, so no booking is ever refused because of it. Only
the late-horizon tighten (drop to `min_selling_limit` once remaining inventory falls
under 5% of capacity) ever refuses anything. Measured consequence: the score is
identical to the cent for any limit from ~11,400 up to the 15,000 cap — which is why
`runs/price_only_long/` reports the same `1,049,488 / 730,305` for
`myopic_greedy@joint` (fixed limit) and `myopic_greedy@price_only` (analytic
controller, limit 15,000).

So myopic is effectively a **price-only policy with an open limit**, and a joint
policy beating it is partly beating a one-lever opponent. That is the comparison the
soft-aware table is making; read it that way, and read `price_only_pace_ppo` as the
honest one-lever benchmark. Before trusting any baseline, check whether its
constraints bind at all — an unbinding constraint is a knob that looks tuned and
does nothing.

## 2. Check for policy collapse

Look at action diversity over episodes:

- Fraction of days at min or max price
- Number of distinct prices / selling limits
- Within-episode price volatility and number of meaningful price changes

A “good” score with ~1 unique price (a degenerate always-max-price policy) is a bad policy.

## 3. Read the booking-curve path

Average (or plot) price, selling limit, accepted bookings, and remaining inventory
versus **days prior**. Sensible perishable RM often:

- Opens with a calendar-aware price (weekend / peak higher)
- Uses selling limits as an overbooking buffer for cancels / no-shows
- May **discount late** to clear leftover seats (perishable inventory)
- Tightens limits when nearly full

Red flags: always max price; always min price; selling limit stuck at the floor
while seats remain; systematic large oversell.

## 4. Check state dependence (not just time)

Calendar: weekend vs weekday, peak vs off-peak mean prices should move the right way
if demand is higher then.

Inventory: at a **fixed** days-prior window, does price rise when fill is higher?
Raw corr(price, fill) can be misleading because late days are both fuller and
cheaper if the policy discounts to clear. Condition on time bins.

## 5. Stress and robustness

- Held-out months / seeds (already default)
- Extra demand noise or elasticity misspecification (not done here)
- Offline: score on logged behavior if you have production traces later (not done here)

## 6. Explain a few episodes end-to-end

Pick 2–3 trajectories (peak weekend, soft weekday). Narrate price/SL choices and
ending load. If you cannot tell a coherent RM story, do not trust the leaderboard.

## Commands in this repo

```bash
# Leaderboard
rprl-eval -c configs/default.yaml --model artifacts/tree_long/best/rl_best.zip --episodes 30

# BC warm-start from myopic → SAC fine-tune
rprl-bc-sac -c configs/experiment_bc_sac.yaml
rprl-eval -c configs/default.yaml --model artifacts/bc_sac/rl_bc_sac_final.zip --algo sac --episodes 30

# Explainability pack for the shipped joint SAC (figures + episode/rollout CSVs)
python scripts/explain_rl_best.py   # writes runs/explain_rl_best/
```

## BC → SAC warm-start

When pure RL undersells or explores poorly, clone `myopic_greedy` on **train**
months into the SAC actor (MSE on normalized actions), seed the replay buffer,
then fine-tune. See `src/reservation_pricing/algorithms/bc.py`,
`train/bc_finetune.py`, and `runs/bc_sac/NOTES.md`.

A shaped training return is not a business metric; every table in `runs/` scores
unshaped revenue and inventory (see `docs/EXPERIMENT_LOG.md` §1).

## Price-only policies

When evaluating `control.price_only` runs, compare against the same baselines on
the **same** env factory (wrapper applies the configured SL controller). Do not
mix joint-action checkpoints with price-only envs. SL diversity comes from the
controller (analytic vs optimize_1d), not from the RL action; judge the **price**
policy and whether controller SL is sane (tighten near full, buffer early).

## Pace reward + soft-day upweight

When policies chronically undersell on weekdays / off-peak, enable `env.reward.pace_reward`
and `soft_day_upweight` (see `configs/experiment_price_only_pace_ppo.yaml`). Judge success
on **held-out business metrics** (especially `undersell>1500` and selection score), not on
shaped training return — pace/soft-day terms are training-only.


## Soft-day-aware evaluation

Overall `undersell>1500` mixes two very different regimes:

1. **Peak / rich** episodes — demand is high enough that a good policy *should*
   fill; leftover inventory is a real failure.
2. **Soft** episodes — mid-horizon tree base is low and/or the calendar is
   weekday / off-peak (e.g. June weekday). Even always-min-price / myopic at max
   SL still ends with remain ≫ 1500 (structural ceiling). Penalizing those
   episodes the same way as peak undersell over-ranks blunt floor-pricing and
   under-ranks safe high-revenue policies.

### Classification (`eval.soft_aware`)

Documented thresholds (defaults in `configs/default.yaml`):

| Signal | Default |
| --- | --- |
| peak months | 7, 8, 11, 12 |
| weekend DOW | 5, 6 |
| soft focus months | 6 (June held-out) |
| mid-horizon base threshold | 90.0 |

- **`rule: structural`** (default): soft if `base < threshold` **OR**
  `(weekday AND month ∈ soft_focus_months)`. Matches the oracle soft-day ceiling
  set (13 of the 30 held-out seeds every run uses).
- **`rule: any`**: soft if weekday **OR** off-peak month **OR** low base
  (broader; most held-out June days become soft).

Peak / rich = not soft.

### Stratified KPIs

Report **overall | soft only | peak only**.

| Slice | Lead with | De-emphasize |
| --- | --- | --- |
| **Soft** | revenue vs soft oracle (**gap to oracle**), fraction of days at/near floor (“did we try?”), oversell | remain>1500 / undersell rate |
| **Peak** | revenue, load/remain, oversell, score | — (undersell *does* matter) |
| **Overall** | `score_aware` (below) | `undersell>1500` as **secondary / diagnostic** only |

Soft oracle (`soft_oracle: min_price` or `myopic`): same seeds, always floor price
(or myopic) — the achievable soft revenue ceiling under the demand model.

### Soft-aware selection scores

```
score_peak  = mean_rev_peak − λ · mean_shortfall_peak
score_soft  = mean_rev_soft − gap_to_oracle          # soft_score_mode: gap_to_oracle
            | mean_rev_soft − μ · mean_oversell_amt  # soft_score_mode: oversell_only
score_aware = score_peak + score_soft
```

- Shortfall on peak uses the usual `max(remain,0) + 2·max(−remain,0)`.
- Soft score **does not** subtract undersell shortfall (structural leftover is
  expected). Prefer gap-to-oracle, or oversell-only if no oracle roll is available.
- Legacy overall `score = rev − λ·shortfall` and `undersell>1500` stay in the
  artifacts for continuity but are marked diagnostic when soft-aware is on.

### Paired intervals

A point estimate on thirty nights is not a ranking. Compare policies on the
**same** seeds, resample that seed list with replacement, and recompute
`score_aware` on each resample. The interval is the 2.5 and 97.5 percentiles of
the paired difference. Bootstrapping the finished score would treat a soft night
and a peak night as interchangeable, and `score_aware` is a stratified sum.

An interval that covers zero is a **tie**. Do not rank through it.

```bash
rprl-eval -c configs/default.yaml --soft-aware --interval \
  --baseline-policy myopic_greedy --episodes 30 \
  --model artifacts/bc_sac/rl_bc_sac_final.zip --algo sac \
  --out-dir runs/scratch/soft_aware_eval
```

The headline intervals are computed from the saved episode table, not from a
new rollout:

```bash
python runs/joint_vs_price_only_soft_aware/intervals.py
```

That writes `paired_intervals.csv` beside the point estimates and does not
rewrite them. Seeds 0–29 stay the reported set. A longer list changes the
soft/peak split, so extend it only for a tie you still want to separate.

### Commands

```bash
# Soft-aware leaderboard (writes soft_aware_comparison.md + comparison.md)
rprl-eval -c configs/default.yaml --soft-aware --episodes 30 \
  --model artifacts/bc_sac/rl_bc_sac_final.zip --algo sac \
  --out-dir runs/scratch/soft_aware_eval

# Multi-policy demo (bc_sac+safe_sl vs pace_ppo vs myopic)
python runs/soft_aware_report_demo/run_demo.py
```
