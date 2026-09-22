# Experiment log: amphitheater reservation pricing RL

This document traces the project from the earlier prototype through the final
soft-aware joint vs price-only comparison. It is meant to be read with the
artifacts and configs in this repo.

**Held-out protocol (unless noted):** `tree_elastic` demand, months June & December,
30 episodes, `score = revenue − 200 × capacity_shortfall` (legacy) or soft-aware
scores defined below.

> **2026-09-21 correction.** The price-only controller (`envs/price_only.py`) was
> computing its selling-limit / early-promo decisions one day ahead of the day
> `ReservationEnv.step()` actually settles them against. Fixed; tables that go
> through it were regenerated. Measured effect: **pace PPO moved 832,319 →
> 831,842 (−0.06%)** — inside the run-to-run noise the README already documents
> for SB3 training, and it changes no comparison or ranking in this log. Every
> other row here (joint policies, the vanilla `price_only_ppo_analytic`
> checkpoint) was bit-identical *with respect to this fix*, since the bug only
> bites when the analytic selling limit is close enough to bind, which happens
> only for the soft-day-tuned pace policy. A separate, smaller drift (<0.01%)
> from rebuilding the venv on a different torch/BLAS build is documented in
> `runs/both_goals/NOTES.md` so the two are not conflated.

---

## 1. Framework rebuild

The project began from an earlier prototype whose MDP sketch — capacity 10 000,
100-day horizon, continuous **price + selling limit** — was right and whose
training, demand realism, and scoreboard were not. Built a config-driven package
(`reservation_pricing/`) in its place:

- Gymnasium env, SB3 **PPO / SAC** (TD3 is registered and smoke-tested but
  appears in no published result), classical baselines
- Default demand: **tree (GBT) base + linear price elasticity** (production-like);
  `linear_legacy` kept for A/B
- Registries for demand, algorithms, later selling-limit controllers
- Docs: `DESIGN.md`, `EXTENDING.md`, `EVALUATING_POLICIES.md`

**Takeaway:** One YAML selects demand / env / algo; the prototype is not the runtime path.

---

## 2. Longer training on tree demand

| Policy | Legacy score | Notes |
| --- | ---: | --- |
| SAC @200k (`rl_best`) | ~823k | Best pure joint before BC |
| PPO long @300k | ~817k | Beat baselines; some sellout |
| PPO screen @50k | ~755k | After fixing HP-merge bug |
| myopic / fixed_80 | ~578k / ~590k | |

**Takeaways**

- HP-merge bug had made early grids meaningless; fixing it unlocked real gains.
- Pure joint RL beat myopic/fixed_80 on legacy score.
- Soft undersell (~43% episodes remain>1500) remained.

Artifacts: `artifacts/tree_long/best/`.

---

## 3. BC → SAC (joint)

Clone myopic → warm SAC actor → fine-tune (`rprl-bc-sac`).

| Policy | Legacy score | Oversell | Undersell>1500 |
| --- | ---: | ---: | ---: |
| **BC→SAC final** | **~871k** | **0.40** | 0.433 |
| `rl_best` | ~823k | 0.10 | 0.433 |
| BC only | ~667k | 0.00 | 0.433 |

**Takeaways**

- BC→SAC wins legacy score via higher fill/revenue.
- Oversell jumps to 40% — unsafe if oversell is constrained.
- Undersell rate unchanged (not fixed by BC).

Artifacts: `artifacts/bc_sac/rl_bc_sac_final.zip`.

---

## 4. Price-only RL + analytic / 1D SL

Decomposed actions: RL outputs **price only**; SL from analytic or 1D-optimize controller.

| Policy | Legacy score | Oversell | Undersell>1500 |
| --- | ---: | ---: | ---: |
| price-only PPO @200k | ~827k | **0.00** | 0.467 |
| price-only SAC @150k | ~818k | **0.00** | 0.433 |
| BC→SAC | ~871k | 0.40 | 0.433 |

**Takeaways**

- Price-only beats joint PPO and ≈ `rl_best` with **zero oversell**.
- Does not beat BC→SAC on legacy score.
- Selling limit was not the soft-undersell bottleneck (limits already open).

Configs: `experiment_price_only_*.yaml`. Artifacts: `artifacts/price_only_long/`.

---

## 5. Soft-day undersell campaigns

### 5a. Pace reward + soft-day upweight (price-only PPO)

| Policy | Legacy score | Undersell>1500 | Oversell |
| --- | ---: | ---: | ---: |
| pace PPO | ~832k | 0.433 | 0.00 |
| price-only PPO | ~827k | 0.467 | 0.00 |

Small fill/score win; undersell floor unchanged.

### 5b. Early promo (force $80 when tree base soft)

Worse than pace on score (~820k); undersell tied at 0.433.

### 5c. Oracle ceiling + MPC + safe SL (“both goals”)

| Goal | Result |
| --- | --- |
| Score + low oversell | **Safe SL on BC→SAC:** oversell 0.40→0.00, score 871k→863k (−0.8%) |
| Soft undersell | **Structural:** oracle soft eps still remain>1500 even at $80; MPC no-op vs pace |

**Takeaway:** Soft empty seats are largely **uncontrollable demand**, not a missing RL trick.

Results: `runs/both_goals/` (safe SL + MPC tables), `runs/oracle_ceiling/` (the $80
oracle sweep). Config: `configs/experiment_bc_sac_safe_sl.yaml`. The safe-SL row reuses
`artifacts/bc_sac/rl_bc_sac_final.zip` under the wrapper — no separate checkpoint.

---

## 6. Soft-aware evaluation (KPI redesign)

Do **not** lead with overall `undersell>1500`.

- Split **soft vs peak** episodes.
- Soft KPIs: revenue, **gap to $80 oracle**, floor use, oversell (no undersell term).
- Peak KPIs: revenue, remain/load, oversell, shortfall score.
- Selection: `score_aware = score_peak + score_soft`.

```bash
rprl-eval --soft-aware ...
```

The two scores rank policies differently, and that is the point of the redesign:
the legacy score charges every unsold seat, so a policy is rewarded for chasing
fill it cannot get on soft nights; `score_aware` charges soft nights only for
falling below the floor-price oracle, so peak revenue net of oversell is what
separates the policies (§7). Under it the unconstrained BC→SAC still scores
highest, but the cap that removes its denied admission costs 0.6% rather than the
0.8% the legacy score charged (§5c) — the capped policy becomes the clear package.

Docs: `docs/EVALUATING_POLICIES.md` (soft-aware section).

---

## 7. Final soft-aware joint vs price-only table

**Source of truth for numbers:**  
`runs/joint_vs_price_only_soft_aware/soft_aware_table.csv`  
(and `soft_aware_comparison.md`).

**Protocol:** 30 held-out episodes, seeds 0–29 (the same thirty every other run
uses), soft-aware classification → 13 soft / 17 peak; all policies soft
**gap_to_oracle = 0**.

| policy | type | score_aware | peak_rev | peak_remain | peak_oversell | soft_floor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| joint BC→SAC raw | joint | **2.094M** | 1.310M | 50 | **0.71** | 0.36 |
| **joint BC→SAC + safe SL** | joint | **2.081M** | 1.280M | 293 | **0.00** | 0.36 |
| joint SAC `rl_best` | joint | 2.033M | 1.255M | 411 | 0.18 | 0.15 |
| joint PPO long_007 | joint | 2.011M | 1.265M | 285 | 0.35 | 0.53 |
| price-only pace PPO | price-only | 2.010M | 1.237M | 421 | **0.00** | 1.00 |
| price-only PPO | price-only | 2.003M | 1.264M | 590 | 0.00 | 0.94 |
| myopic | baseline | 1.972M | 1.186M | 423 | 0.00 | 0.00 |
| fixed_80 | baseline | 1.816M | 1.042M | 415 | 0.00 | 1.00 |

Raw BC→SAC scores highest by filling hardest, and denies admission on 12 of the
17 peak nights doing it. Under the safe-SL cap it keeps 99.4% of that score with
no denied admission, which is why it is the recommended package. The three
SAC-based joint policies sit above every price-only policy; the best price-only
policy (pace PPO) trails the capped BC→SAC by 3.6% and `rl_best` by 1.2%.

> **Seed sensitivity (2026-09-22).** This table was first published on a
> different thirty seeds (0–4 plus 1000–1024, an 18 soft / 12 peak split) that no
> other run used. On that draw joint SAC `rl_best` led (2.057M) ahead of raw
> BC→SAC (2.044M), the capped BC→SAC (2.040M) and pace PPO (1.991M). Moving to
> the common seeds 0–29 reordered the three joint policies and left their lead
> over every price-only policy intact; the order *within* that group is inside
> seed noise and should be read that way.

**Models used**

| Row | Artifact | Config |
| --- | --- | --- |
| joint SAC `rl_best` | `artifacts/tree_long/best/rl_best.zip` | `configs/default.yaml` |
| BC→SAC raw | `artifacts/bc_sac/rl_bc_sac_final.zip` | `configs/default.yaml` |
| BC→SAC + safe SL | same zip + safe SL wrapper | `configs/experiment_bc_sac_safe_sl.yaml` |
| joint PPO | `artifacts/tree_long/best/rl_ppo_long_007.zip` | `configs/default.yaml` |
| pace PPO | `artifacts/pace_ppo/rl_pace_ppo.zip` | `configs/experiment_price_only_pace_ppo.yaml` |
| price-only PPO | `artifacts/price_only_long/rl_ppo_analytic.zip` | `configs/experiment_price_only_ppo.yaml` |

In the CSV the row named `rl_sac` is BC→SAC **under the safe-SL wrapper** (it is
the `--model` of the run, evaluated under `experiment_bc_sac_safe_sl.yaml`); the
raw BC→SAC and `rl_best` rows carry the `joint_*` names.

Reproduce (needs the five checkpoints; see README § Model checkpoints):

```bash
source .venv/bin/activate
python runs/joint_vs_price_only_soft_aware/REPRODUCE.py
```

---

## 8. Explainability: what the joint policy learned

Joint SAC `rl_best` is the pure joint policy — no imitation, no cap — so it is
the one whose behaviour says what the two levers buy; this section says **what
it learned**. Figures and the figure-by-figure guide: `runs/explain_rl_best/`.

```bash
python scripts/explain_rl_best.py   # 30 held-out seeds, configs/default.yaml
```

**What the policy sees:** the observation is **model-free** — inventory / booking
state plus calendar one-hots. Tree `predict_base` appears only in post-hoc analysis
(figures 03/04/07), never as an RL input.

| Slice | Metric | `rl_best` | myopic |
| --- | --- | ---: | ---: |
| all steps | mean weekend price | ~$109 | — |
| all steps | mean weekday price | ~$84 | — |
| soft episodes | mean revenue | ~$863k | ~$871k |
| peak episodes | mean revenue | ~$1.26M | ~$1.19M |

**What it learned**

- **Booking-curve pricing.** Myopic is flat (~$92); SAC prices higher mid-curve and
  cuts sharply in the last ~15 days — protect early, clear late.
- **Calendar premium.** Weekend sits ~$25–35 above weekday (peak ~$117 around 40
  days prior), recovered from day-of-week / month one-hots alone.
- **The joint action is used.** Selling limit tracks remaining inventory rather
  than sitting at a bound — this is the "joint" in joint SAC.
- **Implicit demand awareness.** Chosen price correlates with tree base demand at
  mid-horizon even though that signal is not in the observation. Correlation, not
  a causal input.

**Where the lift comes from:** peak episodes, by roughly ~$70k mean revenue. Soft
episodes are essentially tied with myopic (slightly below in this run) — the same
structural demand floor found in 5c, seen from the policy side.

---

## 9. Ablation: does the second lever earn its place?

Section 7's 1.2–3.6% joint-vs-price-only gap admits two readings — the joint
policies use the limit well, or their *price* policy is better and the limit is
along for the ride. This re-scores each joint policy with its price untouched and
its limit replaced by a constant.

**Protocol:** same 30 held-out seeds, same `score_aware`.
Results: `runs/ablate_selling_limit/` (`python runs/ablate_selling_limit/run_ablation.py`).

| policy | learned limit | pinned open 15,000 | pinned flat 12,353 |
| --- | ---: | ---: | ---: |
| joint SAC `rl_best` | **2.033M** | 1.689M (−344k) | 1.914M (−120k) |
| joint BC→SAC raw | **2.094M** | 1.855M (−239k) | 1.943M (−152k) |
| joint PPO `long_007` | **2.011M** | 1.903M (−108k) | 1.921M (−90k) |

**Takeaways**

- **The gap is not a price-policy artifact.** Removing the limit costs 90k–344k,
  an order of magnitude more than the 1.2–3.6% that separates joint from price-only.
- **A sensible constant does not recover it.** `rl_best`'s limit *averages* 12,578,
  within 2% of the flat 12,353 — and pinning it there still costs 120k. The value
  is not where the limit sits, it is **when it moves** (§8's figure 05, priced).
- **The limit is what holds denied admission down.** Peak oversell goes 0.18 → 0.82
  for `rl_best` the moment the limit stops moving.
- **Caveat:** pinned arms are off-distribution — each policy priced for the limit it
  learned. This measures how tightly the levers are **coupled**, not what a policy
  trained for a fixed limit would score. That comparison is the price-only row in
  §7 (2.003M–2.010M, above every pinned arm), which is why §7's gap stays the
  number to quote for the second lever's worth.

---

## 10. Does the oversell cap transfer?

§5c and §7 only ever cap BC→SAC. Joint SAC (0.18) and joint PPO (0.35) also deny
admission and had never been wrapped. Same cap
(`configs/experiment_bc_sac_safe_sl.yaml`), same seeds, no retraining.
Results: `runs/oversell_cap_transfer/`.

| policy | uncapped | capped | given up | peak oversell |
| --- | ---: | ---: | ---: | --- |
| joint BC→SAC raw | 2.094M | **2.081M** | **0.62%** | 0.71 → **0.00** |
| joint SAC `rl_best` | 2.033M | **2.009M** | 1.21% | 0.18 → **0.00** |
| joint PPO `long_007` | 2.011M | **1.962M** | 2.42% | 0.35 → **0.00** |

**Takeaways**

- **The cap generalises.** Zero denied admission on all three, for 0.6–2.4% of
  score, no retraining. The method claim in §5c holds beyond the policy it was
  built for.
- **The cost inverts.** The policy that oversells *most* is *cheapest* to fix.
  BC→SAC is filling so far past capacity that the clipped bookings were already
  paying the double oversell penalty; joint PPO oversells rarely, so the cap takes
  bookings it was being paid for. Oversell volume does not predict the cost of
  safety — whether the marginal booking earns or costs does.
- **It narrows §7.** Among policies that deny admission on no peak night:
  BC→SAC + cap **2.081M**, pace PPO 2.010M, joint SAC + cap 2.009M, price-only PPO
  2.003M, joint PPO + cap 1.962M. Capped joint SAC lands 1,198 behind pace PPO — a
  tie at this seed count — and capped joint PPO falls below both price-only
  policies. So under a safety constraint, "both levers beat one" is a property of
  **the behaviour-cloned policy**, not of joint control in general. The second
  lever is still load-bearing (§9), but having it is not by itself enough.

---

## Experiment takeaways

1. **The earlier prototype fell short on engineering and metrics**, not because “RL cannot do RM.”
2. **Tree + elasticity** demand is a better production analogy than linear-only.
3. **Joint continuous (price, SL)** can work (SAC / BC→SAC) but oversell must be
   managed (safe SL) or you accept risk for legacy score.
4. **Price-only + analytic SL** is a safe PPO path with zero oversell; it clears
   myopic on `score_aware` but trails every SAC-based joint policy on peak harvest.
5. **Soft-day undersell is mostly structural** under current demand — do not use
   overall remain>1500 as the primary KPI; use soft-aware eval.
6. **The second lever is load-bearing, and it is the *movement* that pays** (§9).
   Pinning a joint policy's limit — even to a sensible constant near its own
   average — costs 90k–344k and sends peak denied admission to 0.82.
7. **The cap generalises to every joint policy, but the advantage does not** (§10).
   All three reach zero denied admission for 0.6–2.4%; once constrained that way,
   only BC→SAC still beats the price-only policies.
8. **Recommended packages**
   - Best `score_aware` with zero denied admission: **BC→SAC + safe SL**
   - Highest raw `score_aware`, if 71% peak oversell is acceptable: **BC→SAC**
   - The pure joint policy, and the one to read for what the levers buy: **joint SAC `rl_best`**
   - Simple 1D + zero oversell: **pace PPO** — and at this seed count it ties capped joint SAC
9. Further soft-fill gains need **demand model / data changes**, not more vanilla RL.

---

## Key paths

| Path | Role |
| --- | --- |
| `docs/EXPERIMENT_LOG.md` | This narrative |
| `docs/DESIGN.md` | Architecture |
| `docs/EVALUATING_POLICIES.md` | Soft-aware + checklist |
| `runs/joint_vs_price_only_soft_aware/` | Final table CSV/MD |
| `runs/explain_rl_best/` | Figures + guide behind section 8 |
| `runs/both_goals/` | Safe SL / MPC tables behind section 5c |
| `runs/oracle_ceiling/` | $80 oracle soft-day ceiling |
| `runs/tree_long/` | Section 2 sweep table |
| `runs/bc_sac/` | Section 3 table |
| `runs/price_only_long/` | Section 4 tables |
| `runs/pace_ppo/` | Section 5a table |
| `runs/promo_ppo/` | Section 5b table |
| `runs/ablate_selling_limit/` | Section 9 second-lever ablation |
| `runs/oversell_cap_transfer/` | Section 10 cap transfer across joint policies |
| `runs/soft_aware_eval/` | Section 6 demo of the stratified report |
| `runs/tree_demand_sanity.md` | Early 30k-step sanity check, superseded by section 2 |
| `artifacts/tree_long/best/rl_best.zip` | Joint SAC (pure joint policy) |
| `artifacts/bc_sac/rl_bc_sac_final.zip` | BC→SAC (recommended, under the safe-SL cap) |
| `artifacts/pace_ppo/rl_pace_ppo.zip` | Best price-only PPO |
