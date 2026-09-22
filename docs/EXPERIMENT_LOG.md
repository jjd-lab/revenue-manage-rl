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
| price-only SAC @150k | ~802k | **0.00** | 0.433 |
| BC→SAC | ~871k | 0.40 | 0.433 |

**Takeaways**

- Price-only PPO sits just above `rl_best` with **zero oversell**. The SAC figure is a 2026-09-22 retrain of a checkpoint that had not been kept; it is below `rl_best`.
- Neither beats BC→SAC on legacy score.
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

Worse than pace on score (~821k vs ~832k); undersell tied at 0.433. Regenerated 2026-09-22 from a new training; the loss to pace stands.

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

Raw BC→SAC denies admission on 12 of the 17 peak nights. The safe-SL cap takes
that rate to zero. The point estimates give back 0.6% of score for the cap; the
interval on that difference covers zero, so the score cost is a **tie**. The cap
is the recommended package because it sits in that top tie and denies admission
on no peak night.

**Paired intervals, seeds 0–29.** Source of truth:
`runs/joint_vs_price_only_soft_aware/paired_intervals.csv`
(`python runs/joint_vs_price_only_soft_aware/intervals.py` — it reads the saved
episodes and does not rewrite this table). 10,000 resamples of the seed list;
`score_aware` is recomputed on each resample. An interval that covers zero is a
**tie**. Do not rank through it.

- **BC→SAC raw and BC→SAC + cap are a tie with each other, and both beat every
  other row.** Capped minus raw is −13.0k (interval −39.0k to +14.7k). Capped
  minus pace PPO is +71.5k (+31.2k to +105.8k): the 3.6% point gap, and the
  interval sits above zero. Raw minus pace is +84.5k (+39.7k to +124.0k). Both
  also clear `rl_best`, joint PPO, price-only PPO, and myopic.
- **`rl_best`, joint PPO, pace PPO, price-only PPO, and myopic are mutually
  tied.** `rl_best` minus pace is +23.4k (−0.6k to +47.5k): the 1.2% point gap
  is a tie. Joint PPO minus pace is +1.0k (−72.9k to +70.7k). Every other pair
  inside this group covers zero, including both price-only rows against myopic.
- **Fixed $80 is below every other row.** Those intervals sit entirely on one
  side of zero.

The joint-versus-price-only comparison these thirty nights support, on the
published checkpoints, is the BC→SAC lead over pace. "The three SAC policies
sit above every price-only policy" does not.

> **Seed sensitivity (2026-09-22).** This table was first published on a
> different thirty seeds (0–4 plus 1000–1024, an 18 soft / 12 peak split) that no
> other run used. On that draw joint SAC `rl_best` led (2.057M) ahead of raw
> BC→SAC (2.044M), the capped BC→SAC (2.040M) and pace PPO (1.991M). Moving to
> the common seeds 0–29 reordered the joint policies. The intervals above are
> the uncertainty on this draw: only the two BC→SAC rows are separated from the
> price-only policies. A longer seed list would change the soft/peak split, so
> it waits on a tie that is still worth separating. `rl_best` versus pace is the
> closest.

**Training seeds (2026-09-22).** Those intervals are one training of each
policy, both at seed 42. Retraining pace PPO and BC→SAC at seeds 43, 44, and 46
(`runs/training_seeds/`; seed 42 left as the shipped zips; same nights 0–29)
shows the capped lead does **not** repeat. Seeds 42, 43, and 44 sit above their
matched pace run; seed 46 is a tie (+8.0k, −7.9k to +25.2k) and sits below the
published pace checkpoint (−50.3k, −78.8k to −24.1k). Capped `score_aware`
across the four seeds runs from 1.960M to 2.103M, a spread of 144k, wider than
the night-level interval on the published pair (+31.2k to +105.8k). The
published lead is training-seed sensitive. The same wrapper also fails to take
every retrain to zero peak denied admission (0.24 and 0.47 on seeds 43 and 44).
The table above, and `paired_intervals.csv` beside it, stay the seed-42 record.

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
python runs/joint_vs_price_only_soft_aware/run_headline.py
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

Section 7's point estimates put the joint policies above the price-only ones.
The intervals keep that lead only for BC→SAC; `rl_best` and joint PPO tie the
price-only rows. Either way the gap admits two readings — the joint policies
use the limit well, or their *price* policy is better and the limit is along
for the ride. This re-scores each joint policy with its price untouched and
its limit replaced by a constant.

**Protocol:** same 30 held-out seeds, same `score_aware`.
Results: `runs/ablate_selling_limit/` (`python runs/ablate_selling_limit/run_ablation.py`).

| policy | learned limit | pinned open 15,000 | pinned flat 12,353 |
| --- | ---: | ---: | ---: |
| joint SAC `rl_best` | **2.033M** | 1.689M (−344k) | 1.914M (−120k) |
| joint BC→SAC raw | **2.094M** | 1.855M (−239k) | 1.943M (−152k) |
| joint PPO `long_007` | **2.011M** | 1.903M (−108k) | 1.921M (−90k) |

**Takeaways**

- **The gap is not a price-policy artifact.** Removing the limit costs 90k–344k.
  Those pin costs are the measurement; §7's intervals are a separate question
  and are not what this table is for.
- **A sensible constant does not recover it.** `rl_best`'s limit *averages* 12,578,
  within 2% of the flat 12,353 — and pinning it there still costs 120k. The value
  is not where the limit sits, it is **when it moves** (§8's figure 05, priced).
- **The limit is what holds denied admission down.** Peak oversell goes 0.18 → 0.82
  for `rl_best` the moment the limit stops moving.
- **Caveat:** pinned arms are off-distribution — each policy priced for the limit it
  learned. This measures how tightly the levers are **coupled**, not what a policy
  trained for a fixed limit would score. That comparison is the price-only row in
  §7 (2.003M–2.010M, above every pinned arm on the point estimates). The number
  to quote for the published checkpoints' lead over a purpose-trained one-lever
  policy is §7's interval for BC→SAC against pace, not the pin costs and not the
  1.2% point gap. That interval is one training seed; a retrain can tie pace (§7,
  training seeds).

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
- **It narrows §7.** Among policies that deny admission on no peak night, the
  point estimates are: BC→SAC + cap **2.081M**, pace PPO 2.010M, joint SAC + cap
  2.009M, price-only PPO 2.003M, joint PPO + cap 1.962M. The saved headline
  episodes cover the first, the second, and the fourth, and §7's interval says
  capped BC→SAC beats both price-only rows. Capped `rl_best` and capped joint PPO
  were not saved night by night, so they have no interval: the point estimates
  put capped `rl_best` 1,198 behind pace and capped joint PPO below both
  price-only policies. The saved neighbour, uncapped `rl_best` versus pace, is a
  tie (§7). Under a safety constraint, "both levers beat one" is a property of
  **the behaviour-cloned policy**, which is the comparison the interval supports.
  The second lever is still load-bearing (§9), but having it is not by itself enough.

---

## 11. Privileged knowledge: what the system is allowed to know

Two audits of information the code has and a real operator would not.

**11a. The cancellation model.** `estimate_keep_rate` reads the env's own
`cancel_lambda`, `cancel_rho_*` and `noshow_*` and replays its Weibull, so the
analytic limit and the oversell cap run on a perfectly specified cancellation
model. Priced in `runs/keep_rate_dependence/` by swapping in fixed guesses:
**0.07%** on the cap, **≤1.11%** on the limits, peak oversell `0.0000` in every
cell — and a round 0.85 guess *beats* the exact model twice out of three. Knowing
the true cancellation physics gives you the keep rate; the score-optimal limit
depends on the reward structure, which is a different problem. **Disclosed, not
fixed.**

**11b. The demand forecast.** Everything else in this log assumes decision code
consults the model that generates the world. `demand.forecast` splits them
(`docs/DESIGN.md` § Forecast vs truth) and `runs/forecast_misspecification/`
re-scores under a wrong one, without retraining:

| policy | consults forecast for | elasticity −0.9 | elasticity −1.5 | level stale |
| --- | --- | ---: | ---: | ---: |
| myopic | price | **−3.47%** | **−5.82%** | 0.00% |
| myopic @ `optimize_1d` | price + limit | −2.83% | −4.69% | **−1.06%** |
| pace PPO + MPC | MPC trigger + lookahead | +0.69% | 0.00% | **−0.99%** |
| pace PPO (analytic limit) | nothing | 0.00% | 0.00% | 0.00% |
| joint SAC `rl_best` | nothing | 0.00% | 0.00% | 0.00% |
| joint BC→SAC raw | nothing | 0.00% | 0.00% | 0.00% |

**Takeaways**

- **A plausible forecast error is as large as the lead this log can support.**
  Under a perfect forecast the resolved joint-vs-price-only gap is capped BC→SAC
  over pace, 3.6%, with the interval above zero (§7). A wrong elasticity costs
  myopic 3.5–5.8% on its own. The 1.2% point gap (`rl_best` over pace) is a tie.
- **The joint policies do not move at all** — identical scores and mean prices to
  the cent. Their observation is booking state plus calendar one-hots (§8), so
  they consult no model. That is the operator-facing property: not a one-percent
  lead (the 1.2% point gap is a tie) but "does not need your forecast to be right."
  Indifference by construction, not learned robustness — see the caveats in that
  directory.
- **A demand-*level* error moves no pricing decision, ever.** Demand is
  `base * (1 + e(p−ref)/ref)`, so base cancels out of the argmax and myopic's price
  is `ref(1−e)/(−2e) = $91.67` however wrong the level is. That is also why §8's
  myopic line is "a flat $92" — a closed form, not a quirk. Level errors reach only
  the MPC and `optimize_1d`. **To misspecify pricing, perturb elasticity.**

**Also corrected here:** `mix_alpha=0.25` was originally chosen by reading
held-out seeds 0–29 — the same thirty §7 reports on. Re-run on a disjoint block
(seeds 100–129, `runs/both_goals/validate_mix_alpha.py`) the identical rule picks
**0.25** again, for the same reason both times: `mix_alpha=0.4` scores highest but
denies admission on 41–45% of peak nights. The shipped value is not an artefact of
the test set.

---

## 12. What a non-decreasing price costs, and whether a policy can learn it

F4's weekend path marks down late: about $112.85 on day 16 to $89.08 on day 1.
`control.price_monotone` with `direction: up` forbids that. `mode: clamp`
raises an offending action to the floor; `mode: ratchet` reads the action as a
move from the floor, so no two actions charge the same price; `mode: penalty`
constrains nothing and subtracts `penalty * (violation_dollars / price_span)`
from the step reward. The first step of each episode is exempt, because
`reset()`'s price is a placeholder. Promo and MPC are rejected alongside it:
both rewrite price inside `PriceOnlyWrapper` after the outer clamp. Results:
`runs/price_monotone_up/`.

Same reference throughout: frozen joint SAC `rl_best`, seeds 0–29,
`score_aware`, paired against the unconstrained arm. Penalty weights are chosen
on seeds 100–129 (`penalty_selection.csv`, `penalty_hw_selection.csv`); those
scores are not results.

| arm | score_aware | paired vs unconstrained | 95% interval | charged decreases | clamped steps |
| --- | ---: | ---: | --- | ---: | ---: |
| unconstrained `rl_best` | 2,033,264 | 0 | — | 1,672 | 0 |
| **clamp, frozen** | **2,077,976** | **+44,712** | [20,945, 66,164] | **0** | 2,610 |
| penalty_hw 1, retrained | 2,048,563 | +15,299 | [−39,344, 74,833] | 1,377 | 0 |
| **bc_clone** | **2,038,966** | **+5,702** | [−42,250, 51,133] | **0** | 1,812 |
| bc_clone_ratchet | 2,024,192 | −9,072 | [−63,648, 38,982] | **0** | 0 |
| peak_only | 1,964,902 | −68,362 | [−124,006, −12,670] | 528 | 0 |
| penalty 10, retrained | 1,951,555 | −81,709 | [−134,174, −32,539] | 1,323 | 0 |
| ratchet_pace | 1,935,085 | −98,179 | [−162,419, −31,938] | **0** | 0 |
| bc_ratchet | 1,935,036 | −98,228 | [−230,925, 16,571] | **0** | 0 |
| clamp, retrained | 1,923,106 | −110,158 | [−146,418, −73,687] | **0** | 0 |
| ratchet | 1,921,207 | −112,057 | [−144,200, −80,287] | **0** | 0 |
| bc_clamp | 1,911,249 | −122,015 | [−147,486, −94,954] | **0** | 1,851 |

**Takeaways**

- **The markdown is real, and it was losing money.** The unconstrained policy
  decreases price on every held-out night. Clamping it with no retrain overrides
  **2,610 of its 3,000 decisions**, removes every decrease, and *raises* score
  by about 45k. The guarantee is free; the late plunge was the mistake.
- **Training under the clamp costs about 110k, and removing the aliasing did not
  help.** Under `clamp` every action below the floor charges the floor, so the
  gradient over that whole region is exactly zero, and the dead region grows as
  the floor rises. `ratchet` removes that by construction and scored 1,921,207 —
  statistically identical to the clamp retrain. The aliasing was real and was
  not the binding constraint.
- **The retrains pin low rather than rise.** The clamp retrain holds $116.31 on
  weekend nights from day 100 to day 1 and sits on the $80 floor on weekdays;
  the ratchet opens at $80 and peaks at $82. Under an irreversible ratchet,
  opening at `min_price` keeps the most options, and on soft nights — where the
  revenue-maximizing path genuinely falls — that is close to right. Pace shaping
  restores the shape (open $80.9, climb to $94.5) for about 14k of score.
  `peak_only`, which frees weekday off-peak nights, recovers 42k.
- **Cloning works; fine-tuning is what destroys it.** Behaviour cloning of the
  clamped policy scores 2,038,966 with zero markdowns — a tie with the
  unconstrained reference and 116k above the clamp retrain. Fine-tuning then
  undoes it, at every setting tried: 1,966,735 at 200k steps and lr 3e-4,
  1,937,339 at lr 3e-5, 1,808,182 at 20k steps. SAC walks away from a policy
  worth 2.04M back into the same basin. It is not an optimization failure.
  The training reward rises from 54.6 to 96.2 over the same fine-tune. The
  reward charges about $730 per unsold seat and forfeits the whole utilization
  bonus on any oversold night, while `score_aware` charges nothing for unsold
  soft seats and $400 per oversold seat. The fine-tune fills soft nights at the
  $80 floor and stops overselling peak nights by selling less. That rules out
  a KL-tethered fine-tune, whose best case is the clone. The recipe is **clone
  and stop**.
- **A penalty does not buy a guarantee at any weight.** Charged against
  yesterday's price, a slow staircase is a series of small fines; weight 10
  marks down on all 30 nights. Charged against the episode's high-water mark and
  selected properly, weight 1 is the best-scoring retrained arm (2,048,563) and
  **still marks down 1,377 times**. If the promise is "later buyers never pay
  less", it has to be a clamp.

The operator-facing answer: train unconstrained and deploy behind the clamp.

---

## 13. What the training objective does to the policy

§12's diagnostic showed SAC optimizing its reward faithfully while the score
fell. This asks how much of a policy's behaviour comes from its objective. The
new reward is one rule for every night, one a venue could state without a
soft/peak label: revenue − $200 per unsold seat − $400 per denied admission, no
utilization bonus (`configs/experiment_objective_cu200_sac.yaml`, overriding
`env:` weights only). Joint SAC, seed 7, 200k steps, against `rl_best` with the
same hyperparameters. Results: `runs/objective/`.

- **+54,486 on `score_aware`** (2,087,750; interval [26,377, 80,566]). Behind
  the cap, +86,851 (2,095,566).
- **All of it is peak nights, and it comes from overbooking.** Denied admission
  on 11 of 17 peak nights against 3. Peak price $100.61 against $97.11, and 258
  fewer empty seats. Soft nights are unchanged ($83.22 against $83.73).
- **The cap no longer reaches zero.** 2 of 17 peak nights still deny admission
  behind it, where every policy in §10 reached zero.
- Each policy wins on its own objective. One training seed, and `rl_best` is
  from 2026-09-17 code.

## Experiment takeaways

1. The earlier prototype fell short on the engineering and on the metrics.
2. Tree base demand plus linear elasticity is the demand model the published tables use. `linear_legacy` is the older alternative.
3. Joint BC to SAC sets the price and the selling limit and has denied admission on 0.71 of peak nights. The cap, applied after training, takes that share to zero on the published checkpoint. The paired interval on the score cost covers zero, so the capped and uncapped scores are a tie (§7).
4. Pace PPO sets only the price, and denied admission on these thirty nights is zero. On `score_aware` it ties myopic, Joint SAC, and Joint PPO. It trails both Joint BC to SAC rows (§7).
5. On the thirteen soft nights, the best price with the selling limit wide open still leaves more than 1,500 seats empty. Do not rank policies by overall `remain > 1500`.
6. Pinning a joint policy's selling limit and leaving the price alone costs $90,000 to $344,000 and sends the share of peak nights with denied admission to 0.82 (§9).
7. The cap takes all three joint policies to zero denied admission on the published checkpoints, for 0.6 to 2.4 percent of score. After that, only Joint BC to SAC still beats the price-only policies (§10). The 3.6 percent lead over pace PPO does not repeat on every training seed. Seeds 43 and 44 beat their matched pace run. Seed 46 ties, and it scores below the published pace checkpoint. On seeds 43 and 44 the cap leaves denied admission on 0.24 and 0.47 of peak nights (§7).
8. Which policy to use, matching the public page:
   - Best score with denied admission at zero: Joint BC to SAC with the cap.
   - If denied admission on 12 of 17 peak nights is acceptable: raw Joint BC to SAC. It ties the capped policy on score.
   - If you want the weekend and weekday price paths: Joint SAC.
   - If the policy should set only the price: pace PPO.
   - If the demand forecast might be wrong: a joint policy. Those policies see bookings and the calendar. They do not read a demand model (§11). A wrong elasticity costs myopic 3.5 to 5.8 percent. It costs the joint policies nothing. A wrong cancellation model is worth at most 1.11 percent.
9. Further fill on the soft nights needs a different demand model. It does not come from another training run of these policies.
10. Clamping the published Joint SAC so its price never falls raises the score by about $45,000 and removes every markdown, with no retraining. Retraining under that clamp costs about $110,000. A penalty in the reward does not stop the markdowns (§12).
11. The training objective sets how much a joint policy overbooks. Charging a flat $400 per denied admission and removing the cliff on the fill bonus raises Joint SAC's score by about $54,000. All of the gain is peak nights, with denied admission on 11 of 17 of them, and the cap no longer takes that to zero (§13).

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
| `runs/keep_rate_dependence/` | Section 11a: what the true cancellation model is worth |
| `runs/forecast_misspecification/` | Section 11b: policies under a wrong demand forecast |
| `runs/soft_aware_report_demo/` | Section 6 demo of the stratified report |
| `runs/objective/` | Section 13: the training objective and overbooking |
| `runs/training_seeds/` | Whether the §7 BC→SAC lead repeats across training seeds |
| `runs/price_monotone_up/` | Section 12: cost of a non-decreasing price |
| `artifacts/tree_long/best/rl_best.zip` | Joint SAC (pure joint policy) |
| `artifacts/bc_sac/rl_bc_sac_final.zip` | BC→SAC (recommended, under the safe-SL cap) |
| `artifacts/pace_ppo/rl_pace_ppo.zip` | Best price-only PPO |
