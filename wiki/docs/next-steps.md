---
type: decision
title: Session findings and next steps
description: Plain-language record of the 2026-09-22 release-and-audit session, and the queued work with executable instructions for a coding agent.
tags: [decision, roadmap, handover]
timestamp: 2026-09-22
---

# Session findings and next steps

Two audiences. **Part 1** is plain language, for a human deciding what to do
next. **Part 2** is instructions for a coding agent picking up the work, and
assumes no memory of the session.

---

# Part 1 — What happened, in plain language

## Why the session happened

The repo was about to be made public. The ask was a last review: code, content,
claims, the website. It turned into three things — a release cleanup, a
correction of two claims that were wrong, and four new experiments that the
review's own questions provoked.

## The setup in one paragraph

A 10,000-seat venue sells tickets for one night, over 100 days. Every day the
operator picks two things: **the price**, and **the selling limit** — the most
bookings it will hold. The limit goes above 10,000 on purpose, because people
cancel and don't show up; stopping at exactly 10,000 guarantees empty seats, and
an empty seat can never be sold again. Two ways to fail: turn away ticket-holders
at the door, or open to empty seats. The project asks whether a computer should
control **both** dials or just the price.

## What we found

**1. The repo was measuring against a rigged opponent — by accident.**
The simple "myopic" rulebook we compare against has a selling limit, but that
limit **never does anything**. It is set at 12,353, and the rulebook never holds
more than 11,402 bookings. Like a credit card limit of $50,000 when you never
spend more than $3,000 — printed on the card, no effect on your life. So our
"two-dial" benchmark was really a one-dial benchmark, and nobody had noticed.
Now documented. → `docs/EVALUATING_POLICIES.md`

**2. But the learned second dial is very real.**
Tape a trained policy's dial in one position and it loses 5–17% — several times
the 1.2–3.6% margin the whole project argues over. The striking part: the dial's
*average* position is almost exactly the fixed value we compare it against, and
pinning it there *still* costs 6%. **It's a thermostat.** Holding your house at a
fixed 20°C gives the same average as a thermostat that reacts to the weather, and
a completely different experience. The value isn't where the dial sits; it's that
it keeps moving. → `runs/ablate_selling_limit/`

**3. The safety cap works on every policy — and costs the opposite of what you'd guess.**
A "cap" stops a policy overbooking. We'd only ever tested it on one policy. It
works on all three, taking denied admission to zero for 0.6–2.4% of score. But
**the policy that oversells most is the cheapest to fix.** Think of each extra
booking as a bet that someone will cancel. The wild overbooker was losing that bet
71% of the time, so stopping costs it almost nothing. The careful one was winning
its bets, so stopping genuinely hurts. *How often a system misbehaves doesn't
predict what fixing it costs.* → `runs/oversell_cap_transfer/`

**4. The biggest finding: the RL policies don't need a forecast to be right.**
Everything in the repo quietly assumed the operator has a *perfect* demand
forecast — something no real venue gets. We split "what the world does" from
"what the operator believes" and made the forecast wrong. The simple rulebook lost
3.5–5.8% — more than the entire margin the project was arguing over. **The RL
policies lost exactly nothing.** Identical scores, identical prices to the cent,
because they never look at a forecast at all: they only see the booking count and
the calendar. That's the thing an operator actually buys — not "1% better", but
"doesn't need your forecast to be right." → `runs/forecast_misspecification/`

**5. Two of my own claims were wrong, and got corrected.**
I said some constants were frozen because changing them would alter every table.
Measured: they change nothing. And my first version of finding #4 returned "no
effect" for everything — that wasn't a result, it was a broken experiment. Chasing
it found something better: demand multiplies, so an error in *how much* demand
there is cancels out of the price calculation entirely. Only an error in *how
price-sensitive* people are can move a price. That trap is now written down so the
next person doesn't lose an afternoon to it. → `docs/DESIGN.md` § Forecast vs truth

## What this means

The project's headline used to be "two dials beat one, by 1.2–3.6%." That is
still true, but it is now the *least* interesting thing it can say, because a
plausible forecast error is worth more than that entire margin. The stronger claim
is **robustness**: these policies keep working when the forecast doesn't.

## The honest caveats, kept everywhere

- The RL policies are unaffected by a bad forecast **by construction, not by
  merit** — they never had a forecast to lose.
- Only **one** comparison is genuinely safe to make right now (see Task 1).
- Everything rests on 30 simulated nights of synthetic demand.

---

# Part 2 — Instructions for a coding agent

You are picking up a public repo at `github.com/jjd-lab/revenue-manage-rl`. Read
`CLAUDE.md` and `docs/EXPERIMENT_LOG.md` §7–§11 first.

## Ground rules — do not break these

1. **Frozen:** `env.capacity`, the price band, the selling-limit bounds and the
   reward weights in `configs/default.yaml`. Observations are normalised against
   them; changing one invalidates every checkpoint and every table.
2. **`configs/default.yaml` and `src/reservation_pricing/configs/default.yaml`
   must stay byte-identical.** A test enforces it. Edit both.
3. **Never hand-edit a generated file under `runs/`** (`*.csv`, `*.png`,
   `comparison.md`, `soft_aware_*.md`). Rerun the script beside it. `NOTES.md` is
   hand-written.
4. **All evaluation uses held-out seeds 0–29. Seeds 100–129 are for choosing
   hyperparameters and must never be reported as results.**
5. Before every commit: `ruff check . && ruff format --check . && pytest -m "not slow" -q`.
   Run the full `pytest -q` before touching anything in `runs/`.
6. **Do not "fix" `estimate_keep_rate` to stop reading env parameters.** It is a
   known, measured, deliberately-kept leak worth ≤1.11%. See
   `runs/keep_rate_dependence/NOTES.md`.
7. **Do not switch GitHub Pages to a branch source.** Branch-deploy cannot serve
   `/site`; it is deployed by `.github/workflows/pages.yml`.

## Task 1 — Seed protocol (do this first)

**Goal.** Make policy comparisons carry uncertainty, so a reader can tell a real
gap from noise.

**Why.** Right now every table reports a single number. The headline ranking
already flipped once between two different 30-seed draws. And the gap between
capped joint SAC (2,008,716) and pace PPO (2,009,914) is **1,198** on a score of
two million — far inside noise, currently presented as if it were a fact.

**The key measurement, already done — read this before choosing an N.** Pairing
is what matters, not sample size. Every policy is evaluated on the *same* nights,
so night-to-night variation cancels:

| quantity | value |
| --- | ---: |
| revenue std *across nights*, one policy | 231,113 |
| std of the **paired difference**, joint SAC − pace PPO | **13,274** |

A 17× reduction. Resolution of a paired 95% interval: **±4,847 at n=30**, ±2,655
at n=100, ±1,877 at n=200, ±1,187 at n=500. So the joint-SAC-vs-pace-PPO revenue
gap (12,981) is *already* resolvable at 30 nights — but the three joint policies
against each other (paired std 47,000–54,000) need roughly **n=70–100**.

**Do this:**

1. Add `src/reservation_pricing/evaluate/intervals.py` with a paired bootstrap:
   resample the seed list with replacement (10,000 draws, `numpy` default_rng
   seeded at 0), and for each draw **recompute `score_aware` from scratch** for
   both policies on the resampled seeds. Return mean difference and the 2.5/97.5
   percentiles. Do **not** bootstrap the final score — it is a stratified
   aggregate over soft/peak, so it must be recomputed per resample.
2. Add `--interval` to `rprl-eval --soft-aware`, and a `paired_ci_low` /
   `paired_ci_high` column to `evaluate/report.py`'s table, relative to a
   `--baseline-policy`.
3. Extend the headline run to **seeds 0–99** (keep 0–29 as a subset so old numbers
   stay checkable), regenerate `runs/joint_vs_price_only_soft_aware/`, and report
   intervals in `docs/EXPERIMENT_LOG.md` §7.
4. Rewrite §7's prose so any pair whose interval spans zero is called **a tie**,
   explicitly. Expect the three joint policies to become ties with each other.

**Verify.** `pytest -q` green; the bootstrap returns a zero-width interval when
given a policy against itself; §7 contains no ranking claim that its own interval
contradicts.

**Gotcha.** Extending seeds changes the soft/peak split (13/17 at n=30). Report
the new split and do not compare score levels across seed counts.

## Task 2 — Housekeeping (do this after Task 1, not before)

**Goal.** Remove the last two "cannot be reproduced" caveats.

**Why after.** `runs/promo_ppo/` and `runs/price_only_long/` contain rows from
checkpoints nobody kept, and `promo_ppo` still carries a pre-fix `pace_ppo` number
behind a note. Fixing them means retraining and regenerating — and if Task 1
changes the seed count, the regeneration happens twice. **Retrain once, after the
protocol is settled.**

**Do this:**

1. `rprl-train -c configs/experiment_price_only_promo_ppo.yaml` (~4 min) and
   `-c configs/experiment_price_only_sac_long.yaml` (~32 min).
2. Copy the finals to the curated paths in README § Model checkpoints.
3. Rerun `runs/promo_ppo/final_eval.py` and `runs/price_only_long/final_eval.py`.
4. Update both `NOTES.md` from the new CSVs and **delete the stale-number
   caveats** — including the promo pre-fix note.

**Gotcha.** SB3 is not bit-reproducible, so the numbers will move slightly. The
prose around them moves too: `runs/promo_ppo/NOTES.md` claims "promo regresses vs
pace by −12.2k" — recompute it, don't copy it.

## Task 3 — Experiments, ranked

**3a. Cap frontier (cheap, eval-only, highest value).**
`mix_alpha` is sampled at three points {0, 0.25, 0.4}. Sweep `mix_alpha` ×
`overbook_factor` and plot the score-vs-denied-admission frontier. That curve —
"each unit of overbooking risk buys this much score" — is the deliverable an
operator actually wants, and it drops straight into figure F2 on the site. Select
on seeds 100–129, report on the reported set.

**3b. Misspecification under training (medium, needs retraining).**
`runs/forecast_misspecification/` shows RL is indifferent to a bad forecast *by
construction*. The open question: would a forecast-fed policy **trained under**
misspecification beat both? Train a price-only PPO with `demand.forecast` enabled
so it learns against a wrong model, then evaluate against the true world. This is
the natural sequel and nobody has run it.

**3c. Multiple training seeds (medium, needs retraining).**
Every policy is one training run at seed 42, so "SAC beats PPO" could be a
lottery. Retrain 3–5 seeds per policy, report mean ± spread. Pairs naturally with
Task 1.

**3d. Demand-pressure sweep (larger, needs retraining).**
We learned myopic's limit never binds *at this demand level*, and binds at $80.
The two-lever advantage may scale with capacity pressure. Scale base demand
0.8×–1.5×, retrain at each level, compare joint vs price-only. Converts a
single-point result into a curve. **Retraining is required** — evaluating today's
policies at a different demand level only measures out-of-distribution behaviour.

## What not to bother with

- Rewriting `estimate_keep_rate` (measured at ≤1.11%; see ground rule 6).
- Tuning the myopic baseline's `1.05`/`0.85` (its limit never binds; and they are
  the behaviour-cloning target, so changing them breaks `rprl-bc-sac`
  reproduction of the shipped checkpoint).
- Degrading demand *level* to test pricing robustness — it provably cannot move a
  price. Perturb `elasticity`. See `docs/DESIGN.md` § Forecast vs truth.
