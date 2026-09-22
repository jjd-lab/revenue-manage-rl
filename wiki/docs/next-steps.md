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

The project's headline used to be "two dials beat one, by 1.2–3.6%." Paired
intervals on the same thirty nights keep the 3.6% (capped BC→SAC over pace) and
call the 1.2% a tie. A plausible forecast error is as large as the lead that
remains. The stronger claim is **robustness**: these policies keep working when
the forecast doesn't.

## The honest caveats, kept everywhere

- The RL policies are unaffected by a bad forecast **by construction, not by
  merit** — they never had a forecast to lose.
- Rank only a pair whose interval sits off zero (§7). The rest of the headline
  table is ties.
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

## Task 1 — Seed protocol (intervals done; more seeds only if a tie is worth it)

**Status (2026-09-22).** Paired intervals on seeds 0–29 are in
`runs/joint_vs_price_only_soft_aware/paired_intervals.csv`. §7 of the experiment
log names the ties. Do not extend to seeds 0–99 unless a pair you still want to
separate has an interval that covers zero. The closest is `rl_best` versus pace
PPO. Extending the seed list changes the soft/peak split, so score levels would
stop matching the rest of `runs/`. Housekeeping and the cap frontier stay behind
this, and they are not unblocked by the intervals.

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
against each other (paired std 47,000–54,000) need roughly **n=70–100**. That
prediction was about revenue. The `score_aware` intervals are the ones that
count, and they did not make the three joint policies mutual ties. Use the CSV.

**Done, on seeds 0–29.** `evaluate/intervals.py` resamples the seed list
(10,000 draws, `default_rng` seeded at 0) and recomputes `score_aware` per
resample. `rprl-eval --soft-aware --interval --baseline-policy` writes
`paired_ci_low` / `paired_ci_high` on the soft-aware table. The legacy `score`
table in `evaluate/report.py` was left alone: §7's claim is `score_aware`. The
prose treats a covering-zero interval as a tie. The three joint policies did
**not** all become ties with each other: raw and capped BC→SAC are separated
from `rl_best` and joint PPO. What became ties is the middle of the table —
`rl_best`, joint PPO, pace PPO, price-only PPO, and myopic.

**Not done — only if you still want to separate a tie:**

1. Extend the headline run to **seeds 0–99** (keep 0–29 as a subset so old numbers
   stay checkable) and regenerate `runs/joint_vs_price_only_soft_aware/`. Do this
   only for a pair whose interval covers zero and that you still want to rank.
   `rl_best` versus pace is the closest. Otherwise leave the seed count alone.

**Verify.** `pytest -q` green; the bootstrap returns a zero-width interval when
given a policy against itself; §7 contains no ranking claim that its own interval
contradicts.

**Gotcha.** Extending seeds changes the soft/peak split (13/17 at n=30). Report
the new split and do not compare score levels across seed counts.

## Task 2 — Housekeeping (done 2026-09-22)

**Status.** Promo PPO and price-only SAC were retrained at seed 42, copied to the curated paths, and both tables were regenerated. The "not kept" caveats are gone. The shipped price-only PPO zip was not retrained. Promo still loses to pace (−10.8k). The SAC retrain scores 802k, below `rl_best`, and still does not beat BC→SAC.

**Goal.** Remove the last two "cannot be reproduced" caveats.

**Why after.** `runs/promo_ppo/` and `runs/price_only_long/` contain rows from
checkpoints nobody kept, and `promo_ppo` still carries a pre-fix `pace_ppo` number
behind a note. Fixing them means retraining and regenerating — and if the seed
count later changes, the regeneration happens twice. The interval protocol is
settled at seeds 0–29 unless someone chooses to separate a tie. **Retrain once,
after that choice, or now if the seed count is staying at 30.**

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

**3c. Multiple training seeds — the headline pair is done (2026-09-22).**
The narrowed question, BC→SAC against pace only, is answered in
`runs/training_seeds/NOTES.md`. The lead does not repeat: one of the four
training seeds ties its matched pace run, so the published §7 lead is
training-seed sensitive. Seed 42 was not retrained. Evaluation nights stayed
0–29. A sweep of every other policy is not required to read that result.
The cap frontier (3a) stays queued. Housekeeping (Task 2) is done.

**3d. Demand-pressure sweep (larger, needs retraining).**
We learned myopic's limit never binds *at this demand level*, and binds at $80.
The two-lever advantage may scale with capacity pressure. Scale base demand
0.8×–1.5×, retrain at each level, compare joint vs price-only. Converts a
single-point result into a curve. **Retraining is required** — evaluating today's
policies at a different demand level only measures out-of-distribution behaviour.

## Task 4 — Monotone price constraint (built 2026-09-22)

**Status (2026-09-22).** Built, then pushed on. Twelve arms in
`runs/price_monotone_up/NOTES.md`. Clamping the frozen `rl_best` overrides
2,610 of its 3,000 decisions, removes every markdown and *raises*
`score_aware` about 45k — the guarantee is free as a filter. Nothing trained
under the constraint matched it. `mode: ratchet`, added to remove the clamp's
action aliasing, changed nothing (1,921,207 vs the clamp retrain's 1,923,106):
the retrains pin at `min_price`, which is the maximum-optionality move under an
irreversible ratchet. Cloning the clamped policy scores 2,038,966 with zero
markdowns, a tie with the unconstrained reference; fine-tuning destroys it at
every setting tried. **Clone and stop.** No penalty weight buys the guarantee,
including the corrected high-water reference.

**Still open.** `max_step` for `mode: ratchet` was fixed at $1.0 and never
selected — the plan called for a sweep over `{0.5, 1.0, 2.0}` on seeds 100–129,
and the observed drift (a neutral action climbs $0.5/day) suggests smaller
values are worth trying. Whether a fine-tune that is *constrained to stay near
the clone* (a KL or trust-region term, which SB3's SAC does not have) preserves
the clone's score is untested, and it is the obvious next question given that
every unconstrained fine-tune destroyed it.

**Goal.** Add a config-driven constraint that forbids price from *decreasing*
as the event date approaches — `direction: up`, so later buyers never pay less
than earlier ones — then measure what the guarantee costs per night.

**Why.** F4's weekend mean price path (`scripts/build_site_figures.py`,
`runs/explain_rl_best/rollouts.csv`, policy `rl_best`, `weekend == 1`) opens
at $108.50 on day 100, is $104.32 on day 80, crests at $117.30 on day 37
(day 42 is $116.74), and ends at $89.08 on day 1. Day 21 is still $114.67.
The markdown a venue cannot run is the late plunge, from about $112.85 on
day 16 down to $89.08 on day 1: a later buyer paying less than an earlier one.
All 30 episodes in that file contain at least one decrease, so an unconstrained
arm will show a violation count above zero. **Expected outcome, stated up
front:** forbidding the plunge removes the policy's clearance mechanism, so
the constrained retrain will very likely score *below* the unconstrained one.
The number is the deliverable, not a win.

**Where it plugs in.** `make_env` in
`src/reservation_pricing/envs/factory.py` is the single choke point for
training, evaluation and tuning, so one wrapper there covers every entry point.
After the first real step, the last executed price is observation index 2 in
`ReservationEnv._raw_obs` (`src/reservation_pricing/envs/reservation.py`), so a
constraint defined against yesterday's price keeps the MDP Markov **without
touching the frozen observation layout or normalization bounds** — no
checkpoint and no table in `runs/` is invalidated. At `reset()` that slot is
the $80 placeholder; step 1 says how to exempt it.
`src/reservation_pricing/envs/oversell_guard.py` is the structural template for
the wrapper (unscale the action, let a controller clamp it, re-scale, forward,
annotate `info`); `src/reservation_pricing/controls/oversell_cap.py` is the
template for the controller (`get_*` factory returning `None` when disabled,
`last_*` diagnostic attributes, `**_ignored` in the constructor).

**Do this:**

1. Add `src/reservation_pricing/controls/price_monotone.py` —
   `MonotonePriceControl` with `enabled`, `direction` (`up`|`down`), `mode`
   (`project`|`penalty`), `tolerance`, `max_step`, `penalty`,
   `apply_when_days_prior_le`; method `apply(env, price) -> float` mirroring
   `OversellCap.project`'s shape (reset diagnostics → early-return untouched
   when disabled → compute the bound → one-sided projection →
   `last_clamped` with a `1e-6` tolerance → clip). Needs a local
   `_clip_price(env, price)` — `_clip_sl` in `selling_limit.py` is SL-only.
   Factory `get_price_monotone(cfg)` returning `None` when absent/disabled.
   Knob defaults, and the experiment leaves them there:
   `tolerance: 0` (a smaller move is neither a violation nor a projection;
   the `1e-6` flag is only float-noise, same as `OversellCap`),
   `max_step: null` (no per-day cap; a set value also limits the change to
   `last ± max_step`), `apply_when_days_prior_le: null` (every decision day).
   If the window is set, judge it on the decision day `days_prior - 1` — the
   day `PriceOnlyWrapper` already shows every price controller
   (`docs/DESIGN.md` § Control layer). `OversellGuardEnv` reads `days_prior`
   with no shift; copying that makes the window a day late.
   Track a controller-local `last_executed_price`, `None` until a real step
   has executed, and clear it in the wrapper's `reset()`.
   `ReservationEnv.reset()` sets `self.price = self.min_price` ($80) as a
   *placeholder*, not a decision. The band is $80–$120, so for `direction: up`
   that placeholder is already the floor and would not pin the opener; for
   `direction: down` it ceilings the opener at $80. Either way, leave the
   first step exempt. Write `last_executed_price` from `info["price"]` *after*
   `env.step` (the price actually charged), and clear it on every `reset()`
   so episode N's closing price cannot floor episode N+1.
   `OversellGuardEnv.reset` clears no controller state; do not copy that.
   Penalty mode lives in the wrapper, not the env — it subtracts
   `penalty * (violation_dollars / price_span)` from the returned reward and
   does not clamp, so no new weight enters the frozen `env.reward` block.
   `price_span` is `max_price - min_price` (40). A step's shaped revenue is
   `accepted * price * revenue_scale` (`revenue_scale` is `0.0001`), about 1
   on a typical day, which is why the penalty sweep below is `{1, 10, 100}`.
2. Add `src/reservation_pricing/envs/price_guard.py` —
   `MonotonePriceEnv(gym.Wrapper)`, outermost, reading `action[0]` so it
   handles both the 2D joint action space and the 1D price-only space. Reuse
   the `_unscale`/`_scale` idiom and the `__getattr__` forwarding from
   `oversell_guard.py` (Gymnasium 1.x wrappers do not forward attributes).
   `reset()` clears `last_executed_price` before returning. Write
   `info["price_monotone"]`, `["price_monotone_floor"]`,
   `["price_monotone_clamped"]`, `["price_monotone_violation"]`,
   `["policy_price"]`.
3. **`validate_config` must reject `early_promo`, its `promo` alias, and `mpc`
   when any of them is enabled together with `price_monotone.enabled`. This
   is not optional.** `get_early_promo` treats `control.promo` as the same
   controller as `control.early_promo`. Both promo and MPC override price
   *inside* `PriceOnlyWrapper` (`src/reservation_pricing/envs/price_only.py`,
   the RL price → early_promo → mpc → SL-controller chain) — i.e. *after* any
   outer projection — so without the rejection the guarantee is **silently
   violated** in exactly those configs (`experiment_price_only_promo_ppo.yaml`,
   `experiment_pace_mpc.yaml`). Check the `enabled` flags, not mere presence
   of the block: the default config carries both blocks with `enabled: false`.
4. Wire it in, each step a silent failure if skipped:
   - Add `price_monotone` to `_CONTROL_OVERRIDE_KEYS` in `envs/factory.py`.
     A key left out of that tuple is not popped: it stays in `overrides`, is
     copied onto the env dict, then dropped by the `_ENV_KEYS` filter, and
     never merged into `control`. No error.
   - Call `get_price_monotone(control)` and wrap only when it returns a
     controller, after both the price-only branch and the oversell-guard
     branch, so the wrapper is outermost on either action space. When the
     factory returns `None`, do not wrap at all — that is what keeps the
     default config a true no-op, same as the oversell cap.
   - Add `price_monotone` to the hardcoded bare-block exclusion tuple in
     `controls/oversell_cap.py` (not derived from that constant). That branch
     keeps every other key, and `OversellCap` swallows unknown kwargs via
     `**_ignored`. `make_env` takes the nested `safe_sl` branch today, so this
     is defensive; skip it and a bare block absorbs the monotone config.
     `get_price_mpc` / `get_early_promo` are a different path: a monotone
     block handed to them returns `None` or raises on `mode`. Leave them.
   - Add an inert `control.price_monotone` block (`enabled: false`,
     `direction: up`, `mode: clamp`, `tolerance: 0`, `max_step: null`,
     `penalty: 10`, `apply_when_days_prior_le: null`) to
     `configs/default.yaml` **and** `src/reservation_pricing/configs/default.yaml`
     — identically; a test enforces they stay byte-identical.
   - Add `KNOWN_MONOTONE_DIRECTIONS`/`KNOWN_MONOTONE_MODES` to
     `config/load.py` and extend the schema assertion in `tests/test_configs.py`
     so `price_monotone.enabled` is false on the default config. The existing
     assertion is a superset, so a missing block would otherwise still pass.
   - Export from `controls/__init__.py` and `envs/__init__.py`.
5. Add `configs/experiment_monotone_up_sac.yaml` and
   `experiment_monotone_up_penalty_sac.yaml`, matching
   `configs/tree_long_sac.yaml`'s hyperparameters (200k steps, seed 7), with
   `mode: clamp` and `mode: penalty` respectively.
6. Add `tests/test_price_monotone.py`: a unit test on a `SimpleNamespace` fake
   env (per `tests/test_controls_extra.py`) plus an integration test that loads
   a shipped config and asserts on `info`.
7. Write `runs/price_monotone_up/run_monotone.py`, modelled on
   `runs/oversell_cap_transfer/run_cap_transfer.py` (arms are env factories
   built from two configs; the soft oracle is collected once on the plain
   factory and shared, which is what makes the comparison paired). Four arms
   on held-out seeds 0–29 with `score_aware`:
   - Arm 0 — frozen `artifacts/tree_long/best/rl_best.zip`, unconstrained. Reference.
   - Arm 1 — the same frozen checkpoint under `mode: clamp`, no retraining.
     Free, and it prices the constraint against a policy that was never told
     about it.
   - Arm 2 — SAC retrained under `mode: clamp` (~30–40 min).
   - Arm 3 — SAC retrained under `mode: penalty`. The weight is a sweep, not
     one training. Train `penalty` in `{1, 10, 100}` (one 200k-step SAC each,
     seed 7, ~30–40 min each): a $4 markdown then costs 0.1, 1, or 10 against
     a step reward of about 1. Rank the three on seeds 100–129. Evaluate and
     report only the winner on seeds 0–29. Seeds 100–129 never appear as a
     result.
   `evaluate_policy_soft_aware` returns `(summary, rows)`. The cap-transfer
   script discards `rows` (`sa, _`). This script keeps them. Build
   `{seed: EpisodeMetrics}` with `episode_from_mapping` and pass those maps
   to `paired_difference` (`src/reservation_pricing/evaluate/intervals.py`).
   Do not call `write_saved_intervals` — that needs `episode_metrics.csv`,
   `soft_aware_summary.json`, and `soft_aware_table.csv`, which this script
   does not produce. Episode metrics store `mean_price` only, so count
   decreases on successive `info["price"]` values during the rollout. Skip
   `reset()`'s placeholder $80. Arm 0's decrease count is > 0; every
   `mode: clamp` arm's count is 0.
8. Redraw the price path for the retrained checkpoint with
   `scripts/explain_rl_best.py --config ... --model ... --out
   runs/price_monotone_up/explain/` (it already takes those three flags; no new
   plotting code needed). That figure is how to judge whether F4 now reads
   intuitively.
9. Update `docs/DESIGN.md` § Control layer (a fifth control), `docs/EXTENDING.md`
   (the YAML block), `docs/EXPERIMENT_LOG.md` (a new section), and sync this
   wiki per `wiki/SCHEMA.md`.

**Verify.**

- Zero violations under `mode: clamp`, asserted on the env's own
  **`info["price"]`** (the price `ReservationEnv.step` charged — this is what
  catches a silent inner override by promo/MPC). Compare executed prices from
  the second step on; `reset()`'s $80 is not a prior decision. The
  unconstrained arm's violation count is > 0 (30/30 episodes in the published
  rollouts already decrease), proving the constraint actually binds.
- `cmp configs/default.yaml src/reservation_pricing/configs/default.yaml` → exit 0.
- Default config unchanged in effect: with `enabled: false` the factory does
  not wrap, the full `pytest -q` suite still passes with the new tests added,
  and no number elsewhere in `runs/` moves.
- The redrawn price path in `runs/price_monotone_up/explain/` is non-decreasing
  as `days_prior` falls. The late clearance — weekend mean about $112.85 at
  day 16 down to $89.08 at day 1 — is gone. Day 21 ($114.67) was never the
  markdown.
- `ruff check . && ruff format --check . && pytest -q` (full suite, since
  `runs/` is touched) before any commit.

**Gotcha.** `env.capacity`, the price band, the selling-limit bounds and the
`env.reward` weights stay untouched — the new checkpoints go to
`artifacts/price_monotone/` (untracked, retrain locally). Never hand-edit a
generated file under `runs/price_monotone_up/`; rerun the script beside it.

## What not to bother with

- Rewriting `estimate_keep_rate` (measured at ≤1.11%; see ground rule 6).
- Tuning the myopic baseline's `1.05`/`0.85` (its limit never binds; and they are
  the behaviour-cloning target, so changing them breaks `rprl-bc-sac`
  reproduction of the shipped checkpoint).
- Degrading demand *level* to test pricing robustness — it provably cannot move a
  price. Perturb `elasticity`. See `docs/DESIGN.md` § Forecast vs truth.
