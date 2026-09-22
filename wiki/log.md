# Log

Append-only. Newest at the bottom. Format: `## [YYYY-MM-DD] {ingest|decision|review} | Title`.

## [2026-09-21] ingest | Bootstrap wiki and CLAUDE.md

Created `wiki/` (SCHEMA, index, log, three `docs/` pages) and a thin `CLAUDE.md`.
Deliberately did not mirror the root `docs/` set — the wiki routes to it and covers
only the gaps: [[dev-commands]] and [[conventions]]. Also created
the `.venv` the repo's scripts assume and confirmed the project installs and tests
clean on Python 3.13 with torch 2.14 / SB3 2.9.

## [2026-09-21] ingest | Section 8 (explainability) and dead-link repair

Added section 8 to `docs/EXPERIMENT_LOG.md`: the `runs/explain_rl_best/` findings
(model-free observation, ~$109 weekend vs ~$84 weekday, booking-curve pricing, SL
tracking remaining inventory, lift concentrated on peak) previously existed only in
that folder's README, so the narrative ended without saying what the winning policy
learned. Key paths gained rows for `runs/explain_rl_best/`, `runs/both_goals/`, and
`runs/oracle_ceiling/`. Repaired references to four files that were never kept:
`runs/bc_sac/final_eval.py` and `runs/tree_long/policy_deepdive/` (replaced with
working `rprl-eval` / `explain_rl_best.py` commands) and `artifacts/both_goals/`
(the safe-SL row reuses the BC→SAC checkpoint under a wrapper). Smoke tests 20/20.


## [2026-09-21] decision | artifacts/ untracked: runs are findings, artifacts are raw material

Gitignored `artifacts/` wholesale rather than committing the five checkpoints.
Rationale is the findings/raw-material split, not size: a checkpoint is a
regenerable input (over half its bytes are Adam optimizer state), while `runs/` is
the claim the repo makes and stays tracked. Cost: a fresh clone cannot run
`REPRODUCE.py` or `rprl-eval` until retrained, and SB3 is not bit-reproducible
across machines, so regenerated numbers land near but not on the published tables.
README gained a *Model checkpoints* table mapping each expected path to its
training config. Recorded in [[conventions]].

## [2026-09-21] decision | Prune orphan configs; keep tree-world provenance

`configs/` went 30 → 20. Removed four redundancies (`ppo.yaml`/`sac.yaml`, which
duplicated `algo_ppo.yaml`/`algo_sac.yaml`; `experiment_bc_sac_safe_sl_mpc.yaml`,
whose own header says it is not used; `experiment_legacy_ppo.yaml`, covered by
`demand_linear_legacy.yaml` + `algo_ppo.yaml`) and the six-config linear-world
`push_further` sweep (`long_003/004/007`, `long_sac`, `fix_fill`, `push_screen`)
together with `runs/example_results.md`, its only results file. That batch wrote to
`runs/push_further/` and `artifacts/push_further/`, neither of which was ever kept,
and its numbers are on `linear_legacy` demand (~1.5M revenue) sitting unlabelled
next to tree-world tables (~1.1M) — actively misleading to a new reader.

Kept all five `tree_long_*.yaml`: `_007` and `_sac` trained the two shipped joint
checkpoints, and `_000`/`_003`/`_screen` are the retrain recipes for rows in the
section-2 table. Deleting those would make a published table unreproducible, which
is a worse trade than five extra files. Also gave `runs/tree_long/`, `bc_sac/`,
`price_only_long/`, `pace_ppo/` and `promo_ppo/` rows in the Key paths table — the
audit found they were reachable only via a config's `log_dir`. Smoke tests 20/20.

## [2026-09-21] decision | Experiment IDs: one name across config, run, checkpoint

Every runnable config now sets `train.run_name`, and the trainers read it. They
previously honoured only the CLI flag, so a config-driven run fell back to
`ppo_20260917_143502_s42` — that is why the curated zips (`rl_best.zip`,
`rl_pace_ppo.zip`) had to be renamed by hand and why four `tree_long_*` configs
writing to one directory would interleave. IDs reuse the historical run-directory
names (`ppo_pace`, `ppo_analytic`, `sac_optimize1d`, `long_sac`), so a retrain lands
beside the existing records instead of contradicting them. One test locks both
directions: config supplies the name, `--run-name` still overrides.

Deliberately did **not** rename the existing checkpoints to match: `rl_best.zip`
alone has 22 references, 18 inside generated `runs/` tables that are dated records.
`docs/EXPERIMENTS.md` carries the correspondence instead, indexed in [[conventions]].

## [2026-09-21] review | Source review and repo hygiene

Reviewed `src/` (8 findings, recorded separately — the load-bearing ones are an
off-by-one between the price-only controllers and `ReservationEnv.step`, and
`config/load.py` silently returning `{}` on a non-editable install). Also removed 20
byte-identical `* 2.py` duplicates and an empty `runs/price_only_long 2/`, all
untracked macOS copies. Dependency audit: every declared dependency is imported and
every third-party import is declared; the venv's five top-level packages are exactly
`pip install -e ".[dev]"`.

## [2026-09-21] ingest | Remove docs/wiki overlap rather than merging the trees

Considered folding root `docs/` into the wiki and decided against it: the two have
different readers — `docs/` is written for someone reading the project cold, the
wiki for whoever maintains it. Fixed the actual overlap instead, in the
direction SCHEMA's own lint requires. The README now owns the command reference
(gained the `rprl-*` table that only [[dev-commands]] had); [[dev-commands]] went
71 -> 50 lines and keeps only what the README does not say: why the venv must sit
at the repo root, what each extra buys, the limits of the smoke suite, and the
reproduction gotchas. Also corrected the README layout block, which still called
`artifacts/` shipped.

## [2026-09-21] decision | Name controls after the metric they move

`controls/safe_sl.py` -> `controls/oversell_cap.py` (`SafeSLProjector` ->
`OversellCap`) and `envs/safe_sl_wrapper.py` -> `envs/oversell_guard.py`
(`SafeSLWrapper` -> `OversellGuardEnv`). The old names stacked four problems: `SL`
is an abbreviation the repo elsewhere spells `selling_limit`, "safe" is a judgment
rather than a mechanism, "wrapper" spends the filename on an implementation
pattern, and the two files differed only by that word. Every table in `runs/`
already calls the thing they move `oversell_rate`, so the code now uses the data's
vocabulary.

The config block and `info[...]` keys stay `safe_sl`: `runs/*/final_eval.py` build
that block by name, so renaming it would break published experiments. The retained
spelling is explained in the module docstring.

Also deleted `env.py` and `config_utils.py` (re-export shims for an earlier layout,
zero importers) and `demand/multiproduct.py` (a placeholder that raised on use and
was never registered) - `docs/EXTENDING.md` keeps the migration sketch, which is
the part with value. README and CLAUDE.md now describe the problem as two levers,
price and availability, rather than pricing alone.

## [2026-09-21] ingest | Give soft-aware reporting one home; share run-name logic

`metrics.py` 497 -> 273 lines. The soft-aware *reporting* half
(`aggregate_soft_aware`, `soft_aware_table`, `_slice_stats`, `_round_or_nan`) moved
next to the orchestration that calls it in `evaluate/soft_aware.py`. `SoftAwareConfig`
and `classify_soft` stay in `metrics.py`: classification happens *during* the
rollout, reporting happens after, and that split is the layering rather than an
accident. Moving them too would have made `metrics` and `evaluate.soft_aware`
import each other.

Both trainers now call one `resolve_run_name` (`train/runner.py`) instead of
repeating the precedence chain - the duplication that made the earlier `run_name`
fix need the same edit twice.

`pyflakes` caught what the tests could not: the moved block referenced `Sequence`
with no import, invisible at runtime because the module uses
`from __future__ import annotations`. Worth keeping in the loop for moves like this.

## [2026-09-21] decision | Keep the venv in `.venv.nosync/` with `.venv` symlinked to it

The checkout lives in a cloud-synced folder. Sync duplicated files as `name 2.py`
and set the macOS hidden flag on the editable install's `.pth` files, which
Python 3.13 skips — so `import reservation_pricing` failed while `pytest` still
passed (`pythonpath = ["src"]` never needs the install). The venv now lives in
`.venv.nosync/`, a name the sync client excludes, with `.venv` symlinked to it.
Symptom, check, and fix are in [[dev-commands]].

## [2026-09-21] ingest | Fix six of the eight source-review findings

Fixed the ones that cannot change a published number:

- `algorithms/bc.py` branched on the SAC-only `get_action_dist_params`, so the
  advertised `td3` BC path raised `AttributeError` *after* collecting the whole
  expert dataset. TD3's actor is deterministic and has no `log_std`.
- `evaluate/soft_aware.py` built the soft oracle at a literal `80.0` and the caller
  never passed the env's floor, so `gap_to_oracle` was measured against a
  non-oracle on any other price band. `min_price` is now required, not defaulted.
- `envs/reservation.py` accepted `use_held_out=True` with empty `held_out_months`,
  satisfied neither return branch, and silently sampled the full year - the
  held-out protocol failing open. Now raises at construction.
- `cli.py` dropped `--trials` unless `--method optuna` was typed too, even when the
  config already selected optuna.
- `tune/runner.py`'s `run_optuna` wrote its summary without `mkdir` (the grid path
  does it), so a finished sweep could raise and lose its return value.
- Corrected the comment on the `is_soft is None` branch in `aggregate_soft_aware`:
  it fills in a hand-built `EpisodeMetrics`, it does **not** re-classify a rollout,
  because `run_episode` always records a bool.

Regression tests for the first three; the TD3 one was confirmed to exercise the
real failure (`stable_baselines3.td3.policies.Actor` has no
`get_action_dist_params`), and the oracle test drives the caller with a floor of 60
so the old hard-coded 80 lands inside the band and yields a different revenue.

Still open by decision: the price-only controller off-by-one (changing it
invalidates published pace/price-only numbers) and `config/load.py` returning `{}`
on a missing default.

## [2026-09-21] ingest | Commit editor settings

Added `.vscode/settings.json`. It points at `.venv/bin/python` (the symlink) rather
than `.venv.nosync`, so the file stays valid on a machine with an ordinary `.venv`,
and it enables pytest, adds `src` to the analysis path, and excludes the venv and
`artifacts/` from search and watching.

## [2026-09-21] decision | Config loading fails loudly; full installability deferred

`load_config` returned `{}` when `configs/default.yaml` could not be read, and
`validate_config` is too light to catch it - so a non-editable install trained on
`linear_legacy` demand with constructor-default reward weights while looking
healthy. It now raises, naming the path and what to do. `package_root` also stopped
counting directory levels (`parents[3]`) and walks up for the ancestor holding
`configs/default.yaml`, which is correct for a checkout at any depth.

Did **not** make `pip install .` work, despite planning to: `configs/` lives beside
the package, not inside it, and setuptools package-data can only ship files within
the package. Real installability means relocating `configs/` into
`src/reservation_pricing/` and updating the 24 files that reference `configs/...`
paths - a separate decision, and speculative until someone actually wants to
install this without a checkout. Failing loudly removes the dangerous half either
way.

## [2026-09-21] decision | Fix the price-only decision-day off-by-one; regenerate affected tables

`envs/price_only.py` computed the selling-limit and early-promo decisions before
`ReservationEnv.step()` decrements `days_prior`, so the controller planned for a
day the env would not charge the booking against (`price_mpc.py` already
compensated with `dp - 1`; the other two controllers did not). Fixed by decrementing
`env.days_prior` for the span of each controller call and restoring it before
`env.step()`, matching the MPC's convention.

**Measured, not assumed.** Reproduced the shipped `pace_ppo` checkpoint on all 30
held-out seeds before and after: score moved **832,319 → 831,842 (−0.06%)**. The vanilla
`price_only_ppo_analytic` checkpoint and every price-only baseline were **bit-identical**
before and after (verified to 11 significant figures) - the bug only bites when the
analytic selling limit sits close enough to bind, which only the soft-day-tuned pace
policy does.

Regenerated every table reachable from a shipped checkpoint:
`runs/joint_vs_price_only_soft_aware/`, `runs/pace_ppo/`, `runs/both_goals/`,
`runs/soft_aware_eval/`. Left `runs/price_only_long/` untouched deliberately - its
`sac_optimize1d` row needs a checkpoint that was never shipped, and a full regen
would have silently dropped that row from a hand-maintained table; confirmed by a
separate isolated run that its reproducible rows are unaffected anyway.
`docs/EXPERIMENT_LOG.md` gained a correction note at the top and one updated cell
in the §7 table (the only rounded value that crossed a display boundary).

**Also found, unrelated to this fix:** re-running the *unfixed* code after the
venv rebuild no longer reproduces the exact numbers committed earlier in this
session (confirmed via an isolated single-policy rerun with zero price-only code
involved) - a <0.001% drift from a different torch/BLAS build, not from any code
change. Documented in `both_goals/NOTES.md` rather than conflated with the real
fix, so a reader doesn't misattribute noise to the bug or vice versa.

Added a regression test (`test_price_only_controller_sees_the_decision_day`) and
confirmed it fails against the pre-fix code (`[100] == [99]`) before trusting it.

## [2026-09-21] ingest | Public reading of the experiment

Added `site/index.html`, a chart-led page for someone who is not going to read
the experiment log: the single-resource problem, four differences from the
classical revenue-management line, and the soft-aware result. The charts are
drawn by `scripts/build_site_figures.py` from committed CSVs in `runs/`, so they
do not depend on checkpoints.

## [2026-09-21] ingest | requirements.txt pinning the environment behind runs/

Added a root `requirements.txt` that pins the exact versions the shipped `runs/`
tables were produced with (SB3 2.9.0, torch 2.14.0, numpy 2.5.3, Python 3.13) and
carries `-e .` plus the `dev` extra, so one `pip install -r requirements.txt`
reproduces the full environment. `pyproject.toml` keeps the allowed ranges; the
pins exist because a venv rebuild has already moved eval numbers by ~0.001% with
no code change. README *Install* now offers both paths; rationale in [[dev-commands]].

## [2026-09-22] decision | Pre-release pass: one seed set, one report writer, packaged config, CI

Final review before the repo goes public. Code: the price MPC now sees the
decision day like the other controllers (rerun of `runs/both_goals/` unchanged
to the cent); `evaluate/report.py` is the one writer behind the four
`runs/*/final_eval.py` scripts and stores checkpoint paths relative to the repo
root (verified: every rerunnable row reproduces exactly, the two exceptions being
the documented venv drift on `rl_best` and the decision-day fix on `pace_ppo`);
`configs/default.yaml` now carries the whole schema, every control off, with a
byte-identical packaged copy so `pip install .` works; `rprl-tune --trials`
no longer crashes under grid; the explainability script uses the canonical
booking-curve encoding (its `base_demand` column was wrong on 60 of 101 days,
the soft/peak split was not); `cumulative_mat_BOH` → `cumulative_mat_boh` and
`room` → `headroom`. Tests 27 → 69 across seven files, `slow` marker for the
three that train; ruff, pre-commit, CI on 3.10/3.13, LICENSE, CONTRIBUTING,
CITATION.

**The headline table moved to seeds 0–29**, the set every other run already
used (it alone had been on 0–4 + 1000–1024). On the common seeds the ranking
changed: raw BC→SAC 2.094M > BC→SAC + safe SL 2.081M > joint SAC 2.033M > joint
PPO 2.011M > pace PPO 2.010M > price-only PPO 2.003M > myopic 1.972M. The three
SAC-based joint policies lead every price-only policy on both draws; their order
among themselves does not survive a change of seeds and is documented as seed
noise in `docs/EXPERIMENT_LOG.md` §7. The recommended package is now the capped
BC→SAC (99.4% of the raw score, zero denied admission); README, the site, and
the takeaways say so.

Hygiene: ten tracked `*log*.txt` files (pre-fix numbers, another machine's
paths) removed; absolute paths gone from every table; timezone suffixes and
machine references gone from `runs/`; this log trimmed of home-directory and
sibling-project detail. `site/` rebuilt: all five charts from one script and one
palette, dot chart instead of a truncated bar chart, self-hosted fonts, metadata,
repo links. Git history squashed to a single commit for release (tag
`pre-public-squash` kept locally).

## [2026-09-22] decision | Ablation: the second lever is load-bearing, and the movement is what pays

Section 7's 1.2–3.6% joint-vs-price-only gap had two readings — the joint policies
use the selling limit well, or their price policy is simply better and the limit is
along for the ride. Nothing published separated them, so `runs/ablate_selling_limit/`
re-scores each joint policy with its price untouched and its limit pinned, once wide
open and once to the myopic baseline's flat rule.

Removing the limit costs **90k–344k** of `score_aware`, an order of magnitude more
than the gap it was meant to explain, and peak denied admission jumps to 0.82. The
flat-rule arm is the finding: `rl_best`'s limit *averages* 12,578, within 2% of the
flat 12,353 it is pinned to, and that pinning still costs 120k — so the value is not
the level the limit sits at, it is when it moves. That prices §8's figure 05
(limit tracking remaining inventory), which until now was only qualitative.

Recorded as §9 plus takeaway 6 in `docs/EXPERIMENT_LOG.md`, a paragraph in §05 of
the site, and a row in [[conventions]]-adjacent `docs/EXPERIMENTS.md`. The caveat
travels with it everywhere: pinned arms are off-distribution (each policy priced for
the limit it learned), so this measures coupling, not what a policy purpose-trained
for a fixed limit would score — the price-only rows in §7 remain the honest
cross-policy number.

## [2026-09-22] decision | The oversell cap transfers; the joint advantage does not

The cap (`control.safe_sl`) had only ever been applied to BC→SAC.
`runs/oversell_cap_transfer/` runs all three joint policies through it, uncapped
and capped, on the same seeds. Every one reaches **zero** peak denied admission
for 0.6–2.4% of score, so §5c's method claim generalises beyond the policy it was
built for.

Two findings worth the write-up. **The cost inverts:** BC→SAC gives up 0.62% to
remove oversell on 71% of peak nights while joint PPO gives up 2.42% to remove it
on 35% — the heavy overseller is filling so far past capacity that the clipped
bookings were already paying the double penalty, so oversell volume does not
predict the price of safety. **And it narrows §7:** ranked among zero-denied-
admission policies, BC→SAC+cap 2.081M > pace PPO 2.010M > joint SAC+cap 2.009M >
price-only PPO 2.003M > joint PPO+cap 1.962M. Capped joint SAC ties pace PPO
(1,198 apart) and capped joint PPO falls below both price-only policies, so under
a safety constraint "both levers beat one" belongs to the behaviour-cloned policy
specifically, not to joint control in general. Recorded as §10 plus takeaway 7,
and a paragraph in §05 of the site.

Also fixed a published falsehood: the README and two wiki pages said the site is
published "with GitHub Pages from the `/site` folder". Branch-deploy only accepts
`/` or `/docs` (verified: the API rejects `/site` with a 422), so `site/` is now
deployed by `.github/workflows/pages.yml` with Pages in `build_type=workflow`.

## [2026-09-22] decision | Split forecast from truth; price the privileged knowledge

A question about where the cap's constants come from turned into an audit of what
this code is allowed to know that a real operator would not. Two channels, both
now measured rather than argued about.

**The cancellation model** (`runs/keep_rate_dependence/`): `estimate_keep_rate`
reads the env's own `cancel_lambda`, `cancel_rho_*`, `noshow_*` and replays its
Weibull, so the analytic limit and the oversell cap run on a perfectly specified
cancellation model. Priced by swapping in fixed guesses: **0.07%** on the cap,
**≤1.11%** on the limits, peak oversell 0.0000 in every cell — and a round 0.85
guess beats the exact model twice out of three. Knowing the physics gives you the
keep rate; the score-optimal limit depends on the reward structure. Disclosed in
[[conventions]]-adjacent docs, deliberately **not** fixed.

**The demand forecast** (`runs/forecast_misspecification/`): new `demand.forecast`
block plus `demand.protocol.decision_model(env)`, so baselines, the `optimize_1d`
limit, early promo and the MPC can be given a wrong model while the env still
generates from the true one. Env dynamics, soft/peak classification and the
soft-day oracle deliberately keep the truth — misspecifying those would change the
world or the scoreboard, not the operator's information. Result: a wrong
elasticity costs myopic **3.5–5.8%**, more than §7's whole 1.2–3.6% margin, while
the joint policies move **0.00%** with mean prices identical to the cent. Recorded
as §11 with takeaway 9, and a paragraph in §05 of the site.

**A design trap worth remembering.** The first version degraded the demand *level*
and returned +0.00% for everything including myopic — not a null result but a bad
experiment. Demand is `base * (1 + e(p-ref)/ref)`, so base is multiplicative and
cancels out of the price argmax: myopic's price is `ref(1-e)/(-2e) = $91.67`
however wrong the level is (verified at base scaled 1.00/0.75/0.50). That is also
why §8's myopic line is "a flat $92" — a closed form, not a quirk. To misspecify
pricing, perturb elasticity. Written into `docs/DESIGN.md` § Forecast vs truth.

**Also closed:** `mix_alpha=0.25` was originally selected by reading the same
seeds §7 reports on. Re-run on a disjoint block (100–129,
`runs/both_goals/validate_mix_alpha.py`) the identical rule picks 0.25 again,
because 0.4 denies admission on 41–45% of peak nights either way. The shipped
value is not a test-set artefact.

## [2026-09-22] decision | Handover page: session findings and the queued work

Added [[next-steps]], a two-audience page: a plain-language account of what this
session found (the myopic limit is inert; the learned limit is load-bearing and it
is the *movement* that pays; the cap transfers but its cost inverts with oversell
volume; the RL policies need no demand forecast), and executable instructions for
whoever picks the work up.

The seed protocol recommendation is measured, not guessed. Pairing is what matters:
revenue std across nights is 231,113 for one policy, but the std of the *paired*
difference between joint SAC and pace PPO is 13,274 — a 17x reduction, because both
policies see the same nights. A paired 95% interval resolves ±4,847 at n=30 and
±1,877 at n=200, so the big comparisons are already decidable at thirty nights
while the three joint policies against each other need ~70–100. The fix is
therefore **paired bootstrap intervals first, more seeds second** — and the
bootstrap must recompute `score_aware` per resample, since it is a stratified
aggregate rather than a mean.

Housekeeping is explicitly sequenced *after* the seed protocol so the two
regenerations do not happen twice.

## [2026-09-22] decision | Paired intervals on the headline, before more seeds

`score_aware` comparisons now carry a paired bootstrap: resample seeds 0–29,
recompute the stratified score on each draw, and call a pair a tie when the
2.5/97.5 interval covers zero. The headline file is
`runs/joint_vs_price_only_soft_aware/paired_intervals.csv`, from the saved
episodes, so the point-estimate table did not move. On that readout raw and
capped BC→SAC are a tie with each other and a lead over every other row; the
1.2% point gap (`rl_best` over pace) covers zero, and so does every pair among
`rl_best`, joint PPO, pace PPO, price-only PPO, and myopic. §7, the site, and
the README now say that. More seeds stay queued, and only for a tie still worth
separating — extending the list would change the soft/peak split. Protocol:
`docs/EVALUATING_POLICIES.md`. Status: [[next-steps]].

## [2026-09-22] ingest | Gitignore `private/` and `CLAUDE.local.md`

`private/` was documented as gitignored and never shipped, but `.gitignore` did
not list it. It does now, with `CLAUDE.local.md` for personal agent overrides.
`CLAUDE.md` stays tracked: it is the shared agent front door, not a local file.

## [2026-09-22] ingest | Site policies section regrouped

`site/index.html` §03 now groups the roster as not-learned / learned / the cap,
and nests a two-step BC→SAC recipe (copy the myopic rule, then let SAC improve)
under that policy instead of a four-box strip that sat under the whole list.
The cap is no longer visually part of the training pipeline. [[dev-commands]]
still only says how the page is published.

## [2026-09-22] decision | The BC→SAC lead does not repeat across training seeds

Retrained pace PPO and BC→SAC at seeds 43, 44, and 46 into
`artifacts/training_seeds/`, leaving the shipped seed-42 zips in place, and
scored them on nights 0–29. One matched interval covers zero, so the published
§7 lead is training-seed sensitive. Reading: `runs/training_seeds/NOTES.md`.
Both trainers now share `<model_dir>/<run_name>/`; a config with a `bc:` block
is the warm start, and anything else is the standard trainer. The oversell cap
stays an eval wrapper. Status: [[next-steps]].

## [2026-09-22] ingest | Promo and price-only SAC checkpoints regenerated

Retrained the two campaigns whose weights had not been kept: early-promo PPO and
price-only SAC, both at seed 42. Copied the finals to the curated paths and
regenerated `runs/promo_ppo/` and `runs/price_only_long/`. Promo still loses to
pace. The SAC retrain is below `rl_best` and still short of BC→SAC. The shipped
price-only PPO zip was not retrained. Status: [[next-steps]].

## [2026-09-22] ingest | Pre-commit cleanup: gitignore, a stale row, two renames

Fixed `runs/price_only_long/NOTES.md`'s `rl_best_sac@200k` row, which still
carried pre-regeneration numbers disagreeing with its own CSV. Ignored
`runs/**/expert_dataset.npz` (1.2 MB × 3, BC rollout tensors — raw material,
same class as `artifacts/`) before it entered history. Dropped the duplicate
`comparison.md` write in `evaluate/soft_aware.py` (`soft_aware_comparison.md`
is the one name now). Removed `runs/tree_demand_sanity.md` (both places that
cited it called it superseded). Renamed
`runs/joint_vs_price_only_soft_aware/REPRODUCE.py` → `run_headline.py` (the
only SCREAMING_CASE filename in the repo) and `runs/soft_aware_eval/` →
`runs/soft_aware_report_demo/` (every doc already called it a demo). Cleaned
~37 MB of byte-identical duplicate checkpoints and two unreferenced smoke dirs
out of `artifacts/` (local only, no git impact). [[conventions]] now states the
run-directory and driver-script naming rule. Status: [[next-steps]].

## [2026-09-22] decision | Monotone price constraint queued, not built

F4's price path ends in a late markdown that no real venue can run. Wrote the
full spec — a `direction: up` constraint (outermost wrapper, project or
penalty mode, first-step exempt, rejects `early_promo`/`mpc` since both
override price *inside* `PriceOnlyWrapper` after any outer projection) plus a
four-arm experiment — into [[next-steps]] Task 4 as executable instructions for
whoever picks it up next. Expected result stated up front: the constrained
retrain will likely score below the unconstrained one; the number is the
deliverable. No code, config, or `runs/` directory exists yet.

## [2026-09-22] review | Task 4 spec corrected before anyone builds it

Checked the queued monotone-price spec against the code and the published
weekend path. Both hazards hold: promo/MPC (including the `promo` alias)
override price inside `PriceOnlyWrapper` after an outer wrapper, and
`reset()`'s $80 is a placeholder `OversellGuardEnv` would leak across
episodes. Corrected the F4 days (crest is day 37, the plunge is day 16→1,
day 21 was never the markdown), stated knob defaults, and replaced the single
penalty training with a `{1, 10, 100}` sweep ranked on seeds 100–129. The run
script must keep `evaluate_policy_soft_aware`'s per-seed rows for
`paired_difference`. Still no code. Status: [[next-steps]].

## [2026-09-22] ingest | Monotone price constraint, and what it costs

Added `price_monotone` as a fifth control: outermost wrapper, project or
penalty, first step exempt, rejected together with promo (including the
`promo` alias) and MPC. Four arms on seeds 0–29 are in
`runs/price_monotone_up/`. Projecting frozen `rl_best` raises `score_aware`
about 45k and removes every charged markdown. Retraining under the clamp
costs about 110k and flattens the path. Penalty weight 10 still marks down.
`explain_rl_best.py` now records the charged price, so a clamped path is what
the figure shows. Status: [[next-steps]].
