# Does the second lever earn its place? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29
**Score:** `score_aware`, same soft-aware scoring as `docs/EXPERIMENT_LOG.md` §7
**Regenerate:** `python runs/ablate_selling_limit/run_ablation.py`

## The question

Section 7's point estimates put the joint policies above the price-only ones.
The intervals keep a lead only for BC→SAC; `rl_best` and joint PPO tie the
price-only rows. Two explanations fit either gap equally well:

1. the joint policies **use the selling limit well**, or
2. their **price policy** is simply better (SAC, and the BC warm start), and the
   second lever is along for the ride.

Nothing published so far separates them. This run does: each joint policy is
re-scored with its price untouched and its limit replaced by a constant.

## Results

| policy | limit arm | score_aware | peak oversell | mean limit |
| --- | --- | ---: | ---: | ---: |
| joint SAC `rl_best` | **learned** | **2,033,264** | 0.18 | 12,578 |
| | pinned open (15,000) | 1,688,778 | 0.82 | 15,000 |
| | pinned flat (12,353) | 1,913,626 | 0.88 | 12,353 |
| joint BC→SAC raw | **learned** | **2,094,381** | 0.71 | 12,722 |
| | pinned open (15,000) | 1,855,362 | 0.82 | 15,000 |
| | pinned flat (12,353) | 1,942,613 | 0.82 | 12,353 |
| joint PPO `long_007` | **learned** | **2,010,885** | 0.35 | 11,687 |
| | pinned open (15,000) | 1,903,204 | 0.82 | 15,000 |
| | pinned flat (12,353) | 1,921,221 | 0.94 | 12,353 |

Full CSV: `ablation_table.csv`.

## Verdict

| Question | Answer |
| --- | --- |
| Is the joint gap just a better price policy? | **NO** — removing the limit costs 90k–344k. That cost is this table's result; §7's intervals are a separate question |
| Does a sensible *constant* limit recover it? | **NO** — the flat-rule arm still loses 90k–152k |
| Is the limit what holds denied admission down? | **YES** — peak oversell goes 0.18 → 0.82 for `rl_best` once the limit stops moving |

**The flat-rule arm is the interesting one.** Joint SAC's limit *averages* 12,578,
within 2% of the flat 12,353 it is compared against — and pinning it to that
average still costs 119,638. So the value is not the level the limit sits at; it
is **when it moves**. That is the same behaviour figure 05 of
`runs/explain_rl_best/` shows qualitatively (limit tracking remaining inventory),
now with a price attached to it.

## What this does not show

The pinned arms are **off-distribution**. Each policy chose its prices expecting
the limit it learned, so a pinned arm is a policy operating under a rule it was
never trained for. This measures how tightly the two levers are **coupled** — you
cannot strip the limit from a policy that learned to use it — not what a policy
purpose-trained for a fixed limit would score.

That second comparison already exists and is the honest cross-policy number: the
price-only policies, trained for exactly that regime, score 2.003M and 2.010M —
above every pinned arm (1.69M–1.94M) on the point estimates. §7's intervals put
them in a tie with `rl_best` and joint PPO, and below both BC→SAC rows. Quote
this table for the claim that the pin cost is not an artifact of the price
policy, and quote §7's BC→SAC-versus-pace interval for a resolved lead.
