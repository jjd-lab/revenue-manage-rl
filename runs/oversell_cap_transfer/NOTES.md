# Does the oversell cap transfer? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29
**Score:** `score_aware`, same scoring as `docs/EXPERIMENT_LOG.md` §7
**Cap:** `configs/experiment_bc_sac_safe_sl.yaml` — `kind=chance`, `overbook_factor=1.02`, `activate_remain_frac=0.20`, `mix_alpha=0.25`
**Regenerate:** `python runs/oversell_cap_transfer/run_cap_transfer.py`

## The question

The cap had only ever been applied to BC→SAC, where §5c and §7 report it turning
0.71 peak denied admission into 0.00 for 0.6% of score. The site generalises that
into a method claim — *keep overbooking risk in a projection rather than in the
reward*. Joint SAC (0.18) and joint PPO (0.35) also deny admission and had never
been capped. One wrapper, three policies, same seeds.

## Results

| policy | uncapped | capped | score given up | peak oversell |
| --- | ---: | ---: | ---: | --- |
| joint BC→SAC raw | 2,094,381 | **2,081,400** | **0.62%** | 0.71 → **0.00** |
| joint SAC `rl_best` | 2,033,264 | **2,008,716** | 1.21% | 0.18 → **0.00** |
| joint PPO `long_007` | 2,010,885 | **1,962,157** | 2.42% | 0.35 → **0.00** |

Full CSV: `cap_transfer_table.csv`.

## Verdict

| Question | Answer |
| --- | --- |
| Does the cap reach zero denied admission on every joint policy? | **YES** — 0.00 for all three, no retraining |
| Is it cheap for all of them? | **Mostly** — 0.6% to 2.4% of score |
| Does the cost scale with how much oversell is removed? | **NO — it inverts** (see below) |

**The cost inverts, and that is the finding.** The policy that oversells *most*
is the *cheapest* to fix: BC→SAC gives up 0.62% to remove denied admission on 71%
of peak nights, while joint PPO gives up 2.42% to remove it on 35%. The reading:
BC→SAC is filling so far past capacity that the bookings the cap removes were
already being charged the double oversell penalty — clipping them is nearly free.
Joint PPO oversells rarely, so the bookings the cap takes are ones it was actually
being paid for. **A policy that oversells a lot is not necessarily expensive to
make safe; it depends whether the marginal booking was earning or costing.**

## What this does to the §7 ranking

Ranked among policies that deny admission on **no** peak night — the comparison a
venue that cannot turn ticket-holders away would actually make:

| policy | `score_aware` | how it gets to zero |
| --- | ---: | --- |
| **joint BC→SAC + cap** | **2,081,400** | cap |
| price-only pace PPO | 2,009,914 | naturally, one lever |
| joint SAC + cap | 2,008,716 | cap |
| price-only PPO | 2,003,091 | naturally, one lever |
| joint PPO + cap | 1,962,157 | cap |

This **narrows §7's claim**. Uncapped, the intervals separate only BC→SAC from
the price-only rows; `rl_best` and joint PPO tie them (§7). Under a
zero-denied-admission constraint, capped BC→SAC keeps the lead the interval
supports (+3.6% over pace PPO). Capped joint SAC lands 1,198 behind pace PPO and
capped joint PPO falls below both price-only policies on the point estimates;
those two capped rows were not saved night by night, so they have no interval.

So "both levers beat one" is not a property of joint control in general. Under the
safety constraint it is a property of **the behaviour-cloned policy specifically**.
The second lever is still load-bearing (`runs/ablate_selling_limit/` — taking it
away costs 5–17%), but having it is not by itself enough to beat a well-shaped
one-lever policy.
