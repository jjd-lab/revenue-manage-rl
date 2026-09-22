# What does knowing the true cancellation model buy? — NOTES

**Demand:** `tree_elastic` (default)
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29
**Score:** `score_aware`, same scoring as `docs/EXPERIMENT_LOG.md` §7
**Regenerate:** `python runs/keep_rate_dependence/run_probe.py`

## The leak

`controls/selling_limit.py::estimate_keep_rate` reads the environment's **own
generating parameters**:

```
env.cancel_lambda   env.cancel_rho_weekday   env.cancel_rho_weekend
env.noshow_base     env.noshow_dow_coef      env.noshow_month_coef
```

Those are the exact attributes `ReservationEnv.cancel_fn` and `no_show_fn` use to
decide who cancels, and the estimator replays the same Weibull. So the analytic
selling limit and the oversell cap operate with a **perfectly specified
cancellation model** — something a real venue would have to estimate from history
and would get somewhat wrong.

It is not clairvoyance: the estimator cannot see which bookings will cancel, and
it does not see the per-episode noise (`noshow_noise_std`, `cancel_rho_noise_std`)
the env redraws. It is "a forecaster whose model happens to be exactly right,"
the same category as the myopic baseline's perfect demand forecast. But unlike
that one, it had never been disclosed. This directory prices it.

## Results — replace the estimator with fixed guesses

| subject | block | exact model | fixed 0.85 | fixed 0.70 | fixed 0.95 | spread | exact best? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BC→SAC + cap | `safe_sl` | 2,081,400 | 2,081,776 | 2,081,439 | 2,082,832 | **0.07%** | no |
| pace PPO limit | `selling_limit` | 2,009,914 | **2,018,712** | 1,996,375 | 2,014,844 | **1.11%** | no |
| price-only PPO limit | `selling_limit` | 2,003,091 | 2,001,747 | 2,003,091 | 1,996,528 | **0.33%** | yes (tied) |

Peak denied admission is **0.0000 in every cell** — no guess, however wrong,
breaks the cap's guarantee. Full CSV: `keep_rate_table.csv`.

## Verdict

| Question | Answer |
| --- | --- |
| Is the leak real? | **YES** — the estimator reads the simulator's true parameters |
| Does it change any published conclusion? | **NO** — 0.07% on the cap, ≤1.11% on the limits |
| Does privileged knowledge even help? | **NO** — a round 0.85 guess *beats* the exact model twice out of three |

**Why knowing the truth does not pay.** The true cancellation model tells you the
keep *rate*. The score-optimal *limit* depends on the reward structure — what an
empty seat costs versus an oversold one — not on the physics. The estimator is
solving a different problem from the one the score rewards, so being right about
cancellations does not make it right about where to set the limit.

Conclusion: **disclose it, do not fix it.** Rewriting `estimate_keep_rate` to fit
from observed history would add machinery and move results by ~1%. The honest
statement is that the controllers are handed a correct cancellation model, and
that it is worth almost nothing.

## Caveat

`pace PPO` and `price-only PPO` were *trained* against the exact estimator, so
their fixed-guess arms are off-distribution. Read the spread as "not sensitive in
a way that favours privileged knowledge," not as "0.85 is a better setting."

The demand-side counterpart — what happens when the *demand* forecast is wrong —
is `runs/forecast_misspecification/`.
