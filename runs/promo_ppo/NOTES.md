# Early-promo + pace price-only PPO — NOTES

**Demand:** `tree_elastic` (default)  
**Held-out:** 30 episodes, months 6 & 12, seeds 0..29  
**Score:** `mean_true_revenue - 200 * mean_capacity_shortfall`  
**Date:** 2026-09-22  
**Threads:** torch and OpenMP capped at 1 (the trainer's default)

Regenerated from a new training at seed 42. The pace row is the shipped checkpoint, scored again on the same nights (831,842).

## What changed

**Early promo** — config-driven post-process on the RL price inside
`PriceOnlyWrapper` (after the agent acts, before analytic SL + dynamics).
Does **not** reopen joint 2D `(price, SL)`.

When predicted **price-unaware** tree base demand
`< base_demand_threshold` **and** `days_prior >= apply_when_days_prior_ge`,
force price to `promo_price` (`mode: set`).

| knob | value | rationale |
| --- | --- | --- |
| `base_demand_threshold` | 90.0 | matches soft_day_base_threshold; soft weekday mid-horizon base ~87 |
| `promo_price` | 80.0 | `min_price` — strongest soft-day stimulus |
| `mode` | `set` | force promo (alt: `clip` + `max_price_when_soft`) |
| `apply_when_days_prior_ge` | 30 | early/mid only; late clearing left to RL |
| pace + soft-day reward | **on** | keep booking-curve signal from pace experiment |

Config: `configs/experiment_price_only_promo_ppo.yaml`.  
Code: `controls/early_promo.py`, wired in `envs/price_only.py` + `envs/factory.py`.  
Docs: `docs/DESIGN.md`, `docs/EXTENDING.md`.

## What we trained

| run | algo | SL | promo | timesteps | seed | wall clock |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ppo_promo` | PPO (`n_envs=2`) | analytic (`overbook_factor=1.05`) | set@80 if base&lt;90 & dp≥30 | 200 000 | 42 | ~1.3 min |

EvalCallback shaped-reward “best” peaked @40k (same pattern as pace / prior
price-only). **Final** @200k used for primary verdict.

## Held-out results (30 eps)

| policy | mode | mean revenue | load | remain | oversell | undersell>1500 | score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **bc_sac_final** | joint | **1,114,825** | 0.896 | 1036 | 0.40 | 0.433 | **870,576** |
| **pace_ppo@200k** | price_only | 1,072,382 | 0.880 | 1203 | 0.00 | 0.433 | **831,842** |
| price_only_ppo_analytic@200k | price_only | 1,087,615 | 0.870 | 1304 | 0.00 | 0.467 | 826,857 |
| rl_best_sac@200k | joint | 1,085,363 | 0.870 | 1303 | 0.10 | 0.433 | 823,061 |
| **promo_ppo@200k** | price_only+promo | 1,060,692 | 0.880 | 1198 | 0.00 | 0.433 | 821,090 |
| promo_ppo_bestckpt (@40k shaped) | price_only+promo | 1,057,858 | 0.853 | 1468 | 0.00 | 0.433 | 764,302 |
| myopic_greedy | price_only | 1,049,488 | 0.840 | 1596 | 0.00 | 0.433 | 730,305 |
| fixed_price_80 | price_only | 961,979 | 0.880 | 1200 | 0.00 | 0.433 | 722,054 |

Full CSV/JSON: `runs/promo_ppo/comparison_table.csv`, `comparison_summary.json`, `episode_metrics.csv`.

### Promo trigger diagnostics (held-out, final policy)

- Mean fraction of steps with `early_promo_triggered`: **~0.29**
- Fires on soft weekdays (June weekdays ~0.45–0.54 of steps; Dec weekdays ~0.23)
- Never fires on peak weekends (promo_frac=0) — as intended
- **June weekday episodes still remain ~2000–2500** even with price forced to 80
  for half the horizon → undersell>1500 floor is structural at `min_price`

## Verdict (success criteria)

| Question | Answer |
| --- | --- |
| Did **undersell>1500** improve vs **pace_ppo**? | **NO** — tied at **0.433** |
| Did undersell>1500 improve vs **price_only_ppo**? | **YES** — 0.467 → **0.433** (same as pace; no further gain) |
| Did **score** beat **pace_ppo**? | **NO** — 821.1k vs 831.8k (**−10.8k**); revenue lost on forced $80 |
| Did score beat **bc_sac_final**? | **NO** — 821.1k vs 870.6k (−49.5k) |
| Prefer final or EvalCallback best? | **Final** — shaped best@40k undersells more (remain 1468) |

**Bottom line:** Early promo (force `min_price` when tree base &lt; 90 early/mid)
is a **clean, tested control**, but on this held-out set it is **not** a win over
pace-only PPO. Undersell stays at the **0.433 structural floor** (soft June
weekdays cannot fill capacity even at $80). Score **regresses** vs pace because
promo taxes revenue on soft days without buying extra fill. Prefer **pace_ppo**
among price-only runs; **bc_sac_final** still leads overall (with 40% oversell).

## Artifacts

- `artifacts/promo_ppo/rl_promo_ppo.zip` ← copy of `ppo_promo/final_model.zip` @200k
- `artifacts/promo_ppo/rl_promo_ppo_best.zip` ← EvalCallback best (shaped; not preferred)
- `runs/promo_ppo/ppo_promo/` — EvalCallback curve and `train_meta.json`

## Ops / smoke

- `pytest tests/test_smoke.py::test_early_promo_triggers_on_soft_state` — **green**
- `pytest tests/test_smoke.py::test_early_promo_config_loads` — **green**
- Cap Torch/OpenMP threads to 1
