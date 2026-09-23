# Experiment index

One row per experiment. The **ID** is the single name that ties a config to its run
directory and its checkpoint: every runnable config sets `train.run_name` to its ID,
so training writes `<model_dir>/<ID>/` and `<log_dir>/<ID>/` with no guesswork.

`docs/EXPERIMENT_LOG.md` remains the source of truth for what each result *means*;
this page only says where things live. `artifacts/` is untracked — see
"Model checkpoints" in `README.md` — so the checkpoint column is what a retrain
produces, not what you will find after a fresh clone.

## Training experiments

| ID | Config | Results in | Curated checkpoint | Log |
| --- | --- | --- | --- | --- |
| `long_sac` | `tree_long_sac.yaml` | `runs/tree_long/` | `artifacts/tree_long/best/rl_best.zip` | §2, §7 |
| `long_007` | `tree_long_007.yaml` | `runs/tree_long/` | `artifacts/tree_long/best/rl_ppo_long_007.zip` | §2, §7 |
| `long_000` | `tree_long_000.yaml` | `runs/tree_long/` | not kept | §2 |
| `long_003` | `tree_long_003.yaml` | `runs/tree_long/` | not kept | §2 |
| `ppo_screen` | `tree_long_screen.yaml` | `runs/tree_long/` | not kept | §2 |
| `bc_sac` | `experiment_bc_sac.yaml` | `runs/bc_sac/` | `artifacts/bc_sac/rl_bc_sac_final.zip` | §3 |
| `ppo_analytic` | `experiment_price_only_ppo_long.yaml` | `runs/price_only_long/` | `artifacts/price_only_long/rl_ppo_analytic.zip` | §4 |
| `sac_optimize1d` | `experiment_price_only_sac_long.yaml` | `runs/price_only_long/` | `artifacts/price_only_long/rl_sac_optimize1d.zip` | §4 |
| `ppo_pace` | `experiment_price_only_pace_ppo.yaml` | `runs/pace_ppo/` | `artifacts/pace_ppo/rl_pace_ppo.zip` | §5a |
| `ppo_promo` | `experiment_price_only_promo_ppo.yaml` | `runs/promo_ppo/` | `artifacts/promo_ppo/rl_promo_ppo.zip` | §5b |
| `bc_sac_safe_sl` | `experiment_bc_sac_safe_sl.yaml` | `runs/both_goals/` | reuses the `bc_sac` checkpoint | §5c |
| `pace_mpc` | `experiment_pace_mpc.yaml` | `runs/both_goals/` | reuses the `ppo_pace` checkpoint | §5c |
| `monotone_up_clamp` | `experiment_monotone_up_clamp_sac.yaml` | `runs/price_monotone_up/` | `artifacts/price_monotone/monotone_up_clamp/best/best_model.zip` | §12 |
| `monotone_up_penalty_10` | `experiment_monotone_up_penalty_last_sac.yaml` | `runs/price_monotone_up/` | `artifacts/price_monotone/monotone_up_penalty_10/best/best_model.zip` | §12 |
| `cu200` | `experiment_objective_cu200_sac.yaml` | `runs/objective/` | `artifacts/objective/cu200/final_model.zip` | §13 |

`bc_sac_safe_sl` and `pace_mpc` are wrappers, not new policies: safe SL projects a
trained joint action down, and the price MPC post-processes a trained price. Both
run an existing checkpoint under a different config, which is why they ship no
weights of their own.

`monotone_up_clamp` is a retrain, and eight further arms (ratchet, pace
shaping, behaviour cloning, peak-only, high-water penalty) sit beside it in
the same directory. The penalty arms sweep weights `{1, 10, 100}` and
`{0.1, 0.5, 1, 10}`
from `experiment_monotone_up_penalty_last_sac.yaml` (that file is weight 10, run name
`monotone_up_penalty_10`). Weights 1 and 100 load the same file, set
`control.price_monotone.penalty`, and train as `monotone_up_penalty_1` and
`monotone_up_penalty_100`. Selection is seeds 100–129; only the winner is
reported on seeds 0–29.

## Analyses (no training)

| What | Entry point | Results in | Log |
| --- | --- | --- | --- |
| $80 soft-day oracle ceiling | `runs/oracle_ceiling/run_oracle.py` | `runs/oracle_ceiling/` | §5c |
| Final soft-aware head-to-head | `runs/joint_vs_price_only_soft_aware/run_headline.py` | same directory | §7 |
| Soft-aware report demo (two policies) | `runs/soft_aware_report_demo/run_demo.py` | `runs/soft_aware_report_demo/` | §6 |
| Second-lever ablation (pin the selling limit) | `runs/ablate_selling_limit/run_ablation.py` | `runs/ablate_selling_limit/` | §9 |
| Oversell-cap transfer across joint policies | `runs/oversell_cap_transfer/run_cap_transfer.py` | `runs/oversell_cap_transfer/` | §10 |
| Cancellation-model dependence (keep-rate leak) | `runs/keep_rate_dependence/run_probe.py` | `runs/keep_rate_dependence/` | §11a |
| Policies under a wrong demand forecast | `runs/forecast_misspecification/run_misspecification.py` | `runs/forecast_misspecification/` | §11b |
| Cap `mix_alpha` re-selected on validation seeds | `runs/both_goals/validate_mix_alpha.py` | `runs/both_goals/` | §11 |
| Explainability figures | `scripts/explain_rl_best.py` | `runs/explain_rl_best/` | §8 |
| Monotone price, four arms | `runs/price_monotone_up/run_monotone.py` | `runs/price_monotone_up/` | §12 |
| Training objective vs `rl_best`, uncapped and capped | `runs/objective/run_objective.py` | `runs/objective/` | §13 |
| Dynamic-programming planner vs learned policies, true and wrong forecasts | `runs/dp_baseline/run_dp.py` | `runs/dp_baseline/` | §14 |
| Public-page charts | `scripts/build_site_figures.py` | `site/figures/` | — |

Every analysis and every training campaign evaluates on held-out seeds 0–29.

## Checks that retrain

| What | Train | Evaluate | Results in | Log |
| --- | --- | --- | --- | --- |
| Does the BC→SAC lead over pace repeat? Seeds 43, 44, 46; seed 42 stays the shipped zips | `runs/training_seeds/train.py` | `runs/training_seeds/eval.py` | `runs/training_seeds/` | §7 |

`train.py` takes any experiment config. A `bc:` block uses the behaviour-clone
warm start; anything else uses the standard trainer. Both write
`artifacts/training_seeds/<label>_s<seed>/final_model.zip`. The oversell cap is
not a third trainer — `eval.py` wraps the BC zip. The driver refuses seed 42,
held-out nights 0–29, the hyperparameter block 100–129, and the published
checkpoint directories.

## Configs that are not experiments

| Config | Role |
| --- | --- |
| `default.yaml` | Full schema; every other config overrides a subset of it |
| `algo_ppo/algo_sac/algo_td3.yaml` | Algorithm swap only |
| `demand_linear_legacy.yaml` | Demand swap only, for the A/B against `tree_elastic` |
| `experiment_tree_ppo.yaml` | The README quick-start demo; names its run with `--run-name` |
| `experiment_price_only_ppo.yaml` | Short price-only variant used by the tests and as the eval config for the shipped price-only PPO (its `control` block equals the long config's) |
| `experiment_price_only_sac.yaml` | Short price-only SAC variant (`optimize_1d` limit); an example, not a published run |

These deliberately set no `run_name`: a fragment merged onto `default.yaml` has no
single identity, and an ad-hoc run should fall back to the timestamped name rather
than silently overwrite a named experiment's directory.

## Naming rule for new experiments

1. Pick an ID that reads as `<algo>_<variant>` (`ppo_pace`, `sac_optimize1d`).
2. Set `train.run_name` to it, plus `model_dir` / `log_dir` for the family it belongs to.
   The config's filename is a family prefix (`experiment_price_only_*`,
   `tree_long_*`), not the ID; this table is the mapping.
3. Add a row here.

CLI `--run-name` still overrides the config, so a one-off sweep never collides with
a named experiment.
