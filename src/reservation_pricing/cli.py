"""Console entrypoints: rprl-train, rprl-eval, rprl-tune, rprl-baselines, rprl-fit-demand, rprl-bc-sac."""

from __future__ import annotations

import argparse
import json


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="YAML experiment config (merged onto configs/default.yaml)",
    )


def train_main(argv: list[str] | None = None) -> None:
    from reservation_pricing.train import train

    p = argparse.ArgumentParser(description="Train reservation pricing agent from config")
    _add_config(p)
    p.add_argument("--algo", choices=["ppo", "sac", "td3"], default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    meta = train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        algo=args.algo,
        run_name=args.run_name,
        out_dir=args.out_dir,
    )
    keys = ("run_name", "algo", "seed", "total_timesteps", "model_path", "demand_kind")
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2))


def eval_main(argv: list[str] | None = None) -> None:
    from reservation_pricing.evaluate import run_comparison, run_soft_aware_comparison

    p = argparse.ArgumentParser(description="Evaluate baselines and/or RL model")
    _add_config(p)
    p.add_argument("--model", default=None, help="Path to SB3 .zip model")
    p.add_argument("--algo", choices=["ppo", "sac", "td3"], default=None)
    p.add_argument("--episodes", type=int, default=None, help="Overrides eval.n_episodes")
    p.add_argument("--no-held-out", action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--out-dir", default="runs/eval")
    p.add_argument(
        "--soft-aware",
        action="store_true",
        help="Soft-day-aware stratified eval (writes soft_aware_comparison.md)",
    )
    p.add_argument(
        "--interval",
        action="store_true",
        help="Paired bootstrap interval on score_aware vs --baseline-policy (requires --soft-aware)",
    )
    p.add_argument(
        "--baseline-policy",
        default=None,
        help="Policy name the paired interval is relative to (required with --interval)",
    )
    args = p.parse_args(argv)
    if args.interval and not args.soft_aware:
        p.error("--interval requires --soft-aware")
    if args.interval and not args.baseline_policy:
        p.error("--interval requires --baseline-policy")

    kwargs = dict(
        config_path=args.config,
        model_path=args.model,
        algo=args.algo,
        n_episodes=args.episodes,
        held_out=not args.no_held_out,
        out_dir=args.out_dir,
        include_baselines=not args.no_baselines,
    )
    if args.soft_aware and args.interval:
        kwargs["interval"] = True
        kwargs["baseline_policy"] = args.baseline_policy
    runner = run_soft_aware_comparison if args.soft_aware else run_comparison
    out = runner(**kwargs)
    print(out["table"].to_string(index=False))
    print(f"\nWrote results under {args.out_dir}")


def baselines_main(argv: list[str] | None = None) -> None:
    from reservation_pricing.evaluate import run_comparison

    p = argparse.ArgumentParser(description="Run classical baselines only")
    _add_config(p)
    p.add_argument("--episodes", type=int, default=None, help="Overrides eval.n_episodes")
    p.add_argument("--out-dir", default="runs/baselines")
    p.add_argument("--no-held-out", action="store_true")
    args = p.parse_args(argv)

    out = run_comparison(
        config_path=args.config,
        model_path=None,
        n_episodes=args.episodes,
        held_out=not args.no_held_out,
        out_dir=args.out_dir,
        include_baselines=True,
    )
    print(out["table"].to_string(index=False))
    print(f"\nWrote results under {args.out_dir}")


def tune_main(argv: list[str] | None = None) -> None:
    from reservation_pricing.tune import run_tune

    p = argparse.ArgumentParser(description="Tune hyperparameters (grid or Optuna)")
    _add_config(p)
    p.add_argument("--method", choices=["grid", "optuna"], default=None)
    p.add_argument("--timesteps", type=int, default=None, help="Timesteps per trial")
    p.add_argument("--trials", type=int, default=None, help="Optuna trials (grid ignores it)")
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    kwargs: dict = {}
    if args.timesteps is not None:
        kwargs["timesteps_per_trial"] = args.timesteps
    if args.eval_episodes is not None:
        kwargs["eval_episodes"] = args.eval_episodes
    if args.out_dir is not None:
        kwargs["out_dir"] = args.out_dir
    if args.trials is not None:
        kwargs["n_trials"] = args.trials

    result = run_tune(config_path=args.config, method=args.method, **kwargs)
    print(
        json.dumps(
            {"root": result["root"], "best": result.get("best"), "summary": result.get("summary")},
            indent=2,
            default=str,
        )
    )


def fit_demand_main(argv: list[str] | None = None) -> None:
    from reservation_pricing.demand.synthesize import fit_demand_main as _main

    _main(argv)


def bc_sac_main(argv: list[str] | None = None) -> None:
    """BC warm-start from expert baseline then SAC/TD3 fine-tune."""
    from reservation_pricing.train.bc_finetune import bc_finetune

    p = argparse.ArgumentParser(description="BC warm-start + SAC/TD3 fine-tune from config")
    _add_config(p)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    meta = bc_finetune(
        config_path=args.config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        run_name=args.run_name,
        out_dir=args.out_dir,
    )
    keys = (
        "run_name",
        "algo",
        "seed",
        "total_timesteps",
        "model_path",
        "bc_only_path",
        "demand_kind",
        "bc",
    )
    print(json.dumps({k: meta[k] for k in keys if k in meta}, indent=2, default=str))


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    rest = sys.argv[2:]
    {
        "train": train_main,
        "eval": eval_main,
        "tune": tune_main,
        "baselines": baselines_main,
        "fit-demand": fit_demand_main,
        "bc-sac": bc_sac_main,
    }[cmd](rest)
