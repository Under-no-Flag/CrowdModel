from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.g4_sahbo import (
    DEFAULT_BASELINE_CONFIG,
    G4EvaluationCache,
    GridSearchConfig,
    SAHBOConfig,
    run_grid_search,
    run_sahbo,
    save_g4_outputs,
)


def _parse_float_tuple(raw: str, *, expected_len: int | None = None) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}")
    return values


def _parse_three_float_tuple(raw: str) -> tuple[float, float, float]:
    values = _parse_float_tuple(raw, expected_len=3)
    return (values[0], values[1], values[2])


def _parse_state_tuple(raw: str) -> tuple[str, str, str]:
    values = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    if len(values) != 3:
        raise ValueError("Expected exactly three channel states, e.g. FREE,FREE,FREE")
    return (values[0], values[1], values[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 SA-HBO optimization and grid-search comparison.")
    parser.add_argument("--mode", choices=("sahbo", "grid", "both"), default="both")
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_CONFIG))
    parser.add_argument("--output-root", default="codes/results/g4_sahbo_vs_grid")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)

    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--proxy-top-k", type=int, default=3)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--initial-directions", default="FREE,FREE,FREE")
    parser.add_argument("--initial-eta", default="8,8,8")
    parser.add_argument("--eta-lower", type=float, default=1.0)
    parser.add_argument("--eta-upper", type=float, default=12.0)
    parser.add_argument("--eta-step-size", type=float, default=1.2)
    parser.add_argument("--eta-perturbation", type=float, default=0.8)
    parser.add_argument("--sahbo-max-evals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--grid-eta-values", default="1,4,8,12")
    parser.add_argument("--grid-max-evals", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.35)
    args = parser.parse_args()

    simulation_overrides: dict[str, object] = {}
    if args.steps is not None:
        simulation_overrides["steps"] = args.steps
    if args.save_every is not None:
        simulation_overrides["save_every"] = args.save_every
    if args.time_horizon is not None:
        simulation_overrides["time_horizon"] = args.time_horizon

    output_root = Path(args.output_root)
    evaluator = G4EvaluationCache(
        baseline_config=Path(args.baseline_config),
        output_root=output_root,
        simulation_overrides=simulation_overrides or None,
        beta=args.beta,
    )

    sahbo_result = None
    grid_result = None
    if args.mode in {"sahbo", "both"}:
        sahbo_result = run_sahbo(
            evaluator=evaluator,
            config=SAHBOConfig(
                iterations=args.iterations,
                proxy_top_k=args.proxy_top_k,
                neighborhood_radius=args.neighborhood_radius,
                initial_eta=_parse_three_float_tuple(args.initial_eta),
                initial_directions=_parse_state_tuple(args.initial_directions),
                eta_lower_bound=args.eta_lower,
                eta_upper_bound=args.eta_upper,
                eta_step_size=args.eta_step_size,
                eta_perturbation=args.eta_perturbation,
                max_evaluations=args.sahbo_max_evals,
                random_seed=args.seed,
                beta=args.beta,
            ),
        )

    if args.mode in {"grid", "both"}:
        grid_result = run_grid_search(
            evaluator=evaluator,
            config=GridSearchConfig(
                eta_values=_parse_float_tuple(args.grid_eta_values),
                max_evaluations=args.grid_max_evals,
                beta=args.beta,
            ),
        )

    save_g4_outputs(
        output_root=output_root,
        evaluator=evaluator,
        sahbo_result=sahbo_result,
        grid_result=grid_result,
    )


if __name__ == "__main__":
    main()
