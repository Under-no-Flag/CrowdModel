from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from crowd_bellman.config import ObjectiveConfig, SearchConfig
from crowd_bellman.scenes import SimulationConfig
from crowd_bellman.search import run_parameter_search


def _parse_probability_triplet(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "split candidate must contain exactly three comma-separated values"
        )
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split candidate values must be numeric") from exc
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run first-stage parameter search for strategy, eta, and split probabilities.",
    )
    parser.add_argument("--output-root", default="codes/results/parameter_search")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    parser.add_argument("--lambda-j1", type=float, default=1.0)
    parser.add_argument("--lambda-j2", type=float, default=1.0)
    parser.add_argument("--lambda-j5", type=float, default=1.0)
    parser.add_argument("--rho-safe", type=float, default=3.5)
    parser.add_argument("--normalize-objective", action="store_true")
    parser.add_argument("--j1-scale", type=float, default=1.0)
    parser.add_argument("--j2-scale", type=float, default=1.0)
    parser.add_argument("--j5-scale", type=float, default=1.0)
    parser.add_argument("--objective-name", default="day2_search")
    parser.add_argument("--eta-values", nargs="+", type=float, default=None)
    parser.add_argument(
        "--split-candidates",
        nargs="+",
        type=_parse_probability_triplet,
        default=None,
        help="One or more triplets like 0.2,0.3,0.5",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    sim_cfg = SimulationConfig()
    if args.steps is not None:
        sim_cfg = SimulationConfig(**{**asdict(sim_cfg), "steps": args.steps})
    if args.save_every is not None:
        sim_cfg = SimulationConfig(**{**asdict(sim_cfg), "save_every": args.save_every})
    if args.time_horizon is not None:
        sim_cfg = SimulationConfig(**{**asdict(sim_cfg), "time_horizon": args.time_horizon})

    objective_cfg = ObjectiveConfig(
        lambda_j1=args.lambda_j1,
        lambda_j2=args.lambda_j2,
        lambda_j5=args.lambda_j5,
        rho_safe=args.rho_safe,
        use_normalized_terms=args.normalize_objective,
        j1_scale=args.j1_scale,
        j2_scale=args.j2_scale,
        j5_scale=args.j5_scale,
        name=args.objective_name,
    )

    search_cfg_kwargs: dict[str, object] = {"top_k": args.top_k}
    if args.eta_values is not None:
        search_cfg_kwargs["eta_values"] = tuple(float(value) for value in args.eta_values)
    if args.split_candidates is not None:
        search_cfg_kwargs["split_probability_candidates"] = tuple(args.split_candidates)
    search_cfg = SearchConfig(**search_cfg_kwargs)

    run_parameter_search(
        cfg=sim_cfg,
        objective_cfg=objective_cfg,
        search_cfg=search_cfg,
        output_root=Path(args.output_root),
    )


if __name__ == "__main__":
    main()
