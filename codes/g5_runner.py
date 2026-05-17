from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.g5_hcmbo import DEFAULT_BASELINE_CONFIG, HCMBOConfig, run_hcmbo_experiment
from crowd_bellman.plotting import parse_density_contour_levels


def _simulation_overrides(
    *,
    steps: int,
    time_horizon: float,
    bellman_every: int,
    save_every: int,
    density_contour_levels: str,
) -> dict[str, object]:
    return {
        "steps": int(steps),
        "time_horizon": float(time_horizon),
        "bellman_every": int(bellman_every),
        "save_every": int(save_every),
        "density_contour_levels": parse_density_contour_levels(density_contour_levels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G5 V2 HCMBO optimization for z=(s,q).")
    parser.add_argument("--baseline-config", default=str(DEFAULT_BASELINE_CONFIG))
    parser.add_argument("--output-root", default="codes/results/g5_hcmbo_quick")
    parser.add_argument("--time-segments", type=int, default=2)
    parser.add_argument("--direction-candidate-limit", type=int, default=8)
    parser.add_argument("--shortlist-size", type=int, default=2)
    parser.add_argument("--initial-samples", type=int, default=4)
    parser.add_argument("--bo-iterations", type=int, default=2)
    parser.add_argument("--bo-candidate-pool", type=int, default=32)
    parser.add_argument("--dfo-evaluations", type=int, default=2)
    parser.add_argument("--high-fidelity-top-k", type=int, default=3)
    parser.add_argument("--random-search-evaluations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)

    parser.add_argument("--lambda-j1", type=float, default=1.0)
    parser.add_argument("--lambda-j2", type=float, default=1.0)
    parser.add_argument("--lambda-j5", type=float, default=1.0)
    parser.add_argument("--lambda-jb", type=float, default=1.0)
    parser.add_argument("--lambda-jr", type=float, default=0.1)
    parser.add_argument("--j2-scale", type=float, default=0.001)

    parser.add_argument("--screen-steps", type=int, default=80)
    parser.add_argument("--screen-time-horizon", type=float, default=5.0)
    parser.add_argument("--screen-bellman-every", type=int, default=4)
    parser.add_argument("--opt-steps", type=int, default=120)
    parser.add_argument("--opt-time-horizon", type=float, default=8.0)
    parser.add_argument("--opt-bellman-every", type=int, default=4)
    parser.add_argument("--hf-steps", type=int, default=180)
    parser.add_argument("--hf-time-horizon", type=float, default=10.0)
    parser.add_argument("--hf-bellman-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100000)
    parser.add_argument("--density-contour-levels", default="off")
    args = parser.parse_args()

    config = HCMBOConfig(
        time_segments=int(args.time_segments),
        direction_candidate_limit=int(args.direction_candidate_limit),
        shortlist_size=int(args.shortlist_size),
        initial_samples=int(args.initial_samples),
        bo_iterations=int(args.bo_iterations),
        bo_candidate_pool=int(args.bo_candidate_pool),
        dfo_evaluations=int(args.dfo_evaluations),
        high_fidelity_top_k=int(args.high_fidelity_top_k),
        random_search_evaluations=int(args.random_search_evaluations),
        random_seed=int(args.seed),
        lambda_j1=float(args.lambda_j1),
        lambda_j2=float(args.lambda_j2),
        lambda_j5=float(args.lambda_j5),
        lambda_jb=float(args.lambda_jb),
        lambda_jr=float(args.lambda_jr),
        j2_scale=float(args.j2_scale),
    )
    payload = run_hcmbo_experiment(
        baseline_config=Path(args.baseline_config),
        output_root=Path(args.output_root),
        config=config,
        screen_overrides=_simulation_overrides(
            steps=args.screen_steps,
            time_horizon=args.screen_time_horizon,
            bellman_every=args.screen_bellman_every,
            save_every=args.save_every,
            density_contour_levels=args.density_contour_levels,
        ),
        optimization_overrides=_simulation_overrides(
            steps=args.opt_steps,
            time_horizon=args.opt_time_horizon,
            bellman_every=args.opt_bellman_every,
            save_every=args.save_every,
            density_contour_levels=args.density_contour_levels,
        ),
        high_fidelity_overrides=_simulation_overrides(
            steps=args.hf_steps,
            time_horizon=args.hf_time_horizon,
            bellman_every=args.hf_bellman_every,
            save_every=args.save_every,
            density_contour_levels=args.density_contour_levels,
        ),
    )
    best = payload["best_high_fidelity"]
    print(f"G5 HCMBO summary: {payload['outputs']['report']}")
    print(f"best_objective={best['objective_value']}")
    best_directions = ",".join(
        f"{name}:{best.get(f'direction_{name}')}"
        for name in ("top", "middle", "lower_middle", "bottom")
    )
    print(f"best_directions={best_directions}")
    print(f"top_candidates={payload['outputs']['top_candidates']}")


if __name__ == "__main__":
    main()
