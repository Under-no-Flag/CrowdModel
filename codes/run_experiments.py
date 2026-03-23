from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from crowd_bellman.reporting import generate_section_5_1_report
from crowd_bellman.runner import run_cases
from crowd_bellman.scenes import SimulationConfig
from crowd_bellman.validation import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiments and save all outputs under one results folder.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["case1_baseline", "case2_middle_guided"],
    )
    parser.add_argument("--output-root", default="codes/results")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    args = parser.parse_args()

    cfg = SimulationConfig()
    if args.steps is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "steps": args.steps})
    if args.save_every is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "save_every": args.save_every})
    if args.time_horizon is not None:
        cfg = SimulationConfig(**{**asdict(cfg), "time_horizon": args.time_horizon})

    output_root = Path(args.output_root)
    run_cases(cfg=cfg, cases=tuple(args.cases), output_root=output_root)

    if not args.skip_validation:
        run_validation(output_root / "unidirectional_validation")
    if not args.skip_report:
        generate_section_5_1_report(output_root)


if __name__ == "__main__":
    main()
