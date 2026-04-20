from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g3_behavior import G3BehaviorCollector, build_g3_behavior_report
from crowd_bellman.metrics import save_json


G3_CONFIGS = (
    Path("codes/scenes/examples/tour_hardcoded/run_single_stage_approx.toml"),
    Path("codes/scenes/examples/tour_hardcoded/run_uniform_preference.toml"),
    Path("codes/scenes/examples/tour_hardcoded/run.toml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G3 behavior-layer necessity batch.")
    parser.add_argument("--output-root", default="codes/results/g3_behavior_layer")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)
    args = parser.parse_args()

    simulation_overrides: dict[str, object] = {}
    if args.steps is not None:
        simulation_overrides["steps"] = args.steps
    if args.save_every is not None:
        simulation_overrides["save_every"] = args.save_every
    if args.time_horizon is not None:
        simulation_overrides["time_horizon"] = args.time_horizon

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    collectors: dict[str, tuple[G3BehaviorCollector, Path]] = {}

    def observer_factory(**kwargs: object):
        case = kwargs["case"]
        bundle = kwargs["bundle"]
        simulation = kwargs["simulation"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        collector = G3BehaviorCollector(
            case_id=str(case.case_id),
            title=str(case.title),
            walkable=case.walkable,
            region_masks=bundle.region_masks,
            group_names={key: group.name for key, group in case.groups.items()},
            dx=float(simulation.dx),
        )
        collectors[str(case.case_id)] = (collector, case_output_dir)
        return collector.observe

    summaries: list[dict[str, object]] = []
    for config_path in G3_CONFIGS:
        summaries.append(
            run_from_config(
                config_path=config_path,
                output_root=output_root,
                simulation_overrides=simulation_overrides or None,
                write_root_summary=False,
                step_observer_factory=observer_factory,
            )
        )

    behavior_summaries: list[dict[str, object]] = []
    for summary in summaries:
        collector, case_output_dir = collectors[str(summary["case_id"])]
        behavior_summaries.append(collector.save_case_outputs(case_output_dir))

    payload = {
        "experiment_group": "G3",
        "config_paths": [str(path.resolve()) for path in G3_CONFIGS],
        "cases": summaries,
        "behavior_cases": behavior_summaries,
    }
    save_json(output_root / "comparison_summary.json", payload)
    build_g3_behavior_report(
        output_root=output_root,
        case_summaries=summaries,
        behavior_summaries=behavior_summaries,
    )


if __name__ == "__main__":
    main()
