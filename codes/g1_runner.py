from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g1_mechanism import CaseBehaviorCollector, build_g1_mechanism_report
from crowd_bellman.g1_u_bidirectional import BidirectionalUCollector, build_bidirectional_u_report
from crowd_bellman.metrics import save_json
from crowd_bellman.plotting import save_comparison_plot


THREE_CHANNEL_CONFIGS = (
    Path("codes/scenes/examples/three_channel_hardcoded/run_baseline.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_m_only.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_u_only.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_middle_guided.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_top_guided.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_bottom_guided.toml"),
)

BRIDGE_CONFIG = Path("codes/scenes/examples/tour_hardcoded/run.toml")
U_BIDIRECTIONAL_CONFIGS = (
    Path("codes/scenes/examples/three_channel_bidirectional/run_baseline.toml"),
    Path("codes/scenes/examples/three_channel_bidirectional/run_u_middle_east.toml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reframed G1 mechanism-validation experiment and build behavior-aware outputs.")
    parser.add_argument("--output-root", default="codes/results/g1_mechanism_reframed")
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

    behavior_collectors: dict[str, tuple[CaseBehaviorCollector, Path]] = {}
    bidirectional_collectors: dict[str, tuple[BidirectionalUCollector, Path]] = {}

    def observer_factory(**kwargs: object):
        config_path = Path(str(kwargs["config_path"]))
        case = kwargs["case"]
        simulation = kwargs["simulation"]
        bundle = kwargs["bundle"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        if "three_channel_hardcoded" not in str(config_path).replace("\\", "/"):
            return None
        collector = CaseBehaviorCollector(
            case_id=str(case.case_id),
            title=str(case.title),
            walkable=case.walkable,
            region_masks=bundle.region_masks,
            channel_masks=case.channel_masks,
            time_horizon=float(simulation.time_horizon),
        )
        behavior_collectors[str(case.case_id)] = (collector, case_output_dir)
        return collector.observe

    def bidirectional_observer_factory(**kwargs: object):
        config_path = Path(str(kwargs["config_path"]))
        case = kwargs["case"]
        simulation = kwargs["simulation"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        if "three_channel_bidirectional" not in str(config_path).replace("\\", "/"):
            return None
        collector = BidirectionalUCollector(
            case_id=str(case.case_id),
            title=str(case.title),
            walkable=case.walkable,
            channel_masks=case.channel_masks,
            group_names={key: group.name for key, group in case.groups.items()},
            dx=float(simulation.dx),
            time_horizon=float(simulation.time_horizon),
        )
        bidirectional_collectors[str(case.case_id)] = (collector, case_output_dir)
        return collector.observe

    summaries: list[dict[str, object]] = []
    for config_path in THREE_CHANNEL_CONFIGS:
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
        case_id = str(summary["case_id"])
        collector_entry = behavior_collectors.get(case_id)
        if collector_entry is None:
            continue
        collector, case_output_dir = collector_entry
        behavior_summaries.append(collector.save_case_outputs(case_output_dir))

    bridge_summary = run_from_config(
        config_path=BRIDGE_CONFIG,
        output_root=output_root,
        simulation_overrides=simulation_overrides or None,
        write_root_summary=False,
    )

    u_bidirectional_summaries: list[dict[str, object]] = []
    for config_path in U_BIDIRECTIONAL_CONFIGS:
        u_bidirectional_summaries.append(
            run_from_config(
                config_path=config_path,
                output_root=output_root,
                simulation_overrides=simulation_overrides or None,
                write_root_summary=False,
                step_observer_factory=bidirectional_observer_factory,
            )
        )

    u_bidirectional_behavior: list[dict[str, object]] = []
    for summary in u_bidirectional_summaries:
        case_id = str(summary["case_id"])
        collector_entry = bidirectional_collectors.get(case_id)
        if collector_entry is None:
            continue
        collector, case_output_dir = collector_entry
        u_bidirectional_behavior.append(collector.save_case_outputs(case_output_dir))

    payload = {
        "experiment_group": "G1",
        "design_version": "2026-04-17_two_layer",
        "config_paths": [str(path.resolve()) for path in THREE_CHANNEL_CONFIGS],
        "cases": summaries,
        "behavior_cases": behavior_summaries,
        "bridge_case": bridge_summary,
        "u_bidirectional_config_paths": [str(path.resolve()) for path in U_BIDIRECTIONAL_CONFIGS],
        "u_bidirectional_cases": u_bidirectional_summaries,
        "u_bidirectional_behavior_cases": u_bidirectional_behavior,
    }
    save_json(output_root / "comparison_summary.json", payload)
    save_comparison_plot(output_root / "comparison.png", summaries)
    build_g1_mechanism_report(
        output_root=output_root,
        case_summaries=summaries,
        behavior_summaries=behavior_summaries,
        bridge_summary=bridge_summary,
    )
    build_bidirectional_u_report(
        output_root=output_root,
        case_summaries=u_bidirectional_summaries,
        validation_summaries=u_bidirectional_behavior,
    )


if __name__ == "__main__":
    main()
