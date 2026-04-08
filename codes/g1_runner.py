from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.metrics import save_json
from crowd_bellman.plotting import save_comparison_plot


G1_CONFIGS = (
    Path("codes/scenes/examples/three_channel_hardcoded/run_baseline.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_m_only.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_u_only.toml"),
    Path("codes/scenes/examples/three_channel_hardcoded/run_middle_guided.toml"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G1 mechanism-validation batch and build one comparison summary.")
    parser.add_argument("--output-root", default="codes/results/g1_mechanism")
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

    summaries: list[dict[str, object]] = []
    for config_path in G1_CONFIGS:
        summaries.append(
            run_from_config(
                config_path=config_path,
                output_root=output_root,
                simulation_overrides=simulation_overrides or None,
                write_root_summary=False,
            )
        )

    payload = {
        "experiment_group": "G1",
        "config_paths": [str(path.resolve()) for path in G1_CONFIGS],
        "cases": summaries,
    }
    save_json(output_root / "comparison_summary.json", payload)
    save_comparison_plot(output_root / "comparison.png", summaries)


if __name__ == "__main__":
    main()
