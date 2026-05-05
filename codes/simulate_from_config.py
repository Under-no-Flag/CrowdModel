from __future__ import annotations

import argparse
from pathlib import Path

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.plotting import parse_density_contour_levels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven crowd simulation from TOML files.")
    parser.add_argument("--config", required=True, help="Path to the run TOML file")
    parser.add_argument(
        "--density-contour-levels",
        default=None,
        help="Comma-separated density contour values, an integer count, or 'off'.",
    )
    args = parser.parse_args()
    simulation_overrides: dict[str, object] = {}
    if args.density_contour_levels is not None:
        simulation_overrides["density_contour_levels"] = parse_density_contour_levels(args.density_contour_levels)
    run_from_config(Path(args.config), simulation_overrides=simulation_overrides or None)


if __name__ == "__main__":
    main()
