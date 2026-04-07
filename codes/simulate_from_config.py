from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from crowd_bellman.compilers.config_compiler import compile_case, compile_scene
from crowd_bellman.loaders.config_loader import (
    load_population_spec,
    load_route_spec,
    load_run_config,
    load_scene_spec,
)
from crowd_bellman.metrics import save_json
from crowd_bellman.runner import simulate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a config-driven crowd simulation from TOML files.")
    parser.add_argument("--config", required=True, help="Path to the run TOML file")
    args = parser.parse_args()

    run_spec = load_run_config(Path(args.config))
    scene_spec = load_scene_spec(run_spec.scene_path)
    population_spec = load_population_spec(run_spec.population_path)
    route_spec = load_route_spec(run_spec.routes_path)

    bundle = compile_scene(scene_spec=scene_spec, cfg=run_spec.simulation)
    scene, case = compile_case(
        bundle=bundle,
        population_spec=population_spec,
        route_spec=route_spec,
    )

    case_output_dir = run_spec.output_root / case.case_id
    summary = simulate_case(
        cfg=run_spec.simulation,
        scene=scene,
        case=case,
        output_dir=case_output_dir,
        objective_cfg=run_spec.objective,
    )
    save_json(
        run_spec.output_root / "config_run_summary.json",
        {
            "config_path": str(run_spec.config_path),
            "scene_path": str(run_spec.scene_path),
            "population_path": str(run_spec.population_path),
            "routes_path": str(run_spec.routes_path),
            "output_root": str(run_spec.output_root),
            "simulation": asdict(run_spec.simulation),
            "objective": asdict(run_spec.objective),
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
