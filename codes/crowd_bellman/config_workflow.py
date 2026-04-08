from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .compilers.config_compiler import compile_case, compile_scene
from .loaders.config_loader import (
    load_population_spec,
    load_route_spec,
    load_run_config,
    load_scene_spec,
)
from .metrics import save_json
from .runner import simulate_case
from .scenes import SimulationConfig


def run_from_config(
    config_path: Path,
    *,
    output_root: Path | None = None,
    simulation_overrides: dict[str, object] | None = None,
    write_root_summary: bool = True,
) -> dict[str, object]:
    run_spec = load_run_config(config_path)

    simulation = run_spec.simulation
    if simulation_overrides:
        simulation = SimulationConfig(**{**asdict(simulation), **simulation_overrides})

    scene_spec = load_scene_spec(run_spec.scene_path)
    population_spec = load_population_spec(run_spec.population_path)
    route_spec = load_route_spec(run_spec.routes_path)

    bundle = compile_scene(scene_spec=scene_spec, cfg=simulation)
    scene, case = compile_case(
        bundle=bundle,
        population_spec=population_spec,
        route_spec=route_spec,
    )

    run_output_root = output_root if output_root is not None else run_spec.output_root
    case_output_dir = run_output_root / case.case_id

    summary = simulate_case(
        cfg=simulation,
        scene=scene,
        case=case,
        output_dir=case_output_dir,
        objective_cfg=run_spec.objective,
    )
    summary["config_path"] = str(run_spec.config_path)
    summary["scene_path"] = str(run_spec.scene_path)
    summary["population_path"] = str(run_spec.population_path)
    summary["routes_path"] = str(run_spec.routes_path)

    if write_root_summary:
        save_json(
            run_output_root / "config_run_summary.json",
            {
                "config_path": str(run_spec.config_path),
                "scene_path": str(run_spec.scene_path),
                "population_path": str(run_spec.population_path),
                "routes_path": str(run_spec.routes_path),
                "output_root": str(run_output_root),
                "simulation": asdict(simulation),
                "objective": asdict(run_spec.objective),
                "summary": summary,
            },
        )

    return summary
