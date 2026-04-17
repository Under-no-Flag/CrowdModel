from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
from typing import Callable

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


StepObserverFactory = Callable[..., Callable[[dict[str, object]], None] | None]


def _write_config_snapshot(
    *,
    config_path: Path,
    scene_path: Path,
    population_path: Path,
    routes_path: Path,
    snapshot_dir: Path,
    output_root: Path,
    case_output_dir: Path,
    simulation: SimulationConfig,
    objective: dict[str, object],
) -> dict[str, object]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    source_files = {
        "run": config_path.resolve(),
        "scene": scene_path.resolve(),
        "population": population_path.resolve(),
        "routes": routes_path.resolve(),
    }
    copied_files: dict[str, str] = {}
    for label, source_path in source_files.items():
        destination_path = snapshot_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied_files[label] = str(destination_path)

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_root": str(output_root),
        "case_output_dir": str(case_output_dir),
        "source_files": {label: str(path) for label, path in source_files.items()},
        "copied_files": copied_files,
        "effective_simulation": asdict(simulation),
        "effective_objective": objective,
    }
    save_json(snapshot_dir / "snapshot_manifest.json", manifest)
    return {
        "directory": str(snapshot_dir),
        "manifest_path": str(snapshot_dir / "snapshot_manifest.json"),
        "files": copied_files,
    }


def run_from_config(
    config_path: Path,
    *,
    output_root: Path | None = None,
    simulation_overrides: dict[str, object] | None = None,
    write_root_summary: bool = True,
    step_observer_factory: StepObserverFactory | None = None,
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

    run_output_root = (output_root if output_root is not None else run_spec.output_root).resolve()
    case_output_dir = run_output_root / case.case_id
    step_observer = (
        step_observer_factory(
            config_path=run_spec.config_path,
            run_spec=run_spec,
            bundle=bundle,
            scene=scene,
            case=case,
            output_root=run_output_root,
            case_output_dir=case_output_dir,
            simulation=simulation,
        )
        if step_observer_factory is not None
        else None
    )

    summary = simulate_case(
        cfg=simulation,
        scene=scene,
        case=case,
        output_dir=case_output_dir,
        objective_cfg=run_spec.objective,
        step_observer=step_observer,
    )
    summary["config_path"] = str(run_spec.config_path)
    summary["scene_path"] = str(run_spec.scene_path)
    summary["population_path"] = str(run_spec.population_path)
    summary["routes_path"] = str(run_spec.routes_path)
    summary["config_snapshot"] = _write_config_snapshot(
        config_path=run_spec.config_path,
        scene_path=run_spec.scene_path,
        population_path=run_spec.population_path,
        routes_path=run_spec.routes_path,
        snapshot_dir=case_output_dir / "config_snapshot",
        output_root=run_output_root,
        case_output_dir=case_output_dir,
        simulation=simulation,
        objective=summary["objective_config"],
    )
    summary["metadata"] = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "output_root": str(run_output_root),
        "case_output_dir": str(case_output_dir),
        "config_source_files": {
            "run": str(run_spec.config_path),
            "scene": str(run_spec.scene_path),
            "population": str(run_spec.population_path),
            "routes": str(run_spec.routes_path),
        },
    }
    save_json(case_output_dir / "summary.json", summary)

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
                "config_snapshot": summary["config_snapshot"],
                "summary": summary,
            },
        )

    return summary
