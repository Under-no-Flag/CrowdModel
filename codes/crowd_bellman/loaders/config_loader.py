from __future__ import annotations

from pathlib import Path
import tomllib

from ..config import ObjectiveConfig
from ..scenes import SimulationConfig
from ..spec.experiment_spec import RunConfigSpec
from ..spec.population_spec import InflowGroupSpec, InitialGroupSpec, PopulationSpec
from ..spec.route_spec import CaseRouteSpec, ControlSpec, StageSpec, TransitionTargetSpec
from ..spec.scene_spec import ChannelSpec, NamedRegionSelectionSpec, RectRegionSpec, SceneSpec


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_relative_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _as_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"{field_name} must be a string or a list of strings")


def _as_optional_string_tuple(value: object, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _as_string_tuple(value, field_name)


def _as_point(value: object, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int | float) for item in value)
    ):
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{field_name} must be a list with exactly two numeric values")


def _as_group_key(value: object, field_name: str) -> tuple[int, int]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int | float) for item in value)
    ):
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{field_name} must be a list with exactly two numeric values")


def load_run_config(path: Path) -> RunConfigSpec:
    path = path.resolve()
    raw = _load_toml(path)
    base_dir = path.parent

    sim_table = raw.get("simulation", {})
    if not isinstance(sim_table, dict):
        raise ValueError("[simulation] must be a table")
    simulation = SimulationConfig(**sim_table)

    objective_table = raw.get("objective", {})
    if not isinstance(objective_table, dict):
        raise ValueError("[objective] must be a table")
    objective = ObjectiveConfig(**objective_table)

    scene_table = raw.get("scene", {})
    population_table = raw.get("population", {})
    routes_table = raw.get("routes", {})
    outputs_table = raw.get("outputs", {})
    if not isinstance(scene_table, dict) or "file" not in scene_table:
        raise ValueError("[scene] must contain file")
    if not isinstance(population_table, dict) or "file" not in population_table:
        raise ValueError("[population] must contain file")
    if not isinstance(routes_table, dict) or "file" not in routes_table:
        raise ValueError("[routes] must contain file")
    if outputs_table and not isinstance(outputs_table, dict):
        raise ValueError("[outputs] must be a table when provided")

    output_root_raw = outputs_table.get("output_root", "codes/results/from_config")
    if not isinstance(output_root_raw, str):
        raise ValueError("outputs.output_root must be a string")

    return RunConfigSpec(
        config_path=path,
        simulation=simulation,
        objective=objective,
        scene_path=_resolve_relative_path(base_dir, str(scene_table["file"])),
        population_path=_resolve_relative_path(base_dir, str(population_table["file"])),
        routes_path=_resolve_relative_path(base_dir, str(routes_table["file"])),
        output_root=_resolve_relative_path(base_dir, output_root_raw),
    )


def load_scene_spec(path: Path) -> SceneSpec:
    raw_doc = _load_toml(path)
    raw = raw_doc.get("scene", raw_doc)
    if not isinstance(raw, dict):
        raise ValueError("scene file must contain a table")

    region_specs: list[RectRegionSpec] = []
    for item in raw.get("regions", []):
        if not isinstance(item, dict):
            raise ValueError("scene.regions entries must be tables")
        region_specs.append(
            RectRegionSpec(
                name=str(item["name"]),
                x0=int(item["x0"]),
                x1=int(item["x1"]),
                y0=int(item["y0"]),
                y1=int(item["y1"]),
            )
        )

    exit_specs: list[NamedRegionSelectionSpec] = []
    for item in raw.get("exits", []):
        if not isinstance(item, dict):
            raise ValueError("scene.exits entries must be tables")
        exit_specs.append(
            NamedRegionSelectionSpec(
                name=str(item["name"]),
                regions=_as_string_tuple(item.get("regions", item.get("region")), f"exit {item.get('name', '')}"),
            )
        )

    channel_specs: list[ChannelSpec] = []
    for item in raw.get("channels", []):
        if not isinstance(item, dict):
            raise ValueError("scene.channels entries must be tables")
        probe_x = item.get("probe_x")
        channel_specs.append(
            ChannelSpec(
                name=str(item["name"]),
                regions=_as_string_tuple(item.get("regions", item.get("region")), f"channel {item.get('name', '')}"),
                probe_x=int(probe_x) if probe_x is not None else None,
            )
        )

    return SceneSpec(
        block_boundaries=bool(raw.get("block_boundaries", True)),
        regions=tuple(region_specs),
        obstacles=_as_string_tuple(raw.get("obstacles", []), "scene.obstacles"),
        exits=tuple(exit_specs),
        channels=tuple(channel_specs),
    )


def load_population_spec(path: Path) -> PopulationSpec:
    raw_doc = _load_toml(path)
    raw = raw_doc.get("population", raw_doc)
    if not isinstance(raw, dict):
        raise ValueError("population file must contain a table")

    groups: list[InitialGroupSpec] = []
    for item in raw.get("initial_groups", []):
        if not isinstance(item, dict):
            raise ValueError("population.initial_groups entries must be tables")
        groups.append(
            InitialGroupSpec(
                group_id=str(item["group_id"]),
                stage_id=str(item["stage_id"]),
                region=str(item["region"]),
                density=float(item["density"]),
            )
        )

    inflow_groups: list[InflowGroupSpec] = []
    for item in raw.get("inflow_groups", []):
        if not isinstance(item, dict):
            raise ValueError("population.inflow_groups entries must be tables")
        time_end = item.get("time_end")
        rho_cap = item.get("rho_cap")
        inflow_groups.append(
            InflowGroupSpec(
                group_id=str(item["group_id"]),
                stage_id=str(item["stage_id"]),
                region=str(item["region"]),
                rate=float(item["rate"]),
                time_start=float(item.get("time_start", 0.0)),
                time_end=float(time_end) if time_end is not None else None,
                rho_cap=float(rho_cap) if rho_cap is not None else None,
            )
        )
    return PopulationSpec(initial_groups=tuple(groups), inflow_groups=tuple(inflow_groups))


def load_route_spec(path: Path) -> CaseRouteSpec:
    raw_doc = _load_toml(path)
    raw = raw_doc.get("routes", raw_doc)
    if not isinstance(raw, dict):
        raise ValueError("routes file must contain a table")
    case_table = raw.get("case", {})
    if not isinstance(case_table, dict):
        raise ValueError("[case] must be a table in routes file")

    stage_specs: list[StageSpec] = []
    for item in raw.get("stages", []):
        if not isinstance(item, dict):
            raise ValueError("stages entries must be tables")

        controls: list[ControlSpec] = []
        for control in item.get("controls", []):
            if not isinstance(control, dict):
                raise ValueError("stage controls entries must be tables")
            controls.append(
                ControlSpec(
                    mode=str(control.get("mode", "identity")),
                    region=str(control["region"]) if control.get("region") is not None else None,
                    alpha=float(control["alpha"]) if control.get("alpha") is not None else None,
                    beta=float(control["beta"]) if control.get("beta") is not None else None,
                    value=float(control["value"]) if control.get("value") is not None else None,
                    direction=str(control["direction"]) if control.get("direction") is not None else None,
                    target_region=str(control["target_region"]) if control.get("target_region") is not None else None,
                    target_point=_as_point(control.get("target_point"), "control.target_point"),
                    allowed_directions=_as_optional_string_tuple(control.get("allowed_directions"), "control.allowed_directions"),
                )
            )

        targets: list[TransitionTargetSpec] = []
        for target in item.get("targets", []):
            if not isinstance(target, dict):
                raise ValueError("stage targets entries must be tables")
            targets.append(
                TransitionTargetSpec(
                    stage_id=str(target["stage_id"]),
                    probability=float(target["probability"]),
                )
            )

        stage_specs.append(
            StageSpec(
                stage_id=str(item["stage_id"]),
                group_key=_as_group_key(item["group_key"], "stage.group_key"),
                goal_regions=_as_string_tuple(item.get("goal_regions", item.get("goal_region")), "stage.goal_region"),
                sink_regions=_as_optional_string_tuple(item.get("sink_regions", item.get("sink_region")), "stage.sink_region"),
                allowed_directions=_as_optional_string_tuple(item.get("allowed_directions"), "stage.allowed_directions"),
                controls=tuple(controls),
                decision_regions=_as_optional_string_tuple(item.get("decision_regions", item.get("decision_region")), "stage.decision_region"),
                next_stage=str(item["next_stage"]) if item.get("next_stage") is not None else None,
                kappa=float(item.get("kappa", 1.0)),
                targets=tuple(targets),
                transition_direction=str(item.get("transition_direction", "stop")),
            )
        )

    return CaseRouteSpec(
        case_id=str(case_table.get("case_id", "config_case")),
        title=str(case_table.get("title", "Config-driven case")),
        stages=tuple(stage_specs),
    )
