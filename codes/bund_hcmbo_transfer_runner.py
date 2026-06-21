from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from pathlib import Path

from bund_capacity_response_runner import (
    DEFAULT_BASE_DIR,
    _format_scalar,
    _dump_routes_toml,
    _dump_run_toml,
    _load_toml,
    _resolve_path,
)
from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.field_recorder import make_field_data_observer_factory
from crowd_bellman.metrics import save_json


G6_CHANNELS = ("top", "middle", "lower_middle", "bottom")
DEFAULT_CHANNEL_MAP = {
    "top": "channel_1",
    "middle": "channel_2",
    "lower_middle": "channel_3",
    "bottom": "channel_4",
}
DEFAULT_G6_MANIFEST = Path("codes/results/g6_horizontal_comparison/G6_manifest.json")
DEFAULT_CONTROL_FALLBACK = Path("codes/results/g6_horizontal_comparison/hcmbo_proposed/seed_37/G6_best_control.json")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _best_hcmbo_control_path_from_manifest(manifest_path: Path) -> Path:
    manifest = _load_json(manifest_path)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"G6 manifest has invalid runs array: {manifest_path}")
    candidates = []
    for item in runs:
        if not isinstance(item, dict):
            continue
        if item.get("method") != "hcmbo_proposed":
            continue
        if item.get("feasible") is not True:
            continue
        output_dir = item.get("output_dir")
        objective = item.get("best_objective")
        if output_dir is None or objective is None:
            continue
        candidates.append((float(objective), Path(str(output_dir)) / "G6_best_control.json"))
    if not candidates:
        raise ValueError(f"No feasible hcmbo_proposed run found in {manifest_path}")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def resolve_control_path(control_json: str | None) -> Path:
    if control_json:
        return _resolve_path(Path(control_json))
    manifest_path = _resolve_path(DEFAULT_G6_MANIFEST)
    if manifest_path.exists():
        return _best_hcmbo_control_path_from_manifest(manifest_path).resolve()
    return _resolve_path(DEFAULT_CONTROL_FALLBACK)


def parse_channel_mapping(values: list[str] | None) -> dict[str, str]:
    mapping = dict(DEFAULT_CHANNEL_MAP)
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Channel mapping must use g6=bund form, got {value!r}")
        source, target = value.split("=", 1)
        source = source.strip()
        target = target.strip()
        if source not in G6_CHANNELS or not target:
            raise argparse.ArgumentTypeError(f"Invalid channel mapping: {value!r}")
        mapping[source] = target
    return mapping


def _translate_direction_state(state: str) -> str:
    normalized = str(state).upper()
    if normalized == "E":
        return "plus"
    if normalized == "W":
        return "minus"
    if normalized == "FREE":
        return "both"
    if normalized == "CLOSED":
        return "closed"
    raise ValueError(f"Unsupported G6 direction state: {state!r}")


def _direction_controls_from_g6(
    directions: dict[str, object],
    *,
    channel_map: dict[str, str],
    alpha: float,
    beta: float,
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for g6_channel in G6_CHANNELS:
        if g6_channel not in directions:
            raise ValueError(f"G6 control is missing direction for channel {g6_channel!r}")
        bund_channel = channel_map[g6_channel]
        direction = _translate_direction_state(str(directions[g6_channel]))
        if direction == "closed":
            controls.append({"mode": "closed", "region": bund_channel})
            continue
        controls.append(
            {
                "mode": "region_axis",
                "region": bund_channel,
                "axis_region": bund_channel,
                "direction": direction,
                "alpha": float(alpha),
                "beta": float(beta),
            }
        )
    return controls


def _validate_profile_lengths(q_by_gate: dict[str, object]) -> int:
    segment_count: int | None = None
    for gate_id, raw_profile in q_by_gate.items():
        if not isinstance(raw_profile, list) or not raw_profile:
            raise ValueError(f"q_by_gate[{gate_id!r}] must be a non-empty list")
        if segment_count is None:
            segment_count = len(raw_profile)
        elif len(raw_profile) != segment_count:
            raise ValueError("All q_by_gate profiles must have the same segment count")
    return int(segment_count or 0)


def _capacity_controls_from_q(
    q_by_gate: dict[str, object],
    *,
    channel_map: dict[str, str],
    duration: float,
    waiting_width: int,
    capacity_scale: float,
) -> list[dict[str, object]]:
    segment_count = _validate_profile_lengths(q_by_gate)
    if segment_count <= 0:
        return []
    segment_length = float(duration) / float(segment_count)
    controls: list[dict[str, object]] = []
    for gate_id, raw_profile in q_by_gate.items():
        if ":" not in gate_id:
            raise ValueError(f"q_by_gate key must use channel:side form, got {gate_id!r}")
        g6_channel, side = gate_id.split(":", 1)
        if g6_channel not in channel_map:
            raise ValueError(f"q_by_gate references unmapped G6 channel {g6_channel!r}")
        side = side.lower()
        if side not in {"plus", "minus"}:
            raise ValueError(f"Unsupported q_by_gate side: {side!r}")
        for segment, raw_rate in enumerate(raw_profile):
            rate = float(raw_rate) * float(capacity_scale)
            controls.append(
                {
                    "channel": channel_map[g6_channel],
                    "side": side,
                    "rate": rate,
                    "time_start": float(segment * segment_length),
                    "time_end": float((segment + 1) * segment_length),
                    "waiting_width": int(waiting_width),
                }
            )
    return controls


def _dump_population_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for table_name in ("initial_groups", "inflow_groups"):
        groups = payload.get(table_name, [])
        if not isinstance(groups, list):
            raise ValueError(f"population.{table_name} must be a list")
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"population.{table_name} entries must be tables")
            if lines:
                lines.append("")
            lines.append(f"[[{table_name}]]")
            for key, value in group.items():
                if value is None:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _population_with_scaled_inflows(
    population: dict[str, object],
    inflow_rate_scale: float,
) -> dict[str, object]:
    scale = float(inflow_rate_scale)
    if scale <= 0.0:
        raise ValueError("inflow_rate_scale must be positive")
    scaled = deepcopy(population)
    inflow_groups = scaled.get("inflow_groups", [])
    if not isinstance(inflow_groups, list):
        raise ValueError("population.inflow_groups must be a list")
    for group in inflow_groups:
        if not isinstance(group, dict):
            raise ValueError("population.inflow_groups entries must be tables")
        if "rate" not in group:
            raise ValueError("population.inflow_groups entries must contain rate")
        group["rate"] = float(group["rate"]) * scale
    return scaled


def _routes_from_hcmbo_control(
    base_routes: dict[str, object],
    control: dict[str, object],
    *,
    channel_map: dict[str, str],
    case_id: str,
    duration: float,
    alpha: float,
    beta: float,
    waiting_width: int,
    capacity_scale: float,
    transition_kappa: float | None = None,
    include_direction_controls: bool = True,
    include_capacity_controls: bool = True,
) -> dict[str, object]:
    directions = control.get("directions")
    q_by_gate = control.get("q_by_gate")
    if not isinstance(directions, dict):
        raise ValueError("HCMBO control JSON must contain directions object")
    if not isinstance(q_by_gate, dict):
        raise ValueError("HCMBO control JSON must contain q_by_gate object")

    routes = deepcopy(base_routes)
    case_table = dict(routes.get("case", {}))
    case_table["case_id"] = case_id
    case_table["title"] = "Bund AnyLogic with transferred G6 HCMBO control"
    routes["case"] = case_table

    direction_controls = (
        _direction_controls_from_g6(
            directions,
            channel_map=channel_map,
            alpha=alpha,
            beta=beta,
        )
        if include_direction_controls
        else []
    )
    stages: list[dict[str, object]] = []
    for stage in routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        has_transition = "next_stage" in stage_copy or bool(stage_copy.get("targets"))
        if transition_kappa is not None and has_transition:
            if float(transition_kappa) <= 0.0:
                raise ValueError("transition_kappa must be positive")
            stage_copy["kappa"] = float(transition_kappa)
        existing_controls = stage_copy.get("controls", [])
        copied_controls = (
            [dict(control_item) for control_item in existing_controls if isinstance(control_item, dict)]
            if isinstance(existing_controls, list)
            else []
        )
        copied_controls.extend(dict(control_item) for control_item in direction_controls)
        stage_copy["controls"] = copied_controls
        stages.append(stage_copy)
    routes["stages"] = stages
    routes["capacity_controls"] = (
        _capacity_controls_from_q(
            q_by_gate,
            channel_map=channel_map,
            duration=duration,
            waiting_width=waiting_width,
            capacity_scale=capacity_scale,
        )
        if include_capacity_controls
        else []
    )
    return routes


def _simulation_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in (
        "steps",
        "time_horizon",
        "save_every",
        "bellman_every",
        "rho_max",
        "density_contour_levels",
        "wall_avoidance_weight",
        "wall_avoidance_sigma_cells",
        "wall_avoidance_radius_cells",
        "wall_clearance_cells",
        "block_diagonal_corner_cutting",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.freeze_potentials:
        steps = int(overrides.get("steps", args.steps if args.steps is not None else 200))
        overrides["bellman_every"] = steps + 1
    if args.no_snapshots:
        steps = int(overrides.get("steps", args.steps if args.steps is not None else 200))
        overrides["save_every"] = steps + 1
    return overrides


def _field_data_config_from_args(
    args: argparse.Namespace,
    simulation_overrides: dict[str, object],
) -> dict[str, object]:
    if not args.save_field_data:
        return {"enabled": False}
    save_every = args.field_save_every
    if save_every is None and "save_every" in simulation_overrides:
        save_every = int(simulation_overrides["save_every"])
    config: dict[str, object] = {
        "enabled": True,
        "output_dir_name": str(args.field_output_dir),
        "dtype": str(args.field_dtype),
    }
    if save_every is not None:
        config["save_every"] = int(save_every)
    return config


def _control_enablement_from_args(args: argparse.Namespace) -> tuple[bool, bool]:
    apply_controls = bool(getattr(args, "apply_controls", True))
    return (
        apply_controls and not bool(args.disable_direction_controls),
        apply_controls and not bool(args.disable_capacity_controls),
    )


def _duration_from_base_run(base_run: dict[str, object], simulation_overrides: dict[str, object]) -> float:
    if "time_horizon" in simulation_overrides:
        return float(simulation_overrides["time_horizon"])
    simulation = base_run.get("simulation", {})
    if not isinstance(simulation, dict):
        return 1.0
    return float(simulation.get("time_horizon", 1.0))


def _write_transfer_config(
    *,
    base_dir: Path,
    output_root: Path,
    control: dict[str, object],
    channel_map: dict[str, str],
    case_id: str,
    alpha: float,
    beta: float,
    waiting_width: int,
    capacity_scale: float,
    inflow_rate_scale: float,
    transition_kappa: float | None,
    include_direction_controls: bool,
    include_capacity_controls: bool,
    simulation_overrides: dict[str, object],
) -> Path:
    base_run = _load_toml(base_dir / "run.toml")
    base_routes = _load_toml(base_dir / "routes.toml")
    duration = _duration_from_base_run(base_run, simulation_overrides)

    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    routes_path = generated_dir / f"routes_{case_id}.toml"
    population_path = (base_dir / str(base_run["population"]["file"])).resolve()
    if not math.isclose(float(inflow_rate_scale), 1.0):
        population_path = generated_dir / f"population_{case_id}.toml"
        generated_population = _population_with_scaled_inflows(
            _load_toml(base_dir / str(base_run["population"]["file"])),
            float(inflow_rate_scale),
        )
        population_path.write_text(_dump_population_toml(generated_population), encoding="utf-8")
    run_path = generated_dir / f"run_{case_id}.toml"

    generated_routes = _routes_from_hcmbo_control(
        base_routes,
        control,
        channel_map=channel_map,
        case_id=case_id,
        duration=duration,
        alpha=alpha,
        beta=beta,
        waiting_width=waiting_width,
        capacity_scale=capacity_scale,
        transition_kappa=transition_kappa,
        include_direction_controls=include_direction_controls,
        include_capacity_controls=include_capacity_controls,
    )
    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run.get("objective", {})),
        "scene": {"file": str((base_dir / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str(population_path.resolve())},
        "routes": {"file": str(routes_path.resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["simulation"].update(simulation_overrides)
    generated_run["objective"]["name"] = case_id

    routes_path.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _gate_total(summary: dict[str, object], key: str) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(value) for value in raw.values()))


def _row_from_summary(summary: dict[str, object]) -> dict[str, object]:
    objective = summary.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    return {
        "case_id": str(summary.get("case_id", "")),
        "objective_value": float(summary.get("objective_value", 0.0)),
        "j1_eval": float(objective.get("j1_eval", summary.get("j1_normalized", 0.0))),
        "j2_eval": float(objective.get("j2_eval", summary.get("j2_normalized", 0.0))),
        "j5_eval": float(objective.get("j5_eval", summary.get("j5_normalized", 0.0))),
        "gate_attempted": _gate_total(summary, "gate_attempted_cumulative"),
        "gate_actual": _gate_total(summary, "gate_actual_cumulative"),
        "gate_rejected": _gate_total(summary, "gate_rejected_cumulative"),
        "final_time": float(summary.get("final_time", 0.0)),
        "config_path": str(summary.get("config_path", "")),
    }


def run_hcmbo_transfer(args: argparse.Namespace) -> dict[str, object]:
    base_dir = _resolve_path(Path(args.base_dir))
    output_root = _resolve_path(Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    control_path = resolve_control_path(args.control_json)
    if not control_path.exists():
        raise FileNotFoundError(control_path)
    control = _load_json(control_path)
    channel_map = parse_channel_mapping(args.channel_map)
    simulation_overrides = _simulation_overrides_from_args(args)
    field_data_config = _field_data_config_from_args(args, simulation_overrides)
    direction_controls_enabled, capacity_controls_enabled = _control_enablement_from_args(args)

    config_path = _write_transfer_config(
        base_dir=base_dir,
        output_root=output_root,
        control=control,
        channel_map=channel_map,
        case_id=args.case_id,
        alpha=args.alpha,
        beta=args.beta,
        waiting_width=args.waiting_width,
        capacity_scale=args.capacity_scale,
        inflow_rate_scale=args.inflow_rate_scale,
        transition_kappa=args.transition_kappa,
        include_direction_controls=direction_controls_enabled,
        include_capacity_controls=capacity_controls_enabled,
        simulation_overrides=simulation_overrides,
    )

    payload: dict[str, object] = {
        "base_dir": str(base_dir),
        "output_root": str(output_root),
        "control_json": str(control_path),
        "channel_map": channel_map,
        "case_id": args.case_id,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "waiting_width": int(args.waiting_width),
        "capacity_scale": float(args.capacity_scale),
        "apply_controls": bool(args.apply_controls),
        "direction_controls_enabled": direction_controls_enabled,
        "capacity_controls_enabled": capacity_controls_enabled,
        "inflow_rate_scale": float(args.inflow_rate_scale),
        "transition_kappa": None if args.transition_kappa is None else float(args.transition_kappa),
        "simulation_overrides": simulation_overrides,
        "field_data": field_data_config,
        "generated_config": str(config_path.resolve()),
        "directions": control.get("directions", {}),
        "translated_directions": {
            channel_map[channel]: _translate_direction_state(str(control.get("directions", {}).get(channel)))
            for channel in G6_CHANNELS
        },
    }

    if args.no_run:
        payload["summary"] = None
        save_json(output_root / "bund_hcmbo_transfer_summary.json", payload)
        return payload

    step_observer_factory = None
    if args.save_field_data:
        step_observer_factory = make_field_data_observer_factory(
            save_every=args.field_save_every,
            dtype=args.field_dtype,
            output_dir_name=args.field_output_dir,
        )

    summary = run_from_config(
        config_path=config_path,
        output_root=output_root,
        simulation_overrides=simulation_overrides,
        write_root_summary=False,
        step_observer_factory=step_observer_factory,
        channel_flux_directions=payload["translated_directions"],
    )
    summary["bund_hcmbo_transfer"] = {
        "control_json": str(control_path),
        "channel_map": channel_map,
        "translated_directions": payload["translated_directions"],
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "waiting_width": int(args.waiting_width),
        "capacity_scale": float(args.capacity_scale),
        "apply_controls": bool(args.apply_controls),
        "direction_controls_enabled": direction_controls_enabled,
        "capacity_controls_enabled": capacity_controls_enabled,
        "inflow_rate_scale": float(args.inflow_rate_scale),
        "transition_kappa": None if args.transition_kappa is None else float(args.transition_kappa),
        "field_data": field_data_config,
        "generated_config": str(config_path.resolve()),
    }
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)
    row = _row_from_summary(summary)
    payload["summary"] = summary
    payload["row"] = row
    _write_csv(output_root / "bund_hcmbo_transfer_summary.csv", [row])
    save_json(output_root / "bund_hcmbo_transfer_summary.json", payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transfer a G6 HCMBO best control policy to the Bund AnyLogic four-channel scene and run it."
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Directory containing converted BundScene TOML files.")
    parser.add_argument("--output-root", default="codes/results/bund_hcmbo_transfer", help="Experiment output directory.")
    parser.add_argument("--control-json", default=None, help="Path to G6_best_control.json. Defaults to best feasible hcmbo_proposed run from G6_manifest.json.")
    parser.add_argument("--case-id", default="bund_hcmbo_transfer", help="Generated case id.")
    parser.add_argument("--channel-map", action="append", default=None, help="Override channel mapping, e.g. top=channel_1. Can be repeated.")
    parser.add_argument("--alpha", type=float, default=2.8, help="Metric alpha for transferred direction controls.")
    parser.add_argument("--beta", type=float, default=0.35, help="Metric beta for transferred direction controls.")
    parser.add_argument("--waiting-width", type=int, default=8, help="Cells behind each oriented gate used for waiting-mass diagnostics.")
    parser.add_argument("--capacity-scale", type=float, default=1.0, help="Multiplier applied to transferred G6 q rates.")
    parser.add_argument("--apply-controls", dest="apply_controls", action="store_true", default=True, help="Apply transferred G6 direction and gate-capacity controls. This is the default.")
    parser.add_argument("--no-controls", dest="apply_controls", action="store_false", help="Disable all transferred G6 controls while keeping the same route and population settings.")
    parser.add_argument("--disable-direction-controls", action="store_true", help="Do not write transferred G6 channel direction controls into generated routes.")
    parser.add_argument("--disable-capacity-controls", action="store_true", help="Do not write transferred G6 q gate-capacity controls into generated routes.")
    parser.add_argument("--inflow-rate-scale", type=float, default=1.0, help="Multiplier applied to Bund continuous inflow rates.")
    parser.add_argument("--transition-kappa", type=float, default=None, help="Override kappa for all route stages that transition to another stage.")
    parser.add_argument("--steps", type=int, default=None, help="Override simulation steps.")
    parser.add_argument("--time-horizon", type=float, default=None, help="Override simulation time horizon.")
    parser.add_argument("--save-every", type=int, default=None, help="Override snapshot interval.")
    parser.add_argument("--bellman-every", type=int, default=None, help="Override Bellman recomputation interval.")
    parser.add_argument("--rho-max", type=float, default=None, help="Override maximum density used by the simulation.")
    parser.add_argument("--density-contour-levels", type=int, default=None, help="Override density contour levels.")
    parser.add_argument("--wall-avoidance-weight", type=float, default=None, help="Soft wall-avoidance cost weight. Defaults to simulation config.")
    parser.add_argument("--wall-avoidance-sigma-cells", type=float, default=None, help="Soft wall-avoidance decay length in grid cells.")
    parser.add_argument("--wall-avoidance-radius-cells", type=float, default=None, help="Maximum distance in grid cells affected by soft wall avoidance.")
    parser.add_argument("--wall-clearance-cells", type=int, default=None, help="Optional hard wall clearance in grid cells. Defaults to simulation config.")
    parser.add_argument(
        "--block-diagonal-corner-cutting",
        dest="block_diagonal_corner_cutting",
        action="store_true",
        default=None,
        help="Remove diagonal Bellman moves that cut across blocked wall corners.",
    )
    parser.add_argument(
        "--allow-diagonal-corner-cutting",
        dest="block_diagonal_corner_cutting",
        action="store_false",
        help="Allow legacy diagonal corner cutting.",
    )
    parser.add_argument("--freeze-potentials", action="store_true", help="Set bellman_every to steps + 1 for a faster frozen-potential check.")
    parser.add_argument("--no-snapshots", action="store_true", help="Set save_every to steps + 1.")
    parser.add_argument("--save-field-data", action="store_true", help="Save compressed rho/speed/vx/vy field arrays for high-resolution post-processing.")
    parser.add_argument("--field-save-every", type=int, default=None, help="Step interval for field array snapshots. Defaults to effective simulation save_every.")
    parser.add_argument("--field-dtype", choices=("float32", "float64"), default="float32", help="Numeric dtype used for saved field arrays.")
    parser.add_argument("--field-output-dir", default="fields", help="Case-output subdirectory for saved field arrays.")
    parser.add_argument("--no-run", action="store_true", help="Only write generated TOML files, do not run simulation.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_hcmbo_transfer(args)
    print(f"summary: {Path(payload['output_root']) / 'bund_hcmbo_transfer_summary.json'}")
    print(f"generated_config: {payload['generated_config']}")
    print(f"control_json: {payload['control_json']}")
    if payload.get("row"):
        row = payload["row"]
        assert isinstance(row, dict)
        print(f"objective_value: {float(row['objective_value']):.6g}")
        print(f"gate_rejected: {float(row['gate_rejected']):.6g}")


if __name__ == "__main__":
    main()
