from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tomllib

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g2_strategy import G2StrategyCollector, build_g2_strategy_report
from crowd_bellman.metrics import save_json


BASELINE_CONFIG = Path("codes/scenes/examples/g2_multistage_directional/run_baseline.toml")


@dataclass(frozen=True)
class DirectionSetting:
    case_id: str
    title: str
    family: str
    directions: dict[str, str]


DIRECTION_SETTINGS = (
    DirectionSetting(
        case_id="case2_topE_middleW_bottomW",
        title="Case 2: single eastbound top lane, return via middle and bottom",
        family="single_entry",
        directions={"top": "E", "middle": "W", "bottom": "W"},
    ),
    DirectionSetting(
        case_id="case3_topW_middleE_bottomW",
        title="Case 3: single eastbound middle lane, return via top and bottom",
        family="single_entry",
        directions={"top": "W", "middle": "E", "bottom": "W"},
    ),
    DirectionSetting(
        case_id="case4_topW_middleW_bottomE",
        title="Case 4: single eastbound bottom lane, return via top and middle",
        family="single_entry",
        directions={"top": "W", "middle": "W", "bottom": "E"},
    ),
    DirectionSetting(
        case_id="case5_topW_middleE_bottomE",
        title="Case 5: single westbound top lane, entry via middle and bottom",
        family="single_return",
        directions={"top": "W", "middle": "E", "bottom": "E"},
    ),
    DirectionSetting(
        case_id="case6_topE_middleW_bottomE",
        title="Case 6: single westbound middle lane, entry via top and bottom",
        family="single_return",
        directions={"top": "E", "middle": "W", "bottom": "E"},
    ),
    DirectionSetting(
        case_id="case7_topE_middleE_bottomW",
        title="Case 7: single westbound bottom lane, entry via top and middle",
        family="single_return",
        directions={"top": "E", "middle": "E", "bottom": "W"},
    ),
)


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML scalar value: {value!r}")


def _dump_table(table_name: str, payload: dict[str, object]) -> list[str]:
    lines = [f"[{table_name}]"]
    for key, value in payload.items():
        lines.append(f"{key} = {_format_scalar(value)}")
    return lines


def _dump_run_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for table_name in ("simulation", "objective", "scene", "population", "routes", "outputs"):
        table = payload.get(table_name, {})
        if not isinstance(table, dict):
            continue
        if lines:
            lines.append("")
        lines.extend(_dump_table(table_name, table))
    return "\n".join(lines) + "\n"


def _dump_routes_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    case_table = payload.get("case", {})
    if isinstance(case_table, dict):
        lines.extend(_dump_table("case", case_table))

    stages = payload.get("stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lines.append("")
            lines.append("[[stages]]")
            for key, value in stage.items():
                if key in {"controls", "targets"}:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
            controls = stage.get("controls", [])
            if isinstance(controls, list):
                for control in controls:
                    if not isinstance(control, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.controls]]")
                    for key, value in control.items():
                        lines.append(f"{key} = {_format_scalar(value)}")
            targets = stage.get("targets", [])
            if isinstance(targets, list):
                for target in targets:
                    if not isinstance(target, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.targets]]")
                    for key, value in target.items():
                        lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _direction_controls(channel_directions: dict[str, str]) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for channel_name in ("top", "middle", "bottom"):
        direction = channel_directions[channel_name]
        controls.append(
            {
                "mode": "fixed_direction",
                "region": f"{channel_name}_channel",
                "direction": direction,
                "alpha": 10.0,
                "beta": 0.2,
                "allowed_directions": [direction],
            }
        )
    return controls


def _direction_applied_routes(base_routes: dict[str, object], setting: DirectionSetting) -> dict[str, object]:
    routes = {
        "case": dict(base_routes["case"]),
        "stages": [],
    }
    routes["case"]["case_id"] = setting.case_id
    routes["case"]["title"] = setting.title

    stage_controls = _direction_controls(setting.directions)
    for stage in base_routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        copied_controls = [dict(control) for control in stage.get("controls", []) if isinstance(control, dict)]
        if str(stage_copy.get("stage_id")) in {"enter_platform", "return_left"}:
            copied_controls.extend(dict(control) for control in stage_controls)
        if copied_controls:
            stage_copy["controls"] = copied_controls
        routes["stages"].append(stage_copy)
    return routes


def _generated_config_paths(*, output_root: Path, setting: DirectionSetting) -> tuple[Path, Path]:
    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return (
        generated_dir / f"run_{setting.case_id}.toml",
        generated_dir / f"routes_{setting.case_id}.toml",
    )


def _write_direction_config(*, output_root: Path, setting: DirectionSetting) -> Path:
    base_run = _load_toml(BASELINE_CONFIG)
    routes_table = base_run.get("routes", {})
    if not isinstance(routes_table, dict):
        raise ValueError(f"{BASELINE_CONFIG} does not contain [routes]")
    routes_path = (BASELINE_CONFIG.parent / str(routes_table["file"])).resolve()
    base_routes = _load_toml(routes_path)

    run_path, generated_routes_path = _generated_config_paths(output_root=output_root, setting=setting)
    generated_routes = _direction_applied_routes(base_routes, setting)

    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run["objective"]),
        "scene": {"file": str((BASELINE_CONFIG.parent / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str((BASELINE_CONFIG.parent / str(base_run["population"]["file"])).resolve())},
        "routes": {"file": str(generated_routes_path.resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["objective"]["name"] = setting.case_id

    generated_routes_path.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _attach_scan_metadata(
    *,
    summary: dict[str, object],
    output_root: Path,
    label: str,
    family: str,
    directions: dict[str, str],
    template_config: Path,
    is_baseline: bool,
) -> None:
    scan = {
        "setting_label": label,
        "family": family,
        "channel_directions": directions,
        "template_config": str(template_config.resolve()),
        "is_baseline": is_baseline,
        "entry_channels": [name for name, direction in directions.items() if direction == "E"],
        "return_channels": [name for name, direction in directions.items() if direction == "W"],
    }
    summary["g2_scan"] = scan
    metadata = summary.get("metadata")
    if isinstance(metadata, dict):
        metadata["g2_scan"] = scan
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the G2 multistage directional-setting scan.")
    parser.add_argument("--output-root", default="codes/results/g2_multistage_direction_scan")
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

    collectors: dict[str, tuple[G2StrategyCollector, Path]] = {}

    def observer_factory(**kwargs: object):
        case = kwargs["case"]
        run_spec = kwargs["run_spec"]
        simulation = kwargs["simulation"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        collector = G2StrategyCollector(
            case_id=str(case.case_id),
            title=str(case.title),
            walkable=case.walkable,
            channel_masks=case.channel_masks,
            rho_safe=float(run_spec.objective.rho_safe),
            dx=float(simulation.dx),
        )
        collectors[str(case.case_id)] = (collector, case_output_dir)
        return collector.observe

    summaries: list[dict[str, object]] = []

    baseline_summary = run_from_config(
        config_path=BASELINE_CONFIG,
        output_root=output_root,
        simulation_overrides=simulation_overrides or None,
        write_root_summary=False,
        step_observer_factory=observer_factory,
    )
    _attach_scan_metadata(
        summary=baseline_summary,
        output_root=output_root,
        label="baseline",
        family="baseline",
        directions={"top": "FREE", "middle": "FREE", "bottom": "FREE"},
        template_config=BASELINE_CONFIG,
        is_baseline=True,
    )
    summaries.append(baseline_summary)

    generated_config_paths: list[str] = []
    for setting in DIRECTION_SETTINGS:
        generated_run_path = _write_direction_config(output_root=output_root, setting=setting)
        generated_config_paths.append(str(generated_run_path.resolve()))
        summary = run_from_config(
            config_path=generated_run_path,
            output_root=output_root,
            simulation_overrides=simulation_overrides or None,
            write_root_summary=False,
            step_observer_factory=observer_factory,
        )
        _attach_scan_metadata(
            summary=summary,
            output_root=output_root,
            label=setting.case_id,
            family=setting.family,
            directions=setting.directions,
            template_config=BASELINE_CONFIG,
            is_baseline=False,
        )
        summaries.append(summary)

    behavior_summaries: list[dict[str, object]] = []
    for summary in summaries:
        collector, case_output_dir = collectors[str(summary["case_id"])]
        behavior_summaries.append(collector.save_case_outputs(case_output_dir))

    payload = {
        "experiment_group": "G2",
        "design_version": "direction_scan_multistage",
        "baseline_config": str(BASELINE_CONFIG.resolve()),
        "direction_settings": [
            {
                "case_id": setting.case_id,
                "title": setting.title,
                "family": setting.family,
                "channel_directions": setting.directions,
            }
            for setting in DIRECTION_SETTINGS
        ],
        "generated_config_paths": generated_config_paths,
        "cases": summaries,
        "behavior_cases": behavior_summaries,
    }
    save_json(output_root / "comparison_summary.json", payload)
    build_g2_strategy_report(
        output_root=output_root,
        case_summaries=summaries,
        behavior_summaries=behavior_summaries,
    )


if __name__ == "__main__":
    main()
