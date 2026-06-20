from __future__ import annotations

import argparse
import csv
import math
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from bund_capacity_response_runner import (
    DEFAULT_BASE_DIR,
    _dump_routes_toml,
    _dump_run_toml,
    _load_toml,
    _resolve_path,
)
from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.metrics import save_json


DEFAULT_FORWARD_DIRECTIONS = ("channel_3:minus", "channel_4:plus")
SUPPORTED_DIRECTIONS = {"plus", "minus", "both", "closed"}


@dataclass(frozen=True)
class DirectionSetting:
    channel: str
    direction: str


@dataclass(frozen=True)
class DirectionCase:
    case_id: str
    title: str
    family: str
    settings: tuple[DirectionSetting, ...]
    description: str


def parse_direction_setting(value: str) -> DirectionSetting:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Direction setting must use channel:direction form, got {value!r}")
    channel, direction = value.split(":", 1)
    direction = direction.lower()
    if not channel or direction not in SUPPORTED_DIRECTIONS:
        supported = ", ".join(sorted(SUPPORTED_DIRECTIONS))
        raise argparse.ArgumentTypeError(
            f"Direction setting must use channel:direction with one of {supported}; got {value!r}"
        )
    return DirectionSetting(channel=channel, direction=direction)


def _reverse_settings(settings: tuple[DirectionSetting, ...]) -> tuple[DirectionSetting, ...]:
    reverse = {"plus": "minus", "minus": "plus", "both": "both", "closed": "closed"}
    return tuple(
        DirectionSetting(channel=setting.channel, direction=reverse[setting.direction])
        for setting in settings
    )


def _both_settings(settings: tuple[DirectionSetting, ...]) -> tuple[DirectionSetting, ...]:
    return tuple(DirectionSetting(channel=setting.channel, direction="both") for setting in settings)


def _closed_settings(settings: tuple[DirectionSetting, ...]) -> tuple[DirectionSetting, ...]:
    return tuple(DirectionSetting(channel=setting.channel, direction="closed") for setting in settings)


def _direction_controls(settings: tuple[DirectionSetting, ...], *, alpha: float, beta: float) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for setting in settings:
        if setting.direction == "closed":
            controls.append({"mode": "closed", "region": setting.channel})
            continue
        controls.append(
            {
                "mode": "region_axis",
                "region": setting.channel,
                "axis_region": setting.channel,
                "direction": setting.direction,
                "alpha": float(alpha),
                "beta": float(beta),
            }
        )
    return controls


def _routes_for_case(
    base_routes: dict[str, object],
    case: DirectionCase,
    *,
    alpha: float,
    beta: float,
) -> dict[str, object]:
    routes = deepcopy(base_routes)
    case_table = dict(routes.get("case", {}))
    case_table["case_id"] = case.case_id
    case_table["title"] = case.title
    routes["case"] = case_table

    controls = _direction_controls(case.settings, alpha=alpha, beta=beta)
    stages = []
    for stage in routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        existing_controls = stage_copy.get("controls", [])
        copied_controls = [dict(control) for control in existing_controls if isinstance(control, dict)] if isinstance(existing_controls, list) else []
        copied_controls.extend(dict(control) for control in controls)
        if copied_controls:
            stage_copy["controls"] = copied_controls
        stages.append(stage_copy)
    routes["stages"] = stages
    return routes


def _write_case_config(
    *,
    base_dir: Path,
    output_root: Path,
    case: DirectionCase,
    alpha: float,
    beta: float,
) -> Path:
    base_run_path = base_dir / "run.toml"
    base_routes_path = base_dir / "routes.toml"
    base_run = _load_toml(base_run_path)
    base_routes = _load_toml(base_routes_path)

    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    routes_path = generated_dir / f"routes_{case.case_id}.toml"
    run_path = generated_dir / f"run_{case.case_id}.toml"

    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run.get("objective", {})),
        "scene": {"file": str((base_dir / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str((base_dir / str(base_run["population"]["file"])).resolve())},
        "routes": {"file": str(routes_path.resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["objective"]["name"] = case.case_id

    routes_path.write_text(
        _dump_routes_toml(_routes_for_case(base_routes, case, alpha=alpha, beta=beta)),
        encoding="utf-8",
    )
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _direction_cases(
    forward_settings: tuple[DirectionSetting, ...],
    *,
    include_both: bool,
    include_closed: bool,
) -> list[DirectionCase]:
    cases = [
        DirectionCase(
            case_id="bund_s_no_control",
            title="Bund direction no control",
            family="reference",
            settings=(),
            description="Reference run without channel direction controls.",
        ),
        DirectionCase(
            case_id="bund_s_forward",
            title="Bund direction forward",
            family="direction_scan",
            settings=forward_settings,
            description="Route-consistent channel-axis direction controls.",
        ),
        DirectionCase(
            case_id="bund_s_reverse",
            title="Bund direction reverse",
            family="direction_scan",
            settings=_reverse_settings(forward_settings),
            description="Reverse channel-axis direction controls.",
        ),
    ]
    if include_both:
        cases.append(
            DirectionCase(
                case_id="bund_s_axis_both",
                title="Bund direction axis both",
                family="direction_scan",
                settings=_both_settings(forward_settings),
                description="Axis-aligned metric tensor without one-way direction restriction.",
            )
        )
    if include_closed:
        cases.append(
            DirectionCase(
                case_id="bund_s_closed",
                title="Bund direction closed",
                family="stress",
                settings=_closed_settings(forward_settings),
                description="Selected channels closed for a stress response check.",
            )
        )
    return cases


def _simulation_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for key in ("steps", "time_horizon", "save_every", "bellman_every"):
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


def _settings_summary(settings: tuple[DirectionSetting, ...]) -> str:
    if not settings:
        return "none"
    return ";".join(f"{setting.channel}:{setting.direction}" for setting in settings)


def _channel_total(summary: dict[str, object], key: str, channels: tuple[str, ...]) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(raw.get(channel, 0.0)) for channel in channels))


def _row_from_summary(summary: dict[str, object], case: DirectionCase, channels: tuple[str, ...]) -> dict[str, object]:
    objective = summary.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    j1_eval = float(objective.get("j1_eval", summary.get("j1_normalized", 0.0)))
    j2_eval = float(objective.get("j2_eval", summary.get("j2_normalized", 0.0)))
    j5_eval = float(objective.get("j5_eval", summary.get("j5_normalized", 0.0)))
    channel_time_mean_density = summary.get("channel_time_mean_density", {})
    if not isinstance(channel_time_mean_density, dict):
        channel_time_mean_density = {}
    selected_density_mean = (
        sum(float(channel_time_mean_density.get(channel, 0.0)) for channel in channels) / max(len(channels), 1)
    )
    return {
        "case_id": str(summary["case_id"]),
        "title": str(summary.get("title", "")),
        "family": case.family,
        "description": case.description,
        "settings": _settings_summary(case.settings),
        "objective_value": float(summary.get("objective_value", 0.0)),
        "objective_without_j5": float(j1_eval + j2_eval),
        "j1_eval": j1_eval,
        "j2_eval": j2_eval,
        "j5_eval": j5_eval,
        "j1_normalized": float(summary.get("j1_normalized", 0.0)),
        "j2_normalized": float(summary.get("j2_normalized", 0.0)),
        "j5_normalized": float(summary.get("j5_normalized", 0.0)),
        "selected_channel_flux": _channel_total(summary, "channel_flux_cumulative", channels),
        "selected_channel_density_mean": float(selected_density_mean),
        "peak_density": float(summary.get("peak_density_max", 0.0)),
        "sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
        "cap_removed": float(summary.get("final_cap_removed_cumulative", 0.0)),
        "final_time": float(summary.get("final_time", 0.0)),
        "config_path": str(summary.get("config_path", "")),
    }


def _direction_response_conclusion(rows: list[dict[str, object]], *, objective_tolerance: float) -> dict[str, object]:
    objectives = [float(row["objective_value"]) for row in rows]
    objectives_without_j5 = [float(row.get("objective_without_j5", 0.0)) for row in rows]
    flux_values = [float(row.get("selected_channel_flux", 0.0)) for row in rows]
    density_values = [float(row.get("selected_channel_density_mean", 0.0)) for row in rows]
    objective_span = max(objectives) - min(objectives) if objectives else 0.0
    objective_without_j5_span = (
        max(objectives_without_j5) - min(objectives_without_j5)
        if objectives_without_j5
        else 0.0
    )
    flux_span = max(flux_values) - min(flux_values) if flux_values else 0.0
    density_span = max(density_values) - min(density_values) if density_values else 0.0

    if objective_span > objective_tolerance or objective_without_j5_span > objective_tolerance:
        verdict = "supports_direction_control_changes_objective"
    elif flux_span > 1.0e-9 or density_span > 1.0e-9:
        verdict = "state_changes_but_objective_change_below_tolerance"
    else:
        verdict = "no_direction_response_detected"

    return {
        "verdict": verdict,
        "objective_tolerance": float(objective_tolerance),
        "objective_span": float(objective_span),
        "objective_without_j5_span": float(objective_without_j5_span),
        "selected_channel_flux_span": float(flux_span),
        "selected_channel_density_span": float(density_span),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_direction_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if len(rows) < 2:
        return
    labels = [str(row["case_id"]).replace("bund_s_", "").replace("_", " ") for row in rows]
    x_values = list(range(len(rows)))

    fig, ax1 = plt.subplots(1, 1, figsize=(9.0, 5.1), dpi=170)
    ax1.plot(x_values, [float(row["objective_value"]) for row in rows], marker="o", label="objective")
    ax1.plot(x_values, [float(row["objective_without_j5"]) for row in rows], marker="x", label="J1+J2 eval")
    ax1.plot(x_values, [float(row["j1_eval"]) for row in rows], marker="s", label="J1 eval")
    ax1.plot(x_values, [float(row["j2_eval"]) for row in rows], marker="^", label="J2 eval")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels, rotation=10)
    ax1.set_ylabel("objective / evaluated terms")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        x_values,
        [float(row["selected_channel_flux"]) for row in rows],
        color="#D95F02",
        marker="D",
        label="selected channel flux",
    )
    ax2.set_ylabel("selected channel cumulative flux")

    lines, labels_left = ax1.get_legend_handles_labels()
    lines2, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_left + labels_right, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _run_case(
    *,
    base_dir: Path,
    output_root: Path,
    case: DirectionCase,
    alpha: float,
    beta: float,
    channels: tuple[str, ...],
    simulation_overrides: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config_path = _write_case_config(
        base_dir=base_dir,
        output_root=output_root,
        case=case,
        alpha=alpha,
        beta=beta,
    )
    summary = run_from_config(
        config_path=config_path,
        output_root=output_root,
        simulation_overrides=simulation_overrides,
        write_root_summary=False,
    )
    summary["bund_direction_response"] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "settings": [
            {"channel": setting.channel, "direction": setting.direction}
            for setting in case.settings
        ],
        "alpha": float(alpha),
        "beta": float(beta),
        "config_path": str(config_path.resolve()),
    }
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)
    return summary, _row_from_summary(summary, case, channels)


def run_direction_response_experiment(args: argparse.Namespace) -> dict[str, object]:
    base_dir = _resolve_path(Path(args.base_dir))
    output_root = _resolve_path(Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    setting_values = args.forward_direction if args.forward_direction is not None else list(DEFAULT_FORWARD_DIRECTIONS)
    forward_settings = tuple(parse_direction_setting(value) for value in setting_values)
    channels = tuple(setting.channel for setting in forward_settings)
    simulation_overrides = _simulation_overrides_from_args(args)

    rows: list[dict[str, object]] = []
    cases = _direction_cases(
        forward_settings,
        include_both=args.include_both,
        include_closed=args.include_closed,
    )
    if args.case:
        requested = set(args.case)
        cases = [case for case in cases if case.case_id.replace("bund_s_", "") in requested or case.case_id in requested]
        missing = requested - {case.case_id.replace("bund_s_", "") for case in cases} - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"Unknown direction cases requested: {sorted(missing)}")

    for case in cases:
        _summary, row = _run_case(
            base_dir=base_dir,
            output_root=output_root,
            case=case,
            alpha=args.alpha,
            beta=args.beta,
            channels=channels,
            simulation_overrides=simulation_overrides,
        )
        rows.append(row)

    conclusion = _direction_response_conclusion(rows, objective_tolerance=args.objective_tolerance)
    payload = {
        "base_dir": str(base_dir),
        "output_root": str(output_root),
        "forward_settings": [
            {"channel": setting.channel, "direction": setting.direction}
            for setting in forward_settings
        ],
        "simulation_overrides": simulation_overrides,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "rows": rows,
        "conclusion": conclusion,
    }
    _write_csv(output_root / "bund_direction_response_summary.csv", rows)
    save_json(output_root / "bund_direction_response_summary.json", payload)
    if not args.no_plot:
        _save_direction_plot(output_root / "bund_direction_response_levels.png", rows)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether channel direction controls change the objective in the Bund AnyLogic scene. "
            "Direction controls are compiled as region_axis controls on selected channel regions."
        )
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Directory containing BundScene converted TOML files.")
    parser.add_argument("--output-root", default="codes/results/bund_direction_response", help="Experiment output directory.")
    parser.add_argument(
        "--forward-direction",
        action="append",
        default=None,
        help="Route-consistent channel direction as channel:plus/minus/both/closed. Defaults to channel_3:minus and channel_4:plus.",
    )
    parser.add_argument("--alpha", type=float, default=2.8, help="Metric tensor alpha for region_axis direction controls.")
    parser.add_argument("--beta", type=float, default=0.35, help="Metric tensor beta for region_axis direction controls.")
    parser.add_argument("--include-both", action="store_true", help="Include an axis-both case without one-way restriction.")
    parser.add_argument("--include-closed", action="store_true", help="Include a closed-channel stress case.")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only selected cases: no_control, forward, reverse, axis_both, closed. Can be repeated.",
    )
    parser.add_argument("--objective-tolerance", type=float, default=1.0e-4, help="Objective span required to claim direction changed the objective.")
    parser.add_argument("--steps", type=int, default=None, help="Override simulation steps.")
    parser.add_argument("--time-horizon", type=float, default=None, help="Override simulation time horizon.")
    parser.add_argument("--save-every", type=int, default=None, help="Override snapshot interval.")
    parser.add_argument("--bellman-every", type=int, default=None, help="Override Bellman recomputation interval.")
    parser.add_argument("--freeze-potentials", action="store_true", help="Set bellman_every to steps + 1 for a faster frozen-potential scan.")
    parser.add_argument("--no-snapshots", action="store_true", help="Save only the final snapshot for each case.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the summary plot.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_direction_response_experiment(args)
    conclusion = payload["conclusion"]
    print(f"summary: {Path(payload['output_root']) / 'bund_direction_response_summary.json'}")
    print(f"csv: {Path(payload['output_root']) / 'bund_direction_response_summary.csv'}")
    print(f"verdict: {conclusion['verdict']}")
    print(f"objective_span: {conclusion['objective_span']:.6g}")
    print(f"objective_without_j5_span: {conclusion['objective_without_j5_span']:.6g}")
    print(f"selected_channel_flux_span: {conclusion['selected_channel_flux_span']:.6g}")
    print(f"selected_channel_density_span: {conclusion['selected_channel_density_span']:.6g}")


if __name__ == "__main__":
    main()
