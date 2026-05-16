from __future__ import annotations

import argparse
import csv
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g2_strategy import G2StrategyCollector
from crowd_bellman.metrics import save_json
from crowd_bellman.plotting import parse_density_contour_levels


BASELINE_CONFIG = Path("codes/scenes/examples/g2_multistage_directional/run_baseline.toml")
CHANNEL_NAMES = ("top", "middle", "lower_middle", "bottom")
FIXED_DIRECTIONS = {
    "top": "FREE",
    "middle": "E",
    "lower_middle": "W",
    "bottom": "FREE",
}
ACTIVE_SIDES_BY_STATE = {
    "E": ("plus",),
    "W": ("minus",),
    "FREE": ("plus", "minus"),
    "CLOSED": (),
}


@dataclass(frozen=True)
class CapacityControl:
    channel: str
    side: str
    rate: float
    time_start: float = 0.0
    time_end: float | None = None


@dataclass(frozen=True)
class CapacityCase:
    case_id: str
    title: str
    family: str
    controls: tuple[CapacityControl, ...]
    description: str


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
        return repr(float(value)) if isinstance(value, float) else repr(value)
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

    capacity_controls = payload.get("capacity_controls", [])
    if isinstance(capacity_controls, list):
        for control in capacity_controls:
            if not isinstance(control, dict):
                continue
            lines.append("")
            lines.append("[[capacity_controls]]")
            for key, value in control.items():
                if value is None:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _active_gate_ids(directions: dict[str, str]) -> tuple[str, ...]:
    gate_ids: list[str] = []
    for channel in CHANNEL_NAMES:
        state = directions[channel].upper()
        for side in ACTIVE_SIDES_BY_STATE[state]:
            gate_ids.append(f"{channel}:{side}")
    return tuple(gate_ids)


def _direction_controls(directions: dict[str, str]) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for channel in CHANNEL_NAMES:
        state = directions[channel].upper()
        if state == "CLOSED":
            controls.append({"mode": "closed", "region": f"{channel}_channel"})
            continue
        if state == "FREE":
            controls.append(
                {
                    "mode": "fixed_direction",
                    "region": f"{channel}_channel",
                    "direction": "E",
                    "alpha": 2.8,
                    "beta": 0.35,
                    "allowed_directions": ["ALL"],
                }
            )
            continue
        controls.append(
            {
                "mode": "fixed_direction",
                "region": f"{channel}_channel",
                "direction": state,
                "alpha": 2.8,
                "beta": 0.35,
                "allowed_directions": [state],
            }
        )
    return controls


def _apply_direction_controls(base_routes: dict[str, object], case: CapacityCase) -> dict[str, object]:
    routes = {
        "case": dict(base_routes["case"]),
        "stages": [],
        "capacity_controls": [],
    }
    routes["case"]["case_id"] = case.case_id
    routes["case"]["title"] = case.title

    stage_controls = _direction_controls(FIXED_DIRECTIONS)
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

    routes["capacity_controls"] = [
        {
            "channel": control.channel,
            "side": control.side,
            "rate": control.rate,
            "time_start": control.time_start,
            "time_end": control.time_end,
        }
        for control in case.controls
    ]
    return routes


def _write_case_config(*, output_root: Path, case: CapacityCase) -> Path:
    base_run = _load_toml(BASELINE_CONFIG)
    routes_table = base_run.get("routes", {})
    if not isinstance(routes_table, dict):
        raise ValueError(f"{BASELINE_CONFIG} does not contain [routes]")
    base_routes = _load_toml((BASELINE_CONFIG.parent / str(routes_table["file"])).resolve())

    generated_dir = output_root / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    routes_path = generated_dir / f"routes_{case.case_id}.toml"
    run_path = generated_dir / f"run_{case.case_id}.toml"

    generated_routes = _apply_direction_controls(base_routes, case)
    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run["objective"]),
        "scene": {"file": str((BASELINE_CONFIG.parent / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str((BASELINE_CONFIG.parent / str(base_run["population"]["file"])).resolve())},
        "routes": {"file": str(routes_path.resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["objective"]["name"] = case.case_id
    generated_run["objective"]["lambda_jb"] = 0.0
    generated_run["objective"]["lambda_jr"] = 0.0
    generated_run["objective"]["j2_metric"] = "soft"
    generated_run["objective"]["j2_gamma"] = 1.0
    generated_run["objective"]["j2_scale"] = 0.001

    routes_path.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _inf_controls() -> tuple[CapacityControl, ...]:
    return tuple(
        CapacityControl(channel=gate_id.split(":")[0], side=gate_id.split(":")[1], rate=math.inf)
        for gate_id in _active_gate_ids(FIXED_DIRECTIONS)
    )


def _controls_from_multipliers(ref_rates: dict[str, float], multipliers: dict[str, float], *, duration: float) -> tuple[CapacityControl, ...]:
    del duration
    controls: list[CapacityControl] = []
    for gate_id in _active_gate_ids(FIXED_DIRECTIONS):
        channel, side = gate_id.split(":")
        multiplier = multipliers.get(gate_id, multipliers.get("*", 1.0))
        controls.append(
            CapacityControl(
                channel=channel,
                side=side,
                rate=max(ref_rates[gate_id] * multiplier, 0.05),
            )
        )
    return tuple(controls)


def _scheduled_controls(
    ref_rates: dict[str, float],
    *,
    duration: float,
    segments: list[tuple[float, float, dict[str, float]]],
) -> tuple[CapacityControl, ...]:
    controls: list[CapacityControl] = []
    for start_frac, end_frac, multipliers in segments:
        for gate_id in _active_gate_ids(FIXED_DIRECTIONS):
            channel, side = gate_id.split(":")
            multiplier = multipliers.get(gate_id, multipliers.get(f"{side}:*", multipliers.get("*", 1.0)))
            controls.append(
                CapacityControl(
                    channel=channel,
                    side=side,
                    rate=max(ref_rates[gate_id] * multiplier, 0.05),
                    time_start=float(duration * start_frac),
                    time_end=float(duration * end_frac),
                )
            )
    return tuple(controls)


def _normalize_budget(
    ref_rates: dict[str, float],
    multipliers: dict[str, float],
    *,
    target_multiplier: float,
) -> dict[str, float]:
    target_budget = sum(ref_rates.values()) * float(target_multiplier)
    raw_budget = sum(ref_rates[gate_id] * multipliers[gate_id] for gate_id in ref_rates)
    if raw_budget <= 1.0e-12:
        return {gate_id: float(target_multiplier) for gate_id in ref_rates}
    scale = target_budget / raw_budget
    return {gate_id: float(multipliers[gate_id] * scale) for gate_id in ref_rates}


def _build_capacity_cases(ref_rates: dict[str, float], *, duration: float) -> list[CapacityCase]:
    high = {gate_id: 0.9 for gate_id in ref_rates}
    medium = {gate_id: 0.6 for gate_id in ref_rates}
    low = {gate_id: 0.3 for gate_id in ref_rates}

    entry_priority = {
        gate_id: 0.85 if gate_id.endswith(":plus") else 0.45
        for gate_id in ref_rates
    }
    return_priority = {
        gate_id: 0.45 if gate_id.endswith(":plus") else 0.85
        for gate_id in ref_rates
    }
    edge_priority = {
        gate_id: 0.85 if gate_id.startswith("top:") or gate_id.startswith("bottom:") else 0.45
        for gate_id in ref_rates
    }
    entry_priority = _normalize_budget(ref_rates, entry_priority, target_multiplier=0.6)
    return_priority = _normalize_budget(ref_rates, return_priority, target_multiplier=0.6)
    edge_priority = _normalize_budget(ref_rates, edge_priority, target_multiplier=0.6)

    bottleneck_middle = dict(high)
    if "middle:plus" in bottleneck_middle:
        bottleneck_middle["middle:plus"] = 0.3
    bottleneck_return = dict(high)
    if "lower_middle:minus" in bottleneck_return:
        bottleneck_return["lower_middle:minus"] = 0.3

    return [
        CapacityCase(
            case_id="g2v2_q_high",
            title="G2-v2 high capacity",
            family="level_scan",
            controls=_controls_from_multipliers(ref_rates, high, duration=duration),
            description="All active gates near no-cap demand.",
        ),
        CapacityCase(
            case_id="g2v2_q_medium",
            title="G2-v2 medium capacity",
            family="level_scan",
            controls=_controls_from_multipliers(ref_rates, medium, duration=duration),
            description="All active gates partly binding.",
        ),
        CapacityCase(
            case_id="g2v2_q_low",
            title="G2-v2 low capacity",
            family="level_scan",
            controls=_controls_from_multipliers(ref_rates, low, duration=duration),
            description="All active gates strongly binding.",
        ),
        CapacityCase(
            case_id="g2v2_bottleneck_middle_entry",
            title="G2-v2 middle entry bottleneck",
            family="single_bottleneck",
            controls=_controls_from_multipliers(ref_rates, bottleneck_middle, duration=duration),
            description="Only the middle eastbound entry is tightened.",
        ),
        CapacityCase(
            case_id="g2v2_bottleneck_return",
            title="G2-v2 return bottleneck",
            family="single_bottleneck",
            controls=_controls_from_multipliers(ref_rates, bottleneck_return, duration=duration),
            description="Only the lower-middle westbound return gate is tightened.",
        ),
        CapacityCase(
            case_id="g2v2_allocation_uniform",
            title="G2-v2 uniform medium allocation",
            family="allocation",
            controls=_controls_from_multipliers(ref_rates, medium, duration=duration),
            description="Same multiplier for every active gate.",
        ),
        CapacityCase(
            case_id="g2v2_allocation_entry_priority",
            title="G2-v2 entry-priority allocation",
            family="allocation",
            controls=_controls_from_multipliers(ref_rates, entry_priority, duration=duration),
            description="Same total budget, more capacity to plus/eastbound entry gates.",
        ),
        CapacityCase(
            case_id="g2v2_allocation_return_priority",
            title="G2-v2 return-priority allocation",
            family="allocation",
            controls=_controls_from_multipliers(ref_rates, return_priority, duration=duration),
            description="Same total budget, more capacity to minus/westbound return gates.",
        ),
        CapacityCase(
            case_id="g2v2_allocation_edge_priority",
            title="G2-v2 edge-priority allocation",
            family="allocation",
            controls=_controls_from_multipliers(ref_rates, edge_priority, duration=duration),
            description="Same total budget, more capacity to top and bottom gates.",
        ),
        CapacityCase(
            case_id="g2v2_schedule_front_loaded",
            title="G2-v2 front-loaded schedule",
            family="schedule",
            controls=_scheduled_controls(
                ref_rates,
                duration=duration,
                segments=[
                    (0.0, 0.5, {"*": 0.9}),
                    (0.5, 1.0, {"*": 0.35}),
                ],
            ),
            description="High early capacity followed by low late capacity.",
        ),
        CapacityCase(
            case_id="g2v2_schedule_return_priority",
            title="G2-v2 staged entry-return priority",
            family="schedule",
            controls=_scheduled_controls(
                ref_rates,
                duration=duration,
                segments=[
                    (0.0, 0.5, {"plus:*": 0.85, "minus:*": 0.45}),
                    (0.5, 1.0, {"plus:*": 0.45, "minus:*": 0.85}),
                ],
            ),
            description="Entry gates receive early priority, return gates receive late priority.",
        ),
        CapacityCase(
            case_id="g2v2_schedule_smooth_l4",
            title="G2-v2 four-segment smooth schedule",
            family="schedule",
            controls=_scheduled_controls(
                ref_rates,
                duration=duration,
                segments=[
                    (0.0, 0.25, {"*": 0.75}),
                    (0.25, 0.5, {"*": 0.65}),
                    (0.5, 0.75, {"*": 0.55}),
                    (0.75, 1.0, {"*": 0.65}),
                ],
            ),
            description="Four low-jump capacity segments for J_R comparison.",
        ),
    ]


def _run_case(
    *,
    case: CapacityCase,
    output_root: Path,
    simulation_overrides: dict[str, object],
    collectors: dict[str, tuple[G2StrategyCollector, Path]],
) -> dict[str, object]:
    config_path = _write_case_config(output_root=output_root, case=case)

    def observer_factory(**kwargs: object):
        compiled_case = kwargs["case"]
        run_spec = kwargs["run_spec"]
        simulation = kwargs["simulation"]
        case_output_dir = Path(str(kwargs["case_output_dir"]))
        collector = G2StrategyCollector(
            case_id=str(compiled_case.case_id),
            title=str(compiled_case.title),
            walkable=compiled_case.walkable,
            channel_masks=compiled_case.channel_masks,
            rho_safe=float(run_spec.objective.rho_safe),
            dx=float(simulation.dx),
            j2_metric=str(run_spec.objective.j2_metric),
            j2_gamma=float(run_spec.objective.j2_gamma),
        )
        collectors[str(compiled_case.case_id)] = (collector, case_output_dir)
        return collector.observe

    summary = run_from_config(
        config_path=config_path,
        output_root=output_root,
        simulation_overrides=simulation_overrides,
        write_root_summary=False,
        step_observer_factory=observer_factory,
        channel_flux_directions=FIXED_DIRECTIONS,
    )
    summary["g2_v2_capacity_response"] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "fixed_directions": FIXED_DIRECTIONS,
        "capacity_controls": [
            {
                "channel": control.channel,
                "side": control.side,
                "rate": float(control.rate),
                "time_start": float(control.time_start),
                "time_end": None if control.time_end is None else float(control.time_end),
            }
            for control in case.controls
        ],
        "config_path": str(config_path.resolve()),
    }
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)
    return summary


def _ref_rates_from_summary(summary: dict[str, object]) -> dict[str, float]:
    final_time = max(float(summary.get("final_time", 0.0)), 1.0)
    attempted = summary.get("gate_attempted_cumulative", {})
    if not isinstance(attempted, dict):
        raise ValueError("No gate_attempted_cumulative in no-cap summary")
    ref: dict[str, float] = {}
    for gate_id in _active_gate_ids(FIXED_DIRECTIONS):
        value = float(attempted.get(gate_id, 0.0)) / final_time
        ref[gate_id] = max(value, 0.2)
    return ref


def _gate_total(summary: dict[str, object], key: str) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(value) for value in raw.values()))


def _row_from_summary(summary: dict[str, object], behavior: dict[str, object]) -> dict[str, object]:
    meta = summary.get("g2_v2_capacity_response", {})
    if not isinstance(meta, dict):
        meta = {}
    objective = summary.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    objective_config = summary.get("objective_config", {})
    if not isinstance(objective_config, dict):
        objective_config = {}
    normalization_context = summary.get("normalization_context", {})
    if not isinstance(normalization_context, dict):
        normalization_context = {}
    mass_reference = float(normalization_context.get("total_mass_reference", 0.0))
    cap_removed = float(summary.get("final_cap_removed_cumulative", 0.0))
    return {
        "case_id": str(summary["case_id"]),
        "title": str(summary.get("title", "")),
        "family": str(meta.get("family", "reference")),
        "description": str(meta.get("description", "")),
        "j1": float(summary.get("j1_normalized", 0.0)),
        "j2": float(summary.get("j2_normalized", 0.0)),
        "j2_eval": float(objective.get("j2_eval", summary.get("j2_normalized", 0.0))),
        "j2_metric": str(objective_config.get("j2_metric", objective.get("j2_metric", ""))),
        "j2_gamma": float(objective_config.get("j2_gamma", objective.get("j2_gamma", 1.0))),
        "j2_scale": float(objective_config.get("j2_scale", objective.get("j2_scale", 1.0))),
        "j5": float(summary.get("j5_normalized", 0.0)),
        "jb": float(summary.get("jb_waiting_exposure", 0.0)),
        "objective_value": float(summary.get("objective_value", 0.0)),
        "peak_density": float(summary.get("peak_density_max", 0.0)),
        "sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
        "mass_reference": mass_reference,
        "cap_removed": cap_removed,
        "cap_removed_relative": cap_removed / max(mass_reference, 1.0e-12),
        "gate_attempted": _gate_total(summary, "gate_attempted_cumulative"),
        "gate_actual": _gate_total(summary, "gate_actual_cumulative"),
        "gate_rejected": _gate_total(summary, "gate_rejected_cumulative"),
        "waiting_mass_peak": max(
            [float(value) for value in dict(summary.get("gate_waiting_mass_peak", {})).values()] or [0.0]
        ),
        "binding_time_ratio_max": max(
            [float(value) for value in dict(summary.get("gate_binding_time_ratio", {})).values()] or [0.0]
        ),
        "hotspot_x": behavior.get("hotspot_centroid", {}).get("x"),
        "hotspot_y": behavior.get("hotspot_centroid", {}).get("y"),
        **{
            f"flux_share_{channel}": float(dict(summary.get("channel_flux_share", {})).get(channel, 0.0))
            for channel in CHANNEL_NAMES
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _non_dominated(rows: list[dict[str, object]]) -> list[str]:
    ids: list[str] = []
    metrics = ("j1", "j2", "j5", "jb")
    for row in rows:
        dominated = False
        values = [float(row[metric]) for metric in metrics]
        for other in rows:
            if other["case_id"] == row["case_id"]:
                continue
            other_values = [float(other[metric]) for metric in metrics]
            if all(o <= v for o, v in zip(other_values, values)) and any(o < v for o, v in zip(other_values, values)):
                dominated = True
                break
        if not dominated:
            ids.append(str(row["case_id"]))
    return ids


def _save_capacity_levels_plot(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if row["case_id"] in {"g2v2_q_inf", "g2v2_q_high", "g2v2_q_medium", "g2v2_q_low"}]
    order = {"g2v2_q_inf": 0, "g2v2_q_high": 1, "g2v2_q_medium": 2, "g2v2_q_low": 3}
    selected.sort(key=lambda row: order[str(row["case_id"])])
    labels = ["inf", "high", "medium", "low"]
    fig, ax1 = plt.subplots(1, 1, figsize=(8.2, 4.8), dpi=160)
    x = np.arange(len(selected))
    for metric in ("j1", "j2", "j5"):
        ax1.plot(x, [float(row[metric]) for row in selected], marker="o", label=metric)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("normalized objective term")
    ax1.set_title("Capacity-level response")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x, [float(row["jb"]) for row in selected], color="tab:red", marker="s", label="J_B")
    ax2.set_ylabel("waiting exposure J_B")
    lines, labels_left = ax1.get_legend_handles_labels()
    lines2, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_left + labels_right, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_pareto_plot(path: Path, rows: list[dict[str, object]], non_dominated_ids: list[str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.4), dpi=160)
    jb_values = np.array([float(row["jb"]) for row in rows], dtype=float)
    for row in rows:
        marker = "D" if row["case_id"] in non_dominated_ids else "o"
        ax.scatter(
            float(row["j1"]),
            float(row["j2"]),
            c=[float(row["jb"])],
            cmap="magma",
            vmin=float(jb_values.min()),
            vmax=float(jb_values.max()),
            s=90.0 + 180.0 * float(row["j5"]),
            marker=marker,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.88,
        )
        if row["case_id"] in non_dominated_ids:
            ax.annotate(str(row["case_id"]).replace("g2v2_", ""), (float(row["j1"]), float(row["j2"])), fontsize=7)
    ax.set_xlabel("J1 normalized")
    ax.set_ylabel("J2 normalized")
    ax.set_title("G2-v2 capacity trade-off projection")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(ax.collections[0], ax=ax)
    colorbar.set_label("J_B waiting exposure")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_allocation_loads(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [row for row in rows if str(row["family"]) == "allocation"]
    if not selected:
        return
    labels = [str(row["case_id"]).replace("g2v2_allocation_", "") for row in selected]
    x = np.arange(len(selected))
    width = 0.18
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8), dpi=160)
    for idx, channel in enumerate(CHANNEL_NAMES):
        ax.bar(
            x + (idx - 1.5) * width,
            [float(row[f"flux_share_{channel}"]) for row in selected],
            width=width,
            label=channel,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("channel flux share")
    ax.set_title("Same-budget allocation response")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_waiting_timeseries(path: Path, output_root: Path, case_ids: list[str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.8), dpi=160)
    for case_id in case_ids:
        ts_path = output_root / case_id / "timeseries.csv"
        if not ts_path.exists():
            continue
        times: list[float] = []
        waiting: list[float] = []
        with ts_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            gate_columns = [name for name in (reader.fieldnames or []) if name.startswith("gate_waiting_mass_")]
            for row in reader:
                times.append(float(row["time"]))
                waiting.append(sum(float(row[column]) for column in gate_columns))
        if times:
            ax.plot(times, waiting, label=case_id.replace("g2v2_", ""))
    ax.set_xlabel("time")
    ax.set_ylabel("sum gate waiting mass")
    ax.set_title("Waiting mass response")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_hotspot_plot(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), dpi=160)
    color_by_family = {
        "reference": "black",
        "level_scan": "#D95F02",
        "single_bottleneck": "#1B9E77",
        "allocation": "#7570B3",
        "schedule": "#E7298A",
    }
    for row in rows:
        if row["hotspot_x"] is None or row["hotspot_y"] is None:
            continue
        ax.scatter(
            float(row["hotspot_x"]),
            float(row["hotspot_y"]),
            s=60.0 + 120.0 * float(row["j2"]),
            color=color_by_family.get(str(row["family"]), "gray"),
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("High-density hotspot centroid migration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_tradeoff_table(path: Path, rows: list[dict[str, object]], non_dominated_ids: list[str]) -> None:
    sorted_rows = sorted(rows, key=lambda row: (str(row["family"]), str(row["case_id"])))
    lines = [
        "| case | family | J1 | J2 soft | J2 eval | J5 | J_B | rejected | binding | cap removed | non-dominated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted_rows:
        lines.append(
            "| {case} | {family} | {j1:.4f} | {j2:.6f} | {j2_eval:.3f} | {j5:.4f} | {jb:.2f} | {rej:.2f} | {bind:.2f} | {cap:.3%} | {nd} |".format(
                case=str(row["case_id"]),
                family=str(row["family"]),
                j1=float(row["j1"]),
                j2=float(row["j2"]),
                j2_eval=float(row["j2_eval"]),
                j5=float(row["j5"]),
                jb=float(row["jb"]),
                rej=float(row["gate_rejected"]),
                bind=float(row["binding_time_ratio_max"]),
                cap=float(row["cap_removed_relative"]),
                nd="yes" if row["case_id"] in non_dominated_ids else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relative_range(rows: list[dict[str, object]], key: str) -> float:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    return float((values.max() - values.min()) / max(float(np.mean(np.abs(values))), 1.0e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G2-v2 capacity-response validation before optimization.")
    parser.add_argument("--output-root", default="codes/results/g2_capacity_response")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--save-every", type=int, default=100000)
    parser.add_argument("--time-horizon", type=float, default=45.0)
    parser.add_argument("--bellman-every", type=int, default=4)
    parser.add_argument(
        "--density-contour-levels",
        default="off",
        help="Comma-separated density contour values, an integer count, or 'off'.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    simulation_overrides = {
        "steps": args.steps,
        "save_every": args.save_every,
        "time_horizon": args.time_horizon,
        "bellman_every": args.bellman_every,
        "density_contour_levels": parse_density_contour_levels(args.density_contour_levels),
    }

    collectors: dict[str, tuple[G2StrategyCollector, Path]] = {}

    ref_case = CapacityCase(
        case_id="g2v2_q_inf",
        title="G2-v2 no-cap reference",
        family="reference",
        controls=_inf_controls(),
        description="Capacity controls are present but set to infinity to record natural gate demand.",
    )
    summaries = [_run_case(case=ref_case, output_root=output_root, simulation_overrides=simulation_overrides, collectors=collectors)]
    ref_rates = _ref_rates_from_summary(summaries[0])

    for case in _build_capacity_cases(ref_rates, duration=float(args.time_horizon)):
        summaries.append(
            _run_case(
                case=case,
                output_root=output_root,
                simulation_overrides=simulation_overrides,
                collectors=collectors,
            )
        )

    behavior_summaries: list[dict[str, object]] = []
    for summary in summaries:
        collector, case_output_dir = collectors[str(summary["case_id"])]
        behavior_summaries.append(collector.save_case_outputs(case_output_dir))
    behavior_by_case = {str(item["case_id"]): item for item in behavior_summaries}
    rows = [_row_from_summary(summary, behavior_by_case.get(str(summary["case_id"]), {})) for summary in summaries]
    non_dominated_ids = _non_dominated(rows)
    for row in rows:
        row["is_non_dominated"] = str(row["case_id"]) in non_dominated_ids

    _write_csv(output_root / "g2_capacity_response_summary.csv", rows)
    _save_capacity_levels_plot(output_root / "g2_capacity_levels.png", rows)
    _save_pareto_plot(output_root / "g2_capacity_response_pareto.png", rows, non_dominated_ids)
    _save_allocation_loads(output_root / "g2_capacity_allocation_loads.png", rows)
    _save_waiting_timeseries(
        output_root / "g2_waiting_mass_timeseries.png",
        output_root,
        ["g2v2_q_inf", "g2v2_q_medium", "g2v2_q_low", "g2v2_schedule_front_loaded"],
    )
    _save_hotspot_plot(output_root / "g2_capacity_hotspot_migration.png", rows)
    _write_tradeoff_table(output_root / "g2_capacity_tradeoff_table.md", rows, non_dominated_ids)

    relative_ranges = {
        metric: _relative_range(rows, metric)
        for metric in ("j1", "j2", "j2_eval", "j5", "jb", "gate_rejected", "waiting_mass_peak")
    }
    max_cap_removed = max(float(row["cap_removed"]) for row in rows)
    max_cap_removed_relative = max(float(row["cap_removed_relative"]) for row in rows)
    allow_optimization = (
        len(non_dominated_ids) >= 2
        and max(relative_ranges["j1"], relative_ranges["j2"], relative_ranges["j5"], relative_ranges["jb"]) > 0.03
        and max_cap_removed_relative <= 0.02
    )
    report = {
        "experiment_group": "G2-v2",
        "design_version": "capacity_response_before_optimization",
        "fixed_directions": FIXED_DIRECTIONS,
        "baseline_config": str(BASELINE_CONFIG.resolve()),
        "output_root": str(output_root),
        "simulation_overrides": simulation_overrides,
        "reference_gate_rates": ref_rates,
        "case_count": len(rows),
        "non_dominated_cases": non_dominated_ids,
        "relative_ranges": relative_ranges,
        "max_cap_removed": max_cap_removed,
        "max_cap_removed_relative": max_cap_removed_relative,
        "cap_removed_relative_threshold": 0.02,
        "allow_g4_v2_optimization": bool(allow_optimization),
        "outputs": {
            "summary_csv": str(output_root / "g2_capacity_response_summary.csv"),
            "pareto": str(output_root / "g2_capacity_response_pareto.png"),
            "levels": str(output_root / "g2_capacity_levels.png"),
            "allocation_loads": str(output_root / "g2_capacity_allocation_loads.png"),
            "waiting_timeseries": str(output_root / "g2_waiting_mass_timeseries.png"),
            "hotspot_migration": str(output_root / "g2_capacity_hotspot_migration.png"),
            "tradeoff_table": str(output_root / "g2_capacity_tradeoff_table.md"),
        },
        "cases": summaries,
        "behavior_cases": behavior_summaries,
    }
    save_json(output_root / "g2_capacity_response_report.json", report)
    print(f"G2-v2 capacity response report: {output_root / 'g2_capacity_response_report.json'}")
    print(f"allow_g4_v2_optimization={allow_optimization}")
    print(f"non_dominated_cases={', '.join(non_dominated_ids)}")


if __name__ == "__main__":
    main()
