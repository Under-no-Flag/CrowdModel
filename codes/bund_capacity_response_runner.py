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

from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.metrics import save_json


DEFAULT_BASE_DIR = Path("anylogic-scene/BundScene/converted")
DEFAULT_GATES = ("channel_3:minus", "channel_4:plus")


@dataclass(frozen=True)
class GateRef:
    channel: str
    side: str

    @property
    def gate_id(self) -> str:
        return f"{self.channel}:{self.side}"


@dataclass(frozen=True)
class CapacityControl:
    gate: GateRef
    rate: float
    waiting_width: int = 6
    time_start: float = 0.0
    time_end: float | None = None


@dataclass(frozen=True)
class CapacityCase:
    case_id: str
    title: str
    family: str
    controls: tuple[CapacityControl, ...]
    description: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (_repo_root() / path).resolve()


def parse_gate_ref(value: str) -> GateRef:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Gate must use channel:side form, got {value!r}")
    channel, side = value.split(":", 1)
    side = side.lower()
    if not channel or side not in {"plus", "minus"}:
        raise argparse.ArgumentTypeError(f"Gate must use channel:plus or channel:minus form, got {value!r}")
    return GateRef(channel=channel, side=side)


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _format_scalar(value: object) -> str:
    if value is None:
        raise TypeError("None should be omitted from TOML output")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML scalar value: {value!r}")


def _dump_table(table_name: str, payload: dict[str, object]) -> list[str]:
    lines = [f"[{table_name}]"]
    for key, value in payload.items():
        if value is None:
            continue
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
            if not isinstance(controls, list):
                controls = []
            for control in controls:
                if not isinstance(control, dict):
                    continue
                lines.append("")
                lines.append("[[stages.controls]]")
                for key, value in control.items():
                    lines.append(f"{key} = {_format_scalar(value)}")
            targets = stage.get("targets", [])
            if not isinstance(targets, list):
                targets = []
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


def _capacity_controls_to_toml(controls: tuple[CapacityControl, ...]) -> list[dict[str, object]]:
    return [
        {
            "channel": control.gate.channel,
            "side": control.gate.side,
            "rate": float(control.rate),
            "time_start": float(control.time_start),
            "time_end": None if control.time_end is None else float(control.time_end),
            "waiting_width": int(control.waiting_width),
        }
        for control in controls
    ]


def _routes_for_case(base_routes: dict[str, object], case: CapacityCase) -> dict[str, object]:
    routes = deepcopy(base_routes)
    case_table = dict(routes.get("case", {}))
    case_table["case_id"] = case.case_id
    case_table["title"] = case.title
    routes["case"] = case_table
    routes["capacity_controls"] = _capacity_controls_to_toml(case.controls)
    return routes


def _write_case_config(*, base_dir: Path, output_root: Path, case: CapacityCase) -> Path:
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

    routes_path.write_text(_dump_routes_toml(_routes_for_case(base_routes, case)), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _controls_for_rate(gates: tuple[GateRef, ...], rate: float, *, waiting_width: int) -> tuple[CapacityControl, ...]:
    return tuple(
        CapacityControl(gate=gate, rate=float(rate), waiting_width=waiting_width)
        for gate in gates
    )


def _controls_from_ref_rates(
    ref_rates: dict[str, float],
    multiplier: float,
    *,
    waiting_width: int,
) -> tuple[CapacityControl, ...]:
    controls: list[CapacityControl] = []
    for gate_id, ref_rate in ref_rates.items():
        gate = parse_gate_ref(gate_id)
        controls.append(
            CapacityControl(
                gate=gate,
                rate=max(float(ref_rate) * float(multiplier), 1.0e-9),
                waiting_width=waiting_width,
            )
        )
    return tuple(controls)


def _reference_cases(
    gates: tuple[GateRef, ...],
    *,
    probe_rate: float,
    waiting_width: int,
) -> list[CapacityCase]:
    return [
        CapacityCase(
            case_id="bund_q_no_limit",
            title="Bund q no limit",
            family="reference",
            controls=(),
            description="Reference run without internal gate capacity controls.",
        ),
        CapacityCase(
            case_id="bund_q_probe_unlimited",
            title="Bund q probe unlimited",
            family="probe",
            controls=_controls_for_rate(gates, probe_rate, waiting_width=waiting_width),
            description="Large finite q used to measure attempted normal flux at selected gates.",
        ),
    ]


def _ref_rates_from_probe(
    summary: dict[str, object],
    gates: tuple[GateRef, ...],
    *,
    rate_floor: float,
) -> dict[str, float]:
    final_time = max(float(summary.get("final_time", 0.0)), 1.0e-9)
    attempted = summary.get("gate_attempted_cumulative", {})
    if not isinstance(attempted, dict):
        raise ValueError("Probe summary does not contain gate_attempted_cumulative")
    ref_rates: dict[str, float] = {}
    for gate in gates:
        average_attempted_rate = float(attempted.get(gate.gate_id, 0.0)) / final_time
        ref_rates[gate.gate_id] = max(average_attempted_rate, float(rate_floor))
    return ref_rates


def _level_cases(
    ref_rates: dict[str, float],
    *,
    multipliers: tuple[float, ...],
    waiting_width: int,
) -> list[CapacityCase]:
    cases: list[CapacityCase] = []
    for multiplier in multipliers:
        if math.isclose(multiplier, 0.9):
            suffix = "high"
        elif math.isclose(multiplier, 0.6):
            suffix = "medium"
        elif math.isclose(multiplier, 0.3):
            suffix = "low"
        else:
            suffix = f"m{multiplier:g}".replace(".", "p")
        cases.append(
            CapacityCase(
                case_id=f"bund_q_{suffix}",
                title=f"Bund q {suffix}",
                family="level_scan",
                controls=_controls_from_ref_rates(ref_rates, multiplier, waiting_width=waiting_width),
                description=f"Selected gates capped at {multiplier:g} times the probe average attempted rate.",
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


def _gate_total(summary: dict[str, object], key: str, gates: tuple[GateRef, ...]) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(raw.get(gate.gate_id, 0.0)) for gate in gates))


def _gate_max(summary: dict[str, object], key: str, gates: tuple[GateRef, ...]) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return max([float(raw.get(gate.gate_id, 0.0)) for gate in gates] or [0.0])


def _mean_control_rate(case: CapacityCase) -> float:
    if not case.controls:
        return math.inf
    return float(sum(control.rate for control in case.controls) / len(case.controls))


def _row_from_summary(summary: dict[str, object], case: CapacityCase, gates: tuple[GateRef, ...]) -> dict[str, object]:
    objective = summary.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    return {
        "case_id": str(summary["case_id"]),
        "title": str(summary.get("title", "")),
        "family": case.family,
        "description": case.description,
        "mean_control_rate": _mean_control_rate(case),
        "objective_value": float(summary.get("objective_value", 0.0)),
        "j1_eval": float(objective.get("j1_eval", summary.get("j1_normalized", 0.0))),
        "j2_eval": float(objective.get("j2_eval", summary.get("j2_normalized", 0.0))),
        "j5_eval": float(objective.get("j5_eval", summary.get("j5_normalized", 0.0))),
        "j1_normalized": float(summary.get("j1_normalized", 0.0)),
        "j2_normalized": float(summary.get("j2_normalized", 0.0)),
        "j5_normalized": float(summary.get("j5_normalized", 0.0)),
        "peak_density": float(summary.get("peak_density_max", 0.0)),
        "sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
        "cap_removed": float(summary.get("final_cap_removed_cumulative", 0.0)),
        "gate_attempted": _gate_total(summary, "gate_attempted_cumulative", gates),
        "gate_allowed": _gate_total(summary, "gate_allowed_cumulative", gates),
        "gate_actual": _gate_total(summary, "gate_actual_cumulative", gates),
        "gate_rejected": _gate_total(summary, "gate_rejected_cumulative", gates),
        "binding_time_ratio_max": _gate_max(summary, "gate_binding_time_ratio", gates),
        "waiting_mass_peak": _gate_max(summary, "gate_waiting_mass_peak", gates),
        "final_time": float(summary.get("final_time", 0.0)),
        "config_path": str(summary.get("config_path", "")),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _capacity_response_conclusion(rows: list[dict[str, object]], *, objective_tolerance: float) -> dict[str, object]:
    compared = [
        row
        for row in rows
        if row["case_id"] == "bund_q_no_limit" or row["family"] == "level_scan"
    ]
    objectives = [float(row["objective_value"]) for row in compared]
    objective_span = max(objectives) - min(objectives) if objectives else 0.0
    max_rejected = max([float(row["gate_rejected"]) for row in compared] or [0.0])
    max_binding = max([float(row["binding_time_ratio_max"]) for row in compared] or [0.0])
    max_attempted = max([float(row["gate_attempted"]) for row in rows] or [0.0])

    if max_attempted <= 1.0e-9:
        verdict = "inconclusive_no_selected_gate_demand"
    elif objective_span > objective_tolerance and (max_rejected > 1.0e-9 or max_binding > 0.0):
        verdict = "supports_capacity_control_changes_objective"
    elif max_rejected > 1.0e-9 or max_binding > 0.0:
        verdict = "gate_bound_but_objective_change_below_tolerance"
    else:
        verdict = "no_capacity_response_detected"

    return {
        "verdict": verdict,
        "objective_tolerance": float(objective_tolerance),
        "objective_span": float(objective_span),
        "max_gate_attempted": float(max_attempted),
        "max_gate_rejected": float(max_rejected),
        "max_binding_time_ratio": float(max_binding),
    }


def _save_level_plot(path: Path, rows: list[dict[str, object]]) -> None:
    selected = [
        row
        for row in rows
        if row["case_id"] in {"bund_q_no_limit", "bund_q_high", "bund_q_medium", "bund_q_low"}
    ]
    order = {"bund_q_no_limit": 0, "bund_q_high": 1, "bund_q_medium": 2, "bund_q_low": 3}
    selected.sort(key=lambda row: order[str(row["case_id"])])
    if len(selected) < 2:
        return

    labels = [
        str(row["case_id"]).replace("bund_q_", "").replace("_", " ")
        for row in selected
    ]
    x_values = list(range(len(selected)))

    fig, ax1 = plt.subplots(1, 1, figsize=(8.8, 5.0), dpi=170)
    ax1.plot(x_values, [float(row["objective_value"]) for row in selected], marker="o", label="objective")
    ax1.plot(x_values, [float(row["j1_eval"]) for row in selected], marker="s", label="J1 eval")
    ax1.plot(x_values, [float(row["j2_eval"]) for row in selected], marker="^", label="J2 eval")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("objective / evaluated terms")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        x_values,
        [float(row["gate_rejected"]) for row in selected],
        color="#D95F02",
        marker="D",
        label="rejected gate flow",
    )
    ax2.set_ylabel("cumulative rejected flow")

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
    case: CapacityCase,
    gates: tuple[GateRef, ...],
    simulation_overrides: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    config_path = _write_case_config(base_dir=base_dir, output_root=output_root, case=case)
    summary = run_from_config(
        config_path=config_path,
        output_root=output_root,
        simulation_overrides=simulation_overrides,
        write_root_summary=False,
    )
    summary["bund_capacity_response"] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "selected_gates": [gate.gate_id for gate in gates],
        "capacity_controls": _capacity_controls_to_toml(case.controls),
        "config_path": str(config_path.resolve()),
    }
    case_output_dir = output_root / str(summary["case_id"])
    save_json(case_output_dir / "summary.json", summary)
    return summary, _row_from_summary(summary, case, gates)


def run_capacity_response_experiment(args: argparse.Namespace) -> dict[str, object]:
    base_dir = _resolve_path(Path(args.base_dir))
    output_root = _resolve_path(Path(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    gate_values = args.gate if args.gate is not None else list(DEFAULT_GATES)
    gates = tuple(parse_gate_ref(value) for value in gate_values)
    multipliers = tuple(float(value) for value in args.level_multipliers.split(",") if value.strip())
    if not multipliers:
        raise ValueError("At least one level multiplier is required")

    simulation_overrides = _simulation_overrides_from_args(args)
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for case in _reference_cases(gates, probe_rate=args.probe_rate, waiting_width=args.waiting_width):
        summary, row = _run_case(
            base_dir=base_dir,
            output_root=output_root,
            case=case,
            gates=gates,
            simulation_overrides=simulation_overrides,
        )
        summaries[case.case_id] = summary
        rows.append(row)

    ref_rates = _ref_rates_from_probe(
        summaries["bund_q_probe_unlimited"],
        gates,
        rate_floor=args.reference_rate_floor,
    )
    save_json(output_root / "bund_capacity_reference_rates.json", ref_rates)

    for case in _level_cases(ref_rates, multipliers=multipliers, waiting_width=args.waiting_width):
        summary, row = _run_case(
            base_dir=base_dir,
            output_root=output_root,
            case=case,
            gates=gates,
            simulation_overrides=simulation_overrides,
        )
        summaries[case.case_id] = summary
        rows.append(row)

    conclusion = _capacity_response_conclusion(rows, objective_tolerance=args.objective_tolerance)
    payload = {
        "base_dir": str(base_dir),
        "output_root": str(output_root),
        "selected_gates": [gate.gate_id for gate in gates],
        "simulation_overrides": simulation_overrides,
        "reference_rates": ref_rates,
        "rows": rows,
        "conclusion": conclusion,
    }
    _write_csv(output_root / "bund_capacity_response_summary.csv", rows)
    save_json(output_root / "bund_capacity_response_summary.json", payload)
    if not args.no_plot:
        _save_level_plot(output_root / "bund_capacity_response_levels.png", rows)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether q capacity controls change the objective in the Bund AnyLogic scene. "
            "The script first probes attempted normal flux, then runs high/medium/low q levels."
        )
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Directory containing BundScene converted TOML files.")
    parser.add_argument("--output-root", default="codes/results/bund_capacity_response", help="Experiment output directory.")
    parser.add_argument("--gate", action="append", default=None, help="Selected gate as channel:plus/minus. Can be repeated. Defaults to channel_3:minus and channel_4:plus.")
    parser.add_argument("--level-multipliers", default="0.9,0.6,0.3", help="Comma-separated q multipliers relative to probe average attempted rates.")
    parser.add_argument("--probe-rate", type=float, default=1.0e6, help="Large finite q used for the probe case.")
    parser.add_argument("--reference-rate-floor", type=float, default=0.05, help="Minimum reference rate if the probe average is tiny.")
    parser.add_argument("--waiting-width", type=int, default=8, help="Cells behind each oriented gate used for waiting-mass diagnostics.")
    parser.add_argument("--objective-tolerance", type=float, default=1.0e-4, help="Objective span required to claim q changed the objective.")
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
    payload = run_capacity_response_experiment(args)
    conclusion = payload["conclusion"]
    print(f"summary: {Path(payload['output_root']) / 'bund_capacity_response_summary.json'}")
    print(f"csv: {Path(payload['output_root']) / 'bund_capacity_response_summary.csv'}")
    print(f"verdict: {conclusion['verdict']}")
    print(f"objective_span: {conclusion['objective_span']:.6g}")
    print(f"max_gate_rejected: {conclusion['max_gate_rejected']:.6g}")
    print(f"max_binding_time_ratio: {conclusion['max_binding_time_ratio']:.6g}")


if __name__ == "__main__":
    main()
