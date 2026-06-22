from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bund_capacity_response_runner import _dump_routes_toml, _dump_run_toml
from bund_pass_through_ablation_runner import (
    MONITORED_REGIONS,
    VARIANT_SPECS,
    _dump_scene_toml,
    _load_toml,
    add_center_goal_regions,
    make_route_variant,
)
from crowd_bellman.metrics import channel_flux_variance


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("anylogic-scene/BundScene/converted")


@dataclass(frozen=True)
class CaseSpec:
    label: str
    case_id: str
    apply_controls: bool


@dataclass(frozen=True)
class CaseCommand:
    spec: CaseSpec
    output_root: Path
    argv: list[str]
    stdout_log: Path
    stderr_log: Path


DEFAULT_CASE_SPECS = (
    CaseSpec(label="controlled", case_id="controlled", apply_controls=True),
    CaseSpec(label="uncontrolled", case_id="uncontrolled", apply_controls=False),
)


def _format_cli_number(value: int | float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12g}"


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def prepare_route_variant_base_dir(
    *,
    source_base_dir: Path,
    comparison_root: Path,
    route_variant: str,
    transition_kappa: float,
) -> Path:
    if route_variant == "base":
        return source_base_dir
    if route_variant not in VARIANT_SPECS:
        raise ValueError(f"Unsupported route variant: {route_variant}")

    generated_base_dir = comparison_root / "_generated_base" / route_variant
    generated_base_dir.mkdir(parents=True, exist_ok=True)

    base_run = _load_toml(source_base_dir / "run.toml")
    scene_file = source_base_dir / str(base_run["scene"]["file"])
    routes_file = source_base_dir / str(base_run["routes"]["file"])
    population_file = source_base_dir / str(base_run["population"]["file"])

    scene = _load_toml(scene_file)
    routes = _load_toml(routes_file)
    variant = VARIANT_SPECS[route_variant]
    if variant.mode == "center_goal":
        scene = add_center_goal_regions(scene, MONITORED_REGIONS)
    routes = make_route_variant(routes, variant, case_id=route_variant, kappa=transition_kappa)

    scene_path = generated_base_dir / "scene.toml"
    routes_path = generated_base_dir / "routes.toml"
    population_path = generated_base_dir / "population.toml"
    run_path = generated_base_dir / "run.toml"
    scene_path.write_text(_dump_scene_toml(scene), encoding="utf-8")
    routes_path.write_text(_dump_routes_toml(routes), encoding="utf-8")
    population_path.write_text(population_file.read_text(encoding="utf-8"), encoding="utf-8")

    generated_run = dict(base_run)
    generated_run["scene"] = {"file": scene_path.name}
    generated_run["routes"] = {"file": routes_path.name}
    generated_run["population"] = {"file": population_path.name}
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return generated_base_dir


def build_case_commands(
    *,
    comparison_root: Path,
    base_dir: Path,
    control_json: Path | None,
    steps: int,
    time_horizon: float,
    save_every: int,
    rho_max: float,
    inflow_rate_scale: float,
    transition_kappa: float,
    capacity_scale: float,
    field_dtype: str,
    python_executable: str,
) -> dict[str, CaseCommand]:
    commands: dict[str, CaseCommand] = {}
    runner_path = REPO_ROOT / "codes" / "bund_hcmbo_transfer_runner.py"
    for spec in DEFAULT_CASE_SPECS:
        output_root = comparison_root / spec.case_id
        argv = [
            python_executable,
            str(runner_path),
            "--base-dir",
            str(_resolve_path(base_dir)),
            "--output-root",
            str(output_root),
            "--case-id",
            spec.case_id,
            "--steps",
            str(int(steps)),
            "--time-horizon",
            _format_cli_number(time_horizon),
            "--save-every",
            str(int(save_every)),
            "--rho-max",
            _format_cli_number(rho_max),
            "--inflow-rate-scale",
            _format_cli_number(inflow_rate_scale),
            "--transition-kappa",
            _format_cli_number(transition_kappa),
            "--capacity-scale",
            _format_cli_number(capacity_scale),
            "--save-field-data",
            "--field-save-every",
            str(int(save_every)),
            "--field-dtype",
            field_dtype,
        ]
        if control_json is not None:
            argv.extend(["--control-json", str(_resolve_path(control_json))])
        argv.append("--apply-controls" if spec.apply_controls else "--no-controls")
        commands[spec.label] = CaseCommand(
            spec=spec,
            output_root=output_root,
            argv=argv,
            stdout_log=comparison_root / "logs" / f"{spec.label}.stdout.log",
            stderr_log=comparison_root / "logs" / f"{spec.label}.stderr.log",
        )
    return commands


def run_case_command(command: CaseCommand) -> None:
    command.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    command.output_root.mkdir(parents=True, exist_ok=True)
    command_file = command.output_root / "command.json"
    command_file.write_text(
        json.dumps(
            {
                "label": command.spec.label,
                "case_id": command.spec.case_id,
                "output_root": str(command.output_root),
                "argv": command.argv,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with command.stdout_log.open("w", encoding="utf-8") as stdout, command.stderr_log.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command.argv, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{command.spec.label} failed with exit code {completed.returncode}; see {command.stderr_log}")


def run_cases_parallel(commands: dict[str, CaseCommand], *, max_workers: int) -> None:
    worker_count = max(1, min(int(max_workers), len(commands)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_case_command, command): label for label, command in commands.items()}
        for future in as_completed(futures):
            future.result()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _fields_dir(case_dir: Path) -> Path:
    fields_dir = case_dir / "fields"
    if not (fields_dir / "fields_manifest.json").exists():
        raise FileNotFoundError(fields_dir / "fields_manifest.json")
    return fields_dir


def load_field_timeseries(fields_dir: Path, *, label: str) -> list[dict[str, object]]:
    manifest = _load_json(fields_dir / "fields_manifest.json")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"fields_manifest files must be a list: {fields_dir}")
    rows: list[dict[str, object]] = []
    for entry in sorted((item for item in files if isinstance(item, dict)), key=lambda item: int(item["step"])):
        field_path = fields_dir / str(entry["file"])
        with np.load(field_path) as payload:
            rho = np.asarray(payload["rho"], dtype=float)
            speed = np.asarray(payload["speed"], dtype=float)
            active = rho > 1.0e-8
            rows.append(
                {
                    "case": label,
                    "step": int(payload["step"][0]) if "step" in payload.files else int(entry["step"]),
                    "time": float(payload["time"][0]) if "time" in payload.files else float(entry.get("time", 0.0)),
                    "density_sum": float(np.sum(rho)),
                    "density_mean": float(np.mean(rho)),
                    "density_max": float(np.max(rho)),
                    "density_nonzero_cells": int(np.count_nonzero(active)),
                    "speed_mean": float(np.mean(speed[active])) if np.any(active) else 0.0,
                    "speed_max": float(np.max(speed)) if speed.size else 0.0,
                }
            )
    return rows


def _dict_total(summary: dict[str, object], key: str) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(value) for value in raw.values()))


def _float_from(mapping: dict[str, object], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def summarize_case_outputs(spec: CaseSpec, case_dir: Path, summary: dict[str, object]) -> dict[str, object]:
    objective_terms = summary.get("objective_terms_normalized", {})
    if not isinstance(objective_terms, dict):
        objective_terms = {}
    row: dict[str, object] = {
        "label": spec.label,
        "case_id": str(summary.get("case_id", spec.case_id)),
        "case_dir": str(case_dir),
        "apply_controls": bool(spec.apply_controls),
        "objective_value": _float_from(summary, "objective_value"),
        "j1_normalized": _float_from(summary, "j1_normalized", _float_from(objective_terms, "j1_total_travel_time")),
        "j2_normalized": _float_from(summary, "j2_normalized", _float_from(objective_terms, "j2_high_density_exposure")),
        "j5_normalized": _float_from(summary, "j5_normalized", _float_from(objective_terms, "j5_channel_flux_variance")),
        "final_time": _float_from(summary, "final_time"),
        "final_sink_cumulative": _float_from(summary, "final_sink_cumulative"),
        "final_inflow_cumulative": _float_from(summary, "final_inflow_cumulative"),
        "final_mass": _float_from(summary, "final_mass"),
        "peak_density_max": _float_from(summary, "peak_density_max"),
        "mean_density_avg": _float_from(summary, "mean_density_avg"),
        "velocity_discontinuity_avg": _float_from(summary, "velocity_discontinuity_avg"),
        "density_gradient_avg": _float_from(summary, "density_gradient_avg"),
        "gate_rejected_total": _dict_total(summary, "gate_rejected_cumulative"),
        "gate_attempted_total": _dict_total(summary, "gate_attempted_cumulative"),
        "gate_actual_total": _dict_total(summary, "gate_actual_cumulative"),
    }
    channel_flux_share = summary.get("channel_flux_share", {})
    if isinstance(channel_flux_share, dict):
        for channel, value in sorted(channel_flux_share.items()):
            row[f"channel_flux_share.{channel}"] = float(value)
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(rows: list[dict[str, object]], metrics: list[str], output: Path, *, title: str, ylabel: str) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * len(labels) + 1.4 * len(metrics)), 4.2), dpi=180)
    for index, metric in enumerate(metrics):
        values = [float(row.get(metric, 0.0)) for row in rows]
        ax.bar(x + (index - (len(metrics) - 1) / 2.0) * width, values, width=width, label=metric)
    ax.set_xticks(x, labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_timeseries(rows: list[dict[str, object]], metric: str, output: Path, *, title: str, ylabel: str) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=180)
    for label, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: int(item["step"]))
        ax.plot([float(item["time"]) for item in ordered], [float(item[metric]) for item in ordered], label=label, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_text(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _objective_setting(summary: dict[str, object], key: str, default: float) -> float:
    for container_key in ("objective", "objective_config"):
        raw = summary.get(container_key, {})
        if isinstance(raw, dict) and key in raw:
            return _float_text(raw.get(key), default)
    return float(default)


def _channel_columns(fieldnames: Iterable[str], prefix: str) -> list[str]:
    return sorted(name for name in fieldnames if name.startswith(prefix))


def _build_j5_series(
    timeseries_rows: list[dict[str, str]],
    *,
    final_j5_eval: float,
    j5_scale: float,
) -> tuple[list[float], str]:
    if not timeseries_rows:
        return [], "missing"
    fieldnames = list(timeseries_rows[0])
    flux_columns = _channel_columns(fieldnames, "channel_flux_cumulative_")
    if flux_columns:
        raw_values = [
            channel_flux_variance(
                {column.removeprefix("channel_flux_cumulative_"): _float_text(row.get(column)) for column in flux_columns}
            )
            for row in timeseries_rows
        ]
        final_raw = raw_values[-1] if raw_values else 0.0
        if final_raw > 1.0e-12 and math.isfinite(final_raw):
            factor = float(final_j5_eval) / final_raw
            return [float(value * factor) for value in raw_values], "channel_flux_cumulative_scaled_to_summary"
        return [0.0 for _ in raw_values], "channel_flux_cumulative_zero"

    density_columns = _channel_columns(fieldnames, "channel_density_")
    if not density_columns:
        return [0.0 for _ in timeseries_rows], "unavailable"

    cumulative = {column.removeprefix("channel_density_"): 0.0 for column in density_columns}
    proxy_values: list[float] = []
    for row in timeseries_rows:
        dt = _float_text(row.get("dt"))
        for column in density_columns:
            channel = column.removeprefix("channel_density_")
            cumulative[channel] += max(_float_text(row.get(column)), 0.0) * max(dt, 0.0)
        proxy_values.append(channel_flux_variance(cumulative))

    final_proxy = proxy_values[-1] if proxy_values else 0.0
    if final_proxy > 1.0e-12 and math.isfinite(final_proxy):
        factor = float(final_j5_eval) / final_proxy
        return [float(value * factor) for value in proxy_values], "channel_density_exposure_variance_proxy_scaled_to_summary"
    if len(proxy_values) <= 1:
        return [float(final_j5_eval) for _ in proxy_values], "channel_density_exposure_proxy_fallback"
    return [
        float(final_j5_eval) * index / float(len(proxy_values) - 1)
        for index, _ in enumerate(proxy_values)
    ], "linear_scaled_to_summary"


def load_objective_timeseries(case_dir: Path, *, label: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    summary = _load_json(case_dir / "summary.json")
    rows = _read_csv_rows(case_dir / "timeseries.csv")
    if not rows:
        return [], {"case": label, "j5_source": "missing", "rows": 0}

    objective_payload = summary.get("objective", {})
    if not isinstance(objective_payload, dict):
        objective_payload = {}
    normalization_context = summary.get("normalization_context", {})
    if not isinstance(normalization_context, dict):
        normalization_context = {}

    j1_denominator = _float_text(normalization_context.get("j1_denominator"))
    j2_denominator = _float_text(normalization_context.get("j2_denominator"))
    j1_scale = _objective_setting(summary, "j1_scale", 1.0)
    j2_scale = _objective_setting(summary, "j2_scale", 0.001)
    j5_scale = _objective_setting(summary, "j5_scale", 1.0)
    lambda_j1 = _objective_setting(summary, "lambda_j1", 1.0)
    lambda_j2 = _objective_setting(summary, "lambda_j2", 1.0)
    lambda_j5 = _objective_setting(summary, "lambda_j5", 1.0)
    final_j5_eval = _float_text(objective_payload.get("j5_eval"), _float_from(summary, "j5_normalized") / max(j5_scale, 1.0e-12))
    j5_series, j5_source = _build_j5_series(rows, final_j5_eval=final_j5_eval, j5_scale=j5_scale)

    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        j1_raw = _float_text(row.get("travel_time_cumulative"))
        j2_raw = _float_text(row.get("high_density_exposure_cumulative"), _float_text(row.get("j2_safety_risk_cumulative")))
        j1_normalized = j1_raw / j1_denominator if j1_denominator > 1.0e-12 else 0.0
        j2_normalized = j2_raw / j2_denominator if j2_denominator > 1.0e-12 else 0.0
        j1_eval = j1_normalized / max(j1_scale, 1.0e-12)
        j2_eval = j2_normalized / max(j2_scale, 1.0e-12)
        j5_eval = float(j5_series[index]) if index < len(j5_series) else 0.0
        objective_value = lambda_j1 * j1_eval + lambda_j2 * j2_eval + lambda_j5 * j5_eval
        output_rows.append(
            {
                "case": label,
                "step": int(index),
                "time": _float_text(row.get("time")),
                "j1_raw": j1_raw,
                "j2_raw": j2_raw,
                "j1_normalized": j1_normalized,
                "j2_normalized": j2_normalized,
                "j5_eval": j5_eval,
                "j1_eval": j1_eval,
                "j2_eval": j2_eval,
                "objective_value": objective_value,
                "j5_source": j5_source,
            }
        )

    metadata = {
        "case": label,
        "rows": len(output_rows),
        "j5_source": j5_source,
        "final_objective_from_timeseries": output_rows[-1]["objective_value"],
        "summary_objective_value": _float_from(summary, "objective_value"),
        "final_j1_eval_from_timeseries": output_rows[-1]["j1_eval"],
        "summary_j1_eval": _float_text(objective_payload.get("j1_eval"), _float_from(summary, "j1_normalized")),
        "final_j2_eval_from_timeseries": output_rows[-1]["j2_eval"],
        "summary_j2_eval": _float_text(objective_payload.get("j2_eval"), _float_from(summary, "j2_normalized") / max(j2_scale, 1.0e-12)),
        "final_j5_eval_from_timeseries": output_rows[-1]["j5_eval"],
        "summary_j5_eval": final_j5_eval,
    }
    return output_rows, metadata


def _smooth(values: list[float], window: int = 9) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size < 3 or window <= 1:
        return data
    actual_window = min(int(window), data.size)
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window <= 1:
        return data
    kernel = np.ones(actual_window, dtype=float) / float(actual_window)
    pad = actual_window // 2
    padded = np.pad(data, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _group_objective_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    return {label: sorted(items, key=lambda item: float(item["time"])) for label, items in sorted(grouped.items())}


def _plot_objective_stacked_timeseries(rows: list[dict[str, object]], output: Path) -> None:
    grouped = _group_objective_rows(rows)
    if not grouped:
        return
    fig, axes = plt.subplots(len(grouped), 1, figsize=(11.5, 3.4 * len(grouped)), dpi=180, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = ["#cfe3ff", "#3b82f6", "#7c3aed"]
    labels = ["J1 travel", "J2 density risk", "J5 balance"]
    for ax, (case, items) in zip(axes, grouped.items()):
        time = np.asarray([float(item["time"]) for item in items], dtype=float)
        j1 = _smooth([float(item["j1_eval"]) for item in items])
        j2 = _smooth([float(item["j2_eval"]) for item in items])
        j5 = _smooth([float(item["j5_eval"]) for item in items])
        objective = _smooth([float(item["objective_value"]) for item in items])
        ax.stackplot(time, j1, j2, j5, labels=labels, colors=colors, alpha=0.96)
        ax.plot(time, objective, color="#4338ca", linewidth=2.0, label="J total")
        ax.set_title(f"{case}: objective terms over time")
        ax.set_ylabel("J contribution")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", ncols=4, fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_objective_line_timeseries(rows: list[dict[str, object]], output: Path) -> None:
    grouped = _group_objective_rows(rows)
    if not grouped:
        return
    metrics = [
        ("j1_eval", "J1 travel"),
        ("j2_eval", "J2 density risk"),
        ("j5_eval", "J5 balance"),
        ("objective_value", "J total"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11.0, 9.0), dpi=180, sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for case, items in grouped.items():
            time = [float(item["time"]) for item in items]
            values = _smooth([float(item[metric]) for item in items])
            ax.plot(time, values, linewidth=1.8, label=case)
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def generate_objective_timeseries_outputs(comparison_root: Path) -> dict[str, object]:
    objective_rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for spec in DEFAULT_CASE_SPECS:
        rows, payload = load_objective_timeseries(comparison_root / spec.case_id / spec.case_id, label=spec.label)
        objective_rows.extend(rows)
        metadata.append(payload)
    _write_csv(comparison_root / "objective_timeseries.csv", objective_rows)
    (comparison_root / "objective_timeseries_metadata.json").write_text(
        json.dumps({"cases": metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    figures_dir = comparison_root / "figures"
    _plot_objective_stacked_timeseries(objective_rows, figures_dir / "all_j_timeseries_stacked.png")
    _plot_objective_line_timeseries(objective_rows, figures_dir / "all_j_timeseries_lines.png")
    return {"objective_timeseries_csv": str(comparison_root / "objective_timeseries.csv"), "objective_timeseries_metadata": metadata}


def _plot_channel_flux_share(rows: list[dict[str, object]], output: Path) -> None:
    channels = sorted({key.split(".", 1)[1] for row in rows for key in row if key.startswith("channel_flux_share.")})
    if not channels:
        return
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(channels), dtype=float)
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=180)
    for index, row in enumerate(rows):
        values = [float(row.get(f"channel_flux_share.{channel}", 0.0)) for channel in channels]
        ax.bar(x + (index - (len(rows) - 1) / 2.0) * width, values, width=width, label=labels[index])
    ax.set_xticks(x, channels)
    ax.set_title("Channel Flux Share")
    ax.set_ylabel("share")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_visualization_plan(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Bund Control Comparison Visualization Plan",
                "",
                "- `comparison_summary.csv`: controlled/uncontrolled final metrics and flattened channel shares.",
                "- `field_timeseries.csv`: every saved field frame, including density sum/mean/max and speed mean/max.",
                "- `objective_timeseries.csv`: per-step J1/J2/J5/J-total time series reconstructed from case timeseries.",
                "- `objective_timeseries_metadata.json`: reconstruction source and final-value consistency checks.",
                "- `figures/final_metrics_bar.png`: objective, sink, mass, peak density summary.",
                "- `figures/objective_terms_bar.png`: normalized J1/J2/J5 comparison.",
                "- `figures/all_j_timeseries_stacked.png`: stacked J1/J2/J5 contribution curves over time.",
                "- `figures/all_j_timeseries_lines.png`: line comparison for J1, J2, J5, and total objective over time.",
                "- `figures/density_sum_timeseries.png`: saved density mass proxy over time.",
                "- `figures/density_max_timeseries.png`: peak density over time.",
                "- `figures/density_nonzero_cells_timeseries.png`: occupied-area proxy over time.",
                "- `figures/speed_mean_timeseries.png`: density-active mean speed over time.",
                "- `figures/channel_flux_share_bar.png`: final channel flux share comparison.",
                "- `figures/gate_totals_bar.png`: attempted/actual/rejected gate flow totals.",
                "- `<case>/refined_density_vector_walls_vmax6_full_every10`: high-resolution density frames with continuous vector walls.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def generate_comparison_outputs(comparison_root: Path) -> dict[str, object]:
    summary_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    case_dirs: dict[str, str] = {}
    for spec in DEFAULT_CASE_SPECS:
        case_dir = comparison_root / spec.case_id / spec.case_id
        summary = _load_json(case_dir / "summary.json")
        summary_rows.append(summarize_case_outputs(spec, case_dir, summary))
        field_rows.extend(load_field_timeseries(_fields_dir(case_dir), label=spec.label))
        case_dirs[spec.label] = str(case_dir)

    _write_csv(comparison_root / "comparison_summary.csv", summary_rows)
    _write_csv(comparison_root / "field_timeseries.csv", field_rows)
    figures_dir = comparison_root / "figures"
    _plot_bar(
        summary_rows,
        ["objective_value", "final_sink_cumulative", "final_mass", "peak_density_max"],
        figures_dir / "final_metrics_bar.png",
        title="Bund Control Comparison: Final Metrics",
        ylabel="value",
    )
    _plot_bar(
        summary_rows,
        ["j1_normalized", "j2_normalized", "j5_normalized"],
        figures_dir / "objective_terms_bar.png",
        title="Normalized Objective Terms",
        ylabel="normalized value",
    )
    _plot_bar(
        summary_rows,
        ["gate_attempted_total", "gate_actual_total", "gate_rejected_total"],
        figures_dir / "gate_totals_bar.png",
        title="Gate Flow Totals",
        ylabel="cumulative flow",
    )
    _plot_channel_flux_share(summary_rows, figures_dir / "channel_flux_share_bar.png")
    _plot_timeseries(field_rows, "density_sum", figures_dir / "density_sum_timeseries.png", title="Density Sum Over Time", ylabel="sum(rho)")
    _plot_timeseries(field_rows, "density_max", figures_dir / "density_max_timeseries.png", title="Peak Density Over Time", ylabel="max(rho)")
    _plot_timeseries(
        field_rows,
        "density_nonzero_cells",
        figures_dir / "density_nonzero_cells_timeseries.png",
        title="Occupied Cell Count Over Time",
        ylabel="cells",
    )
    _plot_timeseries(field_rows, "speed_mean", figures_dir / "speed_mean_timeseries.png", title="Mean Active Speed Over Time", ylabel="speed")
    objective_outputs = generate_objective_timeseries_outputs(comparison_root)
    write_visualization_plan(comparison_root / "visualization_plan.md")
    return {"summary_rows": summary_rows, "field_rows": field_rows, "case_dirs": case_dirs, **objective_outputs}


def render_high_resolution_density(
    comparison_root: Path,
    *,
    python_executable: str,
    scale: int,
    smooth_sigma: float,
    vmax: float,
    dpi: int,
    max_workers: int,
) -> None:
    renderer = REPO_ROOT / "codes" / "render_refined_density_heatmap.py"
    commands: list[tuple[str, list[str], Path, Path]] = []
    for spec in DEFAULT_CASE_SPECS:
        case_dir = comparison_root / spec.case_id / spec.case_id
        output_dir = case_dir / "refined_density_vector_walls_vmax6_full_every10"
        argv = [
            python_executable,
            str(renderer),
            str(case_dir),
            "--all",
            "--output-dir",
            str(output_dir),
            "--scale",
            str(int(scale)),
            "--smooth-sigma",
            _format_cli_number(smooth_sigma),
            "--color-scale",
            "absolute",
            "--vmax",
            _format_cli_number(vmax),
            "--cmap",
            "low-density",
            "--norm",
            "power",
            "--gamma",
            "0.42",
            "--fusion-mode",
            "wall-preserve",
            "--density-alpha",
            "1.0",
            "--overlay-threshold",
            "0.005",
            "--alpha-gamma",
            "0.32",
            "--dpi",
            str(int(dpi)),
            "--no-crop",
            "--no-background",
            "--nonwalkable-fill",
            "zero",
            "--wall-overlay",
            "vector",
            "--vector-wall-color",
            "#111111",
        ]
        commands.append(
            (
                spec.label,
                argv,
                comparison_root / "logs" / f"{spec.label}.hq.stdout.log",
                comparison_root / "logs" / f"{spec.label}.hq.stderr.log",
            )
        )

    def run_render(item: tuple[str, list[str], Path, Path]) -> None:
        label, argv, stdout_path, stderr_path = item
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(argv, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{label} high-resolution rendering failed with exit code {completed.returncode}; see {stderr_path}")

    worker_count = max(1, min(int(max_workers), len(commands)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_render, command) for command in commands]
        for future in as_completed(futures):
            future.result()


def write_comparison_config(
    comparison_root: Path,
    args: argparse.Namespace,
    commands: dict[str, CaseCommand],
    *,
    effective_base_dir: Path,
) -> None:
    comparison_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "comparison_root": str(comparison_root),
        "parameters": vars(args),
        "effective_base_dir": str(effective_base_dir),
        "control_json": None if args.control_json is None else str(_resolve_path(Path(args.control_json))),
        "cases": {
            label: {
                "spec": asdict(command.spec),
                "output_root": str(command.output_root),
                "stdout_log": str(command.stdout_log),
                "stderr_log": str(command.stderr_log),
                "argv": command.argv,
            }
            for label, command in commands.items()
        },
    }
    (comparison_root / "comparison_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    comparison_root = _resolve_path(args.output_root) if args.output_root else (
        REPO_ROOT / "codes" / "results" / f"bund_control_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    base_dir = prepare_route_variant_base_dir(
        source_base_dir=_resolve_path(Path(args.base_dir)),
        comparison_root=comparison_root,
        route_variant=args.route_variant,
        transition_kappa=args.transition_kappa,
    )
    commands = build_case_commands(
        comparison_root=comparison_root,
        base_dir=base_dir,
        control_json=None if args.control_json is None else Path(args.control_json),
        steps=args.steps,
        time_horizon=args.time_horizon,
        save_every=args.save_every,
        rho_max=args.rho_max,
        inflow_rate_scale=args.inflow_rate_scale,
        transition_kappa=args.transition_kappa,
        capacity_scale=args.capacity_scale,
        field_dtype=args.field_dtype,
        python_executable=args.python_executable,
    )
    write_comparison_config(comparison_root, args, commands, effective_base_dir=base_dir)
    if not args.skip_runs:
        run_cases_parallel(commands, max_workers=args.max_workers)
    outputs = generate_comparison_outputs(comparison_root)
    if not args.skip_hq:
        render_high_resolution_density(
            comparison_root,
            python_executable=args.python_executable,
            scale=args.hq_scale,
            smooth_sigma=args.hq_smooth_sigma,
            vmax=args.hq_vmax,
            dpi=args.hq_dpi,
            max_workers=args.max_workers,
        )
    manifest = {
        "comparison_root": str(comparison_root),
        "route_variant": str(args.route_variant),
        "effective_base_dir": str(base_dir),
        "cases": outputs["case_dirs"],
        "comparison_summary_csv": str(comparison_root / "comparison_summary.csv"),
        "field_timeseries_csv": str(comparison_root / "field_timeseries.csv"),
        "figures_dir": str(comparison_root / "figures"),
        "visualization_plan": str(comparison_root / "visualization_plan.md"),
        "objective_timeseries_csv": str(comparison_root / "objective_timeseries.csv"),
        "high_resolution_density_dirs": {
            spec.label: str(comparison_root / spec.case_id / spec.case_id / "refined_density_vector_walls_vmax6_full_every10")
            for spec in DEFAULT_CASE_SPECS
        },
    }
    (comparison_root / "comparison_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled vs uncontrolled Bund AnyLogic comparison and post-process visualizations.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Converted BundScene directory containing run.toml/scene.toml/routes.toml.")
    parser.add_argument(
        "--route-variant",
        choices=("base", *tuple(VARIANT_SPECS)),
        default="base",
        help="Route geometry variant used before applying controlled/uncontrolled settings.",
    )
    parser.add_argument("--output-root", default=None, help="Comparison output root. Defaults to codes/results/bund_control_comparison_<timestamp>.")
    parser.add_argument("--control-json", default=None, help="HCMBO control JSON used for the controlled case. Defaults to the transfer runner's G6 control.")
    parser.add_argument("--steps", type=int, default=1600, help="Simulation steps for each case.")
    parser.add_argument("--time-horizon", type=float, default=50.0, help="Physical time horizon for each case.")
    parser.add_argument("--save-every", type=int, default=10, help="Snapshot and field-data save interval in steps.")
    parser.add_argument("--rho-max", type=float, default=4.0, help="Maximum density for both cases.")
    parser.add_argument("--inflow-rate-scale", type=float, default=6.0, help="Continuous inflow multiplier for both cases.")
    parser.add_argument("--transition-kappa", type=float, default=32.0, help="Stage transition kappa for both cases.")
    parser.add_argument("--capacity-scale", type=float, default=2.0, help="Transferred G6 gate-capacity multiplier for the controlled case.")
    parser.add_argument("--field-dtype", choices=("float32", "float64"), default="float32", help="Saved field array dtype.")
    parser.add_argument("--max-workers", type=int, default=2, help="Parallel workers for case runs and high-resolution rendering.")
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used for child processes.")
    parser.add_argument("--skip-runs", action="store_true", help="Do not run simulations; only post-process existing outputs.")
    parser.add_argument("--skip-hq", action="store_true", help="Do not render high-resolution density PNGs.")
    parser.add_argument("--hq-scale", type=int, default=8, help="Density upsampling scale for high-resolution PNGs.")
    parser.add_argument("--hq-smooth-sigma", type=float, default=5.0, help="Smoothing sigma for high-resolution PNGs.")
    parser.add_argument("--hq-vmax", type=float, default=6.0, help="Absolute density vmax used in high-resolution PNGs.")
    parser.add_argument("--hq-dpi", type=int, default=240, help="DPI for high-resolution PNGs.")
    return parser


def main() -> None:
    manifest = run_comparison(build_arg_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
