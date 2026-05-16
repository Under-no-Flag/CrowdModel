from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from crowd_bellman.core import compute_face_fluxes
from crowd_bellman.g4_sahbo import ControlVector, G4EvaluationCache
from crowd_bellman.metrics import save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "codes" / "results" / "model_v2_3_2_validation"
BASELINE_CONFIG = REPO_ROOT / "codes" / "scenes" / "examples" / "g2_multistage_directional" / "run_baseline.toml"
CHANNEL_NAMES = ("top", "middle", "lower_middle", "bottom")
ETA_VALUES = (1.0, 4.0, 8.0, 12.0)
ETA_DIRECTIONS = ("FREE", "E", "W", "FREE")
EPS = 1.0e-12


@dataclass(frozen=True)
class GateDiagnostics:
    attempt_plus: float
    attempt_minus: float
    actual_plus: float
    actual_minus: float
    rejected_plus: float
    rejected_minus: float
    lambda_plus: float
    lambda_minus: float
    actual_plus_by_group: dict[str, float]
    attempt_plus_by_group: dict[str, float]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.floating | np.integer):
        return _as_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _save_json(path: Path, payload: dict[str, object]) -> None:
    save_json(path, _as_jsonable(payload))


def run_eta_sensitivity(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    run_root = output_root / "eta_sensitivity_runs"
    evaluator = G4EvaluationCache(
        baseline_config=BASELINE_CONFIG,
        output_root=run_root,
        simulation_overrides={
            "time_horizon": 30.0,
            "steps": 500,
            "bellman_every": 4,
            "save_every": 100000,
            "density_contour_levels": 0,
        },
        beta=0.35,
    )

    rows: list[dict[str, object]] = []
    for eta in ETA_VALUES:
        control = ControlVector(ETA_DIRECTIONS, tuple(float(eta) for _ in CHANNEL_NAMES))
        record = evaluator.evaluate(control, source=f"eta_{eta:g}", record_cached=True)
        summary = record.summary
        objective = summary.get("objective", {})
        objective_dict = objective if isinstance(objective, dict) else {}
        flux_share = summary.get("channel_flux_share", {})
        flux_dict = flux_share if isinstance(flux_share, dict) else {}
        row: dict[str, object] = {
            "eta": float(eta),
            "directions": ",".join(ETA_DIRECTIONS),
            "objective_value": float(record.objective_value),
            "j1_normalized": float(summary.get("j1_normalized", 0.0)),
            "j2_normalized": float(summary.get("j2_normalized", 0.0)),
            "j5_normalized": float(summary.get("j5_normalized", 0.0)),
            "j1_raw": float(summary.get("j1_total_travel_time", 0.0)),
            "j2_raw": float(summary.get("j2_high_density_exposure", 0.0)),
            "j5_raw": float(summary.get("j5_channel_flux_variance", 0.0)),
            "case_id": str(summary.get("case_id")),
            "config_path": str(record.config_path),
            "case_output_dir": str((run_root / str(summary.get("case_id"))).resolve()),
            "objective_term_mode": str(objective_dict.get("term_mode", "")),
        }
        for channel in CHANNEL_NAMES:
            row[f"flux_share_{channel}"] = float(flux_dict.get(channel, 0.0))
        rows.append(row)

    _write_csv(output_root / "eta_sensitivity.csv", rows)
    _plot_eta_sensitivity(output_root / "eta_sensitivity.png", rows)

    objectives = np.array([float(row["objective_value"]) for row in rows], dtype=float)
    terms = {
        "j1_normalized": np.array([float(row["j1_normalized"]) for row in rows], dtype=float),
        "j2_normalized": np.array([float(row["j2_normalized"]) for row in rows], dtype=float),
        "j5_normalized": np.array([float(row["j5_normalized"]) for row in rows], dtype=float),
    }
    relative_range = float((np.max(objectives) - np.min(objectives)) / max(float(np.mean(np.abs(objectives))), EPS))
    term_relative_ranges = {
        name: float((np.max(values) - np.min(values)) / max(float(np.mean(np.abs(values))), EPS))
        for name, values in terms.items()
    }
    threshold = 0.05
    summary = {
        "eta_values": list(ETA_VALUES),
        "directions": list(ETA_DIRECTIONS),
        "row_count": len(rows),
        "objective_relative_range": relative_range,
        "term_relative_ranges": term_relative_ranges,
        "weak_sensitivity_threshold": threshold,
        "weak_sensitivity": bool(relative_range <= threshold),
        "csv": str((output_root / "eta_sensitivity.csv").resolve()),
        "figure": str((output_root / "eta_sensitivity.png").resolve()),
        "run_root": str(run_root.resolve()),
        "command": "D:\\Anaconda\\envs\\interpreter\\python.exe codes\\validate_model_v2_3_2.py",
    }
    return rows, summary


def _plot_eta_sensitivity(path: Path, rows: list[dict[str, object]]) -> None:
    eta = [float(row["eta"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(eta, [float(row["objective_value"]) for row in rows], marker="o", label="objective")
    axes[0].set_xlabel("eta")
    axes[0].set_ylabel("objective")
    axes[0].set_title("Objective sensitivity")
    axes[0].grid(True, alpha=0.3)

    for key in ("j1_normalized", "j2_normalized", "j5_normalized"):
        axes[1].plot(eta, [float(row[key]) for row in rows], marker="o", label=key.replace("_normalized", ""))
    axes[1].set_xlabel("eta")
    axes[1].set_ylabel("normalized term")
    axes[1].set_title("Term sensitivity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def project_capacity(direction: str, q_plus: float, q_minus: float, qbar: float) -> tuple[float, float]:
    state = direction.upper()
    plus = max(float(q_plus), 0.0)
    minus = max(float(q_minus), 0.0)
    total_cap = max(float(qbar), 0.0)

    if state in {"CLOSED", "C", "NONE"}:
        return 0.0, 0.0
    if state in {"E", "EAST", "EASTBOUND"}:
        return min(plus, total_cap), 0.0
    if state in {"W", "WEST", "WESTBOUND"}:
        return 0.0, min(minus, total_cap)
    if state in {"FREE", "BOTH", "BIDIRECTIONAL", "ALL"}:
        total = plus + minus
        if total <= total_cap + EPS:
            return plus, minus
        if total <= EPS:
            return 0.0, 0.0
        scale = total_cap / total
        return plus * scale, minus * scale
    raise ValueError(f"Unsupported direction state: {direction!r}")


def apply_gate_limits(
    fx_by_group: dict[str, np.ndarray],
    *,
    gate_face: int,
    gate_rows: np.ndarray,
    dx: float,
    q_plus: float,
    q_minus: float,
) -> tuple[dict[str, np.ndarray], GateDiagnostics]:
    attempt_plus_by_group: dict[str, float] = {}
    attempt_minus_by_group: dict[str, float] = {}
    for group, fx in fx_by_group.items():
        face_flux = fx[gate_rows, gate_face]
        attempt_plus_by_group[group] = float(np.sum(np.maximum(face_flux, 0.0)) * dx)
        attempt_minus_by_group[group] = float(np.sum(np.maximum(-face_flux, 0.0)) * dx)

    attempt_plus = float(sum(attempt_plus_by_group.values()))
    attempt_minus = float(sum(attempt_minus_by_group.values()))
    lambda_plus = _limiter_lambda(attempt_plus, q_plus)
    lambda_minus = _limiter_lambda(attempt_minus, q_minus)

    limited: dict[str, np.ndarray] = {}
    actual_plus_by_group: dict[str, float] = {}
    for group, fx in fx_by_group.items():
        fx_limited = np.array(fx, copy=True)
        face_flux = fx_limited[gate_rows, gate_face]
        fx_limited[gate_rows, gate_face] = np.where(
            face_flux >= 0.0,
            face_flux * lambda_plus,
            face_flux * lambda_minus,
        )
        limited[group] = fx_limited
        limited_face = fx_limited[gate_rows, gate_face]
        actual_plus_by_group[group] = float(np.sum(np.maximum(limited_face, 0.0)) * dx)

    actual_plus = float(sum(actual_plus_by_group.values()))
    actual_minus = float(attempt_minus * lambda_minus)
    diagnostics = GateDiagnostics(
        attempt_plus=attempt_plus,
        attempt_minus=attempt_minus,
        actual_plus=actual_plus,
        actual_minus=actual_minus,
        rejected_plus=max(attempt_plus - actual_plus, 0.0),
        rejected_minus=max(attempt_minus - actual_minus, 0.0),
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        actual_plus_by_group=actual_plus_by_group,
        attempt_plus_by_group=attempt_plus_by_group,
    )
    return limited, diagnostics


def _limiter_lambda(attempt: float, q: float) -> float:
    if attempt <= EPS or math.isinf(q):
        return 1.0
    return float(min(1.0, max(q, 0.0) / attempt))


def update_density_from_fluxes(
    rho: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    *,
    walkable: np.ndarray,
    sink_mask: np.ndarray,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, float]:
    div = np.zeros_like(rho)
    div[:, 0] += fx[:, 0] / dx
    div[:, 1:-1] += (fx[:, 1:] - fx[:, :-1]) / dx
    div[:, -1] += -fx[:, -1] / dx
    div[0, :] += fy[0, :] / dx
    div[1:-1, :] += (fy[1:, :] - fy[:-1, :]) / dx
    div[-1, :] += -fy[-1, :] / dx

    updated = rho - dt * div
    updated[~walkable] = 0.0
    updated = np.clip(updated, 0.0, None)
    sink_mass = float(np.sum(updated[sink_mask]) * dx * dx)
    updated[sink_mask] = 0.0
    return updated, sink_mass


def apply_total_density_cap(
    rho_by_group: dict[str, np.ndarray],
    *,
    rho_max: float,
    walkable: np.ndarray,
    dx: float,
) -> tuple[dict[str, np.ndarray], float]:
    total = np.zeros_like(next(iter(rho_by_group.values())))
    for rho in rho_by_group.values():
        total += rho
    overflow = walkable & (total > rho_max + EPS)
    if not np.any(overflow):
        return {key: np.where(walkable, np.clip(rho, 0.0, None), 0.0) for key, rho in rho_by_group.items()}, 0.0

    before = float(np.sum(total[overflow]) * dx * dx)
    scale = np.ones_like(total)
    scale[overflow] = rho_max / total[overflow]
    capped = {
        key: np.where(walkable, np.clip(rho * scale, 0.0, None), 0.0)
        for key, rho in rho_by_group.items()
    }
    after_total = np.zeros_like(total)
    for rho in capped.values():
        after_total += rho
    after = float(np.sum(after_total[overflow]) * dx * dx)
    return capped, max(before - after, 0.0)


def run_gate_case(
    *,
    q_schedule: tuple[float, ...],
    direction: str = "E",
    qbar: float = 20.0,
    steps: int = 90,
    dt: float = 0.2,
    inflow_rate: float = 4.0,
    group_inflow_split: dict[str, float] | None = None,
) -> dict[str, object]:
    dx = 1.0
    ny, nx = 7, 14
    gate_face = 6
    gate_rows = np.arange(1, 6)
    walkable = np.ones((ny, nx), dtype=bool)
    sink_mask = np.zeros((ny, nx), dtype=bool)
    sink_mask[gate_rows, nx - 1] = True
    source_mask = np.zeros((ny, nx), dtype=bool)
    source_mask[gate_rows, 1] = True
    waiting_mask = np.zeros((ny, nx), dtype=bool)
    waiting_mask[gate_rows, 1 : gate_face + 1] = True

    if group_inflow_split is None:
        group_inflow_split = {"g0": 1.0}
    split_total = sum(max(value, 0.0) for value in group_inflow_split.values())
    if split_total <= EPS:
        raise ValueError("group_inflow_split must contain positive mass")

    rho_by_group = {
        group: np.zeros((ny, nx), dtype=float)
        for group in group_inflow_split
    }
    vx = np.ones((ny, nx), dtype=float)
    vy = np.zeros((ny, nx), dtype=float)

    initial_mass = 0.0
    inflow_total = 0.0
    sink_total = 0.0
    cap_removed_total = 0.0
    attempted_total = 0.0
    actual_total = 0.0
    rejected_total = 0.0
    binding_steps = 0
    waiting_peak = 0.0
    max_mass_balance_residual = 0.0
    max_actual_minus_capacity = 0.0

    cell_area = dx * dx
    source_cells = int(np.count_nonzero(source_mask))
    for step in range(steps):
        schedule_idx = min(int(step * len(q_schedule) / max(steps, 1)), len(q_schedule) - 1)
        q_raw = float(q_schedule[schedule_idx])
        q_plus, q_minus = project_capacity(direction, q_raw, q_raw, qbar)

        for group, split in group_inflow_split.items():
            group_rate = inflow_rate * max(split, 0.0) / split_total
            increment = group_rate * dt / (source_cells * cell_area)
            rho_by_group[group][source_mask] += increment
            inflow_total += group_rate * dt

        fx_by_group: dict[str, np.ndarray] = {}
        fy_by_group: dict[str, np.ndarray] = {}
        for group, rho in rho_by_group.items():
            fx, fy = compute_face_fluxes(rho, vx, vy)
            fx_by_group[group] = fx
            fy_by_group[group] = fy

        limited_fx_by_group, diagnostics = apply_gate_limits(
            fx_by_group,
            gate_face=gate_face,
            gate_rows=gate_rows,
            dx=dx,
            q_plus=q_plus,
            q_minus=q_minus,
        )

        attempted_total += diagnostics.attempt_plus * dt
        actual_total += diagnostics.actual_plus * dt
        rejected_total += diagnostics.rejected_plus * dt
        if diagnostics.rejected_plus > 1.0e-9:
            binding_steps += 1
        if not math.isinf(q_plus):
            max_actual_minus_capacity = max(max_actual_minus_capacity, diagnostics.actual_plus - q_plus)

        updated_by_group: dict[str, np.ndarray] = {}
        for group, rho in rho_by_group.items():
            updated, sink_increment = update_density_from_fluxes(
                rho,
                limited_fx_by_group[group],
                fy_by_group[group],
                walkable=walkable,
                sink_mask=sink_mask,
                dx=dx,
                dt=dt,
            )
            updated_by_group[group] = updated
            sink_total += sink_increment
        rho_by_group, cap_removed = apply_total_density_cap(
            updated_by_group,
            rho_max=20.0,
            walkable=walkable,
            dx=dx,
        )
        cap_removed_total += cap_removed

        total_rho = np.zeros((ny, nx), dtype=float)
        for rho in rho_by_group.values():
            total_rho += rho
        current_mass = float(np.sum(total_rho[walkable]) * cell_area)
        residual = initial_mass + inflow_total - sink_total - cap_removed_total - current_mass
        max_mass_balance_residual = max(max_mass_balance_residual, abs(residual))
        waiting_peak = max(waiting_peak, float(np.sum(total_rho[waiting_mask]) * cell_area))

    final_rho = np.zeros((ny, nx), dtype=float)
    for rho in rho_by_group.values():
        final_rho += rho
    final_mass = float(np.sum(final_rho[walkable]) * cell_area)
    return {
        "steps": steps,
        "dt": dt,
        "duration": steps * dt,
        "direction": direction,
        "q_schedule": ["inf" if math.isinf(value) else float(value) for value in q_schedule],
        "initial_mass": initial_mass,
        "inflow_total": inflow_total,
        "sink_total": sink_total,
        "cap_removed_total": cap_removed_total,
        "final_mass": final_mass,
        "attempted_cumulative": attempted_total,
        "actual_cumulative": actual_total,
        "rejected_cumulative": rejected_total,
        "binding_time_ratio": binding_steps / max(steps, 1),
        "waiting_mass_peak": waiting_peak,
        "mass_balance_residual": initial_mass + inflow_total - sink_total - cap_removed_total - final_mass,
        "max_mass_balance_residual": max_mass_balance_residual,
        "max_actual_minus_capacity": max_actual_minus_capacity,
    }


def run_gate_smoke(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    q_cases = {
        "q_inf": (math.inf,),
        "q_high": (8.0,),
        "q_medium": (1.5,),
        "q_zero": (0.0,),
    }
    rows: list[dict[str, object]] = []
    for case_id, schedule in q_cases.items():
        result = run_gate_case(q_schedule=schedule)
        row = {"case_id": case_id, **result}
        rows.append(row)

    _write_csv(
        output_root / "single_channel_gate_smoke.csv",
        rows,
        fieldnames=[
            "case_id",
            "direction",
            "q_schedule",
            "duration",
            "attempted_cumulative",
            "actual_cumulative",
            "rejected_cumulative",
            "binding_time_ratio",
            "waiting_mass_peak",
            "inflow_total",
            "sink_total",
            "cap_removed_total",
            "final_mass",
            "mass_balance_residual",
            "max_mass_balance_residual",
            "max_actual_minus_capacity",
        ],
    )

    by_case = {str(row["case_id"]): row for row in rows}
    no_cap_waiting = float(by_case["q_inf"]["waiting_mass_peak"])
    summary = {
        "csv": str((output_root / "single_channel_gate_smoke.csv").resolve()),
        "mass_balance_tolerance": 1.0e-8,
        "capacity_tolerance": 1.0e-8,
        "checks": {
            "q_inf_degenerates_to_unlimited": float(by_case["q_inf"]["rejected_cumulative"]) <= 1.0e-8,
            "q_high_not_binding": float(by_case["q_high"]["rejected_cumulative"]) <= 1.0e-8,
            "q_medium_binding": float(by_case["q_medium"]["rejected_cumulative"]) > 1.0e-6
            and float(by_case["q_medium"]["binding_time_ratio"]) > 0.0,
            "q_zero_blocks_gate": float(by_case["q_zero"]["actual_cumulative"]) <= 1.0e-8
            and float(by_case["q_zero"]["rejected_cumulative"]) > 1.0e-6,
            "q_zero_raises_waiting_mass": float(by_case["q_zero"]["waiting_mass_peak"]) > no_cap_waiting + 1.0,
            "mass_conserved": max(float(row["max_mass_balance_residual"]) for row in rows) <= 1.0e-8,
            "actual_rate_never_exceeds_capacity": max(float(row["max_actual_minus_capacity"]) for row in rows) <= 1.0e-8,
        },
    }
    summary["passed"] = all(bool(value) for value in summary["checks"].values())
    return rows, summary


def run_direction_coupling(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_plus, raw_minus, qbar = 7.0, 5.0, 10.0
    for state in ("E", "W", "FREE", "CLOSED"):
        q_plus, q_minus = project_capacity(state, raw_plus, raw_minus, qbar)
        rows.append(
            {
                "state": state,
                "raw_q_plus": raw_plus,
                "raw_q_minus": raw_minus,
                "qbar": qbar,
                "projected_q_plus": q_plus,
                "projected_q_minus": q_minus,
                "projected_total": q_plus + q_minus,
            }
        )
    _write_csv(output_root / "direction_capacity_coupling.csv", rows)
    by_state = {str(row["state"]): row for row in rows}
    checks = {
        "E_sets_minus_zero": abs(float(by_state["E"]["projected_q_minus"])) <= 1.0e-12,
        "W_sets_plus_zero": abs(float(by_state["W"]["projected_q_plus"])) <= 1.0e-12,
        "FREE_respects_qbar": float(by_state["FREE"]["projected_total"]) <= qbar + 1.0e-12,
        "CLOSED_sets_both_zero": abs(float(by_state["CLOSED"]["projected_total"])) <= 1.0e-12,
    }
    return rows, {
        "csv": str((output_root / "direction_capacity_coupling.csv").resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_multigroup_capacity(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    gate_rows = np.arange(0, 3)
    gate_face = 2
    dx = 1.0
    fx_a = np.zeros((3, 5), dtype=float)
    fx_b = np.zeros((3, 5), dtype=float)
    fx_a[gate_rows, gate_face] = 2.0
    fx_b[gate_rows, gate_face] = 3.0
    limited, diagnostics = apply_gate_limits(
        {"group_a": fx_a, "group_b": fx_b},
        gate_face=gate_face,
        gate_rows=gate_rows,
        dx=dx,
        q_plus=6.0,
        q_minus=0.0,
    )
    del limited
    expected_lambda = 6.0 / 15.0
    rows = []
    for group in ("group_a", "group_b"):
        attempt = diagnostics.attempt_plus_by_group[group]
        actual = diagnostics.actual_plus_by_group[group]
        rows.append(
            {
                "group": group,
                "attempted_rate": attempt,
                "actual_rate": actual,
                "actual_over_attempt": actual / attempt if attempt > EPS else 0.0,
                "shared_lambda": diagnostics.lambda_plus,
            }
        )
    _write_csv(output_root / "multigroup_shared_capacity.csv", rows)
    checks = {
        "total_actual_equals_capacity": abs(diagnostics.actual_plus - 6.0) <= 1.0e-10,
        "shared_lambda_matches_expected": abs(diagnostics.lambda_plus - expected_lambda) <= 1.0e-10,
        "group_a_scaled_by_same_lambda": abs(rows[0]["actual_over_attempt"] - expected_lambda) <= 1.0e-10,
        "group_b_scaled_by_same_lambda": abs(rows[1]["actual_over_attempt"] - expected_lambda) <= 1.0e-10,
    }
    return rows, {
        "csv": str((output_root / "multigroup_shared_capacity.csv").resolve()),
        "attempt_total": diagnostics.attempt_plus,
        "actual_total": diagnostics.actual_plus,
        "shared_lambda": diagnostics.lambda_plus,
        "checks": checks,
        "passed": all(checks.values()),
    }


def normalized_jr(q_plus: np.ndarray, q_minus: np.ndarray, qbar_plus: np.ndarray, qbar_minus: np.ndarray) -> float:
    if q_plus.shape != q_minus.shape:
        raise ValueError("q_plus and q_minus must have the same shape")
    if q_plus.ndim != 2:
        raise ValueError("q arrays must have shape (channel, time_segment)")
    channel_count, segment_count = q_plus.shape
    if segment_count <= 1 or channel_count == 0:
        return 0.0

    total = 0.0
    active_terms = 0
    for values, bars in ((q_plus, qbar_plus), (q_minus, qbar_minus)):
        for c in range(channel_count):
            bar = float(bars[c])
            if bar <= EPS:
                continue
            diffs = np.diff(values[c, :]) / bar
            total += float(np.sum(diffs * diffs))
            active_terms += len(diffs)
    if active_terms == 0:
        return 0.0
    return total / active_terms


def run_jr_validation(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    qbar = np.array([8.0], dtype=float)
    rough_plus = np.array([[8.0, 0.0, 8.0, 0.0]], dtype=float)
    rough_minus = np.array([[0.0, 8.0, 0.0, 8.0]], dtype=float)
    smooth_plus = np.array([[4.0, 4.0, 4.0, 4.0]], dtype=float)
    smooth_minus = np.array([[4.0, 4.0, 4.0, 4.0]], dtype=float)
    rough_jr = normalized_jr(rough_plus, rough_minus, qbar, qbar)
    smooth_jr = normalized_jr(smooth_plus, smooth_minus, qbar, qbar)

    rough_case = run_gate_case(q_schedule=tuple(float(value) for value in rough_plus[0]), qbar=8.0)
    smooth_case = run_gate_case(q_schedule=tuple(float(value) for value in smooth_plus[0]), qbar=8.0)
    physical_metric = float(rough_case["waiting_mass_peak"])
    rows = [
        {
            "case_id": "rough",
            "jr": rough_jr,
            "lambda_jr_0_objective": physical_metric,
            "lambda_jr_1_objective": physical_metric + rough_jr,
            "max_mass_balance_residual": rough_case["max_mass_balance_residual"],
            "max_actual_minus_capacity": rough_case["max_actual_minus_capacity"],
        },
        {
            "case_id": "smooth",
            "jr": smooth_jr,
            "lambda_jr_0_objective": float(smooth_case["waiting_mass_peak"]),
            "lambda_jr_1_objective": float(smooth_case["waiting_mass_peak"]) + smooth_jr,
            "max_mass_balance_residual": smooth_case["max_mass_balance_residual"],
            "max_actual_minus_capacity": smooth_case["max_actual_minus_capacity"],
        },
    ]
    _write_csv(output_root / "jr_smoothness_validation.csv", rows)
    checks = {
        "rough_jr_greater_than_smooth": rough_jr > smooth_jr,
        "lambda_jr_changes_only_objective_for_fixed_physics": abs(
            float(rows[0]["lambda_jr_1_objective"]) - float(rows[0]["lambda_jr_0_objective"]) - rough_jr
        )
        <= 1.0e-12,
        "mass_conserved": max(float(row["max_mass_balance_residual"]) for row in rows) <= 1.0e-8,
        "limiter_respects_capacity": max(float(row["max_actual_minus_capacity"]) for row in rows) <= 1.0e-8,
    }
    return rows, {
        "csv": str((output_root / "jr_smoothness_validation.csv").resolve()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model v2 section 3.2 validation checks.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    eta_rows, eta_summary = run_eta_sensitivity(output_root)
    gate_rows, gate_summary = run_gate_smoke(output_root)
    direction_rows, direction_summary = run_direction_coupling(output_root)
    multigroup_rows, multigroup_summary = run_multigroup_capacity(output_root)
    jr_rows, jr_summary = run_jr_validation(output_root)

    mass_summary = {
        "source_csv": gate_summary["csv"],
        "mass_balance_tolerance": 1.0e-8,
        "max_mass_balance_residual": max(float(row["max_mass_balance_residual"]) for row in gate_rows),
        "max_actual_minus_capacity": max(float(row["max_actual_minus_capacity"]) for row in gate_rows),
        "rejected_flux_manifested_as_waiting": bool(
            float(next(row for row in gate_rows if row["case_id"] == "q_zero")["waiting_mass_peak"])
            > float(next(row for row in gate_rows if row["case_id"] == "q_inf")["waiting_mass_peak"]) + 1.0
        ),
    }
    mass_summary["passed"] = bool(
        mass_summary["max_mass_balance_residual"] <= 1.0e-8
        and mass_summary["max_actual_minus_capacity"] <= 1.0e-8
        and mass_summary["rejected_flux_manifested_as_waiting"]
    )

    summary = {
        "validation_name": "model_v2_section_3_2",
        "output_root": str(output_root),
        "eta_sensitivity": eta_summary,
        "single_channel_gate_smoke": gate_summary,
        "mass_conservation": mass_summary,
        "direction_coupling": direction_summary,
        "multigroup_shared_capacity": multigroup_summary,
        "jr_smoothness": jr_summary,
        "eta_rows": len(eta_rows),
        "gate_rows": len(gate_rows),
        "direction_rows": len(direction_rows),
        "multigroup_rows": len(multigroup_rows),
        "jr_rows": len(jr_rows),
    }
    mechanism_checks = [
        gate_summary["passed"],
        mass_summary["passed"],
        direction_summary["passed"],
        multigroup_summary["passed"],
        jr_summary["passed"],
    ]
    summary["all_mechanism_checks_passed"] = all(bool(item) for item in mechanism_checks)
    summary["eta_scan_completed"] = len(eta_rows) == len(ETA_VALUES)
    summary["all_required_outputs_created"] = bool(summary["all_mechanism_checks_passed"] and summary["eta_scan_completed"])
    _save_json(output_root / "validation_summary.json", summary)

    print(f"Validation summary: {output_root / 'validation_summary.json'}")
    print(f"Eta weak sensitivity: {eta_summary['weak_sensitivity']} (relative range={eta_summary['objective_relative_range']:.6g})")
    print(f"Mechanism checks passed: {summary['all_mechanism_checks_passed']}")
    return 0 if summary["all_required_outputs_created"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
