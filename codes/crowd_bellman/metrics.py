from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ObjectiveConfig


@dataclass
class CaseStats:
    times: list[float] = field(default_factory=list)
    mean_density: list[float] = field(default_factory=list)
    peak_density: list[float] = field(default_factory=list)
    sink_cumulative: list[float] = field(default_factory=list)
    velocity_discontinuity: list[float] = field(default_factory=list)
    density_gradient_intensity: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
    travel_time_cumulative: list[float] = field(default_factory=list)
    high_density_exposure_cumulative: list[float] = field(default_factory=list)
    inflow_cumulative: list[float] = field(default_factory=list)
    channel_density: dict[str, list[float]] = field(default_factory=dict)
    channel_flux_cumulative: dict[str, float] = field(default_factory=dict)


def init_case_stats(channel_names: list[str]) -> CaseStats:
    return CaseStats(
        channel_density={name: [] for name in channel_names},
        channel_flux_cumulative={name: 0.0 for name in channel_names},
    )


def velocity_discontinuity_metric(
    walkable: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> float:
    jump_x = np.sqrt((vx[:, 1:] - vx[:, :-1]) ** 2 + (vy[:, 1:] - vy[:, :-1]) ** 2)
    jump_y = np.sqrt((vx[1:, :] - vx[:-1, :]) ** 2 + (vy[1:, :] - vy[:-1, :]) ** 2)
    valid_x = walkable[:, 1:] & walkable[:, :-1]
    valid_y = walkable[1:, :] & walkable[:-1, :]
    total = 0.0
    count = 0
    if np.any(valid_x):
        total += float(np.mean(jump_x[valid_x]))
        count += 1
    if np.any(valid_y):
        total += float(np.mean(jump_y[valid_y]))
        count += 1
    return total / max(count, 1)


def density_gradient_metric(walkable: np.ndarray, rho: np.ndarray, dx: float) -> float:
    gy, gx = np.gradient(rho, dx, dx)
    grad = np.sqrt(gx * gx + gy * gy)
    if not np.any(walkable):
        return 0.0
    return float(np.mean(grad[walkable]))


def channel_flux_increment(
    fx: np.ndarray,
    probe_x: int,
    channel_mask: np.ndarray,
    dx: float,
    dt: float,
) -> float:
    face_index = int(np.clip(probe_x, 0, fx.shape[1] - 1))
    right_col = min(face_index + 1, channel_mask.shape[1] - 1)
    y_valid = np.where(channel_mask[:, face_index] | channel_mask[:, right_col])[0]
    if y_valid.size == 0:
        return 0.0
    positive_flux = np.maximum(fx[y_valid, face_index], 0.0)
    return float(np.sum(positive_flux) * dx * dt)


def channel_flux_variance(channel_flux_cumulative: dict[str, float]) -> float:
    values = np.array(list(channel_flux_cumulative.values()), dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.var(values))


def objective_terms_from_stats(stats: CaseStats) -> dict[str, float]:
    return {
        "j1_total_travel_time": float(stats.travel_time_cumulative[-1]) if stats.travel_time_cumulative else 0.0,
        "j2_high_density_exposure": float(stats.high_density_exposure_cumulative[-1]) if stats.high_density_exposure_cumulative else 0.0,
        "j5_channel_flux_variance": channel_flux_variance(stats.channel_flux_cumulative),
    }


def _objective_evaluation_from_terms(
    objective_terms: dict[str, float],
    objective_cfg: ObjectiveConfig,
) -> dict[str, float | str | bool]:
    j1_raw = float(objective_terms["j1_total_travel_time"])
    j2_raw = float(objective_terms["j2_high_density_exposure"])
    j5_raw = float(objective_terms["j5_channel_flux_variance"])

    if objective_cfg.use_normalized_terms:
        j1_eval = j1_raw / objective_cfg.j1_scale
        j2_eval = j2_raw / objective_cfg.j2_scale
        j5_eval = j5_raw / objective_cfg.j5_scale
    else:
        j1_eval = j1_raw
        j2_eval = j2_raw
        j5_eval = j5_raw

    objective_value = (
        objective_cfg.lambda_j1 * j1_eval
        + objective_cfg.lambda_j2 * j2_eval
        + objective_cfg.lambda_j5 * j5_eval
    )

    return {
        "name": objective_cfg.name,
        "lambda_j1": float(objective_cfg.lambda_j1),
        "lambda_j2": float(objective_cfg.lambda_j2),
        "lambda_j5": float(objective_cfg.lambda_j5),
        "rho_safe": float(objective_cfg.rho_safe),
        "use_normalized_terms": bool(objective_cfg.use_normalized_terms),
        "j1_scale": float(objective_cfg.j1_scale),
        "j2_scale": float(objective_cfg.j2_scale),
        "j5_scale": float(objective_cfg.j5_scale),
        "j1_total_travel_time": j1_raw,
        "j2_high_density_exposure": j2_raw,
        "j5_channel_flux_variance": j5_raw,
        "j1_eval": float(j1_eval),
        "j2_eval": float(j2_eval),
        "j5_eval": float(j5_eval),
        "objective_value": float(objective_value),
    }


def compute_objective_terms(
    stats: CaseStats,
    objective_cfg: ObjectiveConfig,
) -> dict[str, float | str | bool]:
    return _objective_evaluation_from_terms(objective_terms_from_stats(stats), objective_cfg)


def extract_objective_terms(summary: dict[str, object]) -> dict[str, float]:
    raw_terms = summary.get("objective_terms")
    if isinstance(raw_terms, dict):
        j1 = float(raw_terms.get("j1_total_travel_time", 0.0))
        j2 = float(raw_terms.get("j2_high_density_exposure", 0.0))
        j5 = float(raw_terms.get("j5_channel_flux_variance", 0.0))
    else:
        j1 = float(summary.get("j1_total_travel_time", 0.0))
        j2 = float(summary.get("j2_high_density_exposure", 0.0))
        j5 = float(summary.get("j5_channel_flux_variance", 0.0))

    return {
        "j1_total_travel_time": j1,
        "j2_high_density_exposure": j2,
        "j5_channel_flux_variance": j5,
    }


def evaluate_objective_from_summary(
    summary: dict[str, object],
    objective_cfg: ObjectiveConfig,
) -> dict[str, float | str | bool]:
    return _objective_evaluation_from_terms(extract_objective_terms(summary), objective_cfg)


def evaluate_objective_batch_from_summary(
    summary: dict[str, object],
    objective_cfgs: Iterable[ObjectiveConfig],
) -> list[dict[str, float | str | bool]]:
    objective_terms = extract_objective_terms(summary)
    return [
        _objective_evaluation_from_terms(objective_terms, objective_cfg)
        for objective_cfg in objective_cfgs
    ]


def record_step(
    stats: CaseStats,
    time_value: float,
    rho: np.ndarray,
    walkable: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    fx: np.ndarray,
    sink_total: float,
    dt: float,
    dx: float,
    rho_safe: float,
    channel_masks: dict[str, np.ndarray],
    probe_x: dict[str, int],
    inflow_total: float,
) -> None:
    cell_area = dx * dx
    walkable_rho = rho[walkable]

    stats.times.append(time_value)
    stats.mean_density.append(float(np.mean(walkable_rho)))
    stats.peak_density.append(float(np.max(walkable_rho)))
    stats.sink_cumulative.append(sink_total)
    stats.velocity_discontinuity.append(velocity_discontinuity_metric(walkable, vx, vy))
    stats.density_gradient_intensity.append(density_gradient_metric(walkable, rho, dx))
    stats.dt.append(dt)
    stats.inflow_cumulative.append(inflow_total)

    travel_increment = float(np.sum(walkable_rho) * cell_area * dt)
    exposure_increment = float(np.sum(walkable_rho > rho_safe) * cell_area * dt)
    prev_j1 = stats.travel_time_cumulative[-1] if stats.travel_time_cumulative else 0.0
    prev_j2 = stats.high_density_exposure_cumulative[-1] if stats.high_density_exposure_cumulative else 0.0
    stats.travel_time_cumulative.append(prev_j1 + travel_increment)
    stats.high_density_exposure_cumulative.append(prev_j2 + exposure_increment)

    for name, channel_mask in channel_masks.items():
        stats.channel_density[name].append(float(np.mean(rho[channel_mask])) if np.any(channel_mask) else 0.0)
        stats.channel_flux_cumulative[name] += channel_flux_increment(
            fx=fx,
            probe_x=probe_x[name],
            channel_mask=channel_mask,
            dx=dx,
            dt=dt,
        )


def build_summary(
    case_id: str,
    title: str,
    stats: CaseStats,
    objective_cfg: ObjectiveConfig | None = None,
) -> dict[str, object]:
    objective_terms = objective_terms_from_stats(stats)
    channel_flux_total = sum(stats.channel_flux_cumulative.values())
    if channel_flux_total <= 1.0e-12:
        channel_share = {name: 0.0 for name in stats.channel_flux_cumulative}
    else:
        channel_share = {
            name: value / channel_flux_total
            for name, value in stats.channel_flux_cumulative.items()
        }

    channel_time_mean_density = {
        name: float(np.mean(series)) if series else 0.0
        for name, series in stats.channel_density.items()
    }

    summary = {
        "case_id": case_id,
        "title": title,
        "final_time": float(stats.times[-1]) if stats.times else 0.0,
        "final_sink_cumulative": float(stats.sink_cumulative[-1]) if stats.sink_cumulative else 0.0,
        "final_inflow_cumulative": float(stats.inflow_cumulative[-1]) if stats.inflow_cumulative else 0.0,
        "mean_density_avg": float(np.mean(stats.mean_density)) if stats.mean_density else 0.0,
        "peak_density_max": float(np.max(stats.peak_density)) if stats.peak_density else 0.0,
        "velocity_discontinuity_avg": float(np.mean(stats.velocity_discontinuity)) if stats.velocity_discontinuity else 0.0,
        "density_gradient_avg": float(np.mean(stats.density_gradient_intensity)) if stats.density_gradient_intensity else 0.0,
        "channel_time_mean_density": channel_time_mean_density,
        "channel_flux_cumulative": {k: float(v) for k, v in stats.channel_flux_cumulative.items()},
        "channel_flux_share": channel_share,
        "objective_terms": objective_terms,
        "j1_total_travel_time": float(objective_terms["j1_total_travel_time"]),
        "j2_high_density_exposure": float(objective_terms["j2_high_density_exposure"]),
        "j5_channel_flux_variance": float(objective_terms["j5_channel_flux_variance"]),
    }
    if objective_cfg is not None:
        summary["objective"] = compute_objective_terms(stats, objective_cfg)
        summary["objective_value"] = float(summary["objective"]["objective_value"])
    return summary


def save_case_timeseries(path: Path, stats: CaseStats) -> None:
    fieldnames = [
        "time",
        "dt",
        "mean_density",
        "peak_density",
        "sink_cumulative",
        "velocity_discontinuity",
        "density_gradient_intensity",
        "travel_time_cumulative",
        "high_density_exposure_cumulative",
        "inflow_cumulative",
    ] + [f"channel_density_{name}" for name in stats.channel_density]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, time_value in enumerate(stats.times):
            row = {
                "time": time_value,
                "dt": stats.dt[idx],
                "mean_density": stats.mean_density[idx],
                "peak_density": stats.peak_density[idx],
                "sink_cumulative": stats.sink_cumulative[idx],
                "velocity_discontinuity": stats.velocity_discontinuity[idx],
                "density_gradient_intensity": stats.density_gradient_intensity[idx],
                "travel_time_cumulative": stats.travel_time_cumulative[idx],
                "high_density_exposure_cumulative": stats.high_density_exposure_cumulative[idx],
                "inflow_cumulative": stats.inflow_cumulative[idx],
            }
            for name, series in stats.channel_density.items():
                row[f"channel_density_{name}"] = series[idx]
            writer.writerow(row)


def save_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
