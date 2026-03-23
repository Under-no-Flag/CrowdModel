from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CaseStats:
    times: list[float] = field(default_factory=list)
    mean_density: list[float] = field(default_factory=list)
    peak_density: list[float] = field(default_factory=list)
    sink_cumulative: list[float] = field(default_factory=list)
    velocity_discontinuity: list[float] = field(default_factory=list)
    density_gradient_intensity: list[float] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)
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
    channel_masks: dict[str, np.ndarray],
    probe_x: dict[str, int],
) -> None:
    stats.times.append(time_value)
    stats.mean_density.append(float(np.mean(rho[walkable])))
    stats.peak_density.append(float(np.max(rho[walkable])))
    stats.sink_cumulative.append(sink_total)
    stats.velocity_discontinuity.append(velocity_discontinuity_metric(walkable, vx, vy))
    stats.density_gradient_intensity.append(density_gradient_metric(walkable, rho, dx))
    stats.dt.append(dt)

    for name, channel_mask in channel_masks.items():
        stats.channel_density[name].append(float(np.mean(rho[channel_mask])) if np.any(channel_mask) else 0.0)
        stats.channel_flux_cumulative[name] += channel_flux_increment(
            fx=fx,
            probe_x=probe_x[name],
            channel_mask=channel_mask,
            dx=dx,
            dt=dt,
        )


def build_summary(case_id: str, title: str, stats: CaseStats) -> dict[str, object]:
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

    return {
        "case_id": case_id,
        "title": title,
        "final_time": float(stats.times[-1]) if stats.times else 0.0,
        "final_sink_cumulative": float(stats.sink_cumulative[-1]) if stats.sink_cumulative else 0.0,
        "mean_density_avg": float(np.mean(stats.mean_density)) if stats.mean_density else 0.0,
        "peak_density_max": float(np.max(stats.peak_density)) if stats.peak_density else 0.0,
        "velocity_discontinuity_avg": float(np.mean(stats.velocity_discontinuity)) if stats.velocity_discontinuity else 0.0,
        "density_gradient_avg": float(np.mean(stats.density_gradient_intensity)) if stats.density_gradient_intensity else 0.0,
        "channel_time_mean_density": channel_time_mean_density,
        "channel_flux_cumulative": {k: float(v) for k, v in stats.channel_flux_cumulative.items()},
        "channel_flux_share": channel_share,
    }


def save_case_timeseries(path: Path, stats: CaseStats) -> None:
    fieldnames = [
        "time",
        "dt",
        "mean_density",
        "peak_density",
        "sink_cumulative",
        "velocity_discontinuity",
        "density_gradient_intensity",
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
            }
            for name, series in stats.channel_density.items():
                row[f"channel_density_{name}"] = series[idx]
            writer.writerow(row)


def save_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
