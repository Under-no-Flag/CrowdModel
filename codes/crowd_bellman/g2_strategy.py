from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .metrics import save_json


@dataclass
class G2StrategyCollector:
    case_id: str
    title: str
    walkable: np.ndarray
    channel_masks: dict[str, np.ndarray]
    rho_safe: float
    dx: float

    def __post_init__(self) -> None:
        self.cell_area = float(self.dx * self.dx)
        self.channel_peak_density: dict[str, float] = {name: 0.0 for name in self.channel_masks}
        self.channel_peak_time: dict[str, float] = {name: 0.0 for name in self.channel_masks}
        self.channel_high_density_exposure: dict[str, float] = {name: 0.0 for name in self.channel_masks}
        self.hotspot_weight = 0.0
        self.hotspot_x_sum = 0.0
        self.hotspot_y_sum = 0.0
        self.global_peak_value = 0.0
        self.global_peak_time = 0.0
        self.global_peak_x = 0.0
        self.global_peak_y = 0.0

    def observe(self, snapshot: dict[str, object]) -> None:
        rho = np.asarray(snapshot["rho"], dtype=float)
        time_value = float(snapshot["time"])
        dt = float(snapshot["dt"])

        current_peak = float(np.max(rho[self.walkable])) if np.any(self.walkable) else 0.0
        if current_peak >= self.global_peak_value:
            peak_index = np.unravel_index(int(np.argmax(rho)), rho.shape)
            self.global_peak_value = current_peak
            self.global_peak_time = time_value
            self.global_peak_y = float(peak_index[0])
            self.global_peak_x = float(peak_index[1])

        yy, xx = np.indices(rho.shape)
        high_density_mask = self.walkable & (rho > self.rho_safe)
        if np.any(high_density_mask):
            weights = rho[high_density_mask] * dt
            self.hotspot_weight += float(np.sum(weights))
            self.hotspot_x_sum += float(np.sum(xx[high_density_mask] * weights))
            self.hotspot_y_sum += float(np.sum(yy[high_density_mask] * weights))

        for channel_name, channel_mask in self.channel_masks.items():
            active_mask = channel_mask & self.walkable
            if not np.any(active_mask):
                continue
            channel_peak = float(np.max(rho[active_mask]))
            if channel_peak >= self.channel_peak_density[channel_name]:
                self.channel_peak_density[channel_name] = channel_peak
                self.channel_peak_time[channel_name] = time_value
            self.channel_high_density_exposure[channel_name] += float(np.sum(rho[active_mask] > self.rho_safe) * self.cell_area * dt)

    def save_case_outputs(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        hotspot_centroid = {
            "x": None if self.hotspot_weight <= 1.0e-12 else float(self.hotspot_x_sum / self.hotspot_weight),
            "y": None if self.hotspot_weight <= 1.0e-12 else float(self.hotspot_y_sum / self.hotspot_weight),
        }
        payload = {
            "case_id": self.case_id,
            "title": self.title,
            "channel_peak_density": {name: float(value) for name, value in self.channel_peak_density.items()},
            "channel_peak_time": {name: float(value) for name, value in self.channel_peak_time.items()},
            "channel_high_density_exposure": {name: float(value) for name, value in self.channel_high_density_exposure.items()},
            "hotspot_centroid": hotspot_centroid,
            "global_peak": {
                "density": float(self.global_peak_value),
                "time": float(self.global_peak_time),
                "x": float(self.global_peak_x),
                "y": float(self.global_peak_y),
            },
        }
        save_json(output_dir / "g2_strategy_summary.json", payload)
        return payload


def build_g2_strategy_report(
    *,
    output_root: Path,
    case_summaries: list[dict[str, object]],
    behavior_summaries: list[dict[str, object]],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    behavior_by_case = {str(item["case_id"]): item for item in behavior_summaries}
    rows: list[dict[str, object]] = []
    for summary in case_summaries:
        case_id = str(summary["case_id"])
        behavior = behavior_by_case.get(case_id, {})
        scan = summary.get("g2_scan", {})
        directions = scan.get("channel_directions", {})
        row = {
            "case_id": case_id,
            "title": summary.get("title"),
            "family": scan.get("family", "unknown"),
            "setting_label": scan.get("setting_label", case_id),
            "is_baseline": bool(scan.get("is_baseline", False)),
            "direction_top": directions.get("top"),
            "direction_middle": directions.get("middle"),
            "direction_bottom": directions.get("bottom"),
            "entry_channels": ",".join(scan.get("entry_channels", [])),
            "return_channels": ",".join(scan.get("return_channels", [])),
            "j1": summary.get("j1_normalized", summary.get("j1_total_travel_time")),
            "j2": summary.get("j2_normalized", summary.get("j2_high_density_exposure")),
            "j5": summary.get("j5_normalized", summary.get("j5_channel_flux_variance")),
            "j1_raw": summary.get("j1_total_travel_time"),
            "j2_raw": summary.get("j2_high_density_exposure"),
            "j5_raw": summary.get("j5_channel_flux_variance"),
            "objective_value": summary.get("objective_value"),
            "sink_cumulative": summary.get("final_sink_cumulative"),
            "peak_density": summary.get("peak_density_max"),
            "hotspot_x": behavior.get("hotspot_centroid", {}).get("x"),
            "hotspot_y": behavior.get("hotspot_centroid", {}).get("y"),
            "global_peak_time": behavior.get("global_peak", {}).get("time"),
        }
        for channel_name in ("top", "middle", "bottom"):
            row[f"flux_{channel_name}"] = summary.get("channel_flux_cumulative", {}).get(channel_name)
            row[f"flux_share_{channel_name}"] = summary.get("channel_flux_share", {}).get(channel_name)
            row[f"peak_time_{channel_name}"] = behavior.get("channel_peak_time", {}).get(channel_name)
            row[f"exposure_{channel_name}"] = behavior.get("channel_high_density_exposure", {}).get(channel_name)
            row[f"peak_density_{channel_name}"] = behavior.get("channel_peak_density", {}).get(channel_name)
        rows.append(row)

    non_dominated_case_ids = _non_dominated_case_ids(rows)
    for row in rows:
        row["is_non_dominated"] = str(row["case_id"]) in non_dominated_case_ids

    _save_csv(output_root / "g2_direction_metrics.csv", rows)
    _save_pareto_plot(output_root / "g2_direction_pareto.png", rows)
    _save_metric_bar_plot(output_root / "g2_direction_objectives.png", rows)
    _save_channel_plot(output_root / "g2_direction_channel_loads.png", rows)
    _save_hotspot_plot(output_root / "g2_direction_hotspot_migration.png", rows)

    report = {
        "experiment_group": "G2",
        "design_version": "direction_scan_multistage",
        "cases": case_summaries,
        "behavior_cases": behavior_summaries,
        "non_dominated_cases": non_dominated_case_ids,
        "families": sorted({str(row["family"]) for row in rows}),
    }
    save_json(output_root / "g2_strategy_summary.json", report)
    return report


def _non_dominated_case_ids(rows: list[dict[str, object]]) -> list[str]:
    case_ids: list[str] = []
    for row in rows:
        dominated = False
        j1 = float(row["j1"])
        j2 = float(row["j2"])
        j5 = float(row["j5"])
        for other in rows:
            if other["case_id"] == row["case_id"]:
                continue
            o1 = float(other["j1"])
            o2 = float(other["j2"])
            o5 = float(other["j5"])
            if (o1 <= j1 and o2 <= j2 and o5 <= j5) and (o1 < j1 or o2 < j2 or o5 < j5):
                dominated = True
                break
        if not dominated:
            case_ids.append(str(row["case_id"]))
    return case_ids


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _label(row: dict[str, object]) -> str:
    if bool(row["is_baseline"]):
        return "baseline"
    top = row.get("direction_top")
    middle = row.get("direction_middle")
    bottom = row.get("direction_bottom")
    return f"T:{top} M:{middle} B:{bottom}"


def _family_color(family: str) -> str:
    if family == "baseline":
        return "black"
    if family == "single_entry":
        return "#F58518"
    if family == "single_return":
        return "#4C78A8"
    return "gray"


def _save_pareto_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), dpi=150)
    for row in rows:
        family = str(row["family"])
        marker = "D" if bool(row["is_non_dominated"]) else "o"
        ax.scatter(
            float(row["j1"]),
            float(row["j2"]),
            color=_family_color(family),
            s=115 if bool(row["is_non_dominated"]) else 75,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(_label(row), (float(row["j1"]), float(row["j2"])), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("~J1")
    ax.set_ylabel("~J2")
    ax.set_title("G2 Pareto view under normalized objective terms")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_metric_bar_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [_label(row) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.2), dpi=150)
    ax.bar(x - width, [float(row["j1"]) for row in rows], width=width, label="~J1")
    ax.bar(x, [float(row["j2"]) for row in rows], width=width, label="~J2")
    ax.bar(x + width, [float(row["j5"]) for row in rows], width=width, label="~J5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("G2 normalized objective terms by direction setting")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_channel_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [_label(row) for row in rows]
    x = np.arange(len(rows))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=150)
    for offset, channel_name in zip((-width, 0.0, width), ("top", "middle", "bottom")):
        axes[0].bar(
            x + offset,
            [float(row.get(f"flux_share_{channel_name}") or 0.0) for row in rows],
            width=width,
            label=channel_name,
        )
        axes[1].bar(
            x + offset,
            [float(row.get(f"peak_time_{channel_name}") or 0.0) for row in rows],
            width=width,
            label=channel_name,
        )
    axes[0].set_title("Channel flux share by direction setting")
    axes[1].set_title("Channel peak-load timing by direction setting")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_hotspot_plot(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=150)
    for row in rows:
        x = row.get("hotspot_x")
        y = row.get("hotspot_y")
        if x is None or y is None:
            continue
        family = str(row["family"])
        ax.scatter(float(x), float(y), s=70, color=_family_color(family))
        ax.annotate(_label(row), (float(x), float(y)), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("hotspot centroid x")
    ax.set_ylabel("hotspot centroid y")
    ax.set_title("G2 hotspot migration under channel-direction scan")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
