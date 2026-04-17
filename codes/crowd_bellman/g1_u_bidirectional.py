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
class BidirectionalUCollector:
    case_id: str
    title: str
    walkable: np.ndarray
    channel_masks: dict[str, np.ndarray]
    group_names: dict[tuple[int, int], str]
    dx: float
    time_horizon: float
    observation_start_fraction: float = 0.1

    def __post_init__(self) -> None:
        self.middle_mask = self.channel_masks["middle"] & self.walkable
        self.channel_order = tuple(self.channel_masks.keys())
        self.observation_start_time = float(self.time_horizon) * float(self.observation_start_fraction)
        self.middle_total_mass_time = 0.0
        self.middle_westbound_mass_time = 0.0
        self.middle_counterflow_mass_time = 0.0
        self.middle_head_on_cell_time = 0.0
        self.total_observed_time = 0.0
        self.channel_directional_mass_time = {
            name: {"eastbound": 0.0, "westbound": 0.0}
            for name in self.channel_order
        }
        self.observed_steps = 0

    def observe(self, snapshot: dict[str, object]) -> None:
        time_value = float(snapshot["time"])
        if time_value < self.observation_start_time:
            return

        dt = float(snapshot["dt"])
        rho_by_group = snapshot["rho_by_group"]
        vx_by_group = snapshot["vx_by_group"]
        cell_area = float(self.dx * self.dx)

        east_mass = np.zeros(self.walkable.shape, dtype=float)
        west_mass = np.zeros(self.walkable.shape, dtype=float)
        for key, rho in rho_by_group.items():
            vx = np.asarray(vx_by_group[key], dtype=float)
            rho_arr = np.asarray(rho, dtype=float)
            east_mass += np.where(vx > 1.0e-8, rho_arr, 0.0)
            west_mass += np.where(vx < -1.0e-8, rho_arr, 0.0)

        middle_east = east_mass[self.middle_mask]
        middle_west = west_mass[self.middle_mask]
        middle_total = middle_east + middle_west

        self.middle_total_mass_time += float(np.sum(middle_total) * cell_area * dt)
        self.middle_westbound_mass_time += float(np.sum(middle_west) * cell_area * dt)
        self.middle_counterflow_mass_time += float(np.sum(np.minimum(middle_east, middle_west)) * cell_area * dt)
        self.middle_head_on_cell_time += float(np.sum((middle_east > 1.0e-8) & (middle_west > 1.0e-8)) * cell_area * dt)
        self.total_observed_time += dt

        for channel_name, channel_mask in self.channel_masks.items():
            channel_mask = channel_mask & self.walkable
            self.channel_directional_mass_time[channel_name]["eastbound"] += float(np.sum(east_mass[channel_mask]) * cell_area * dt)
            self.channel_directional_mass_time[channel_name]["westbound"] += float(np.sum(west_mass[channel_mask]) * cell_area * dt)

        self.observed_steps += 1

    def save_case_outputs(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        middle_westbound_share = (
            self.middle_westbound_mass_time / self.middle_total_mass_time
            if self.middle_total_mass_time > 1.0e-12
            else None
        )
        channel_directional_share: dict[str, dict[str, float]] = {}
        for channel_name, directional in self.channel_directional_mass_time.items():
            total = directional["eastbound"] + directional["westbound"]
            if total <= 1.0e-12:
                channel_directional_share[channel_name] = {"eastbound": 0.0, "westbound": 0.0}
                continue
            channel_directional_share[channel_name] = {
                "eastbound": float(directional["eastbound"] / total),
                "westbound": float(directional["westbound"] / total),
            }

        payload = {
            "case_id": self.case_id,
            "title": self.title,
            "observed_steps": int(self.observed_steps),
            "observation_start_time": float(self.observation_start_time),
            "middle_total_mass_time": float(self.middle_total_mass_time),
            "middle_westbound_mass_time": float(self.middle_westbound_mass_time),
            "middle_westbound_share": None if middle_westbound_share is None else float(middle_westbound_share),
            "middle_counterflow_mass_time": float(self.middle_counterflow_mass_time),
            "middle_head_on_cell_time": float(self.middle_head_on_cell_time),
            "channel_directional_mass_time": {
                name: {direction: float(value) for direction, value in directional.items()}
                for name, directional in self.channel_directional_mass_time.items()
            },
            "channel_directional_share": channel_directional_share,
        }
        save_json(output_dir / "u_bidirectional_summary.json", payload)
        return payload


def build_bidirectional_u_report(
    *,
    output_root: Path,
    case_summaries: list[dict[str, object]],
    validation_summaries: list[dict[str, object]],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_by_case = {str(item["case_id"]): item for item in case_summaries}
    rows: list[dict[str, object]] = []
    for validation in validation_summaries:
        case_id = str(validation["case_id"])
        summary = summary_by_case.get(case_id, {})
        row = {
            "case_id": case_id,
            "sink_cumulative": summary.get("final_sink_cumulative"),
            "peak_density": summary.get("peak_density_max"),
            "middle_westbound_share": validation.get("middle_westbound_share"),
            "middle_counterflow_mass_time": validation.get("middle_counterflow_mass_time"),
            "middle_head_on_cell_time": validation.get("middle_head_on_cell_time"),
        }
        for channel_name in ("top", "middle", "bottom"):
            directional = validation.get("channel_directional_share", {}).get(channel_name, {})
            row[f"{channel_name}_eastbound_share"] = directional.get("eastbound", 0.0)
            row[f"{channel_name}_westbound_share"] = directional.get("westbound", 0.0)
        rows.append(row)

    _save_csv(output_root / "g1_u_bidirectional_metrics.csv", rows)
    _save_plot(output_root / "g1_u_bidirectional_validation.png", rows)

    baseline = next((row for row in rows if row["case_id"] == "case_u_bidirectional_baseline"), None)
    ruled = next((row for row in rows if row["case_id"] == "case_u_bidirectional_middle_rule"), None)
    report = {
        "experiment_name": "G1_U_bidirectional_validation",
        "cases": case_summaries,
        "validation_cases": validation_summaries,
        "comparison": {
            "baseline_case": None if baseline is None else baseline["case_id"],
            "rule_case": None if ruled is None else ruled["case_id"],
            "baseline_middle_westbound_share": None if baseline is None else baseline["middle_westbound_share"],
            "rule_middle_westbound_share": None if ruled is None else ruled["middle_westbound_share"],
            "baseline_middle_counterflow_mass_time": None if baseline is None else baseline["middle_counterflow_mass_time"],
            "rule_middle_counterflow_mass_time": None if ruled is None else ruled["middle_counterflow_mass_time"],
            "baseline_middle_head_on_cell_time": None if baseline is None else baseline["middle_head_on_cell_time"],
            "rule_middle_head_on_cell_time": None if ruled is None else ruled["middle_head_on_cell_time"],
        },
    }
    save_json(output_root / "g1_u_bidirectional_summary.json", report)
    return report


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(row["case_id"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    axes[0].bar(x - width, [float(row.get("middle_westbound_share") or 0.0) for row in rows], width=width, label="middle westbound share")
    axes[0].bar(x, [float(row.get("middle_counterflow_mass_time") or 0.0) for row in rows], width=width, label="counterflow mass-time")
    axes[0].bar(x + width, [float(row.get("middle_head_on_cell_time") or 0.0) for row in rows], width=width, label="head-on cell-time")
    axes[0].set_title("Middle-channel counterflow suppression")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    bar_width = 0.35
    axes[1].bar(x - bar_width / 2, [float(row.get("middle_eastbound_share") or 0.0) for row in rows], width=bar_width, label="middle eastbound")
    axes[1].bar(x + bar_width / 2, [float(row.get("middle_westbound_share") or 0.0) for row in rows], width=bar_width, label="middle westbound")
    axes[1].set_title("Directional split inside middle channel")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
