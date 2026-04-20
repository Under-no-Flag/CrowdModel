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
class G3BehaviorCollector:
    case_id: str
    title: str
    walkable: np.ndarray
    region_masks: dict[str, np.ndarray]
    group_names: dict[tuple[int, int], str]
    dx: float

    def __post_init__(self) -> None:
        self.cell_area = float(self.dx * self.dx)
        self.times: list[float] = []
        self.group_mass_timeseries: dict[str, list[float]] = {name: [] for name in self.group_names.values()}
        tracked_regions = (
            "stage1_goal",
            "stage2_goal",
            "route8_goal",
            "route9_goal",
            "route10_goal",
        )
        self.region_mass_timeseries: dict[str, list[float]] = {
            name: []
            for name in tracked_regions
            if name in self.region_masks
        }

    def observe(self, snapshot: dict[str, object]) -> None:
        self.times.append(float(snapshot["time"]))
        rho_by_group = snapshot["rho_by_group"]
        total_rho = np.asarray(snapshot["rho"], dtype=float)
        for group_key, group_name in self.group_names.items():
            rho = np.asarray(rho_by_group[group_key], dtype=float)
            self.group_mass_timeseries[group_name].append(float(np.sum(rho[self.walkable]) * self.cell_area))
        for region_name, series in self.region_mass_timeseries.items():
            mask = self.region_masks[region_name] & self.walkable
            series.append(float(np.sum(total_rho[mask]) * self.cell_area))

    def save_case_outputs(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        peak_region_mass = {
            name: {
                "peak_mass": float(np.max(series)) if series else 0.0,
                "peak_time": float(self.times[int(np.argmax(series))]) if series else 0.0,
            }
            for name, series in self.region_mass_timeseries.items()
        }
        payload = {
            "case_id": self.case_id,
            "title": self.title,
            "times": self.times,
            "group_mass_timeseries": self.group_mass_timeseries,
            "region_mass_timeseries": self.region_mass_timeseries,
            "peak_region_mass": peak_region_mass,
        }
        save_json(output_dir / "g3_behavior_summary.json", payload)
        _save_region_timeseries_plot(output_dir / "g3_region_masses.png", payload)
        return payload


def build_g3_behavior_report(
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
        row = {
            "case_id": case_id,
            "title": summary.get("title"),
            "j1": summary.get("j1_total_travel_time"),
            "j2": summary.get("j2_high_density_exposure"),
            "j5": summary.get("j5_channel_flux_variance"),
            "sink_cumulative": summary.get("final_sink_cumulative"),
            "peak_density": summary.get("peak_density_max"),
            "entry_1_2_flux_share": summary.get("channel_flux_share", {}).get("entry_1_2"),
            "exit_8_flux_share": summary.get("channel_flux_share", {}).get("exit_8"),
            "exit_9_flux_share": summary.get("channel_flux_share", {}).get("exit_9"),
            "exit_10_flux_share": summary.get("channel_flux_share", {}).get("exit_10"),
        }
        for region_name in ("stage1_goal", "stage2_goal", "route8_goal", "route9_goal", "route10_goal"):
            peak_info = behavior.get("peak_region_mass", {}).get(region_name, {})
            row[f"{region_name}_peak_mass"] = peak_info.get("peak_mass")
            row[f"{region_name}_peak_time"] = peak_info.get("peak_time")
        rows.append(row)

    _save_csv(output_root / "g3_behavior_metrics.csv", rows)
    _save_exit_split_plot(output_root / "g3_exit_split.png", rows)
    _save_behavior_terms_plot(output_root / "g3_behavior_terms.png", rows)

    report = {
        "experiment_group": "G3",
        "cases": case_summaries,
        "behavior_cases": behavior_summaries,
        "comparison_summary": {
            "single_stage_case": "case5_single_stage_approx",
            "uniform_preference_case": "case5_multistage_uniform_preference",
            "full_behavior_case": "case5_multistage_tour",
        },
    }
    save_json(output_root / "g3_behavior_summary.json", report)
    return report


def _save_region_timeseries_plot(path: Path, payload: dict[str, object]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5), dpi=150)
    times = payload["times"]
    for region_name, series in payload["region_mass_timeseries"].items():
        ax.plot(times, series, label=region_name)
    ax.set_xlabel("time")
    ax.set_ylabel("mass")
    ax.set_title(str(payload["title"]))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_exit_split_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(row["case_id"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.8), dpi=150)
    for offset, channel_name in zip((-width, 0.0, width), ("exit_8", "exit_9", "exit_10")):
        ax.bar(x + offset, [float(row[f"{channel_name}_flux_share"] or 0.0) for row in rows], width=width, label=channel_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title("G3 exit-load split under different behavior layers")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_behavior_terms_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    labels = [str(row["case_id"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=150)
    for offset, field_name in zip((-width, 0.0, width), ("j1", "j2", "j5")):
        axes[0].bar(x + offset, [float(row[field_name]) for row in rows], width=width, label=field_name)
    for offset, region_name in zip((-width, 0.0, width), ("stage1_goal_peak_mass", "stage2_goal_peak_mass", "route10_goal_peak_mass")):
        axes[1].bar(x + offset, [float(row.get(region_name) or 0.0) for row in rows], width=width, label=region_name)
    axes[0].set_title("G3 system terms by behavior layer")
    axes[1].set_title("G3 retained mass peaks in key regions")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
