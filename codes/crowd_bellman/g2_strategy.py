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
            row = {
                "case_id": case_id,
                "title": summary.get("title"),
                "strategy": scan.get("strategy", "unknown"),
                "eta": scan.get("eta"),
                "is_baseline": bool(scan.get("is_baseline", False)),
                "j1": summary.get("j1_total_travel_time"),
                "j2": summary.get("j2_high_density_exposure"),
                "j5": summary.get("j5_channel_flux_variance"),
                "objective_value": summary.get("objective_value"),
                "sink_cumulative": summary.get("final_sink_cumulative"),
                "peak_density": summary.get("peak_density_max"),
                "hotspot_x": behavior.get("hotspot_centroid", {}).get("x"),
                "hotspot_y": behavior.get("hotspot_centroid", {}).get("y"),
                "peak_time_top": behavior.get("channel_peak_time", {}).get("top"),
                "peak_time_middle": behavior.get("channel_peak_time", {}).get("middle"),
                "peak_time_bottom": behavior.get("channel_peak_time", {}).get("bottom"),
            }
            for channel_name in ("top", "middle", "bottom"):
                row[f"flux_{channel_name}"] = summary.get("channel_flux_cumulative", {}).get(channel_name)
                row[f"flux_share_{channel_name}"] = summary.get("channel_flux_share", {}).get(channel_name)
                row[f"exposure_{channel_name}"] = behavior.get("channel_high_density_exposure", {}).get(channel_name)
                row[f"peak_density_{channel_name}"] = behavior.get("channel_peak_density", {}).get(channel_name)
            rows.append(row)

        non_dominated_case_ids = _non_dominated_case_ids(rows)
        for row in rows:
            row["is_non_dominated"] = str(row["case_id"]) in non_dominated_case_ids

        _save_csv(output_root / "g2_eta_metrics.csv", rows)
        _save_pareto_plot(output_root / "g2_eta_pareto.png", rows)
        _save_heatmaps(output_root / "g2_eta_surfaces.png", rows)
        _save_response_curves(output_root / "g2_eta_curves.png", rows)
        _save_channel_load_plot(output_root / "g2_eta_channel_loads.png", rows)
        _save_hotspot_plot(output_root / "g2_eta_hotspot_migration.png", rows)

        report = {
            "experiment_group": "G2",
            "design_version": "eta_scan",
            "cases": case_summaries,
            "behavior_cases": behavior_summaries,
            "non_dominated_cases": non_dominated_case_ids,
            "eta_values": sorted({float(row["eta"]) for row in rows if row["eta"] is not None}),
            "strategies": sorted({str(row["strategy"]) for row in rows}),
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


def _save_pareto_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    strategy_color = {
        "baseline": "black",
        "middle": "#F58518",
        "top": "#4C78A8",
        "bottom": "#54A24B",
    }
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), dpi=150)
    for row in rows:
        strategy = str(row["strategy"])
        label = f'{strategy}@{row["eta"]}' if row["eta"] is not None else strategy
        marker = "D" if bool(row["is_non_dominated"]) else "o"
        ax.scatter(
            float(row["j1"]),
            float(row["j2"]),
            color=strategy_color.get(strategy, "gray"),
            s=110 if bool(row["is_non_dominated"]) else 70,
            marker=marker,
            edgecolors="white",
            linewidths=0.8,
        )
        ax.annotate(label, (float(row["j1"]), float(row["j2"])), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("J1")
    ax.set_ylabel("J2")
    ax.set_title("G2 Pareto view with eta sweep")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_heatmaps(path: Path, rows: list[dict[str, object]]) -> None:
    guided_rows = [row for row in rows if not bool(row["is_baseline"])]
    if not guided_rows:
        return
    strategies = [name for name in ("middle", "top", "bottom") if any(str(row["strategy"]) == name for row in guided_rows)]
    etas = sorted({float(row["eta"]) for row in guided_rows if row["eta"] is not None})
    metric_defs = (
        ("j1", "J1 response surface"),
        ("j2", "J2 response surface"),
        ("j5", "J5 response surface"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=150)
    for ax, (metric_key, title) in zip(axes, metric_defs):
        grid = np.full((len(strategies), len(etas)), np.nan, dtype=float)
        for row in guided_rows:
            strategy = str(row["strategy"])
            eta = float(row["eta"])
            if strategy not in strategies:
                continue
            grid[strategies.index(strategy), etas.index(eta)] = float(row[metric_key])
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(etas)))
        ax.set_xticklabels([f"{value:g}" for value in etas])
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_xlabel("eta")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("strategy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_response_curves(path: Path, rows: list[dict[str, object]]) -> None:
    guided_rows = [row for row in rows if not bool(row["is_baseline"])]
    if not guided_rows:
        return
    baseline_row = next((row for row in rows if bool(row["is_baseline"])), None)
    metric_defs = (
        ("j1", "J1"),
        ("j2", "J2"),
        ("j5", "J5"),
    )
    strategy_color = {
        "middle": "#F58518",
        "top": "#4C78A8",
        "bottom": "#54A24B",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=150)
    for ax, (metric_key, title) in zip(axes, metric_defs):
        for strategy in ("middle", "top", "bottom"):
            strategy_rows = sorted(
                [row for row in guided_rows if str(row["strategy"]) == strategy],
                key=lambda item: float(item["eta"]),
            )
            if not strategy_rows:
                continue
            ax.plot(
                [float(row["eta"]) for row in strategy_rows],
                [float(row[metric_key]) for row in strategy_rows],
                marker="o",
                label=strategy,
                color=strategy_color[strategy],
            )
        if baseline_row is not None:
            ax.axhline(float(baseline_row[metric_key]), linestyle="--", color="black", linewidth=1.0, label="baseline")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("eta")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("metric value")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_channel_load_plot(path: Path, rows: list[dict[str, object]]) -> None:
    guided_rows = [row for row in rows if not bool(row["is_baseline"])]
    if not guided_rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    for strategy in ("middle", "top", "bottom"):
        strategy_rows = sorted(
            [row for row in guided_rows if str(row["strategy"]) == strategy],
            key=lambda item: float(item["eta"]),
        )
        if not strategy_rows:
            continue
        axes[0].plot(
            [float(row["eta"]) for row in strategy_rows],
            [float(row.get(f"flux_{strategy}") or 0.0) for row in strategy_rows],
            marker="o",
            label=strategy,
        )
        axes[1].plot(
            [float(row["eta"]) for row in strategy_rows],
            [float(row.get(f"peak_time_{strategy}") or 0.0) for row in strategy_rows],
            marker="o",
            label=strategy,
        )
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_title("Guided-channel cumulative load vs eta")
    axes[1].set_title("Guided-channel peak-load timing vs eta")
    for ax in axes:
        ax.set_xlabel("eta")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_hotspot_plot(path: Path, rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5), dpi=150)
    strategy_color = {
        "baseline": "black",
        "middle": "#F58518",
        "top": "#4C78A8",
        "bottom": "#54A24B",
    }
    for row in rows:
        x = row.get("hotspot_x")
        y = row.get("hotspot_y")
        if x is None or y is None:
            continue
        strategy = str(row["strategy"])
        label = f'{strategy}@{row["eta"]}' if row["eta"] is not None else strategy
        ax.scatter(float(x), float(y), s=70, color=strategy_color.get(strategy, "gray"))
        ax.annotate(label, (float(x), float(y)), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("hotspot centroid x")
    ax.set_ylabel("hotspot centroid y")
    ax.set_title("G2 hotspot migration under eta sweep")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
