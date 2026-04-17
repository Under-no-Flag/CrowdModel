from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from .metrics import save_json


def _crop_bounds(mask: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    points = np.argwhere(mask)
    if points.size == 0:
        return (0, mask.shape[1], 0, mask.shape[0])
    y0 = max(0, int(points[:, 0].min()) - pad)
    y1 = min(mask.shape[0], int(points[:, 0].max()) + pad + 1)
    x0 = max(0, int(points[:, 1].min()) - pad)
    x1 = min(mask.shape[1], int(points[:, 1].max()) + pad + 1)
    return (x0, x1, y0, y1)


def _trace_capture_domains(
    *,
    ux: np.ndarray,
    uy: np.ndarray,
    source_mask: np.ndarray,
    walkable: np.ndarray,
    channel_masks: dict[str, np.ndarray],
    channel_order: tuple[str, ...],
    max_steps: int = 256,
    step_size: float = 1.0,
) -> np.ndarray:
    capture = np.full(ux.shape, -1, dtype=int)
    height, width = ux.shape
    for start_y, start_x in np.argwhere(source_mask):
        x = float(start_x) + 0.5
        y = float(start_y) + 0.5
        resolved = -1
        for _ in range(max_steps):
            grid_x = int(np.clip(np.floor(x), 0, width - 1))
            grid_y = int(np.clip(np.floor(y), 0, height - 1))
            if not walkable[grid_y, grid_x]:
                break
            for channel_index, channel_name in enumerate(channel_order):
                if channel_masks[channel_name][grid_y, grid_x]:
                    resolved = channel_index
                    break
            if resolved >= 0:
                break
            dx = float(ux[grid_y, grid_x])
            dy = float(uy[grid_y, grid_x])
            norm = float(np.hypot(dx, dy))
            if norm <= 1.0e-8:
                break
            x += step_size * dx / norm
            y += step_size * dy / norm
            if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
                break
        capture[start_y, start_x] = resolved
    return capture


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    if np.isnan(value):
        return None
    return float(value)


@dataclass
class CaseBehaviorCollector:
    case_id: str
    title: str
    walkable: np.ndarray
    region_masks: dict[str, np.ndarray]
    channel_masks: dict[str, np.ndarray]
    time_horizon: float
    observation_start_fraction: float = 0.15
    rho_threshold: float = 0.05
    alignment_angle_deg: float = 15.0

    def __post_init__(self) -> None:
        self.channel_order = tuple(self.channel_masks.keys())
        feeder_mask = self.region_masks.get("feeder_band")
        if feeder_mask is None:
            feeder_mask = np.zeros_like(self.walkable, dtype=bool)
        self.feeder_mask = feeder_mask & self.walkable
        self.channels_union = np.zeros_like(self.walkable, dtype=bool)
        for channel_mask in self.channel_masks.values():
            self.channels_union |= channel_mask
        self.analysis_mask = (self.feeder_mask | self.channels_union) & self.walkable
        self.observation_start_time = float(self.time_horizon) * float(self.observation_start_fraction)
        self.direction_mass = np.zeros_like(self.walkable, dtype=float)
        self.direction_x_sum = np.zeros_like(self.walkable, dtype=float)
        self.direction_y_sum = np.zeros_like(self.walkable, dtype=float)
        self.density_sum = np.zeros_like(self.walkable, dtype=float)
        self.approach_angle_bins = np.linspace(-90.0, 90.0, 37)
        self.approach_angle_hist = np.zeros(self.approach_angle_bins.size - 1, dtype=float)
        self.approach_angle_weight = 0.0
        self.approach_angle_sum = 0.0
        self.approach_angle_sq_sum = 0.0
        self.alignment_x_sum = 0.0
        self.alignment_weight = 0.0
        self.channel_mass = {name: 0.0 for name in self.channel_order}
        self.channel_reverse_mass = {name: 0.0 for name in self.channel_order}
        self.channel_consistency_sum = {name: 0.0 for name in self.channel_order}
        self.observed_steps = 0
        self._cache: dict[str, object] | None = None

    def observe(self, snapshot: dict[str, object]) -> None:
        time_value = float(snapshot["time"])
        if time_value < self.observation_start_time:
            return

        rho = np.asarray(snapshot["rho"], dtype=float)
        vx = np.asarray(snapshot["vx"], dtype=float)
        vy = np.asarray(snapshot["vy"], dtype=float)
        speed = np.hypot(vx, vy)
        valid = self.walkable & (rho > self.rho_threshold) & (speed > 1.0e-8)
        if not np.any(valid):
            return

        dir_x = np.zeros_like(vx)
        dir_y = np.zeros_like(vy)
        dir_x[valid] = vx[valid] / speed[valid]
        dir_y[valid] = vy[valid] / speed[valid]

        analysis_valid = valid & self.analysis_mask
        self.direction_mass[analysis_valid] += rho[analysis_valid]
        self.direction_x_sum[analysis_valid] += rho[analysis_valid] * dir_x[analysis_valid]
        self.direction_y_sum[analysis_valid] += rho[analysis_valid] * dir_y[analysis_valid]
        self.density_sum[self.analysis_mask] += rho[self.analysis_mask]

        feeder_valid = valid & self.feeder_mask
        if np.any(feeder_valid):
            approach_angles = np.degrees(np.arctan2(vy[feeder_valid], vx[feeder_valid]))
            weights = rho[feeder_valid]
            self.approach_angle_hist += np.histogram(
                approach_angles,
                bins=self.approach_angle_bins,
                weights=weights,
            )[0]
            self.approach_angle_weight += float(weights.sum())
            self.approach_angle_sum += float(np.sum(weights * approach_angles))
            self.approach_angle_sq_sum += float(np.sum(weights * approach_angles * approach_angles))

            _, xx = np.indices(rho.shape)
            alignment_valid = feeder_valid & (np.abs(np.degrees(np.arctan2(vy, vx))) <= self.alignment_angle_deg)
            if np.any(alignment_valid):
                alignment_weights = rho[alignment_valid]
                self.alignment_x_sum += float(np.sum(xx[alignment_valid] * alignment_weights))
                self.alignment_weight += float(np.sum(alignment_weights))

        for channel_name, channel_mask in self.channel_masks.items():
            channel_valid = valid & channel_mask
            if not np.any(channel_valid):
                continue
            channel_weights = rho[channel_valid]
            self.channel_mass[channel_name] += float(np.sum(channel_weights))
            self.channel_reverse_mass[channel_name] += float(np.sum(channel_weights[vx[channel_valid] < 0.0]))
            self.channel_consistency_sum[channel_name] += float(np.sum(channel_weights * dir_x[channel_valid]))

        self.observed_steps += 1

    def _finalize(self) -> dict[str, object]:
        if self._cache is not None:
            return self._cache

        mean_density = np.zeros_like(self.walkable, dtype=float)
        if self.observed_steps > 0:
            mean_density[self.analysis_mask] = self.density_sum[self.analysis_mask] / float(self.observed_steps)

        mean_dir_x = np.zeros_like(self.walkable, dtype=float)
        mean_dir_y = np.zeros_like(self.walkable, dtype=float)
        valid_direction = self.direction_mass > 1.0e-8
        mean_dir_x[valid_direction] = self.direction_x_sum[valid_direction] / self.direction_mass[valid_direction]
        mean_dir_y[valid_direction] = self.direction_y_sum[valid_direction] / self.direction_mass[valid_direction]

        capture_map = _trace_capture_domains(
            ux=mean_dir_x,
            uy=mean_dir_y,
            source_mask=self.feeder_mask,
            walkable=self.walkable,
            channel_masks=self.channel_masks,
            channel_order=self.channel_order,
        )
        capture_counts = {
            channel_name: int(np.sum(capture_map[self.feeder_mask] == channel_index))
            for channel_index, channel_name in enumerate(self.channel_order)
        }
        unresolved = int(np.sum(capture_map[self.feeder_mask] < 0))
        total_capture = max(1, int(np.sum(self.feeder_mask)))
        capture_share = {
            channel_name: float(count / total_capture)
            for channel_name, count in capture_counts.items()
        }
        capture_share["unresolved"] = float(unresolved / total_capture)

        angle_prob = np.zeros_like(self.approach_angle_hist)
        if self.approach_angle_weight > 1.0e-8:
            angle_prob = self.approach_angle_hist / self.approach_angle_weight
        angle_centers = 0.5 * (self.approach_angle_bins[:-1] + self.approach_angle_bins[1:])
        angle_mean = self.approach_angle_sum / self.approach_angle_weight if self.approach_angle_weight > 1.0e-8 else np.nan
        angle_var = (
            self.approach_angle_sq_sum / self.approach_angle_weight - angle_mean * angle_mean
            if self.approach_angle_weight > 1.0e-8
            else np.nan
        )
        angle_std = np.sqrt(max(float(angle_var), 0.0)) if not np.isnan(angle_var) else np.nan
        alignment_x_mean = self.alignment_x_sum / self.alignment_weight if self.alignment_weight > 1.0e-8 else np.nan

        channel_behavior: dict[str, dict[str, float | None]] = {}
        for channel_name in self.channel_order:
            mass = self.channel_mass[channel_name]
            reverse_share = self.channel_reverse_mass[channel_name] / mass if mass > 1.0e-8 else np.nan
            consistency = self.channel_consistency_sum[channel_name] / mass if mass > 1.0e-8 else np.nan
            channel_behavior[channel_name] = {
                "observed_mass": float(mass),
                "reverse_direction_share": _safe_float(reverse_share),
                "direction_consistency_east": _safe_float(consistency),
                "capture_share": float(capture_share.get(channel_name, 0.0)),
            }

        payload = {
            "summary": {
                "case_id": self.case_id,
                "title": self.title,
                "observed_steps": int(self.observed_steps),
                "observation_start_time": float(self.observation_start_time),
                "approach_angle_mean_deg": _safe_float(angle_mean),
                "approach_angle_std_deg": _safe_float(angle_std),
                "alignment_x_mean": _safe_float(alignment_x_mean),
                "capture_share": capture_share,
                "channel_behavior": channel_behavior,
                "approach_angle_histogram": {
                    "centers_deg": [float(value) for value in angle_centers],
                    "probability": [float(value) for value in angle_prob],
                },
            },
            "mean_density": mean_density,
            "mean_dir_x": mean_dir_x,
            "mean_dir_y": mean_dir_y,
            "capture_map": capture_map,
        }
        self._cache = payload
        return payload

    def save_case_outputs(self, output_dir: Path) -> dict[str, object]:
        payload = self._finalize()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "behavior_summary.json", payload["summary"])
        self._save_direction_plot(
            path=output_dir / "behavior_direction.png",
            mean_density=np.asarray(payload["mean_density"], dtype=float),
            mean_dir_x=np.asarray(payload["mean_dir_x"], dtype=float),
            mean_dir_y=np.asarray(payload["mean_dir_y"], dtype=float),
        )
        self._save_capture_plot(
            path=output_dir / "capture_domains.png",
            capture_map=np.asarray(payload["capture_map"], dtype=int),
        )
        return payload["summary"]

    def _save_direction_plot(
        self,
        *,
        path: Path,
        mean_density: np.ndarray,
        mean_dir_x: np.ndarray,
        mean_dir_y: np.ndarray,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        crop_mask = self.analysis_mask | (~self.walkable)
        x0, x1, y0, y1 = _crop_bounds(crop_mask, pad=6)

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5), dpi=150)
        density = mean_density.copy()
        density[~self.walkable] = np.nan
        im = ax.imshow(density[y0:y1, x0:x1], origin="lower", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean density")

        step = 3
        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        ux = mean_dir_x[y0:y1:step, x0:x1:step]
        uy = mean_dir_y[y0:y1:step, x0:x1:step]
        valid = self.analysis_mask[y0:y1:step, x0:x1:step] & (np.hypot(ux, uy) > 1.0e-8)
        ax.quiver(
            xx[valid] - x0,
            yy[valid] - y0,
            ux[valid],
            uy[valid],
            color="white",
            scale=18,
            width=0.003,
        )
        obstacle_y, obstacle_x = np.where(~self.walkable[y0:y1, x0:x1])
        ax.scatter(obstacle_x, obstacle_y, s=2, c="black", marker="s", linewidths=0)
        ax.set_title(f"{self.title}: mean local direction field")
        ax.set_xlim(0, x1 - x0 - 1)
        ax.set_ylim(0, y1 - y0 - 1)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _save_capture_plot(self, *, path: Path, capture_map: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        crop_mask = self.analysis_mask | (~self.walkable)
        x0, x1, y0, y1 = _crop_bounds(crop_mask, pad=6)
        capture_view = np.full(capture_map.shape, np.nan, dtype=float)
        for channel_index, _channel_name in enumerate(self.channel_order):
            capture_view[capture_map == channel_index] = float(channel_index)
        cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B"])

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5), dpi=150)
        ax.imshow(capture_view[y0:y1, x0:x1], origin="lower", cmap=cmap, vmin=0.0, vmax=max(0, len(self.channel_order) - 1))
        obstacle_y, obstacle_x = np.where(~self.walkable[y0:y1, x0:x1])
        ax.scatter(obstacle_x, obstacle_y, s=2, c="black", marker="s", linewidths=0)
        ax.set_title(f"{self.title}: feeder capture domains")
        legend_handles = [
            Line2D([0], [0], marker="s", linestyle="", color=cmap(index / max(1, len(self.channel_order) - 1)), label=name)
            for index, name in enumerate(self.channel_order)
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        ax.set_xlim(0, x1 - x0 - 1)
        ax.set_ylim(0, y1 - y0 - 1)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def _guided_channel(case_id: str) -> str | None:
    mapping = {
        "case2_middle_guided": "middle",
        "case3_top_guided": "top",
        "case4_bottom_guided": "bottom",
    }
    return mapping.get(case_id)


def build_g1_mechanism_report(
    *,
    output_root: Path,
    case_summaries: list[dict[str, object]],
    behavior_summaries: list[dict[str, object]],
    bridge_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_by_case = {str(item["case_id"]): item for item in case_summaries}
    behavior_by_case = {str(item["case_id"]): item for item in behavior_summaries}

    table_rows: list[dict[str, object]] = []
    for case_id, behavior in behavior_by_case.items():
        summary = summary_by_case.get(case_id, {})
        capture_share = behavior.get("capture_share", {})
        row = {
            "case_id": case_id,
            "objective_value": summary.get("objective_value"),
            "sink_cumulative": summary.get("final_sink_cumulative"),
            "peak_density": summary.get("peak_density_max"),
            "approach_angle_std_deg": behavior.get("approach_angle_std_deg"),
            "alignment_x_mean": behavior.get("alignment_x_mean"),
            "top_capture_share": capture_share.get("top", 0.0),
            "middle_capture_share": capture_share.get("middle", 0.0),
            "bottom_capture_share": capture_share.get("bottom", 0.0),
        }
        for channel_name in ("top", "middle", "bottom"):
            channel_behavior = behavior.get("channel_behavior", {}).get(channel_name, {})
            row[f"{channel_name}_reverse_share"] = channel_behavior.get("reverse_direction_share")
            row[f"{channel_name}_consistency"] = channel_behavior.get("direction_consistency_east")
            flux_share = summary.get("channel_flux_share", {}).get(channel_name)
            row[f"{channel_name}_flux_share"] = flux_share
        table_rows.append(row)

    _save_behavior_csv(output_root / "g1_behavior_metrics.csv", table_rows)
    _save_u_validation_plot(output_root / "g1_u_validation.png", table_rows)
    _save_m_validation_plot(output_root / "g1_m_validation.png", behavior_by_case)
    _save_configuration_plot(output_root / "g1_configuration_sensitivity.png", table_rows)

    report = {
        "experiment_group": "G1",
        "design_version": "2026-04-17_two_layer",
        "cases": case_summaries,
        "behavior_cases": behavior_summaries,
        "mechanism_authenticity": {
            "u_branch": {
                "baseline_case": "case1_baseline",
                "comparison_case": "case_u_only_middle",
                "baseline_middle_reverse_share": behavior_by_case.get("case1_baseline", {}).get("channel_behavior", {}).get("middle", {}).get("reverse_direction_share"),
                "u_only_middle_reverse_share": behavior_by_case.get("case_u_only_middle", {}).get("channel_behavior", {}).get("middle", {}).get("reverse_direction_share"),
                "baseline_middle_capture_share": behavior_by_case.get("case1_baseline", {}).get("capture_share", {}).get("middle"),
                "u_only_middle_capture_share": behavior_by_case.get("case_u_only_middle", {}).get("capture_share", {}).get("middle"),
            },
            "m_branch": {
                "baseline_case": "case1_baseline",
                "comparison_case": "case_m_only_middle",
                "baseline_angle_std_deg": behavior_by_case.get("case1_baseline", {}).get("approach_angle_std_deg"),
                "m_only_angle_std_deg": behavior_by_case.get("case_m_only_middle", {}).get("approach_angle_std_deg"),
                "baseline_alignment_x_mean": behavior_by_case.get("case1_baseline", {}).get("alignment_x_mean"),
                "m_only_alignment_x_mean": behavior_by_case.get("case_m_only_middle", {}).get("alignment_x_mean"),
            },
        },
        "management_usability": {
            "guided_cases": [
                {
                    "case_id": row["case_id"],
                    "guided_channel": _guided_channel(str(row["case_id"])),
                    "dominant_capture_channel": max(
                        ("top", "middle", "bottom"),
                        key=lambda channel_name: float(row.get(f"{channel_name}_capture_share", 0.0) or 0.0),
                    ),
                    "dominant_flux_channel": max(
                        ("top", "middle", "bottom"),
                        key=lambda channel_name: float(row.get(f"{channel_name}_flux_share", 0.0) or 0.0),
                    ),
                }
                for row in table_rows
                if _guided_channel(str(row["case_id"])) is not None
            ],
        },
    }
    if bridge_summary is not None:
        report["behavior_layer_bridge"] = bridge_summary
    save_json(output_root / "g1_mechanism_summary.json", report)
    return report


def _save_behavior_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_u_validation_plot(path: Path, rows: list[dict[str, object]]) -> None:
    focus = [row for row in rows if row["case_id"] in {"case1_baseline", "case_u_only_middle", "case2_middle_guided"}]
    if not focus:
        return
    labels = [str(row["case_id"]) for row in focus]
    reverse_share = [float(row.get("middle_reverse_share") or 0.0) for row in focus]
    consistency = [float(row.get("middle_consistency") or 0.0) for row in focus]
    flux_share = [float(row.get("middle_flux_share") or 0.0) for row in focus]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8), dpi=150)
    ax.bar(x - width, reverse_share, width=width, label="middle reverse share")
    ax.bar(x, consistency, width=width, label="middle east consistency")
    ax.bar(x + width, flux_share, width=width, label="middle flux share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title("G1 / U branch: local direction organization")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_m_validation_plot(path: Path, behavior_by_case: dict[str, dict[str, object]]) -> None:
    focus_ids = [
        "case1_baseline",
        "case_m_only_middle",
        "case2_middle_guided",
        "case3_top_guided",
        "case4_bottom_guided",
    ]
    plotted = []
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=150)
    for case_id in focus_ids:
        behavior = behavior_by_case.get(case_id)
        if behavior is None:
            continue
        histogram = behavior.get("approach_angle_histogram", {})
        centers = histogram.get("centers_deg", [])
        probability = histogram.get("probability", [])
        if not centers or not probability:
            continue
        ax.plot(centers, probability, label=case_id)
        plotted.append(case_id)
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("G1 / M branch: feeder approach-angle distribution")
    ax.set_xlabel("approach angle to east (deg)")
    ax.set_ylabel("density-weighted probability")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_configuration_plot(path: Path, rows: list[dict[str, object]]) -> None:
    focus = [row for row in rows if row["case_id"] in {"case2_middle_guided", "case3_top_guided", "case4_bottom_guided"}]
    if not focus:
        return

    labels = [str(row["case_id"]) for row in focus]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    for offset, channel_name in zip((-width, 0.0, width), ("top", "middle", "bottom")):
        axes[0].bar(
            x + offset,
            [float(row.get(f"{channel_name}_capture_share") or 0.0) for row in focus],
            width=width,
            label=f"{channel_name} capture",
        )
        axes[1].bar(
            x + offset,
            [float(row.get(f"{channel_name}_flux_share") or 0.0) for row in focus],
            width=width,
            label=f"{channel_name} flux",
        )

    axes[0].set_title("Capture-domain sensitivity")
    axes[1].set_title("Measured flux sensitivity")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10)
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
