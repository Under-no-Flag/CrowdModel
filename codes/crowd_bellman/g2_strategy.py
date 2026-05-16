from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from .compilers.config_compiler import compile_scene
from .loaders.config_loader import load_run_config, load_scene_spec
from .metrics import safety_risk_density, save_json
from .plotting import DENSITY_CMAP, DENSITY_INTERPOLATION


CHANNEL_NAMES = ("top", "middle", "lower_middle", "bottom")
OBJECTIVE_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "objective_best_to_poor",
    [(0.0, "#F7FBFF"), (0.5, "#9ECAE1"), (1.0, "#084594")],
)
CHANNEL_LOAD_CMAP = LinearSegmentedColormap.from_list(
    "channel_load_share",
    [
        (0.0, "#F2F2F2"),
        (0.25, "#D0E1F2"),
        (0.5, "#74A9CF"),
        (0.75, "#2B8CBE"),
        (1.0, "#045A8D"),
    ],
)


@dataclass
class G2StrategyCollector:
    case_id: str
    title: str
    walkable: np.ndarray
    channel_masks: dict[str, np.ndarray]
    rho_safe: float
    dx: float
    j2_metric: str = "soft"
    j2_gamma: float = 1.0

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
        risk_density = safety_risk_density(rho, self.rho_safe, self.j2_metric, self.j2_gamma)
        high_density_mask = self.walkable & (risk_density > 0.0)
        if np.any(high_density_mask):
            weights = risk_density[high_density_mask] * dt
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
            self.channel_high_density_exposure[channel_name] += float(
                np.sum(risk_density[active_mask]) * self.cell_area * dt
            )

    def save_case_outputs(self, output_dir: Path) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        hotspot_centroid = {
            "x": None if self.hotspot_weight <= 1.0e-12 else float(self.hotspot_x_sum / self.hotspot_weight),
            "y": None if self.hotspot_weight <= 1.0e-12 else float(self.hotspot_y_sum / self.hotspot_weight),
        }
        payload = {
            "case_id": self.case_id,
            "title": self.title,
            "j2_metric": self.j2_metric,
            "j2_gamma": float(self.j2_gamma),
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
            **{f"direction_{channel_name}": directions.get(channel_name) for channel_name in CHANNEL_NAMES},
            "entry_channels": ",".join(scan.get("entry_channels", [])),
            "return_channels": ",".join(scan.get("return_channels", [])),
            "closed_channels": ",".join(scan.get("closed_channels", [])),
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
        for channel_name in CHANNEL_NAMES:
            row[f"flux_{channel_name}"] = summary.get("channel_flux_cumulative", {}).get(channel_name)
            row[f"flux_share_{channel_name}"] = summary.get("channel_flux_share", {}).get(channel_name)
            row[f"peak_time_{channel_name}"] = behavior.get("channel_peak_time", {}).get(channel_name)
            row[f"exposure_{channel_name}"] = behavior.get("channel_high_density_exposure", {}).get(channel_name)
            row[f"peak_density_{channel_name}"] = behavior.get("channel_peak_density", {}).get(channel_name)
        rows.append(row)

    non_dominated_case_ids = _non_dominated_case_ids(rows)
    for row in rows:
        row["is_non_dominated"] = str(row["case_id"]) in non_dominated_case_ids
        row["short_label"] = _short_label(row)
        row["strategy_code"] = _strategy_code(row)
        row["direction_code"] = _direction_code(row)
        row["control_meaning"] = _control_meaning(row)
        row["objective_sum"] = float(row["j1"]) + float(row["j2"]) + float(row["j5"])

    _save_csv(output_root / "g2_direction_metrics.csv", rows)
    _save_pareto_plot(output_root / "g2_direction_pareto.png", rows)
    _save_metric_bar_plot(output_root / "g2_direction_objectives.png", rows)
    _save_channel_plot(output_root / "g2_direction_channel_loads.png", rows)
    _save_hotspot_plot(output_root / "g2_direction_hotspot_migration.png", rows)
    _save_control_tradeoff_summary(output_root / "g2_control_tradeoff_summary.png", rows)
    _save_spatial_mechanism_summary(output_root / "g2_spatial_mechanism_summary.png", rows)

    report = {
        "experiment_group": "G2",
        "design_version": "direction_scan_multistage_with_closure",
        "cases": case_summaries,
        "behavior_cases": behavior_summaries,
        "non_dominated_cases": non_dominated_case_ids,
        "families": sorted({str(row["family"]) for row in rows}),
        "journal_outputs": {
            "control_tradeoff_summary": str(output_root / "g2_control_tradeoff_summary.png"),
            "spatial_mechanism_summary": str(output_root / "g2_spatial_mechanism_summary.png"),
            "pareto_projection": str(output_root / "g2_direction_pareto.png"),
            "hotspot_migration": str(output_root / "g2_direction_hotspot_migration.png"),
            "metrics_csv": str(output_root / "g2_direction_metrics.csv"),
        },
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
    return str(row.get("short_label") or _short_label(row))


def _case_number(row: dict[str, object]) -> int:
    case_id = str(row.get("case_id", ""))
    if case_id.startswith("case"):
        digits = []
        for char in case_id[4:]:
            if char.isdigit():
                digits.append(char)
            else:
                break
        if digits:
            return int("".join(digits))
    return 999


def _short_label(row: dict[str, object]) -> str:
    return f"C{_case_number(row)}"


def _direction_symbol(value: object) -> str:
    normalized = str(value or "").upper()
    if normalized == "FREE":
        return "F"
    if normalized == "CLOSED":
        return "X"
    if normalized in {"E", "W"}:
        return normalized
    return normalized[:1] or "?"


def _direction_code(row: dict[str, object]) -> str:
    return "".join(_direction_symbol(row.get(f"direction_{channel_name}")) for channel_name in CHANNEL_NAMES)


def _active_position(row: dict[str, object], target_state: str) -> str:
    abbreviations = {"top": "T", "middle": "M", "lower_middle": "LM", "bottom": "B"}
    for channel_name in CHANNEL_NAMES:
        if _direction_symbol(row.get(f"direction_{channel_name}")) == target_state:
            return abbreviations[channel_name]
    return "?"


def _strategy_code(row: dict[str, object]) -> str:
    family = str(row.get("family", ""))
    if family == "baseline":
        return "BSL"
    if family == "single_entry":
        return f"SE-{_active_position(row, 'E')}"
    if family == "single_return":
        return f"SR-{_active_position(row, 'W')}"
    if family == "one_closed":
        return f"CL-{_active_position(row, 'X')}"
    return family


def _control_meaning(row: dict[str, object]) -> str:
    position_names = {"T": "top", "M": "middle", "LM": "lower-middle", "B": "bottom"}
    strategy = _strategy_code(row)
    if strategy == "BSL":
        return "four channels free"
    if strategy.startswith("SE-"):
        return f"{position_names.get(strategy[3:], '?')} channel as the only entry channel"
    if strategy.startswith("SR-"):
        return f"{position_names.get(strategy[3:], '?')} channel as the only return channel"
    if strategy.startswith("CL-"):
        return f"{position_names.get(strategy[3:], '?')} channel closed"
    return strategy


def _family_color(family: str) -> str:
    if family == "baseline":
        return "black"
    if family == "single_entry":
        return "#F58518"
    if family == "single_return":
        return "#4C78A8"
    if family == "one_closed":
        return "#54A24B"
    return "gray"


def _family_label(family: str) -> str:
    labels = {
        "baseline": "BSL",
        "single_entry": "SE",
        "single_return": "SR",
        "one_closed": "CL",
    }
    return labels.get(family, family)


def _channel_short_labels() -> list[str]:
    return ["T", "M", "LM", "B"]


def _set_publication_axes(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.25")
    ax.spines["bottom"].set_color("0.25")
    ax.tick_params(colors="0.15", labelsize=8)
    if grid:
        ax.grid(color="0.88", linewidth=0.45, alpha=0.55)
    else:
        ax.grid(False)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.075,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
    )


def _sorted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=_case_number)


def _objective_sum(row: dict[str, object]) -> float:
    return float(row.get("objective_sum") or (float(row["j1"]) + float(row["j2"]) + float(row["j5"])))


def _leader_offset(label: str) -> tuple[int, int]:
    offsets = {
        "C1": (10, 12),
        "C3": (-34, 10),
        "C4": (-32, -16),
        "C6": (10, -18),
        "C8": (12, 10),
        "C9": (10, -16),
    }
    return offsets.get(label, (8, 8))


def _hotspot_leader_offset(label: str) -> tuple[int, int]:
    offsets = {
        "C1": (44, -34),
        "C3": (-48, -28),
        "C4": (-48, 24),
        "C6": (36, -58),
        "C8": (42, 30),
        "C9": (44, 8),
    }
    return offsets.get(label, (8, 8))


def _label_selected_points(ax: plt.Axes, rows: list[dict[str, object]], *, x_key: str, y_key: str) -> None:
    for row in rows:
        if not bool(row.get("is_non_dominated")):
            continue
        label = _label(row)
        dx, dy = _leader_offset(label)
        ax.annotate(
            label,
            (float(row[x_key]), float(row[y_key])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.6, "shrinkA": 0, "shrinkB": 0},
        )


def _save_pareto_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    rows = _sorted_rows(rows)
    j5_values = np.asarray([float(row["j5"]) for row in rows], dtype=float)
    objectives = np.asarray([_objective_sum(row) for row in rows], dtype=float)
    size_min, size_max = 70.0, 240.0
    size_scale = (objectives - objectives.min()) / max(float(objectives.max() - objectives.min()), 1.0e-12)
    sizes = size_min + (size_max - size_min) * size_scale
    fig, ax = plt.subplots(1, 1, figsize=(7.8, 5.8), dpi=150)
    scatter_for_colorbar = None
    for idx, row in enumerate(rows):
        marker = "D" if bool(row["is_non_dominated"]) else "o"
        ax.scatter(
            float(row["j1"]),
            float(row["j2"]),
            c=[float(row["j5"])],
            cmap="viridis",
            vmin=float(j5_values.min()),
            vmax=float(j5_values.max()),
            s=float(sizes[idx]),
            marker=marker,
            edgecolors=_family_color(str(row["family"])),
            linewidths=1.4 if bool(row["is_non_dominated"]) else 0.9,
            alpha=0.92,
        )
        if scatter_for_colorbar is None:
            scatter_for_colorbar = ax.collections[-1]
    _label_selected_points(ax, rows, x_key="j1", y_key="j2")
    ax.set_xlabel(r"$\tilde J_1$")
    ax.set_ylabel(r"$\tilde J_2$")
    ax.set_title("Three-Objective Pareto Projection")
    ax.grid(alpha=0.18)
    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\tilde J_5$")
    legend_handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="0.65", markeredgecolor="0.2", label="Non-dominated"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.65", markeredgecolor="0.2", label="Dominated"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_metric_bar_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    rows = _sorted_rows(rows)
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
    rows = _sorted_rows(rows)
    labels = [_label(row) for row in rows]
    x = np.arange(len(rows))
    width = 0.8 / max(1, len(CHANNEL_NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=150)
    for idx, channel_name in enumerate(CHANNEL_NAMES):
        offset = (idx - (len(CHANNEL_NAMES) - 1) / 2.0) * width
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
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.8), dpi=180)
    _draw_hotspot_migration(ax, rows)
    ax.legend(handles=_family_legend_handles(), frameon=False, loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_design_matrix(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    rows = _sorted_rows(rows)
    mapping = {"E": 0, "W": 1, "F": 2, "X": 3}
    colors = ["#D98C3A", "#5B7FA6", "#D9D9D9", "#3A3A3A"]
    matrix = np.asarray(
        [[mapping[_direction_symbol(row.get(f"direction_{channel_name}"))] for channel_name in CHANNEL_NAMES] for row in rows],
        dtype=int,
    )
    ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    for y, row in enumerate(rows):
        for x, channel_name in enumerate(CHANNEL_NAMES):
            symbol = _direction_symbol(row.get(f"direction_{channel_name}"))
            ax.text(x, y, symbol, ha="center", va="center", color="white" if symbol == "X" else "black", fontsize=8, weight="bold")
    ax.set_xticks(np.arange(len(CHANNEL_NAMES)))
    ax.set_xticklabels(_channel_short_labels())
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_label(row) for row in rows], fontsize=7)
    ax.set_title("Design matrix", fontsize=10, pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_pareto_projection(ax: plt.Axes, fig: plt.Figure, rows: list[dict[str, object]]) -> None:
    rows = _sorted_rows(rows)
    j5_values = np.asarray([float(row["j5"]) for row in rows], dtype=float)
    objectives = np.asarray([_objective_sum(row) for row in rows], dtype=float)
    size_scale = (objectives - objectives.min()) / max(float(objectives.max() - objectives.min()), 1.0e-12)
    sizes = 55.0 + 185.0 * size_scale
    scatter_for_colorbar = None
    for idx, row in enumerate(rows):
        scatter = ax.scatter(
            float(row["j1"]),
            float(row["j2"]),
            c=[float(row["j5"])],
            cmap="viridis",
            vmin=float(j5_values.min()),
            vmax=float(j5_values.max()),
            s=float(sizes[idx]),
            marker="D" if bool(row["is_non_dominated"]) else "o",
            edgecolors=_family_color(str(row["family"])),
            linewidths=1.3 if bool(row["is_non_dominated"]) else 0.8,
            alpha=0.92,
        )
        if scatter_for_colorbar is None:
            scatter_for_colorbar = scatter
    _label_selected_points(ax, rows, x_key="j1", y_key="j2")
    ax.set_xlabel(r"$\tilde J_1$")
    ax.set_ylabel(r"$\tilde J_2$")
    ax.set_title("Pareto projection", fontsize=10, pad=8)
    _set_publication_axes(ax, grid=True)
    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\tilde J_5$")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0)


def _draw_objective_heatmap(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=_objective_sum)
    metrics = ("j1", "j2", "j5", "objective_sum")
    labels = (r"$\tilde J_1$", r"$\tilde J_2$", r"$\tilde J_5$", r"$\tilde J$")
    values = np.asarray([[float(row[metric]) if metric != "objective_sum" else _objective_sum(row) for metric in metrics] for row in rows], dtype=float)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    scaled = (values - mins) / np.maximum(maxs - mins, 1.0e-12)
    ax.imshow(scaled, cmap=OBJECTIVE_HEATMAP_CMAP, aspect="auto", vmin=0.0, vmax=1.0)
    minima = np.argmin(values, axis=0)
    for x, y in enumerate(minima):
        ax.scatter(x, y, s=22, facecolors="none", edgecolors="white", linewidths=1.2)
        ax.text(
            x,
            y,
            f"{values[y, x]:.3f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if scaled[y, x] > 0.62 else "black",
        )
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_label(row) for row in rows], fontsize=7)
    ax.set_title("Objective summary", fontsize=10, pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_channel_load_heatmap(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    rows = _sorted_rows(rows)
    values = np.asarray([[float(row.get(f"flux_share_{channel_name}") or 0.0) for channel_name in CHANNEL_NAMES] for row in rows], dtype=float)
    ax.imshow(values, cmap=CHANNEL_LOAD_CMAP, aspect="auto", vmin=0.0, vmax=0.8)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if values[y, x] < 0.15:
                continue
            text_color = "white" if values[y, x] > 0.55 * values.max() else "black"
            ax.text(x, y, f"{100.0 * values[y, x]:.0f}%", ha="center", va="center", fontsize=7, color=text_color)
    ax.set_xticks(np.arange(len(CHANNEL_NAMES)))
    ax.set_xticklabels(_channel_short_labels())
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([_label(row) for row in rows], fontsize=7)
    ax.set_title("Channel load", fontsize=10, pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


@lru_cache(maxsize=1)
def _g2_scene_walkable() -> np.ndarray:
    codes_root = Path(__file__).resolve().parents[1]
    run_path = codes_root / "scenes" / "examples" / "g2_multistage_directional" / "run_baseline.toml"
    run_spec = load_run_config(run_path)
    scene_spec = load_scene_spec(run_spec.scene_path)
    bundle = compile_scene(scene_spec=scene_spec, cfg=run_spec.simulation)
    return np.asarray(bundle.scene.walkable, dtype=bool)


def _draw_scene_geometry(ax: plt.Axes) -> None:
    walkable = _g2_scene_walkable()
    density_background = np.zeros_like(walkable, dtype=float)
    density_background[~walkable] = np.nan
    ax.set_facecolor("white")
    ax.imshow(
        density_background,
        origin="lower",
        cmap=DENSITY_CMAP,
        vmin=0.0,
        vmax=5.0,
        interpolation=DENSITY_INTERPOLATION,
        zorder=0,
    )
    obstacle_y, obstacle_x = np.where(~walkable)
    ax.scatter(obstacle_x, obstacle_y, s=1.7, c="black", marker="s", linewidths=0, zorder=1)
    ax.set_xlim(0, walkable.shape[1] - 1)
    ax.set_ylim(0, walkable.shape[0] - 1)
    ax.set_aspect("equal")


def _display_hotspot_xy(row: dict[str, object]) -> tuple[float, float] | None:
    x = row.get("hotspot_x")
    y = row.get("hotspot_y")
    if x is None or y is None:
        return None
    x_value = float(x)
    y_value = float(y)
    walkable = _g2_scene_walkable()
    xi = int(round(x_value))
    yi = int(round(y_value))
    if 0 <= yi < walkable.shape[0] and 0 <= xi < walkable.shape[1] and bool(walkable[yi, xi]):
        return x_value, y_value
    walkable_y, walkable_x = np.where(walkable)
    distances = (walkable_x.astype(float) - x_value) ** 2 + (walkable_y.astype(float) - y_value) ** 2
    nearest = int(np.argmin(distances))
    return float(walkable_x[nearest]), float(walkable_y[nearest])


def _draw_hotspot_migration(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    rows = _sorted_rows(rows)
    _draw_scene_geometry(ax)
    exposure_values = np.asarray([float(row.get("j2_raw") or 0.0) for row in rows], dtype=float)
    size_scale = (exposure_values - exposure_values.min()) / max(float(exposure_values.max() - exposure_values.min()), 1.0e-12)
    for idx, row in enumerate(rows):
        point = _display_hotspot_xy(row)
        if point is None:
            continue
        x, y = point
        ax.scatter(
            x,
            y,
            s=55.0 + 180.0 * float(size_scale[idx]),
            color=_family_color(str(row["family"])),
            alpha=0.82,
            edgecolors="white",
            linewidths=0.75,
            zorder=4,
        )
    for row in rows:
        if not bool(row.get("is_non_dominated")):
            continue
        point = _display_hotspot_xy(row)
        if point is None:
            continue
        x, y = point
        dx, dy = _hotspot_leader_offset(_label(row))
        ax.annotate(
            _label(row),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.6, "shrinkA": 0, "shrinkB": 0},
            zorder=5,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Hotspot migration", fontsize=10, pad=8)
    ax.text(0.02, 0.03, "marker size: exposure", transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="white", alpha=0.78)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _family_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=_family_color(family), label=_family_label(family))
        for family in ("baseline", "single_entry", "single_return", "one_closed")
    ]


def _save_control_tradeoff_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 5.1),
        dpi=220,
        gridspec_kw={"width_ratios": [0.78, 1.08, 0.92], "wspace": 0.28},
    )
    _draw_design_matrix(axes[0], rows)
    _draw_pareto_projection(axes[1], fig, rows)
    _draw_objective_heatmap(axes[2], rows)
    for label, ax in zip(("A", "B", "C"), axes):
        _panel_label(ax, label)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.92, bottom=0.08, wspace=0.28)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_spatial_mechanism_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.8),
        dpi=220,
        gridspec_kw={"width_ratios": [0.88, 1.12], "wspace": 0.20},
    )
    _draw_channel_load_heatmap(axes[0], rows)
    _draw_hotspot_migration(axes[1], rows)
    for label, ax in zip(("A", "B"), axes):
        _panel_label(ax, label)
    fig.legend(handles=_family_legend_handles(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=8)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.91, bottom=0.15, wspace=0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
