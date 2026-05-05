from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .metrics import CaseStats


DensityContourLevels = int | Sequence[float] | None
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "density_blue_to_red",
    ("#00106e", "#0057ff", "#00b7ff", "#24d96b", "#ffe600", "#ff8a00", "#d7191c"),
)
DENSITY_INTERPOLATION = "bilinear"


def _contour_levels_from_data(
    values: np.ndarray,
    levels: DensityContourLevels,
    *,
    default_count: int = 6,
) -> np.ndarray | None:
    if levels is None:
        return None

    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return None

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None

    if isinstance(levels, int):
        if levels <= 0:
            return None
        return np.linspace(vmin, vmax, levels + 2, dtype=float)[1:-1]

    raw = np.asarray(list(levels), dtype=float)
    if raw.size == 0:
        return None
    raw = raw[np.isfinite(raw)]
    raw = raw[(raw > vmin) & (raw < vmax)]
    if raw.size == 0:
        return None
    return np.unique(np.sort(raw))


def draw_density_contours(
    ax: plt.Axes,
    density: np.ndarray,
    levels: DensityContourLevels = None,
    *,
    colors: str = "black",
    linewidths: float = 0.55,
    alpha: float = 0.65,
    linestyles: str = "--",
) -> object | None:
    contour_levels = _contour_levels_from_data(density, levels)
    if contour_levels is None:
        return None
    return ax.contour(
        density,
        levels=contour_levels,
        colors=colors,
        linewidths=linewidths,
        alpha=alpha,
        linestyles=linestyles,
    )


def parse_density_contour_levels(raw: str | None) -> DensityContourLevels:
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.lower() in {"off", "false", "none", "no"}:
        return 0
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        try:
            return int(parts[0])
        except ValueError:
            return (float(parts[0]),)
    return tuple(float(part) for part in parts)


def save_case_snapshot(
    path: Path,
    title: str,
    rho: np.ndarray,
    phi: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    walkable: np.ndarray,
    rho_max: float,
    panel_title: str = "Density and direction",
    density_contour_levels: DensityContourLevels = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(title)

    density = rho.copy()
    density[~walkable] = np.nan
    im0 = axes[0].imshow(
        density,
        origin="lower",
        cmap=DENSITY_CMAP,
        vmin=0.0,
        vmax=rho_max,
        interpolation=DENSITY_INTERPOLATION,
    )
    draw_density_contours(axes[0], density, density_contour_levels)
    axes[0].set_title("Density")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        density,
        origin="lower",
        cmap=DENSITY_CMAP,
        vmin=0.0,
        vmax=rho_max,
        interpolation=DENSITY_INTERPOLATION,
    )
    draw_density_contours(axes[1], density, density_contour_levels)
    axes[1].set_title(panel_title)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    step = 4
    ys, xs = np.mgrid[0:rho.shape[0]:step, 0:rho.shape[1]:step]
    ux_d = ux[0:rho.shape[0]:step, 0:rho.shape[1]:step]
    uy_d = uy[0:rho.shape[0]:step, 0:rho.shape[1]:step]
    walkable_d = walkable[0:rho.shape[0]:step, 0:rho.shape[1]:step]
    axes[1].quiver(xs[walkable_d], ys[walkable_d], ux_d[walkable_d], uy_d[walkable_d], color="white", scale=20)

    oy, ox = np.where(~walkable)
    for ax in axes:
        ax.scatter(ox, oy, s=2, c="black", marker="s", linewidths=0)
        ax.set_xlim(0, rho.shape[1] - 1)
        ax.set_ylim(0, rho.shape[0] - 1)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_timeseries_plot(path: Path, title: str, stats: CaseStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle(title)

    axes[0, 0].plot(stats.times, stats.mean_density, label="mean density")
    axes[0, 0].plot(stats.times, stats.peak_density, label="peak density")
    axes[0, 0].legend()
    axes[0, 0].set_title("Density")

    axes[0, 1].plot(stats.times, stats.sink_cumulative, color="tab:green")
    axes[0, 1].set_title("Sink cumulative throughput")

    for name, series in stats.channel_density.items():
        axes[1, 0].plot(stats.times, series, label=name)
    axes[1, 0].legend()
    axes[1, 0].set_title("Channel mean density")

    axes[1, 1].plot(stats.times, stats.velocity_discontinuity, label="velocity discontinuity")
    axes[1, 1].plot(stats.times, stats.density_gradient_intensity, label="density gradient")
    axes[1, 1].legend()
    axes[1, 1].set_title("Stability metrics")

    for ax in axes.flat:
        ax.set_xlabel("time")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_comparison_plot(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(item["case_id"]) for item in summaries]
    sink = [float(item["final_sink_cumulative"]) for item in summaries]
    peak = [float(item["peak_density_max"]) for item in summaries]
    vel_jump = [float(item["velocity_discontinuity_avg"]) for item in summaries]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
    ax.bar(x - width, sink, width=width, label="sink cumulative")
    ax.bar(x, peak, width=width, label="peak density")
    ax.bar(x + width, vel_jump, width=width, label="velocity discontinuity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Case comparison")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
