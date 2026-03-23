from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .metrics import CaseStats


def save_case_snapshot(
    path: Path,
    title: str,
    rho: np.ndarray,
    phi: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    walkable: np.ndarray,
    rho_max: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(title)

    density = rho.copy()
    density[~walkable] = np.nan
    potential = phi.copy()
    potential[~walkable] = np.nan

    im0 = axes[0].imshow(density, origin="lower", cmap="viridis", vmin=0.0, vmax=rho_max)
    axes[0].set_title("Density")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(potential, origin="lower", cmap="magma")
    axes[1].set_title("Potential and direction")
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
