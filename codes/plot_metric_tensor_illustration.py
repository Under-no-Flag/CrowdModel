"""Illustration of isotropic vs anisotropic metric tensor effects on optimal paths.

Generates a paper figure comparing how M(x) = alpha * tau * tau^T + beta * n * n^T
changes equipotential lines and optimal direction fields within a channel.

In the discrete Bellman update used here, the directional step cost is proportional to
1 / sqrt(u^T M u). Thus M acts as a co-metric / mobility tensor: alpha >> beta makes
motion along tau cheaper than cross-channel motion along n.

Left:  Isotropic M = I -> circular local unit-cost sets, radial direction field
Right: Anisotropic M -> channel-aligned unit-cost sets and flow
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch

from crowd_bellman.core import (
    default_allowed_mask,
    precompute_step_factors,
    recover_optimal_direction,
    solve_bellman,
    tensor_from_tau,
)
from crowd_bellman.plotting import DENSITY_CMAP, DENSITY_INTERPOLATION

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "metric_tensor_illustration"

# ---------------------------------------------------------------------------
# geometry constants
# ---------------------------------------------------------------------------

NY, NX = 80, 120
WALL_TOP, WALL_BOTTOM = 26, 54
EXIT_CY = (WALL_TOP + WALL_BOTTOM) // 2
EXIT_HALF_H = 4
ALPHA = 10.0
BETA = 1
DX = 1.0
F_EPS = 1e-6
SPEED_CONST = 1.0


def _build_geometry(open_domain: bool) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    walkable = np.ones((NY, NX), dtype=bool)
    if not open_domain:
        walkable[:WALL_TOP, :] = False
        walkable[WALL_BOTTOM:, :] = False

    exit_mask = np.zeros((NY, NX), dtype=bool)
    if open_domain:
        exit_mask[EXIT_CY, NX - 2] = True
        exit_center = (NX - 2, EXIT_CY)
    else:
        ey0 = EXIT_CY - EXIT_HALF_H
        ey1 = EXIT_CY + EXIT_HALF_H
        exit_mask[ey0:ey1, -1] = True
        exit_center = (NX - 1, EXIT_CY)

    exit_mask &= walkable
    return walkable, exit_mask, exit_center


# ---------------------------------------------------------------------------
# computation helpers
# ---------------------------------------------------------------------------


def _solve_case(
    walkable: np.ndarray,
    exit_mask: np.ndarray,
    m11: np.ndarray,
    m12: np.ndarray,
    m22: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.where(walkable, SPEED_CONST, 0.0)
    allowed = default_allowed_mask(walkable)
    sf = precompute_step_factors(walkable, DX, m11, m12, m22)
    phi = solve_bellman(walkable, exit_mask, allowed, speed, sf, F_EPS)
    ux, uy = recover_optimal_direction(walkable, exit_mask, allowed, speed, sf, phi, F_EPS)
    return phi, ux, uy


def _isotropic_tensor(walkable: np.ndarray):
    s = walkable.shape
    return np.ones(s), np.zeros(s), np.ones(s)


def _anisotropic_tensor(walkable: np.ndarray):
    channel = np.zeros((NY, NX), dtype=bool)
    channel[WALL_TOP:WALL_BOTTOM, :] = True
    channel &= walkable

    tau_x = np.ones((NY, NX))
    tau_y = np.zeros((NY, NX))
    m11c, m12c, m22c = tensor_from_tau(tau_x, tau_y, float(ALPHA), float(BETA))

    m11 = np.where(channel, m11c, 1.0)
    m12 = np.where(channel, m12c, 0.0)
    m22 = np.where(channel, m22c, 1.0)
    return m11, m12, m22


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------


def _add_unit_cost_set(ax, cx, cy, alpha, beta, scale=4.0):
    """Draw the local unit-cost / velocity ellipse {v : v^T M^{-1} v = 1}.

    Semi-axes are proportional to sqrt(alpha) along tau and sqrt(beta) along n, matching
    the Bellman step cost dx / sqrt(u^T M u). This is the visually relevant set for
    explaining why alpha >> beta favors channel-aligned motion.
    """
    a = np.sqrt(alpha) * scale
    b = np.sqrt(beta) * scale

    # outline
    ellipse = Ellipse(
        (cx, cy), 2 * a, 2 * b, angle=0.0,
        facecolor="none", edgecolor="#d62728", linewidth=1.8, zorder=10,
    )
    ax.add_patch(ellipse)

    # tau axis: longer reachable distance / lower cost
    ax.add_patch(FancyArrowPatch(
        (cx, cy), (cx + a, cy),
        arrowstyle="->", mutation_scale=12, color="#d62728", linewidth=1.8, zorder=11,
    ))

    # n axis: shorter reachable distance / higher cost
    ax.add_patch(FancyArrowPatch(
        (cx, cy), (cx, cy + b),
        arrowstyle="->", mutation_scale=12, color="#1f77b4", linewidth=1.8, zorder=11,
    ))

    # centre dot
    ax.plot(cx, cy, "o", color="#d62728", markersize=3.5, zorder=11)


def _draw_panel(
    fig,
    ax,
    title: str,
    phi: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    walkable: np.ndarray,
    exit_mask: np.ndarray,
    *,
    annotate_tau_n: bool = False,
    unit_cost_alpha: float | None = None,
    unit_cost_beta: float | None = None,
    normal_anchor: tuple[float, float] | None = None,
    exit_center: tuple[float, float] | None = None,
    draw_channel_bounds: bool = False,
):
    phi_plot = phi.copy()
    phi_plot[~walkable] = np.nan

    # --- channel background shading (subtle) ---
    if annotate_tau_n:
        ax.axhspan(WALL_TOP - 0.5, WALL_BOTTOM - 0.5, xmin=0.0, xmax=1.0,
                    facecolor="#d62728", alpha=0.04, zorder=0)

    # --- phi heatmap ---
    im = ax.imshow(phi_plot, origin="lower", cmap=DENSITY_CMAP, alpha=0.55, interpolation=DENSITY_INTERPOLATION)

    # --- phi contours (equipotential / isochrone lines) ---
    vmax = np.nanmax(phi_plot)
    if np.isfinite(vmax) and vmax > 1e-6:
        levels = np.linspace(0, vmax, 16)
        ax.contour(phi_plot, levels=levels, colors="#8B0000", linewidths=0.55, alpha=0.65, linestyles="--")

    # --- optimal direction field (quiver) ---
    qstep = 4
    ys, xs = np.mgrid[0:NY:qstep, 0:NX:qstep]
    uxd = ux[::qstep, ::qstep]
    uyd = uy[::qstep, ::qstep]
    wd = walkable[::qstep, ::qstep]
    mag = np.hypot(uxd, uyd)
    valid = wd & (mag > 1e-8)
    ax.quiver(
        xs[valid], ys[valid], uxd[valid], uyd[valid],
        color="#1a1a1a", scale=18, width=0.0028,
        headwidth=4.5, headlength=5.5, headaxislength=4.5,
    )

    # --- obstacles ---
    oy, ox = np.where(~walkable)
    if ox.size:
        ax.scatter(ox, oy, s=1.0, c="#d9d9d9", marker="s", linewidths=0, zorder=1)

    # --- exit marker ---
    ey, ex = np.where(exit_mask)
    ax.scatter(ex, ey, s=8, c="#2ca02c", marker="s", linewidths=0, zorder=9)

    # --- channel boundary dashed lines ---
    if draw_channel_bounds:
        for yb in (WALL_TOP - 0.5, WALL_BOTTOM - 0.5):
            ax.axhline(y=yb, color="black", linewidth=1.0, linestyle="--", alpha=0.45)

    # --- local unit-cost / velocity sets ---
    if unit_cost_alpha is not None and unit_cost_beta is not None:
        positions = [(25, EXIT_CY), (55, EXIT_CY), (85, EXIT_CY)]
        for cx, cy in positions:
            _add_unit_cost_set(ax, cx, cy, unit_cost_alpha, unit_cost_beta, scale=3.8)

    # --- tau / n annotations ---
    if annotate_tau_n:
        mid_y = EXIT_CY
        # tau label (horizontal arrow + text)
        ax.annotate(
            r"$\boldsymbol{\tau}$",
            xy=(70, mid_y),
            xytext=(80, mid_y + 16),
            fontsize=12, color="#d62728", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.8),
        )
        # n label (vertical arrow + text)
        ax.annotate(
            r"$\mathbf{n}$",
            xy=(40, mid_y + 10),
            xytext=(40, mid_y + 24),
            fontsize=12, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
        )
        # parameter text box
        ax.text(
            0.97, 0.03,
            r"$\alpha = 10.0,\;\beta = 0.2$" "\n"
            r"$c_{\tau}=1/\sqrt{\alpha}=0.32$" "\n"
            r"$c_{n}=1/\sqrt{\beta}=2.24$",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85),
        )

    # --- normal direction annotation (schematic) ---
    if normal_anchor is not None and exit_center is not None:
        ax_x, ax_y = normal_anchor
        ex, ey = exit_center
        vec_x = ex - ax_x
        vec_y = ey - ax_y
        norm = np.hypot(vec_x, vec_y)
        if norm > 1e-6:
            vec_x = vec_x / norm * 10.0
            vec_y = vec_y / norm * 10.0
        ax.annotate(
            r"$-\nabla\phi$",
            xy=(ax_x + vec_x, ax_y + vec_y),
            xytext=(ax_x - 10, ax_y + 8),
            fontsize=12, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
        )

    # --- axis labels ---
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.set_xlim(0, NX - 1)
    ax.set_ylim(0, NY - 1)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    ax.tick_params(labelsize=8)

    # --- colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
    cbar.set_label(r"$\phi(x)$  (minimal travel cost to exit)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    walkable_i, exit_mask_i, exit_center_i = _build_geometry(open_domain=True)
    walkable_a, exit_mask_a, _ = _build_geometry(open_domain=False)

    # --- solve both cases ---
    m11_i, m12_i, m22_i = _isotropic_tensor(walkable_i)
    phi_i, ux_i, uy_i = _solve_case(walkable_i, exit_mask_i, m11_i, m12_i, m22_i)

    m11_a, m12_a, m22_a = _anisotropic_tensor(walkable_a)
    phi_a, ux_a, uy_a = _solve_case(walkable_a, exit_mask_a, m11_a, m12_a, m22_a)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=200)

    _draw_panel(
        fig, axes[0],
        r"(a) Isotropic (open domain): $\mathbf{M} = \mathbf{I}$",
        phi_i, ux_i, uy_i, walkable_i, exit_mask_i,
        unit_cost_alpha=1.0, unit_cost_beta=1.0,
        normal_anchor=(28, EXIT_CY + 10),
        exit_center=exit_center_i,
    )

    _draw_panel(
        fig, axes[1],
        r"(b) Anisotropic: $\mathbf{M} = \alpha\,\boldsymbol{\tau}\boldsymbol{\tau}^{\top}"
        r" + \beta\,\mathbf{n}\mathbf{n}^{\top}$",
        phi_a, ux_a, uy_a, walkable_a, exit_mask_a,
        annotate_tau_n=True,
        unit_cost_alpha=ALPHA, unit_cost_beta=BETA,
        draw_channel_bounds=True,
    )

    # --- shared legend ---
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ca02c", markersize=8,
               label="Exit $\\Gamma_E$"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=6,
               label=r"Unit-cost set $\{\mathbf{v}: \mathbf{v}^{\top}\mathbf{M}^{-1}\mathbf{v}=1\}$"),
        Line2D([0], [0], color="#d62728", linewidth=1.8,
               label=r"$\boldsymbol{\tau}$ direction (along channel)"),
        Line2D([0], [0], color="#1f77b4", linewidth=1.8,
               label=r"$\mathbf{n}$ direction (cross-channel)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Effect of Metric Tensor $\\mathbf{M}(x)$ on Optimal-Path Geometry in a Channel",
        fontsize=14, fontweight="bold", y=1.005,
    )

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.97))

    out_path = OUTPUT_DIR / "metric_tensor_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
