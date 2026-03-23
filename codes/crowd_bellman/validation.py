from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .core import DIRECTIONS, precompute_step_factors, single_direction_mask, solve_bellman


@dataclass(frozen=True)
class ValidationConfig:
    nx: int = 80
    ny: int = 11
    dx: float = 0.5
    vmax: float = 1.5
    rho: float = 1.0
    rho_max: float = 5.0
    f_eps: float = 1.0e-6


def run_validation(output_root: Path) -> dict[str, float]:
    cfg = ValidationConfig()
    walkable = np.ones((cfg.ny, cfg.nx), dtype=bool)
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    exit_mask = np.zeros_like(walkable, dtype=bool)
    exit_mask[1:-1, cfg.nx - 2] = True

    allowed_mask = single_direction_mask(walkable, int(DIRECTIONS.bits[0]))

    speed_value = cfg.vmax * (1.0 - cfg.rho / cfg.rho_max)
    speed = np.full(walkable.shape, speed_value, dtype=float)
    m11 = np.ones_like(speed)
    m12 = np.zeros_like(speed)
    m22 = np.ones_like(speed)
    step_factor = precompute_step_factors(walkable, cfg.dx, m11, m12, m22)

    phi = solve_bellman(
        walkable=walkable,
        exit_mask=exit_mask,
        allowed_mask=allowed_mask,
        speed=speed,
        step_factor=step_factor,
        f_eps=cfg.f_eps,
    )

    x_exit = cfg.nx - 2
    exact = np.full_like(phi, np.nan, dtype=float)
    for x in range(1, x_exit + 1):
        exact[:, x] = (x_exit - x) * cfg.dx / speed_value
    exact[~walkable] = np.nan

    residual = np.full_like(phi, np.nan, dtype=float)
    forward = phi[:, 2 : x_exit + 1]
    current = phi[:, 1:x_exit]
    finite_mask = np.isfinite(forward) & np.isfinite(current)
    residual_slice = np.full_like(current, np.nan, dtype=float)
    residual_slice[finite_mask] = (forward[finite_mask] - current[finite_mask]) / cfg.dx + 1.0 / speed_value
    residual[:, 1:x_exit] = residual_slice
    residual_slice = residual[:, 1:x_exit]
    residual_valid = walkable[:, 1:x_exit]

    phi_error = phi[:, 1 : x_exit + 1] - exact[:, 1 : x_exit + 1]
    result = {
        "speed": float(speed_value),
        "max_abs_phi_error": float(np.nanmax(np.abs(phi_error))),
        "mean_abs_phi_error": float(np.nanmean(np.abs(phi_error))),
        "max_abs_pde_residual": float(np.nanmax(np.abs(residual_slice[residual_valid]))),
        "mean_abs_pde_residual": float(np.nanmean(np.abs(residual_slice[residual_valid]))),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    center_y = cfg.ny // 2
    xs = np.arange(cfg.nx)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(xs[1 : x_exit + 1], phi[center_y, 1 : x_exit + 1], label="Bellman")
    axes[0].plot(xs[1 : x_exit + 1], exact[center_y, 1 : x_exit + 1], "--", label="exact")
    axes[0].set_title("Potential profile")
    axes[0].legend()

    axes[1].plot(xs[1:x_exit], residual[center_y, 1:x_exit])
    axes[1].set_title("Residual of tau · grad(phi) + 1/f")

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("grid x")
    fig.tight_layout()
    fig.savefig(output_root / "one_way_validation.png")
    plt.close(fig)

    with (output_root / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def run_validation_cli() -> None:
    parser = argparse.ArgumentParser(description="Verify the strict one-way HJB reduction.")
    parser.add_argument("--output-root", default="codes/results/unidirectional_validation")
    args = parser.parse_args()
    run_validation(Path(args.output_root))
