from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np


@dataclass(frozen=True)
class DirectionLibrary:
    names: tuple[str, ...]
    bits: np.ndarray
    dy: np.ndarray
    dx: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    step: np.ndarray
    all_mask: np.uint16


def build_eight_directions() -> DirectionLibrary:
    names = ("E", "W", "N", "S", "NE", "NW", "SE", "SW")
    offsets = np.array(
        [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ],
        dtype=int,
    )
    bits = np.array([1 << idx for idx in range(len(names))], dtype=np.uint16)
    step = np.sqrt(np.sum(offsets.astype(float) ** 2, axis=1))
    ux = offsets[:, 1] / step
    uy = offsets[:, 0] / step
    return DirectionLibrary(
        names=names,
        bits=bits,
        dy=offsets[:, 0],
        dx=offsets[:, 1],
        ux=ux,
        uy=uy,
        step=step,
        all_mask=np.uint16(np.sum(bits, dtype=np.uint16)),
    )


DIRECTIONS = build_eight_directions()


def greenshields_speed(rho: np.ndarray, vmax: float, rho_max: float) -> np.ndarray:
    speed = vmax * (1.0 - rho / rho_max)
    return np.clip(speed, 0.0, vmax)


def default_allowed_mask(walkable: np.ndarray) -> np.ndarray:
    mask = np.zeros(walkable.shape, dtype=np.uint16)
    mask[walkable] = DIRECTIONS.all_mask
    return mask


def single_direction_mask(walkable: np.ndarray, bit: int) -> np.ndarray:
    mask = np.zeros(walkable.shape, dtype=np.uint16)
    mask[walkable] = np.uint16(bit)
    return mask


def tensor_from_tau(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_x = -tau_y
    n_y = tau_x
    m11 = alpha * tau_x * tau_x + beta * n_x * n_x
    m12 = alpha * tau_x * tau_y + beta * n_x * n_y
    m22 = alpha * tau_y * tau_y + beta * n_y * n_y
    return m11, m12, m22


def precompute_step_factors(
    walkable: np.ndarray,
    dx: float,
    m11: np.ndarray,
    m12: np.ndarray,
    m22: np.ndarray,
) -> np.ndarray:
    step_factor = np.full((walkable.shape[0], walkable.shape[1], len(DIRECTIONS.names)), np.inf, dtype=float)
    for k in range(len(DIRECTIONS.names)):
        ut_mu = (
            DIRECTIONS.ux[k] * DIRECTIONS.ux[k] * m11
            + 2.0 * DIRECTIONS.ux[k] * DIRECTIONS.uy[k] * m12
            + DIRECTIONS.uy[k] * DIRECTIONS.uy[k] * m22
        )
        denom = np.sqrt(np.maximum(ut_mu, 1.0e-12))
        step_factor[:, :, k] = dx * DIRECTIONS.step[k] / denom
    step_factor[~walkable] = np.inf
    return step_factor


def solve_bellman(
    walkable: np.ndarray,
    exit_mask: np.ndarray,
    allowed_mask: np.ndarray,
    speed: np.ndarray,
    step_factor: np.ndarray,
    f_eps: float,
) -> np.ndarray:
    ny, nx = walkable.shape
    phi = np.full((ny, nx), np.inf, dtype=float)
    speed_safe = np.maximum(speed, f_eps)
    queue: list[tuple[float, int, int]] = []

    for y, x in np.argwhere(exit_mask & walkable):
        phi[y, x] = 0.0
        heappush(queue, (0.0, int(y), int(x)))

    while queue:
        value, y, x = heappop(queue)
        if value > phi[y, x]:
            continue

        for k in range(len(DIRECTIONS.names)):
            py = y - int(DIRECTIONS.dy[k])
            px = x - int(DIRECTIONS.dx[k])
            if py < 0 or py >= ny or px < 0 or px >= nx:
                continue
            if not walkable[py, px]:
                continue
            if (allowed_mask[py, px] & DIRECTIONS.bits[k]) == 0:
                continue

            candidate = value + step_factor[py, px, k] / speed_safe[py, px]
            if candidate + 1.0e-12 < phi[py, px]:
                phi[py, px] = candidate
                heappush(queue, (candidate, py, px))

    finite = np.isfinite(phi) & walkable
    if np.any(finite):
        phi[finite] -= np.min(phi[finite])
    return phi


def recover_optimal_direction(
    walkable: np.ndarray,
    exit_mask: np.ndarray,
    allowed_mask: np.ndarray,
    speed: np.ndarray,
    step_factor: np.ndarray,
    phi: np.ndarray,
    f_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = walkable.shape
    ux = np.zeros((ny, nx), dtype=float)
    uy = np.zeros((ny, nx), dtype=float)
    speed_safe = np.maximum(speed, f_eps)

    for y in range(ny):
        for x in range(nx):
            if (not walkable[y, x]) or exit_mask[y, x] or (not np.isfinite(phi[y, x])):
                continue

            best_value = np.inf
            best_ux = 0.0
            best_uy = 0.0

            for k in range(len(DIRECTIONS.names)):
                if (allowed_mask[y, x] & DIRECTIONS.bits[k]) == 0:
                    continue
                nyy = y + int(DIRECTIONS.dy[k])
                nxx = x + int(DIRECTIONS.dx[k])
                if nyy < 0 or nyy >= ny or nxx < 0 or nxx >= nx:
                    continue
                if (not walkable[nyy, nxx]) or (not np.isfinite(phi[nyy, nxx])):
                    continue
                candidate = phi[nyy, nxx] + step_factor[y, x, k] / speed_safe[y, x]
                if candidate < best_value:
                    best_value = candidate
                    best_ux = DIRECTIONS.ux[k]
                    best_uy = DIRECTIONS.uy[k]

            ux[y, x] = best_ux
            uy[y, x] = best_uy

    return ux, uy


def compute_cfl_dt(
    speed: np.ndarray,
    m11: np.ndarray,
    m22: np.ndarray,
    dx: float,
    cfl: float,
    dt_cap: float,
) -> float:
    local_bound = speed * (np.sqrt(np.maximum(m11, 0.0)) + np.sqrt(np.maximum(m22, 0.0))) / max(dx, 1.0e-12)
    vmax = float(np.max(local_bound))
    if vmax <= 1.0e-12:
        return dt_cap
    return min(dt_cap, cfl / vmax)


def compute_face_fluxes(rho: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vx_face = 0.5 * (vx[:, :-1] + vx[:, 1:])
    vy_face = 0.5 * (vy[:-1, :] + vy[1:, :])

    rho_x = np.where(vx_face >= 0.0, rho[:, :-1], rho[:, 1:])
    rho_y = np.where(vy_face >= 0.0, rho[:-1, :], rho[1:, :])
    fx = vx_face * rho_x
    fy = vy_face * rho_y
    return fx, fy


def update_density(
    rho: np.ndarray,
    walkable: np.ndarray,
    exit_mask: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fx, fy = compute_face_fluxes(rho, vx, vy)

    div_x = np.zeros_like(rho)
    div_y = np.zeros_like(rho)
    div_x[:, 1:-1] = (fx[:, 1:] - fx[:, :-1]) / dx
    div_y[1:-1, :] = (fy[1:, :] - fy[:-1, :]) / dx

    updated = rho - dt * (div_x + div_y)
    updated[~walkable] = 0.0
    updated = np.clip(updated, 0.0, None)

    sink_mass = float(np.sum(updated[exit_mask]) * dx * dx)
    updated[exit_mask] = 0.0
    return updated, fx, fy, sink_mass
