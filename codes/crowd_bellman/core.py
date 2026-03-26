from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Mapping

import numpy as np


GroupKey = tuple[int, int]


@dataclass(frozen=True)
class DirectionLibrary:
    """Discrete control directions used by Bellman/HJB updates."""

    names: tuple[str, ...]
    bits: np.ndarray
    dy: np.ndarray
    dx: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    step: np.ndarray
    all_mask: np.uint16


@dataclass(frozen=True)
class TransitionRule:
    """Fixed-probability splitting rule for stage/route transitions."""

    source: GroupKey
    kappa: float
    decision_mask: np.ndarray
    targets: Mapping[GroupKey, float]


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
    """Shared speed-density relation f(rho)."""

    speed = vmax * (1.0 - rho / rho_max)
    return np.clip(speed, 0.0, vmax)


def default_allowed_mask(walkable: np.ndarray) -> np.ndarray:
    """Allow all discrete directions on walkable cells."""

    mask = np.zeros(walkable.shape, dtype=np.uint16)
    mask[walkable] = DIRECTIONS.all_mask
    return mask


def single_direction_mask(walkable: np.ndarray, bit: int) -> np.ndarray:
    """Allow a single discrete direction on walkable cells."""

    mask = np.zeros(walkable.shape, dtype=np.uint16)
    mask[walkable] = np.uint16(bit)
    return mask


def tensor_from_tau(
    tau_x: np.ndarray,
    tau_y: np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build M = alpha*tau*tau^T + beta*n*n^T from a tangent field tau."""

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
    """Precompute geometric step costs h/sqrt(u^T M u) for each control."""

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
    """Solve the discrete Bellman equation with Dijkstra-like label setting."""

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
    """Recover argmin controls from a solved Bellman value function."""

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


def compute_total_density(rho_by_group: Mapping[GroupKey, np.ndarray]) -> np.ndarray:
    """Compute rho_tot = sum_{s,r} rho_{s,r}."""

    iterator = iter(rho_by_group.values())
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("rho_by_group must contain at least one group") from exc

    rho_tot = np.array(first, copy=True)
    for rho in iterator:
        rho_tot += rho
    return rho_tot


def compute_cfl_dt(
    speed: np.ndarray,
    m11: np.ndarray,
    m22: np.ndarray,
    dx: float,
    cfl: float,
    dt_cap: float,
) -> float:
    """Legacy single-group CFL helper."""

    local_bound = speed * (np.sqrt(np.maximum(m11, 0.0)) + np.sqrt(np.maximum(m22, 0.0))) / max(dx, 1.0e-12)
    vmax = float(np.max(local_bound))
    if vmax <= 1.0e-12:
        return dt_cap
    return min(dt_cap, cfl / vmax)


def compute_cfl_dt_multigroup(
    speed: np.ndarray,
    m11_by_group: Mapping[GroupKey, np.ndarray],
    m22_by_group: Mapping[GroupKey, np.ndarray],
    dx: float,
    cfl: float,
    dt_cap: float,
    transition_out_rate_by_group: Mapping[GroupKey, np.ndarray] | None = None,
) -> float:
    """CFL bound for multigroup advection plus explicit transition sink terms."""

    inv_dx = 1.0 / max(dx, 1.0e-12)
    vmax_transport = 0.0
    for key, m11 in m11_by_group.items():
        m22 = m22_by_group[key]
        local = speed * (np.sqrt(np.maximum(m11, 0.0)) + np.sqrt(np.maximum(m22, 0.0))) * inv_dx
        vmax_transport = max(vmax_transport, float(np.max(local)))

    if vmax_transport <= 1.0e-12:
        dt_transport = dt_cap
    else:
        dt_transport = cfl / vmax_transport

    dt_transition = dt_cap
    if transition_out_rate_by_group:
        rate_max = 0.0
        for rate in transition_out_rate_by_group.values():
            rate_max = max(rate_max, float(np.max(np.maximum(rate, 0.0))))
        if rate_max > 1.0e-12:
            dt_transition = 1.0 / rate_max

    return min(dt_cap, dt_transport, dt_transition)


def compute_face_fluxes(rho: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-order upwind face fluxes for explicit conservation update."""

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
    """Advance one density field with explicit conservative advection and sink mask."""

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


def apply_fixed_probability_splitting(
    rho_by_group: dict[GroupKey, np.ndarray],
    transitions: tuple[TransitionRule, ...],
    dt: float,
    walkable: np.ndarray,
) -> dict[GroupKey, np.ndarray]:
    """Apply Q_{(s,r)->(s+1,q)} = p * kappa * chi_G * rho after advection."""

    deltas: dict[GroupKey, np.ndarray] = {
        key: np.zeros_like(rho) for key, rho in rho_by_group.items()
    }

    for rule in transitions:
        if rule.source not in rho_by_group:
            continue

        source_rho = rho_by_group[rule.source]
        probs = np.array(list(rule.targets.values()), dtype=float)
        prob_sum = float(np.sum(probs))
        if prob_sum <= 1.0e-12:
            continue
        if not np.isclose(prob_sum, 1.0, atol=1.0e-6):
            probs = probs / prob_sum

        mask = rule.decision_mask & walkable
        transferable = dt * max(rule.kappa, 0.0) * source_rho * mask.astype(float)
        transferable = np.minimum(transferable, source_rho)

        deltas[rule.source] -= transferable
        for target, prob in zip(rule.targets.keys(), probs):
            if target not in deltas:
                deltas[target] = np.zeros_like(source_rho)
                rho_by_group[target] = np.zeros_like(source_rho)
            deltas[target] += prob * transferable

    updated: dict[GroupKey, np.ndarray] = {}
    for key, rho in rho_by_group.items():
        delta = deltas.get(key)
        if delta is None:
            updated[key] = np.clip(rho, 0.0, None)
            continue
        result = np.clip(rho + delta, 0.0, None)
        result[~walkable] = 0.0
        updated[key] = result
    return updated


def build_transition_out_rate_maps(
    transitions: tuple[TransitionRule, ...],
    shape: tuple[int, int],
) -> dict[GroupKey, np.ndarray]:
    """Build per-group out-rate fields lambda = kappa * chi_G for dt restriction."""

    rates: dict[GroupKey, np.ndarray] = {}
    for rule in transitions:
        out = rates.setdefault(rule.source, np.zeros(shape, dtype=float))
        out += max(rule.kappa, 0.0) * rule.decision_mask.astype(float)
    return rates
