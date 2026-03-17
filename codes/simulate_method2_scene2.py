import os
from dataclasses import dataclass

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


@dataclass
class SimConfig:
    nx: int = 100
    ny: int = 70
    dx: float = 1.0
    dt: float = 5.0
    steps: int = 200

    vmax: float = 10.0
    rho_max: float = 1.0

    # --- Hughes potential recompute frequency ---
    recompute_phi_every: int = 1   # 每几步重算一次phi（1=每步）

    # Initial density for the crowd (spawn from left)
    rho_init: float = 0.8

    # --- Metric tensor M(x) for channel guidance ---
    # M = diag(M11, M22). In channels set M22 small -> penalize transverse motion (in primal metric)
    M11_default: float = 1.0
    M22_default: float = 1.0
    M22_in_channel: float = 1.0 / 25.0  # 横向权重更小 => 横穿更贵(等价于Gyy更大)

    # Eikonal solver
    eikonal_iters: int = 800
    eikonal_tol: float = 1e-6
    f_eps: float = 1e-3  # 避免 1/f 发散

    # Output
    out_dir: str = os.path.join("codes", "results", "scene1_metricA")
    save_every: int = 1


def greenshields_speed(rho: np.ndarray, vmax: float, rho_max: float) -> np.ndarray:
    f = vmax * (1.0 - rho / rho_max)
    return np.clip(f, 0.0, vmax)


def build_scene(cfg: SimConfig):
    """Builds a simple approximation of scenes_1.png geometry on a grid.

    Returns:
      walkable: bool (ny, nx) True for free space
      spawn_mask: bool (ny, nx) left region initialization
      target_mask: bool (ny, nx) right boundary as destination
      channel_mask: bool (ny, nx) area of the 3 vertical channels
      channel_masks: list of bool masks (top/mid/bot)
    """
    ny, nx = cfg.ny, cfg.nx
    walkable = np.ones((ny, nx), dtype=bool)

    # Outer boundaries are walls
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    # Create a central vertical "barrier" with three door-like channels.
    wall_x0 = int(nx * 0.52)
    wall_w = max(2, int(nx * 0.03))
    wall_x1 = min(nx - 2, wall_x0 + wall_w)

    walkable[:, wall_x0:wall_x1] = False

    # Three channels (openings) through the wall (top/middle/bottom)
    ch_h = max(5, int(ny * 0.10))
    gap = int(ny * 0.07)
    y_centers = [int(ny * 0.25), int(ny * 0.50), int(ny * 0.75)]

    channel_masks = []
    channel_mask = np.zeros((ny, nx), dtype=bool)

    for yc in y_centers:
        y0 = max(1, yc - ch_h // 2)
        y1 = min(ny - 1, yc + ch_h // 2)

        walkable[y0:y1, wall_x0:wall_x1] = True

        m = np.zeros((ny, nx), dtype=bool)
        m[y0:y1, wall_x0:wall_x1] = True
        channel_masks.append(m)
        channel_mask |= m

    # Add some horizontal "teeth" walls to mimic comb obstacles on right side
    tooth_len = int(nx * 0.18)
    tooth_h = max(2, int(ny * 0.02))
    tooth_x0 = wall_x1
    tooth_x1 = min(nx - 2, tooth_x0 + tooth_len)
    for yc in y_centers:
        for dy in (-gap, gap):
            y0 = np.clip(yc + dy - tooth_h // 2, 1, ny - 2)
            y1 = np.clip(yc + dy + tooth_h // 2, 1, ny - 2)
            walkable[y0:y1, tooth_x0:tooth_x1] = False

    # Spawn on left open region (excluding borders)
    spawn_mask = np.zeros((ny, nx), dtype=bool)
    spawn_mask[1:ny - 1, 1:int(nx * 0.18)] = True
    spawn_mask &= walkable

    # Target: right boundary (a vertical segment)
    target_mask = np.zeros((ny, nx), dtype=bool)
    target_mask[1:ny - 1, nx - 2] = True
    target_mask &= walkable

    return walkable, spawn_mask, target_mask, channel_mask, channel_masks


def build_metric_M(cfg: SimConfig, walkable: np.ndarray, channel_mask: np.ndarray):
    """Construct diagonal metric tensor M(x)=diag(M11,M22).
    Channel guidance: in channel, reduce M22 -> suppress y-component in u and increase transverse cost (in primal metric).
    """
    ny, nx = walkable.shape
    M11 = np.full((ny, nx), cfg.M11_default, dtype=float)
    M22 = np.full((ny, nx), cfg.M22_default, dtype=float)

    # Apply anisotropy only inside the channel openings
    M22[channel_mask] = cfg.M22_in_channel

    # Obstacles: values won't be used, but keep finite
    M11[~walkable] = 1.0
    M22[~walkable] = 1.0
    return M11, M22


def eikonal_local_update_diag(
    Tx: float, Ty: float, s: float, M11: float, M22: float, dx: float
) -> float:
    """
    Solve local Godunov update for diagonal anisotropic eikonal:
      sqrt( M11 * phi_x^2 + M22 * phi_y^2 ) = s
    with upwind approx:
      phi_x ≈ (T - Tx)/dx,  phi_y ≈ (T - Ty)/dx,
      Tx=min(left,right), Ty=min(down,up)

    Returns updated T >= min(Tx,Ty) and monotone.
    """
    inf = 1e30
    has_x = np.isfinite(Tx) and Tx < inf
    has_y = np.isfinite(Ty) and Ty < inf

    if (not has_x) and (not has_y):
        return np.inf

    # One-sided (1D) updates if only one direction is available
    if has_x and (not has_y):
        return Tx + dx * s / np.sqrt(M11)
    if has_y and (not has_x):
        return Ty + dx * s / np.sqrt(M22)

    # Two-sided quadratic update
    a = M11
    b = M22
    A = a + b
    B = -2.0 * (a * Tx + b * Ty)
    C = a * Tx * Tx + b * Ty * Ty - (dx * s) * (dx * s)

    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        disc = 0.0
    t = (-B + np.sqrt(disc)) / (2.0 * A)

    # Monotonicity check; if violated, fallback to 1D min
    m = max(Tx, Ty)
    if t < m:
        tx = Tx + dx * s / np.sqrt(M11)
        ty = Ty + dx * s / np.sqrt(M22)
        t = min(tx, ty)

    return t


def compute_phi_hughes_aniso_diag(
    walkable: np.ndarray,
    target_mask: np.ndarray,
    f: np.ndarray,
    M11: np.ndarray,
    M22: np.ndarray,
    dx: float,
    max_iters: int,
    tol: float,
    f_eps: float,
) -> np.ndarray:
    """
    Solve anisotropic Hughes eikonal:
      sqrt(∇phi^T M ∇phi) = 1/f
    with diagonal M=diag(M11,M22) using Gauss-Seidel Godunov updates.
    """
    ny, nx = walkable.shape
    T = np.full((ny, nx), np.inf, dtype=float)
    T[~walkable] = np.inf
    T[target_mask] = 0.0

    # Initialize walkable non-target cells to large finite so relaxation can propagate
    init_mask = walkable & (~target_mask)
    T[init_mask] = 1e6

    # Precompute s = 1/f (with eps)
    f_safe = np.maximum(f, f_eps)
    s = 1.0 / f_safe
    s[target_mask] = 0.0

    for _ in range(max_iters):
        max_delta = 0.0

        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if (not walkable[y, x]) or target_mask[y, x]:
                    continue

                Tx = min(T[y, x - 1], T[y, x + 1])
                Ty = min(T[y - 1, x], T[y + 1, x])

                t_new = eikonal_local_update_diag(
                    Tx=Tx,
                    Ty=Ty,
                    s=s[y, x],
                    M11=M11[y, x],
                    M22=M22[y, x],
                    dx=dx,
                )

                if not np.isfinite(t_new):
                    continue

                if t_new < T[y, x]:
                    delta = T[y, x] - t_new
                    if delta > max_delta:
                        max_delta = delta
                    T[y, x] = t_new

        if max_delta < tol:
            break

    # shift for nicer visualization (doesn't affect gradients)
    finite = np.isfinite(T) & walkable
    if np.any(finite):
        T[finite] -= np.min(T[finite])
    return T


def smooth2d_gauss5(a: np.ndarray) -> np.ndarray:
    k = np.array([1, 4, 6, 4, 1], dtype=float)
    k /= k.sum()

    ap = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    tmp = (k[0]*ap[:, 0:-4] + k[1]*ap[:, 1:-3] + k[2]*ap[:, 2:-2] + k[3]*ap[:, 3:-1] + k[4]*ap[:, 4:])

    tp = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    out = (k[0]*tp[0:-4, :] + k[1]*tp[1:-3, :] + k[2]*tp[2:-2, :] + k[3]*tp[3:-1, :] + k[4]*tp[4:, :])
    return out


def plot_frame(rho, T, ux, uy, walkable, out_path, title, rho_max=1.0):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(title)

    rr = smooth2d_gauss5(rho)
    rr = rr.copy()
    rr[~walkable] = np.nan

    im0 = ax0.imshow(
        rr, origin="lower", cmap="viridis",
        vmin=0.0, vmax=rho_max,
        interpolation="bilinear"
    )
    ax0.set_title("Density ρ")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    oy, ox = np.where(~walkable)
    ax0.scatter(ox, oy, s=3, c="k", marker="s", linewidths=0)

    TT = T.copy()
    TT[~walkable] = np.nan
    im1 = ax1.imshow(
        TT, origin="lower", cmap="magma",
        interpolation="bilinear"
    )
    ax1.set_title("Potential φ and direction")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    step = 3
    ys, xs = np.mgrid[0:T.shape[0]:step, 0:T.shape[1]:step]
    uxs = ux[0:T.shape[0]:step, 0:T.shape[1]:step]
    uys = uy[0:T.shape[0]:step, 0:T.shape[1]:step]
    m = walkable[0:T.shape[0]:step, 0:T.shape[1]:step]
    ax1.quiver(xs[m], ys[m], uxs[m], uys[m], color="white", pivot="mid", scale=25)

    ax1.scatter(ox, oy, s=3, c="k", marker="s", linewidths=0)

    for ax in (ax0, ax1):
        ax.set_xlim(0, rho.shape[1]-1)
        ax.set_ylim(0, rho.shape[0]-1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def grad_central(T: np.ndarray, dx: float, valid: np.ndarray | None = None):
    gy = np.zeros_like(T)
    gx = np.zeros_like(T)

    if valid is None:
        gy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dx)
        gx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
        return gx, gy

    vy = valid[2:, :] & valid[:-2, :]
    gy_inner = np.zeros((T.shape[0] - 2, T.shape[1]), dtype=T.dtype)
    gy_inner[vy] = (T[2:, :][vy] - T[:-2, :][vy]) / (2.0 * dx)
    gy[1:-1, :] = gy_inner

    vx = valid[:, 2:] & valid[:, :-2]
    gx_inner = np.zeros((T.shape[0], T.shape[1] - 2), dtype=T.dtype)
    gx_inner[vx] = (T[:, 2:][vx] - T[:, :-2][vx]) / (2.0 * dx)
    gx[:, 1:-1] = gx_inner

    return gx, gy


def divergence_flux(rho: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float):
    vx_face = 0.5 * (vx + np.roll(vx, -1, axis=1))
    rho_R = np.roll(rho, -1, axis=1)

    Fx_p = np.maximum(vx_face, 0.0) * rho + np.minimum(vx_face, 0.0) * rho_R
    Fx_m = np.roll(Fx_p, 1, axis=1)
    dFx = (Fx_p - Fx_m) / dx
    dFx[:, 0] = 0.0
    dFx[:, -1] = 0.0

    vy_face = 0.5 * (vy + np.roll(vy, -1, axis=0))
    rho_U = np.roll(rho, -1, axis=0)

    Fy_p = np.maximum(vy_face, 0.0) * rho + np.minimum(vy_face, 0.0) * rho_U
    Fy_m = np.roll(Fy_p, 1, axis=0)
    dFy = (Fy_p - Fy_m) / dx
    dFy[0, :] = 0.0
    dFy[-1, :] = 0.0

    return dFx + dFy


def simulate_metric_routeA(cfg: SimConfig):
    walkable, spawn_mask, target_mask, channel_mask, channel_masks  = build_scene(cfg)

    # top_mask, mid_mask, bot_mask = channel_masks

    # # 1) 关闭中间通道（不可通行）
    # walkable[mid_mask] = False

    # # 2) 重新约束 spawn/target（避免在墙里）
    # spawn_mask &= walkable
    # target_mask &= walkable

    # # 3) 只有上下通道做“几何引导”用的 channel_mask
    # channel_mask = top_mask | bot_mask

    M11, M22 = build_metric_M(cfg, walkable, channel_mask)

    rho = np.zeros((cfg.ny, cfg.nx), dtype=float)
    rho[spawn_mask] = cfg.rho_init

    # initial phi/dir
    f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
    phi = compute_phi_hughes_aniso_diag(
        walkable=walkable,
        target_mask=target_mask,
        f=f,
        M11=M11,
        M22=M22,
        dx=cfg.dx,
        max_iters=cfg.eikonal_iters,
        tol=cfg.eikonal_tol,
        f_eps=cfg.f_eps,
    )

    valid = np.isfinite(phi) & walkable
    gx, gy = grad_central(phi, cfg.dx, valid=valid)

    denom = np.sqrt(M11 * gx * gx + M22 * gy * gy) + 1e-12
    ux = np.zeros_like(gx)
    uy = np.zeros_like(gy)
    m = valid & (denom > 1e-12)
    ux[m] = -(M11[m] * gx[m]) / denom[m]
    uy[m] = -(M22[m] * gy[m]) / denom[m]

    os.makedirs(cfg.out_dir, exist_ok=True)

    for k in range(cfg.steps):
        # --- recompute phi from Hughes anisotropic eikonal ---
        if (k % cfg.recompute_phi_every) == 0:
            f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
            phi = compute_phi_hughes_aniso_diag(
                walkable=walkable,
                target_mask=target_mask,
                f=f,
                M11=M11,
                M22=M22,
                dx=cfg.dx,
                max_iters=cfg.eikonal_iters,
                tol=cfg.eikonal_tol,
                f_eps=cfg.f_eps,
            )
            valid = np.isfinite(phi) & walkable
            gx, gy = grad_central(phi, cfg.dx, valid=valid)

            denom = np.sqrt(M11 * gx * gx + M22 * gy * gy) + 1e-12
            ux.fill(0.0)
            uy.fill(0.0)
            m = valid & (denom > 1e-12)
            ux[m] = -(M11[m] * gx[m]) / denom[m]
            uy[m] = -(M22[m] * gy[m]) / denom[m]

        # --- velocity & continuity update ---
        f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
        vx = f * ux
        vy = f * uy

        vx[~walkable] = 0.0
        vy[~walkable] = 0.0
        rho[~walkable] = 0.0

        div = divergence_flux(rho, vx, vy, cfg.dx)
        rho = rho - cfg.dt * div
        rho = np.clip(rho, 0.0, cfg.rho_max)

        # outflow at target
        rho[target_mask] = 0.0

        if (k % cfg.save_every) == 0:
            plot_frame(
                rho, phi, ux, uy, walkable,
                os.path.join(cfg.out_dir, f"routeA_metric_frame_{k:04d}.png"),
                title=f"Hughes-aniso(M) step {k} | Route A",
                rho_max=cfg.rho_max
            )


def main():
    cfg = SimConfig()
    simulate_metric_routeA(cfg)


if __name__ == "__main__":
    main()
