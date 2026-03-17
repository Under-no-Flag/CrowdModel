import os
from dataclasses import dataclass

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ---------------------------
# Config
# ---------------------------
@dataclass
class SimConfig:
    nx: int = 100
    ny: int = 70
    dx: float = 1.0
    dt: float = 5.0          # 你给的是 5.0；代码会自动做 CFL 限制
    steps: int = 200

    vmax: float = 10.0
    rho_max: float = 1.0

    recompute_phi_every: int = 1

    rho_init: float = 0.8

    # Metric M(x)=diag(M11,M22)
    M11_default: float = 1.0
    M22_default: float = 1.0
    M22_in_channel: float = 1.0 / 25.0   # 通道内 M22 小 => 竖向代价大，促使水平通过

    # HJB / Value iteration
    hjb_iters: int = 1200
    hjb_tol: float = 1e-6
    f_eps: float = 1.0

    # Direction control
    strict_oneway_mid: bool = False
    # False: 中通道只禁止 +x，但允许 -x/+y/-y（更符合“不可正向通行”）
    # True : 中通道只允许 -x（严格 U={-tau}）

    # Output
    out_dir: str = os.path.join("codes", "results", "scene1_M_plus_HJB")
    save_every: int = 1

    # CFL safety
    cfl: float = 0.45  # dt_eff <= cfl * dx / vmax


# ---------------------------
# Speed-density
# ---------------------------
def greenshields_speed(rho: np.ndarray, vmax: float, rho_max: float) -> np.ndarray:
    f = vmax * (1.0 - rho / rho_max)
    return np.clip(f, 0.0, vmax)


# ---------------------------
# Scene geometry
# ---------------------------
def build_scene(cfg: SimConfig):
    ny, nx = cfg.ny, cfg.nx
    walkable = np.ones((ny, nx), dtype=bool)

    # Outer walls
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    # Central vertical barrier with 3 openings
    wall_x0 = int(nx * 0.52)
    wall_w = max(2, int(nx * 0.03))
    wall_x1 = min(nx - 2, wall_x0 + wall_w)

    walkable[:, wall_x0:wall_x1] = False

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

    # Right-side comb obstacles
    tooth_len = int(nx * 0.18)
    tooth_h = max(2, int(ny * 0.02))
    tooth_x0 = wall_x1
    tooth_x1 = min(nx - 2, tooth_x0 + tooth_len)
    for yc in y_centers:
        for dy in (-gap, gap):
            y0 = np.clip(yc + dy - tooth_h // 2, 1, ny - 2)
            y1 = np.clip(yc + dy + tooth_h // 2, 1, ny - 2)
            walkable[y0:y1, tooth_x0:tooth_x1] = False

    # Spawn region (left)
    spawn_mask = np.zeros((ny, nx), dtype=bool)
    spawn_mask[1:ny - 1, 1:int(nx * 0.18)] = True
    spawn_mask &= walkable

    # Target (right boundary segment)
    target_mask = np.zeros((ny, nx), dtype=bool)
    target_mask[1:ny - 1, nx - 2] = True
    target_mask &= walkable

    return walkable, spawn_mask, target_mask, channel_mask, channel_masks


# ---------------------------
# Metric M(x)
# ---------------------------
def build_metric_M(cfg: SimConfig, walkable: np.ndarray, channel_masks: list[np.ndarray]):
    """
    这里让 M(x) 表达“通道几何引导”，但为了避免人被吸进“不可正向”的中通道，
    默认只对 上/下 通道施加各向异性（M22 小）。
    """
    ny, nx = walkable.shape
    M11 = np.full((ny, nx), cfg.M11_default, dtype=float)
    M22 = np.full((ny, nx), cfg.M22_default, dtype=float)

    top_mask, mid_mask, bot_mask = channel_masks

    # 只对上/下通道施加引导（更符合你的目标：中间通道不走）
    M22[top_mask] = cfg.M22_in_channel
    M22[bot_mask] = cfg.M22_in_channel

    M11[~walkable] = 1.0
    M22[~walkable] = 1.0
    return M11, M22


# ---------------------------
# Control set U(x) on 4-neighborhood
# ---------------------------
DIR_R = np.uint8(1)   # +x
DIR_L = np.uint8(2)   # -x
DIR_U = np.uint8(4)   # +y
DIR_D = np.uint8(8)   # -y


def build_U_mask(walkable: np.ndarray, channel_masks: list[np.ndarray], strict_oneway_mid: bool):
    """
    U(x) as a bitmask over 4-neighborhood.
    - default (walkable): allow all directions
    - middle channel: forbid +x (right). Two modes:
        strict_oneway_mid=True  -> only allow left  (U={-tau})
        strict_oneway_mid=False -> allow left/up/down (U excludes +tau)
    """
    U = np.zeros_like(walkable, dtype=np.uint8)
    U[walkable] = DIR_R | DIR_L | DIR_U | DIR_D

    mid_mask = channel_masks[1]

    if strict_oneway_mid:
        U[mid_mask] = DIR_L
    else:
        U[mid_mask] = DIR_L | DIR_U | DIR_D

    U[~walkable] = 0
    return U


# ---------------------------
# HJB (value iteration) solver with metric-dependent step costs
# ---------------------------
def compute_phi_HJB_control_metric(
    walkable: np.ndarray,
    target_mask: np.ndarray,
    f: np.ndarray,
    U_mask: np.ndarray,
    M11: np.ndarray,
    M22: np.ndarray,
    dx: float,
    iters: int,
    tol: float,
    f_eps: float,
) -> np.ndarray:
    """
    Discrete HJB:
        phi(x) = min_{u in U(x)} ( phi(x + dx*u) + step_cost(x,u) )

    step_cost for 4-neighborhood, consistent with diagonal metric in 1D:
      move ±x: dx / (f * sqrt(M11))
      move ±y: dx / (f * sqrt(M22))

    This implements the control-constraint Hamilton–Jacobi idea on your grid.
    """
    ny, nx = walkable.shape
    phi = np.full((ny, nx), np.inf, dtype=float)
    phi[~walkable] = np.inf
    phi[target_mask] = 0.0

    init = walkable & (~target_mask)
    phi[init] = 1e6

    f_safe = np.maximum(f, f_eps)
    cx = dx / (f_safe * np.sqrt(np.maximum(M11, 1e-12)))
    cy = dx / (f_safe * np.sqrt(np.maximum(M22, 1e-12)))
    cx[target_mask] = 0.0
    cy[target_mask] = 0.0

    for _ in range(iters):
        max_delta = 0.0

        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if (not walkable[y, x]) or target_mask[y, x]:
                    continue

                um = U_mask[y, x]
                if um == 0:
                    continue

                best = phi[y, x]
                # allowed moves -> successor neighbors
                if (um & DIR_R) and walkable[y, x + 1] and np.isfinite(phi[y, x + 1]):
                    best = min(best, phi[y, x + 1] + cx[y, x])
                if (um & DIR_L) and walkable[y, x - 1] and np.isfinite(phi[y, x - 1]):
                    best = min(best, phi[y, x - 1] + cx[y, x])
                if (um & DIR_U) and walkable[y + 1, x] and np.isfinite(phi[y + 1, x]):
                    best = min(best, phi[y + 1, x] + cy[y, x])
                if (um & DIR_D) and walkable[y - 1, x] and np.isfinite(phi[y - 1, x]):
                    best = min(best, phi[y - 1, x] + cy[y, x])

                if best < phi[y, x]:
                    d = phi[y, x] - best
                    if d > max_delta:
                        max_delta = d
                    phi[y, x] = best

        if max_delta < tol:
            break

    finite = np.isfinite(phi) & walkable
    if np.any(finite):
        phi[finite] -= np.min(phi[finite])
    return phi


def compute_u_star_from_phi_metric(
    walkable: np.ndarray,
    target_mask: np.ndarray,
    phi: np.ndarray,
    f: np.ndarray,
    U_mask: np.ndarray,
    M11: np.ndarray,
    M22: np.ndarray,
    dx: float,
    f_eps: float,
):
    """
    u*(x) = argmin_{u in U(x)} [phi(x+dx*u) + step_cost(x,u)]
    (equivalently argmax of -u·∇phi in continuous form)
    """
    ny, nx = walkable.shape
    ux = np.zeros((ny, nx), dtype=float)
    uy = np.zeros((ny, nx), dtype=float)

    f_safe = np.maximum(f, f_eps)
    cx = dx / (f_safe * np.sqrt(np.maximum(M11, 1e-12)))
    cy = dx / (f_safe * np.sqrt(np.maximum(M22, 1e-12)))

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if (not walkable[y, x]) or target_mask[y, x] or (not np.isfinite(phi[y, x])):
                continue

            um = U_mask[y, x]
            if um == 0:
                continue

            best = np.inf
            bx, by = 0.0, 0.0

            if (um & DIR_R) and walkable[y, x + 1] and np.isfinite(phi[y, x + 1]):
                val = phi[y, x + 1] + cx[y, x]
                if val < best:
                    best = val; bx, by = 1.0, 0.0
            if (um & DIR_L) and walkable[y, x - 1] and np.isfinite(phi[y, x - 1]):
                val = phi[y, x - 1] + cx[y, x]
                if val < best:
                    best = val; bx, by = -1.0, 0.0
            if (um & DIR_U) and walkable[y + 1, x] and np.isfinite(phi[y + 1, x]):
                val = phi[y + 1, x] + cy[y, x]
                if val < best:
                    best = val; bx, by = 0.0, 1.0
            if (um & DIR_D) and walkable[y - 1, x] and np.isfinite(phi[y - 1, x]):
                val = phi[y - 1, x] + cy[y, x]
                if val < best:
                    best = val; bx, by = 0.0, -1.0

            ux[y, x] = bx
            uy[y, x] = by

    return ux, uy


# ---------------------------
# Plot helpers
# ---------------------------
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
    ax1.set_title("Potential φ and direction (u*)")
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


# ---------------------------
# Continuity update (upwind flux)
# ---------------------------
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


# ---------------------------
# Simulation
# ---------------------------
def simulate_M_plus_HJB(cfg: SimConfig):
    walkable, spawn_mask, target_mask, channel_mask, channel_masks = build_scene(cfg)

    # Control set U(x): middle channel forbidden +x
    U_mask = build_U_mask(walkable, channel_masks, strict_oneway_mid=cfg.strict_oneway_mid)

    # Metric M(x): use it to guide (top/bottom channels)
    M11, M22 = build_metric_M(cfg, walkable, channel_masks)

    rho = np.zeros((cfg.ny, cfg.nx), dtype=float)
    rho[spawn_mask] = cfg.rho_init

    os.makedirs(cfg.out_dir, exist_ok=True)

    # initial phi / u*
    f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
    phi = compute_phi_HJB_control_metric(
        walkable=walkable,
        target_mask=target_mask,
        f=f,
        U_mask=U_mask,
        M11=M11,
        M22=M22,
        dx=cfg.dx,
        iters=cfg.hjb_iters,
        tol=cfg.hjb_tol,
        f_eps=cfg.f_eps,
    )
    ux, uy = compute_u_star_from_phi_metric(
        walkable=walkable,
        target_mask=target_mask,
        phi=phi,
        f=f,
        U_mask=U_mask,
        M11=M11,
        M22=M22,
        dx=cfg.dx,
        f_eps=cfg.f_eps,
    )

    for k in range(cfg.steps):
        # recompute phi and u*
        if (k % cfg.recompute_phi_every) == 0:
            f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
            phi = compute_phi_HJB_control_metric(
                walkable=walkable,
                target_mask=target_mask,
                f=f,
                U_mask=U_mask,
                M11=M11,
                M22=M22,
                dx=cfg.dx,
                iters=cfg.hjb_iters,
                tol=cfg.hjb_tol,
                f_eps=cfg.f_eps,
            )
            ux, uy = compute_u_star_from_phi_metric(
                walkable=walkable,
                target_mask=target_mask,
                phi=phi,
                f=f,
                U_mask=U_mask,
                M11=M11,
                M22=M22,
                dx=cfg.dx,
                f_eps=cfg.f_eps,
            )

        # velocity
        f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
        vx = f * ux
        vy = f * uy

        vx[~walkable] = 0.0
        vy[~walkable] = 0.0
        rho[~walkable] = 0.0

        # CFL clamp for stability (important because cfg.dt may be huge)
        dt_eff = min(cfg.dt, cfg.cfl * cfg.dx / max(cfg.vmax, 1e-12))

        div = divergence_flux(rho, vx, vy, cfg.dx)
        rho = rho - dt_eff * div
        rho = np.clip(rho, 0.0, cfg.rho_max)

        # outflow at target
        rho[target_mask] = 0.0

        if (k % cfg.save_every) == 0:
            plot_frame(
                rho, phi, ux, uy, walkable,
                os.path.join(cfg.out_dir, f"M_HJB_frame_{k:04d}.png"),
                title=f"M(x)+HJB control step {k} | mid forbids +x | dt_eff={dt_eff:.3f}",
                rho_max=cfg.rho_max
            )


def main():
    cfg = SimConfig()
    simulate_M_plus_HJB(cfg)


if __name__ == "__main__":
    main()
