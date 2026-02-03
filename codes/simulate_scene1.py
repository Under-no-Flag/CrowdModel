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
    dt: float = 0.5
    steps: int = 200

    vmax: float = 10.0
    rho_max: float = 1.0
    alpha_cost: float = 6.0      # 拥挤惩罚强度（越大越绕开高密度）
    recompute_T_every: int = 1   # 每几步重算一次T（1=每步）

    # Initial density for the second crowd (spawn from left)
    rho_init: float = 0.8

    # Output
    out_dir: str = os.path.join("codes", "results", "scene1")
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
    """
    ny, nx = cfg.ny, cfg.nx
    walkable = np.ones((ny, nx), dtype=bool)

    # Outer boundaries are walls
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    # Create a central vertical "barrier" with three door-like channels.
    # Geometry is an approximation of the provided schematic.
    wall_x0 = int(nx * 0.52)
    wall_w = max(2, int(nx * 0.03))
    wall_x1 = min(nx - 2, wall_x0 + wall_w)

    walkable[:, wall_x0:wall_x1] = False

    # Three channels (openings) through the wall (top/middle/bottom)
    ch_h = max(5, int(ny * 0.10))
    gap = int(ny * 0.07)
    y_centers = [int(ny * 0.25), int(ny * 0.50), int(ny * 0.75)]

    channel_masks = []  # 分别存 top/mid/bot
    channel_mask = np.zeros((ny, nx), dtype=bool)

    for yc in y_centers:
        y0 = max(1, yc - ch_h // 2)
        y1 = min(ny - 1, yc + ch_h // 2)

        walkable[y0:y1, wall_x0:wall_x1] = True

        m = np.zeros((ny, nx), dtype=bool)
        m[y0:y1, wall_x0:wall_x1] = True
        channel_masks.append(m)

        channel_mask |= m   # 总 mask（如果你还想保留）

    # Add some horizontal "teeth" walls to mimic the comb-like obstacles
    # on the right side (purely for visual/flow complexity).
    tooth_len = int(nx * 0.18)
    tooth_h = max(2, int(ny * 0.02))
    tooth_x0 = wall_x1
    tooth_x1 = min(nx - 2, tooth_x0 + tooth_len)
    for yc in y_centers:
        # two teeth around each channel center
        for dy in (-gap, gap):
            y0 = np.clip(yc + dy - tooth_h // 2, 1, ny - 2)
            y1 = np.clip(yc + dy + tooth_h // 2, 1, ny - 2)
            walkable[y0:y1, tooth_x0:tooth_x1] = False

    # Spawn on left open region (excluding borders)
    spawn_mask = np.zeros((ny, nx), dtype=bool)
    spawn_mask[1 : ny - 1, 1 : int(nx * 0.18)] = True
    spawn_mask &= walkable

    # Target: right boundary (a vertical segment)
    target_mask = np.zeros((ny, nx), dtype=bool)
    target_mask[1 : ny - 1, nx - 2] = True
    target_mask &= walkable

    return walkable, spawn_mask, target_mask, channel_mask,channel_masks


def compute_potential_to_target(walkable: np.ndarray,
                                target_mask: np.ndarray,
                                cost: np.ndarray) -> np.ndarray:
    """
    动态势场：T approx 最小累计代价到目标
    cost: (ny,nx) 每格代价，障碍/不可达请给 np.inf
    """
    ny, nx = walkable.shape
    T = np.full((ny, nx), np.inf, dtype=float)

    T[target_mask] = 0.0
    T[~walkable] = np.inf

    # walkable但非target 初始化为大数
    T[np.isinf(T) & walkable] = 1e6

    # 迭代松弛（Gauss-Seidel风格）
    for _ in range(600):
        changed = 0
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                if (not walkable[y, x]) or target_mask[y, x]:
                    continue
                if not np.isfinite(cost[y, x]):
                    continue
                v = min(T[y - 1, x], T[y + 1, x], T[y, x - 1], T[y, x + 1]) + cost[y, x]
                if v < T[y, x] - 1e-6:
                    T[y, x] = v
                    changed += 1
        if changed == 0:
            break

    # 归一化（仅为了画图好看）
    finite = np.isfinite(T)
    if np.any(finite):
        T[finite] -= np.min(T[finite])

    return T

def smooth2d_gauss5(a: np.ndarray) -> np.ndarray:
    k = np.array([1, 4, 6, 4, 1], dtype=float)
    k /= k.sum()

    # x方向卷积
    ap = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    tmp = (k[0]*ap[:, 0:-4] + k[1]*ap[:, 1:-3] + k[2]*ap[:, 2:-2] + k[3]*ap[:, 3:-1] + k[4]*ap[:, 4:])

    # y方向卷积
    tp = np.pad(tmp, ((2, 2), (0, 0)), mode="edge")
    out = (k[0]*tp[0:-4, :] + k[1]*tp[1:-3, :] + k[2]*tp[2:-2, :] + k[3]*tp[3:-1, :] + k[4]*tp[4:, :])
    return out

def plot_frame(rho, T, ux, uy, walkable, out_path, title, rho_max=1.0):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(title)

    # ---- Density panel ----
    rr = smooth2d_gauss5(rho)
    rr = rr.copy()
    rr[~walkable] = np.nan

    im0 = ax0.imshow(
        rr, origin="lower", cmap="viridis",
        vmin=0.0, vmax=rho_max,
        interpolation="bilinear"   # 关键：更自然
    )
    ax0.set_title("Density ρ")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 障碍物点阵（更像你图里的黑点墙）
    oy, ox = np.where(~walkable)
    ax0.scatter(ox, oy, s=3, c="k", marker="s", linewidths=0)

    # ---- Potential + direction panel ----
    TT = T.copy()
    TT[~walkable] = np.nan
    im1 = ax1.imshow(
        TT, origin="lower", cmap="magma",
        interpolation="bilinear"
    )
    ax1.set_title("Potential T and direction")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # quiver（下采样）
    step = 3
    ys, xs = np.mgrid[0:T.shape[0]:step, 0:T.shape[1]:step]
    uxs = ux[0:T.shape[0]:step, 0:T.shape[1]:step]
    uys = uy[0:T.shape[0]:step, 0:T.shape[1]:step]
    m = walkable[0:T.shape[0]:step, 0:T.shape[1]:step]
    ax1.quiver(xs[m], ys[m], uxs[m], uys[m], color="white", pivot="mid", scale=25)

    # 障碍物点阵
    ax1.scatter(ox, oy, s=3, c="k", marker="s", linewidths=0)

    for ax in (ax0, ax1):
        ax.set_xlim(0, rho.shape[1]-1)
        ax.set_ylim(0, rho.shape[0]-1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def grad_central(T: np.ndarray, dx: float, valid: np.ndarray | None = None):
    """Central gradient with optional validity mask.

    If valid is provided, gradients are only computed where both sides are valid;
    otherwise set to 0. This avoids inf/nan propagation from unreachable cells.
    """
    gy = np.zeros_like(T)
    gx = np.zeros_like(T)

    if valid is None:
        gy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dx)
        gx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
        return gx, gy

    # y-gradient valid where up/down are valid
    vy = valid[2:, :] & valid[:-2, :]
    gy_inner = np.zeros((T.shape[0] - 2, T.shape[1]), dtype=T.dtype)
    gy_inner[vy] = (T[2:, :][vy] - T[:-2, :][vy]) / (2.0 * dx)
    gy[1:-1, :] = gy_inner

    # x-gradient valid where left/right are valid
    vx = valid[:, 2:] & valid[:, :-2]
    gx_inner = np.zeros((T.shape[0], T.shape[1] - 2), dtype=T.dtype)
    gx_inner[vx] = (T[:, 2:][vx] - T[:, :-2][vx]) / (2.0 * dx)
    gx[:, 1:-1] = gx_inner

    return gx, gy


def divergence_flux(rho: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float):
    # ----- x 方向：面速度 + 迎风取 rho -----
    vx_face = 0.5 * (vx + np.roll(vx, -1, axis=1))          # v_{i+1/2}
    rho_R   = np.roll(rho, -1, axis=1)                      # rho_{i+1}

    Fx_p = np.maximum(vx_face, 0.0) * rho + np.minimum(vx_face, 0.0) * rho_R
    Fx_m = np.roll(Fx_p, 1, axis=1)                         # F_{i-1/2}
    dFx = (Fx_p - Fx_m) / dx

    # 边界零通量（可按需要改成出流）
    dFx[:, 0] = 0.0
    dFx[:, -1] = 0.0

    # ----- y 方向 -----
    vy_face = 0.5 * (vy + np.roll(vy, -1, axis=0))          # v_{j+1/2}
    rho_U   = np.roll(rho, -1, axis=0)                      # rho_{j+1}

    Fy_p = np.maximum(vy_face, 0.0) * rho + np.minimum(vy_face, 0.0) * rho_U
    Fy_m = np.roll(Fy_p, 1, axis=0)
    dFy = (Fy_p - Fy_m) / dx

    dFy[0, :] = 0.0
    dFy[-1, :] = 0.0

    return dFx + dFy



def plot_density(rho: np.ndarray, walkable: np.ndarray, out_path: str, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    rr = rho.copy()
    rr[~walkable] = np.nan
    im = ax.imshow(rr, origin="lower", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_potential(T: np.ndarray, ux: np.ndarray, uy: np.ndarray, walkable: np.ndarray, out_path: str, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)
    TT = T.copy()
    TT[~walkable] = np.nan
    im = ax.imshow(TT, origin="lower", cmap="magma")

    # quiver (downsample)
    step = 3
    ys, xs = np.mgrid[0 : T.shape[0] : step, 0 : T.shape[1] : step]
    uxs = ux[0 : T.shape[0] : step, 0 : T.shape[1] : step]
    uys = uy[0 : T.shape[0] : step, 0 : T.shape[1] : step]
    m = walkable[0 : T.shape[0] : step, 0 : T.shape[1] : step]
    ax.quiver(xs[m], ys[m], uxs[m], uys[m], color="white", pivot="mid", scale=25)

    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def simulate(case: str, cfg: SimConfig):
    assert case in ("A", "B")

    walkable, spawn_mask, target_mask, channel_mask,channel_masks = build_scene(cfg)

    # Initial density: second crowd from left
    rho = np.zeros((cfg.ny, cfg.nx), dtype=float)
    rho[spawn_mask] = cfg.rho_init

    # 初始化一次 T/方向（避免第0步未定义）
    cost0 = 1.0 + cfg.alpha_cost * (rho / cfg.rho_max)
    cost0[~walkable] = np.inf
    cost0[target_mask] = 1.0

    T = compute_potential_to_target(walkable, target_mask, cost0)
    valid = np.isfinite(T) & walkable
    gx, gy = grad_central(T, cfg.dx, valid=valid)

    gnorm = np.sqrt(gx * gx + gy * gy)
    ux = np.zeros_like(gx)
    uy = np.zeros_like(gy)
    m = valid & (gnorm > 1e-12)
    ux[m] = -gx[m] / gnorm[m]
    uy[m] = -gy[m] / gnorm[m]

    if case == "B":
        mid_mask = channel_masks[1]  # 0=上, 1=中, 2=下
        ux[mid_mask] = np.minimum(ux[mid_mask], 0.0)  # 禁止向右

        ren = np.sqrt(ux*ux + uy*uy) + 1e-12
        ux /= ren
        uy /= ren

    os.makedirs(cfg.out_dir, exist_ok=True)

    for k in range(cfg.steps):

        # ---- 每步（或每N步）重算势场 ----
        if (k % cfg.recompute_T_every) == 0:
            cost = 1.0 + cfg.alpha_cost * (rho / cfg.rho_max)
            cost[~walkable] = np.inf
            cost[target_mask] = 1.0

            T = compute_potential_to_target(walkable, target_mask, cost)
            valid = np.isfinite(T) & walkable
            gx, gy = grad_central(T, cfg.dx, valid=valid)

            gnorm = np.sqrt(gx * gx + gy * gy)
            ux = np.zeros_like(gx)
            uy = np.zeros_like(gy)
            m = valid & (gnorm > 1e-12)
            ux[m] = -gx[m] / gnorm[m]
            uy[m] = -gy[m] / gnorm[m]

            if case == "B":
                mid_mask = channel_masks[1]          # 0=上, 1=中, 2=下
                ux[mid_mask] = np.minimum(ux[mid_mask], 0.0)  # 只禁中间通道向右
                ren = np.sqrt(ux * ux + uy * uy) + 1e-12
                ux /= ren
                uy /= ren


        # ---- 速度场 & 连续性方程更新（必须在循环里！）----
        f = greenshields_speed(rho, cfg.vmax, cfg.rho_max)
        vx = f * ux
        vy = f * uy

        vx[~walkable] = 0.0
        vy[~walkable] = 0.0
        rho[~walkable] = 0.0

        div = divergence_flux(rho, vx, vy, cfg.dx)
        rho = rho - cfg.dt * div
        rho = np.clip(rho, 0.0, cfg.rho_max)

        # people leave at target
        rho[target_mask] = 0.0

        # ---- 保存：密度 + 势场同一张图（也必须在循环里！）----
        if (k % cfg.save_every) == 0:
            plot_frame(
                rho, T, ux, uy, walkable,
                os.path.join(cfg.out_dir, f"{case}_frame_{k:04d}.png"),
                title=f"Hughes model step {k} | case {case}",
                rho_max=cfg.rho_max
            )


def main():
    cfg = SimConfig()
    # simulate("A", cfg)
    simulate("B", cfg)


if __name__ == "__main__":
    main()
