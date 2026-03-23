from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import DIRECTIONS, default_allowed_mask, tensor_from_tau


CHANNEL_NAMES = ("top", "middle", "bottom")


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 96
    ny: int = 72
    dx: float = 0.5
    steps: int = 600
    time_horizon: float = 40.0
    vmax: float = 1.5
    rho_max: float = 5.0
    rho_init: float = 2.2
    bellman_every: int = 1
    bellman_f_eps: float = 0.08
    cfl: float = 0.9
    dt_cap: float = 0.18
    save_every: int = 40


@dataclass(frozen=True)
class BaseScene:
    walkable: np.ndarray
    initial_rho: np.ndarray
    exit_mask: np.ndarray
    channel_masks: dict[str, np.ndarray]
    probe_x: dict[str, int]
    wall_x0: int
    wall_x1: int
    tooth_x1: int
    centers_y: tuple[int, int, int]
    middle_entry: tuple[int, int]


@dataclass(frozen=True)
class CaseModel:
    case_id: str
    title: str
    walkable: np.ndarray
    exit_mask: np.ndarray
    channel_masks: dict[str, np.ndarray]
    probe_x: dict[str, int]
    m11: np.ndarray
    m12: np.ndarray
    m22: np.ndarray
    allowed_mask: np.ndarray


def scene_with_walkable(scene: BaseScene, walkable: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    exit_mask = scene.exit_mask & walkable
    channel_masks = {
        name: mask & walkable
        for name, mask in scene.channel_masks.items()
    }
    probe_x = dict(scene.probe_x)
    return exit_mask, channel_masks, probe_x


def channel_center_y(scene: BaseScene, channel_name: str) -> int:
    try:
        idx = CHANNEL_NAMES.index(channel_name)
    except ValueError as exc:
        raise ValueError(f"Unknown channel name: {channel_name}") from exc
    return scene.centers_y[idx]


def build_three_channel_scene(cfg: SimulationConfig) -> BaseScene:
    ny, nx = cfg.ny, cfg.nx
    walkable = np.ones((ny, nx), dtype=bool)

    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    wall_x0 = int(nx * 0.50)
    wall_w = max(3, int(nx * 0.04))
    wall_x1 = min(nx - 2, wall_x0 + wall_w)
    walkable[:, wall_x0:wall_x1] = False

    centers_y = (int(ny * 0.23), int(ny * 0.50), int(ny * 0.77))
    opening_h = max(8, int(ny * 0.11))
    gap = int(ny * 0.09)

    opening_masks: list[np.ndarray] = []
    for cy in centers_y:
        y0 = max(1, cy - opening_h // 2)
        y1 = min(ny - 1, cy + opening_h // 2)
        walkable[y0:y1, wall_x0:wall_x1] = True
        current = np.zeros((ny, nx), dtype=bool)
        current[y0:y1, wall_x0:wall_x1] = True
        opening_masks.append(current)

    tooth_len = int(nx * 0.20)
    tooth_h = max(2, int(ny * 0.03))
    tooth_x0 = wall_x1
    tooth_x1 = min(nx - 2, tooth_x0 + tooth_len)
    for cy in centers_y:
        for dy in (-gap, gap):
            y0 = int(np.clip(cy + dy - tooth_h // 2, 1, ny - 2))
            y1 = int(np.clip(cy + dy + tooth_h // 2, 1, ny - 2))
            walkable[y0:y1, tooth_x0:tooth_x1] = False

    initial_rho = np.zeros((ny, nx), dtype=float)
    spawn_mask = np.zeros((ny, nx), dtype=bool)
    spawn_mask[1:ny - 1, 1:int(nx * 0.16)] = True
    spawn_mask &= walkable
    initial_rho[spawn_mask] = cfg.rho_init

    exit_mask = np.zeros((ny, nx), dtype=bool)
    exit_mask[1:ny - 1, nx - 2] = True
    exit_mask &= walkable

    channel_masks: dict[str, np.ndarray] = {}
    probe_x: dict[str, int] = {}
    lane_x0 = wall_x0
    lane_x1 = tooth_x1
    for idx, (name, cy) in enumerate(zip(CHANNEL_NAMES, centers_y)):
        y0 = max(1, cy - opening_h // 2)
        y1 = min(ny - 1, cy + opening_h // 2)
        lane = np.zeros((ny, nx), dtype=bool)
        lane[y0:y1, lane_x0:lane_x1] = True
        lane &= walkable
        lane |= opening_masks[idx]
        channel_masks[name] = lane
        probe_x[name] = min(nx - 3, wall_x1 + max(2, tooth_len // 2))

    middle_entry = (wall_x0, centers_y[1])
    return BaseScene(
        walkable=walkable,
        initial_rho=initial_rho,
        exit_mask=exit_mask,
        channel_masks=channel_masks,
        probe_x=probe_x,
        wall_x0=wall_x0,
        wall_x1=wall_x1,
        tooth_x1=tooth_x1,
        centers_y=centers_y,
        middle_entry=middle_entry,
    )


def build_guided_channel_case(
    scene: BaseScene,
    case_id: str,
    title: str,
    guided_channel: str,
) -> CaseModel:
    walkable = scene.walkable
    ny, nx = walkable.shape
    m11 = np.ones((ny, nx), dtype=float)
    m12 = np.zeros((ny, nx), dtype=float)
    m22 = np.ones((ny, nx), dtype=float)
    case_walkable = walkable.copy()

    feeder = np.zeros_like(case_walkable, dtype=bool)
    feeder[:, int(nx * 0.18):scene.wall_x0] = True
    feeder &= case_walkable

    target_y = channel_center_y(scene, guided_channel)
    entry_x = scene.wall_x0
    yy, xx = np.indices(walkable.shape)
    tx = entry_x - xx
    ty = target_y - yy
    norm = np.sqrt(tx * tx + ty * ty)
    tau_x = np.zeros_like(m11)
    tau_y = np.zeros_like(m11)
    mask = feeder & (norm > 1.0e-12)
    tau_x[mask] = tx[mask] / norm[mask]
    tau_y[mask] = ty[mask] / norm[mask]
    feeder_m11, feeder_m12, feeder_m22 = tensor_from_tau(
        tau_x=tau_x,
        tau_y=tau_y,
        alpha=8.0,
        beta=0.35,
    )
    m11[mask] = feeder_m11[mask]
    m12[mask] = feeder_m12[mask]
    m22[mask] = feeder_m22[mask]

    guided_lane = scene.channel_masks[guided_channel] & case_walkable
    lane_tau_x = np.ones_like(m11)
    lane_tau_y = np.zeros_like(m11)
    lane_m11, lane_m12, lane_m22 = tensor_from_tau(
        tau_x=lane_tau_x,
        tau_y=lane_tau_y,
        alpha=10.0,
        beta=0.20,
    )
    m11[guided_lane] = lane_m11[guided_lane]
    m12[guided_lane] = lane_m12[guided_lane]
    m22[guided_lane] = lane_m22[guided_lane]

    penalty_mask = np.zeros_like(case_walkable, dtype=bool)
    for channel_name, mask_channel in scene.channel_masks.items():
        if channel_name == guided_channel:
            continue
        penalty_mask |= mask_channel
    penalty_mask &= case_walkable
    penalty_value = 0.15
    m11[penalty_mask] = penalty_value
    m12[penalty_mask] = 0.0
    m22[penalty_mask] = penalty_value

    east_mask = np.uint16(DIRECTIONS.bits[0])
    allowed_mask = default_allowed_mask(case_walkable)
    allowed_mask[guided_lane] = east_mask

    exit_mask, channel_masks, probe_x = scene_with_walkable(scene, case_walkable)
    return CaseModel(
        case_id=case_id,
        title=title,
        walkable=case_walkable,
        exit_mask=exit_mask,
        channel_masks=channel_masks,
        probe_x=probe_x,
        m11=m11,
        m12=m12,
        m22=m22,
        allowed_mask=allowed_mask,
    )


def build_case_model(case_id: str, scene: BaseScene) -> CaseModel:
    walkable = scene.walkable
    ny, nx = walkable.shape
    m11 = np.ones((ny, nx), dtype=float)
    m12 = np.zeros((ny, nx), dtype=float)
    m22 = np.ones((ny, nx), dtype=float)
    allowed_mask = default_allowed_mask(walkable)

    if case_id == "case1_baseline":
        exit_mask, channel_masks, probe_x = scene_with_walkable(scene, walkable)
        return CaseModel(
            case_id=case_id,
            title="Case 1: baseline",
            walkable=walkable,
            exit_mask=exit_mask,
            channel_masks=channel_masks,
            probe_x=probe_x,
            m11=m11,
            m12=m12,
            m22=m22,
            allowed_mask=allowed_mask,
        )

    if case_id == "case2_middle_guided":
        return build_guided_channel_case(
            scene=scene,
            case_id=case_id,
            title="Case 2: geometry-guided middle one-way channel",
            guided_channel="middle",
        )

    if case_id == "case3_top_guided":
        return build_guided_channel_case(
            scene=scene,
            case_id=case_id,
            title="Case 3: geometry-guided top one-way channel",
            guided_channel="top",
        )

    if case_id == "case4_bottom_guided":
        return build_guided_channel_case(
            scene=scene,
            case_id=case_id,
            title="Case 4: geometry-guided bottom one-way channel",
            guided_channel="bottom",
        )

    raise ValueError(f"Unknown case id: {case_id}")
