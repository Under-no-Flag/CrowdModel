from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import DIRECTIONS, GroupKey, TransitionRule, default_allowed_mask, tensor_from_tau


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
    """Geometric scene and baseline initialization."""

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
class GroupModel:
    """Per-(stage, route) model fields."""

    key: GroupKey
    name: str
    goal_mask: np.ndarray
    sink_mask: np.ndarray
    allowed_mask: np.ndarray
    m11: np.ndarray
    m12: np.ndarray
    m22: np.ndarray


@dataclass(frozen=True)
class CaseModel:
    """Simulation-ready case configuration.

    Legacy single-group fields are kept for backward compatibility.
    """

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
    groups: dict[GroupKey, GroupModel] | None = None
    transitions: tuple[TransitionRule, ...] = ()
    initial_group_density: dict[GroupKey, np.ndarray] | None = None


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


def _directional_tensor_to_target(
    walkable: np.ndarray,
    target_mask: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny, nx = walkable.shape
    yy, xx = np.indices((ny, nx))
    points = np.argwhere(target_mask)
    if points.size == 0:
        return (
            np.ones((ny, nx), dtype=float),
            np.zeros((ny, nx), dtype=float),
            np.ones((ny, nx), dtype=float),
        )

    center_y = float(np.mean(points[:, 0]))
    center_x = float(np.mean(points[:, 1]))
    tx = center_x - xx
    ty = center_y - yy
    norm = np.sqrt(tx * tx + ty * ty)

    tau_x = np.zeros((ny, nx), dtype=float)
    tau_y = np.zeros((ny, nx), dtype=float)
    valid = walkable & (norm > 1.0e-12)
    tau_x[valid] = tx[valid] / norm[valid]
    tau_y[valid] = ty[valid] / norm[valid]

    m11, m12, m22 = tensor_from_tau(tau_x=tau_x, tau_y=tau_y, alpha=alpha, beta=beta)
    m11[~walkable] = 1.0
    m12[~walkable] = 0.0
    m22[~walkable] = 1.0
    return m11, m12, m22


def build_tour_scene(cfg: SimulationConfig) -> BaseScene:
    """Build multi-stage sightseeing scene with dedicated wall openings."""

    ny, nx = cfg.ny, cfg.nx
    walkable = np.ones((ny, nx), dtype=bool)
    walkable[0, :] = False
    walkable[-1, :] = False
    walkable[:, 0] = False
    walkable[:, -1] = False

    wall_x0 = int(nx * 0.52)
    wall_w = max(3, int(nx * 0.04))
    wall_x1 = min(nx - 2, wall_x0 + wall_w)
    walkable[:, wall_x0:wall_x1] = False

    top_centers = (int(ny * 0.17), int(ny * 0.30))
    bottom_centers = (int(ny * 0.67), int(ny * 0.78), int(ny * 0.88))
    all_centers = top_centers + bottom_centers
    opening_h = max(5, int(ny * 0.08))

    openings: dict[str, np.ndarray] = {}
    for idx, cy in enumerate(all_centers):
        y0 = max(1, cy - opening_h // 2)
        y1 = min(ny - 1, cy + opening_h // 2)
        walkable[y0:y1, wall_x0:wall_x1] = True
        key = f"opening_{idx + 1}"
        mask = np.zeros((ny, nx), dtype=bool)
        mask[y0:y1, wall_x0:wall_x1] = True
        openings[key] = mask

    initial_rho = np.zeros((ny, nx), dtype=float)
    spawn_mask = np.zeros((ny, nx), dtype=bool)
    spawn_mask[1:ny - 1, 2:max(3, int(nx * 0.10))] = True
    spawn_mask &= walkable
    initial_rho[spawn_mask] = cfg.rho_init

    left_sink = np.zeros((ny, nx), dtype=bool)
    left_sink[1:ny - 1, 1] = True
    left_sink &= walkable

    channel_masks = {
        "entry_1_2": openings["opening_1"] | openings["opening_2"],
        "exit_8": openings["opening_3"],
        "exit_9": openings["opening_4"],
        "exit_10": openings["opening_5"],
    }

    probe_x = {
        "entry_1_2": wall_x1,
        "exit_8": wall_x0,
        "exit_9": wall_x0,
        "exit_10": wall_x0,
    }

    centers_y = (top_centers[0], int(ny * 0.52), bottom_centers[-1])
    return BaseScene(
        walkable=walkable,
        initial_rho=initial_rho,
        exit_mask=left_sink,
        channel_masks=channel_masks,
        probe_x=probe_x,
        wall_x0=wall_x0,
        wall_x1=wall_x1,
        tooth_x1=wall_x1,
        centers_y=centers_y,
        middle_entry=(wall_x0, centers_y[1]),
    )


def build_multistage_tour_case(scene: BaseScene) -> CaseModel:
    """Create multi-stage multi-route sightseeing case with fixed splitting probs."""

    walkable = scene.walkable
    ny, nx = walkable.shape
    allowed = default_allowed_mask(walkable)

    yy, xx = np.indices((ny, nx))
    right_side = (xx >= scene.wall_x1) & walkable

    # With plotting origin="lower", larger y means "up".
    # Stage 1: move to the platform upper band (enter from lower channel, then go up).
    g1 = right_side & (yy >= int(ny * 0.78))
    # Stage 2: tour downward on platform.
    g2 = right_side & (yy <= int(ny * 0.28))
    # Stage 3: leave from the platform lower side; keep 8/9/10 route split by x-band.
    lower_exit_band = right_side & (yy <= int(ny * 0.20))
    split_left = scene.wall_x1 + max(1, (nx - scene.wall_x1) // 3)
    split_right = scene.wall_x1 + max(2, 2 * (nx - scene.wall_x1) // 3)
    g38 = lower_exit_band & (xx < split_left)
    g39 = lower_exit_band & (xx >= split_left) & (xx < split_right)
    g310 = lower_exit_band & (xx >= split_right)
    left_sink = scene.exit_mask.copy()

    m11_11, m12_11, m22_11 = _directional_tensor_to_target(walkable, g1, alpha=9.0, beta=0.3)
    m11_21, m12_21, m22_21 = _directional_tensor_to_target(walkable, g2, alpha=10.0, beta=0.25)
    m11_38, m12_38, m22_38 = _directional_tensor_to_target(walkable, g38, alpha=9.0, beta=0.3)
    m11_39, m12_39, m22_39 = _directional_tensor_to_target(walkable, g39, alpha=9.0, beta=0.3)
    m11_310, m12_310, m22_310 = _directional_tensor_to_target(walkable, g310, alpha=9.0, beta=0.3)
    m11_41, m12_41, m22_41 = _directional_tensor_to_target(walkable, left_sink, alpha=8.0, beta=0.35)

    groups: dict[GroupKey, GroupModel] = {
        (1, 1): GroupModel(
            key=(1, 1),
            name="stage1_entry",
            goal_mask=g1,
            sink_mask=np.zeros_like(walkable, dtype=bool),
            allowed_mask=allowed,
            m11=m11_11,
            m12=m12_11,
            m22=m22_11,
        ),
        (2, 1): GroupModel(
            key=(2, 1),
            name="stage2_tour_down",
            goal_mask=g2,
            sink_mask=np.zeros_like(walkable, dtype=bool),
            allowed_mask=allowed,
            m11=m11_21,
            m12=m12_21,
            m22=m22_21,
        ),
        (3, 8): GroupModel(
            key=(3, 8),
            name="stage3_route8",
            goal_mask=g38,
            sink_mask=np.zeros_like(walkable, dtype=bool),
            allowed_mask=allowed,
            m11=m11_38,
            m12=m12_38,
            m22=m22_38,
        ),
        (3, 9): GroupModel(
            key=(3, 9),
            name="stage3_route9",
            goal_mask=g39,
            sink_mask=np.zeros_like(walkable, dtype=bool),
            allowed_mask=allowed,
            m11=m11_39,
            m12=m12_39,
            m22=m22_39,
        ),
        (3, 10): GroupModel(
            key=(3, 10),
            name="stage3_route10",
            goal_mask=g310,
            sink_mask=np.zeros_like(walkable, dtype=bool),
            allowed_mask=allowed,
            m11=m11_310,
            m12=m12_310,
            m22=m22_310,
        ),
        (4, 1): GroupModel(
            key=(4, 1),
            name="stage4_return_left",
            goal_mask=left_sink,
            sink_mask=left_sink,
            allowed_mask=allowed,
            m11=m11_41,
            m12=m12_41,
            m22=m22_41,
        ),
    }

    transitions = (
        TransitionRule(source=(1, 1), kappa=2.0, decision_mask=g1, targets={(2, 1): 1.0}),
        TransitionRule(source=(2, 1), kappa=1.8, decision_mask=g2, targets={(3, 8): 0.2, (3, 9): 0.3, (3, 10): 0.5}),
        TransitionRule(source=(3, 8), kappa=2.0, decision_mask=g38, targets={(4, 1): 1.0}),
        TransitionRule(source=(3, 9), kappa=2.0, decision_mask=g39, targets={(4, 1): 1.0}),
        TransitionRule(source=(3, 10), kappa=2.0, decision_mask=g310, targets={(4, 1): 1.0}),
    )

    initial_group_density = {key: np.zeros((ny, nx), dtype=float) for key in groups}
    initial_group_density[(1, 1)] = scene.initial_rho.copy()

    m11 = np.ones((ny, nx), dtype=float)
    m12 = np.zeros((ny, nx), dtype=float)
    m22 = np.ones((ny, nx), dtype=float)

    return CaseModel(
        case_id="case5_multistage_tour",
        title="Case 5: multi-stage sightseeing with fixed-route splitting",
        walkable=walkable,
        exit_mask=scene.exit_mask,
        channel_masks=scene.channel_masks,
        probe_x=scene.probe_x,
        m11=m11,
        m12=m12,
        m22=m22,
        allowed_mask=allowed,
        groups=groups,
        transitions=transitions,
        initial_group_density=initial_group_density,
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

    if case_id == "case5_multistage_tour":
        return build_multistage_tour_case(scene)

    raise ValueError(f"Unknown case id: {case_id}")


def build_scene_for_case(case_id: str, cfg: SimulationConfig, cached_scene: BaseScene | None = None) -> BaseScene:
    """Return case-specific scene while reusing cached common scene where possible."""

    if case_id == "case5_multistage_tour":
        return build_tour_scene(cfg)

    if cached_scene is not None:
        return cached_scene
    return build_three_channel_scene(cfg)
