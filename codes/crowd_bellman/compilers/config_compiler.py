from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core import DIRECTIONS, TransitionRule, default_allowed_mask, tensor_from_tau
from ..scenes import BaseScene, CaseModel, GroupModel, SimulationConfig
from ..spec.population_spec import PopulationSpec
from ..spec.route_spec import CaseRouteSpec, ControlSpec, StageSpec
from ..spec.scene_spec import SceneSpec


@dataclass(frozen=True)
class CompiledSceneBundle:
    scene: BaseScene
    region_masks: dict[str, np.ndarray]
    exit_masks: dict[str, np.ndarray]


def _clip_interval(value0: int, value1: int, limit: int) -> tuple[int, int]:
    lower = max(0, min(limit, int(value0)))
    upper = max(0, min(limit, int(value1)))
    return lower, upper


def _rect_mask(cfg: SimulationConfig, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    xx0, xx1 = _clip_interval(x0, x1, cfg.nx)
    yy0, yy1 = _clip_interval(y0, y1, cfg.ny)
    if xx1 <= xx0 or yy1 <= yy0:
        return mask
    mask[yy0:yy1, xx0:xx1] = True
    return mask


def _selector_mask(
    selector: tuple[str, ...] | None,
    region_masks: dict[str, np.ndarray],
    walkable: np.ndarray,
    *,
    default_to_walkable: bool = False,
) -> np.ndarray:
    if selector is None:
        return walkable.copy() if default_to_walkable else np.zeros_like(walkable, dtype=bool)

    mask = np.zeros_like(walkable, dtype=bool)
    if not selector:
        return walkable.copy() if default_to_walkable else mask

    for name in selector:
        if name in {"*", "__walkable__", "walkable"}:
            mask |= walkable
            continue
        region_mask = region_masks.get(name)
        if region_mask is None:
            raise ValueError(f"Unknown region name: {name}")
        mask |= region_mask
    return mask & walkable


def _channel_probe_x(mask: np.ndarray, probe_x: int | None) -> int:
    if probe_x is not None:
        return int(probe_x)
    points = np.argwhere(mask)
    if points.size == 0:
        return 0
    return int(round(float(np.mean(points[:, 1]))))


def compile_scene(scene_spec: SceneSpec, cfg: SimulationConfig) -> CompiledSceneBundle:
    walkable = np.ones((cfg.ny, cfg.nx), dtype=bool)
    if scene_spec.block_boundaries and cfg.nx >= 2 and cfg.ny >= 2:
        walkable[0, :] = False
        walkable[-1, :] = False
        walkable[:, 0] = False
        walkable[:, -1] = False

    region_masks: dict[str, np.ndarray] = {}
    for region in scene_spec.regions:
        if region.name in region_masks:
            raise ValueError(f"Duplicate region name: {region.name}")
        region_masks[region.name] = _rect_mask(cfg, region.x0, region.x1, region.y0, region.y1)

    for obstacle_name in scene_spec.obstacles:
        obstacle_mask = region_masks.get(obstacle_name)
        if obstacle_mask is None:
            raise ValueError(f"Obstacle region not found: {obstacle_name}")
        walkable[obstacle_mask] = False

    exit_masks: dict[str, np.ndarray] = {}
    exit_union = np.zeros_like(walkable, dtype=bool)
    for exit_spec in scene_spec.exits:
        mask = _selector_mask(exit_spec.regions, region_masks, walkable)
        exit_masks[exit_spec.name] = mask
        exit_union |= mask

    channel_masks: dict[str, np.ndarray] = {}
    probe_x: dict[str, int] = {}
    for channel in scene_spec.channels:
        mask = _selector_mask(channel.regions, region_masks, walkable)
        channel_masks[channel.name] = mask
        probe_x[channel.name] = _channel_probe_x(mask, channel.probe_x)

    centers_y = [0, 0, 0]
    for idx, channel in enumerate(scene_spec.channels[:3]):
        points = np.argwhere(channel_masks[channel.name])
        if points.size > 0:
            centers_y[idx] = int(round(float(np.mean(points[:, 0]))))

    scene = BaseScene(
        walkable=walkable,
        initial_rho=np.zeros((cfg.ny, cfg.nx), dtype=float),
        exit_mask=exit_union,
        channel_masks=channel_masks,
        probe_x=probe_x,
        wall_x0=0,
        wall_x1=0,
        tooth_x1=0,
        centers_y=(centers_y[0], centers_y[1], centers_y[2]),
        middle_entry=(0, 0),
    )
    return CompiledSceneBundle(scene=scene, region_masks=region_masks, exit_masks=exit_masks)


def _direction_bitmask(direction_names: tuple[str, ...] | None) -> np.uint16:
    if direction_names is None or not direction_names:
        return DIRECTIONS.all_mask

    normalized = tuple(name.upper() for name in direction_names)
    if "ALL" in normalized:
        return DIRECTIONS.all_mask

    bitmask = np.uint16(0)
    for name in normalized:
        try:
            index = DIRECTIONS.names.index(name)
        except ValueError as exc:
            raise ValueError(f"Unknown direction name: {name}") from exc
        bitmask = np.uint16(bitmask | DIRECTIONS.bits[index])
    return bitmask


def _constant_direction_vector(direction_name: str) -> tuple[float, float]:
    normalized = direction_name.upper()
    try:
        index = DIRECTIONS.names.index(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown direction name: {direction_name}") from exc
    return float(DIRECTIONS.ux[index]), float(DIRECTIONS.uy[index])


def _tensor_to_region_target(
    walkable: np.ndarray,
    target_mask: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(walkable.shape)
    points = np.argwhere(target_mask)
    if points.size == 0:
        raise ValueError("target region is empty")

    center_y = float(np.mean(points[:, 0]))
    center_x = float(np.mean(points[:, 1]))
    tx = center_x - xx
    ty = center_y - yy
    norm = np.sqrt(tx * tx + ty * ty)

    tau_x = np.zeros_like(tx, dtype=float)
    tau_y = np.zeros_like(ty, dtype=float)
    valid = walkable & (norm > 1.0e-12)
    tau_x[valid] = tx[valid] / norm[valid]
    tau_y[valid] = ty[valid] / norm[valid]
    return tensor_from_tau(tau_x=tau_x, tau_y=tau_y, alpha=alpha, beta=beta)


def _tensor_to_point_target(
    walkable: np.ndarray,
    target_point: tuple[int, int],
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices(walkable.shape)
    center_x = float(target_point[0])
    center_y = float(target_point[1])
    tx = center_x - xx
    ty = center_y - yy
    norm = np.sqrt(tx * tx + ty * ty)

    tau_x = np.zeros_like(tx, dtype=float)
    tau_y = np.zeros_like(ty, dtype=float)
    valid = walkable & (norm > 1.0e-12)
    tau_x[valid] = tx[valid] / norm[valid]
    tau_y[valid] = ty[valid] / norm[valid]
    return tensor_from_tau(tau_x=tau_x, tau_y=tau_y, alpha=alpha, beta=beta)


def _apply_control(
    stage: StageSpec,
    control: ControlSpec,
    *,
    walkable: np.ndarray,
    region_masks: dict[str, np.ndarray],
    goal_mask: np.ndarray,
    m11: np.ndarray,
    m12: np.ndarray,
    m22: np.ndarray,
    allowed_mask: np.ndarray,
) -> None:
    control_region = _selector_mask(
        (control.region,) if control.region is not None else None,
        region_masks,
        walkable,
        default_to_walkable=True,
    )
    if not np.any(control_region):
        return

    mode = control.mode.lower()
    if mode == "identity":
        pass
    elif mode == "isotropic":
        if control.value is None or control.value <= 0.0:
            raise ValueError(f"Control value must be positive for isotropic mode in stage {stage.stage_id}")
        m11[control_region] = control.value
        m12[control_region] = 0.0
        m22[control_region] = control.value
    elif mode == "fixed_direction":
        if control.direction is None or control.alpha is None or control.beta is None:
            raise ValueError(f"fixed_direction control is missing direction/alpha/beta in stage {stage.stage_id}")
        tau_x_value, tau_y_value = _constant_direction_vector(control.direction)
        tau_x = np.full(walkable.shape, tau_x_value, dtype=float)
        tau_y = np.full(walkable.shape, tau_y_value, dtype=float)
        control_m11, control_m12, control_m22 = tensor_from_tau(
            tau_x=tau_x,
            tau_y=tau_y,
            alpha=control.alpha,
            beta=control.beta,
        )
        m11[control_region] = control_m11[control_region]
        m12[control_region] = control_m12[control_region]
        m22[control_region] = control_m22[control_region]
    elif mode == "target_region":
        if control.alpha is None or control.beta is None:
            raise ValueError(f"target_region control is missing alpha/beta in stage {stage.stage_id}")
        target_name = control.target_region
        target_mask = _selector_mask((target_name,), region_masks, walkable) if target_name is not None else goal_mask
        control_m11, control_m12, control_m22 = _tensor_to_region_target(
            walkable=walkable,
            target_mask=target_mask,
            alpha=control.alpha,
            beta=control.beta,
        )
        m11[control_region] = control_m11[control_region]
        m12[control_region] = control_m12[control_region]
        m22[control_region] = control_m22[control_region]
    elif mode == "target_point":
        if control.alpha is None or control.beta is None or control.target_point is None:
            raise ValueError(f"target_point control is missing target_point/alpha/beta in stage {stage.stage_id}")
        control_m11, control_m12, control_m22 = _tensor_to_point_target(
            walkable=walkable,
            target_point=control.target_point,
            alpha=control.alpha,
            beta=control.beta,
        )
        m11[control_region] = control_m11[control_region]
        m12[control_region] = control_m12[control_region]
        m22[control_region] = control_m22[control_region]
    else:
        raise ValueError(f"Unsupported control mode: {control.mode}")

    if control.allowed_directions is not None:
        allowed_mask[control_region] = _direction_bitmask(control.allowed_directions)


def compile_case(
    bundle: CompiledSceneBundle,
    population_spec: PopulationSpec,
    route_spec: CaseRouteSpec,
) -> tuple[BaseScene, CaseModel]:
    if not route_spec.stages:
        raise ValueError("Route spec must define at least one stage")

    walkable = bundle.scene.walkable
    region_masks = bundle.region_masks
    stage_by_id = {stage.stage_id: stage for stage in route_spec.stages}
    if len(stage_by_id) != len(route_spec.stages):
        raise ValueError("Stage ids must be unique")

    group_keys = [stage.group_key for stage in route_spec.stages]
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("Stage group keys must be unique")

    groups: dict[tuple[int, int], GroupModel] = {}
    goal_masks: dict[str, np.ndarray] = {}
    sink_masks: dict[str, np.ndarray] = {}

    for stage in route_spec.stages:
        goal_mask = _selector_mask(stage.goal_regions, region_masks, walkable)
        if not np.any(goal_mask):
            raise ValueError(f"Goal region is empty for stage {stage.stage_id}")

        has_transition = stage.next_stage is not None or bool(stage.targets)
        if stage.sink_regions is not None:
            sink_mask = _selector_mask(stage.sink_regions, region_masks, walkable)
        elif has_transition:
            sink_mask = np.zeros_like(walkable, dtype=bool)
        else:
            sink_mask = goal_mask.copy()

        allowed_mask = default_allowed_mask(walkable)
        if stage.allowed_directions is not None:
            allowed_mask[walkable] = _direction_bitmask(stage.allowed_directions)

        m11 = np.ones(walkable.shape, dtype=float)
        m12 = np.zeros(walkable.shape, dtype=float)
        m22 = np.ones(walkable.shape, dtype=float)
        for control in stage.controls:
            _apply_control(
                stage,
                control,
                walkable=walkable,
                region_masks=region_masks,
                goal_mask=goal_mask,
                m11=m11,
                m12=m12,
                m22=m22,
                allowed_mask=allowed_mask,
            )

        groups[stage.group_key] = GroupModel(
            key=stage.group_key,
            name=stage.stage_id,
            goal_mask=goal_mask,
            sink_mask=sink_mask,
            allowed_mask=allowed_mask,
            m11=m11,
            m12=m12,
            m22=m22,
        )
        goal_masks[stage.stage_id] = goal_mask
        sink_masks[stage.stage_id] = sink_mask

    initial_group_density = {
        stage.group_key: np.zeros(walkable.shape, dtype=float)
        for stage in route_spec.stages
    }
    total_initial = np.zeros(walkable.shape, dtype=float)
    for initial_group in population_spec.initial_groups:
        stage = stage_by_id.get(initial_group.stage_id)
        if stage is None:
            raise ValueError(f"Population references unknown stage: {initial_group.stage_id}")
        region_mask = _selector_mask((initial_group.region,), region_masks, walkable)
        if not np.any(region_mask):
            raise ValueError(f"Population region is empty: {initial_group.region}")
        contribution = initial_group.density * region_mask.astype(float)
        initial_group_density[stage.group_key] += contribution
        total_initial += contribution

    transitions: list[TransitionRule] = []
    for stage in route_spec.stages:
        has_next = stage.next_stage is not None
        has_targets = bool(stage.targets)
        if has_next and has_targets:
            raise ValueError(f"Stage {stage.stage_id} cannot define both next_stage and targets")
        if not has_next and not has_targets:
            continue

        decision_mask = _selector_mask(stage.decision_regions, region_masks, walkable) if stage.decision_regions is not None else goal_masks[stage.stage_id]
        if not np.any(decision_mask):
            raise ValueError(f"Decision region is empty for stage {stage.stage_id}")

        if has_next:
            next_stage = stage_by_id.get(stage.next_stage)
            if next_stage is None:
                raise ValueError(f"Unknown next_stage {stage.next_stage} in stage {stage.stage_id}")
            transitions.append(
                TransitionRule(
                    source=stage.group_key,
                    kappa=stage.kappa,
                    decision_mask=decision_mask,
                    targets={next_stage.group_key: 1.0},
                )
            )
            continue

        target_map: dict[tuple[int, int], float] = {}
        for target in stage.targets:
            target_stage = stage_by_id.get(target.stage_id)
            if target_stage is None:
                raise ValueError(f"Unknown target stage {target.stage_id} in stage {stage.stage_id}")
            target_map[target_stage.group_key] = float(target.probability)

        transitions.append(
            TransitionRule(
                source=stage.group_key,
                kappa=stage.kappa,
                decision_mask=decision_mask,
                targets=target_map,
            )
        )

    first_stage = route_spec.stages[0]
    case_exit_mask = bundle.scene.exit_mask.copy()
    for mask in sink_masks.values():
        case_exit_mask |= mask

    scene = replace(bundle.scene, initial_rho=total_initial)
    case = CaseModel(
        case_id=route_spec.case_id,
        title=route_spec.title,
        walkable=walkable,
        exit_mask=case_exit_mask,
        channel_masks=scene.channel_masks,
        probe_x=scene.probe_x,
        m11=groups[first_stage.group_key].m11,
        m12=groups[first_stage.group_key].m12,
        m22=groups[first_stage.group_key].m22,
        allowed_mask=groups[first_stage.group_key].allowed_mask,
        groups=groups,
        transitions=tuple(transitions),
        initial_group_density=initial_group_density,
    )
    return scene, case
