from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core import DIRECTIONS, DirectionHandoffRule, TransitionRule, default_allowed_mask, tensor_from_tau
from ..scenes import (
    BaseScene,
    CaseModel,
    ChannelGateModel,
    GateCapacitySchedule,
    GroupModel,
    InflowModel,
    SimulationConfig,
)
from ..spec.population_spec import PopulationSpec
from ..spec.route_spec import CapacityControlSpec, CaseRouteSpec, ControlSpec, StageSpec
from ..spec.scene_spec import SceneSpec


@dataclass(frozen=True)
class CompiledSceneBundle:
    scene: BaseScene
    region_masks: dict[str, np.ndarray]
    exit_masks: dict[str, np.ndarray]
    region_axes: dict[str, tuple[float, float]]
    channel_axes: dict[str, tuple[float, float]]


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


def _point_in_polygon(x: float, y: float, points: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    previous_x, previous_y = points[-1]
    for current_x, current_y in points:
        crosses_y = (current_y > y) != (previous_y > y)
        if crosses_y:
            x_intersection = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < x_intersection:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _polygon_mask(cfg: SimulationConfig, points: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    if len(points) < 3:
        return mask

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, x1 = _clip_interval(int(np.floor(min(xs))), int(np.ceil(max(xs))), cfg.nx)
    y0, y1 = _clip_interval(int(np.floor(min(ys))), int(np.ceil(max(ys))), cfg.ny)
    if x1 <= x0 or y1 <= y0:
        return mask

    edges = tuple(zip(points, points[1:] + points[:1]))
    for yy in range(y0, y1):
        cy = float(yy) + 0.5
        for xx in range(x0, x1):
            cx = float(xx) + 0.5
            if _point_in_polygon(cx, cy, points) or any(
                _distance_to_segment(cx, cy, start, end) <= 1.0e-9 for start, end in edges
            ):
                mask[yy, xx] = True
    return mask


def _distance_to_segment(
    x: float,
    y: float,
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-12:
        return float(np.hypot(x - sx, y - sy))
    t = max(0.0, min(1.0, ((x - sx) * dx + (y - sy) * dy) / length2))
    px = sx + t * dx
    py = sy + t * dy
    return float(np.hypot(x - px, y - py))


def _polyline_wall_mask(cfg: SimulationConfig, points: tuple[tuple[float, float], ...], width: float) -> np.ndarray:
    mask = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    if len(points) < 2 or width <= 0.0:
        return mask

    radius = width / 2.0
    for start, end in zip(points[:-1], points[1:]):
        min_x = int(np.floor(min(start[0], end[0]) - radius))
        max_x = int(np.ceil(max(start[0], end[0]) + radius)) + 1
        min_y = int(np.floor(min(start[1], end[1]) - radius))
        max_y = int(np.ceil(max(start[1], end[1]) + radius)) + 1
        x0, x1 = _clip_interval(min_x, max_x, cfg.nx)
        y0, y1 = _clip_interval(min_y, max_y, cfg.ny)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if _distance_to_segment(float(xx), float(yy), start, end) <= radius + 1.0e-9:
                    mask[yy, xx] = True
    return mask


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        expanded[1:, 1:] |= result[:-1, :-1]
        expanded[1:, :-1] |= result[:-1, 1:]
        expanded[:-1, 1:] |= result[1:, :-1]
        expanded[:-1, :-1] |= result[1:, 1:]
        result = expanded
    return result


def _distance_to_mask(source_mask: np.ndarray) -> np.ndarray:
    source = np.asarray(source_mask, dtype=bool)
    distance = np.full(source.shape, np.inf, dtype=float)
    distance[source] = 0.0
    if not np.any(source):
        return distance

    ny, nx = source.shape
    diagonal = float(np.sqrt(2.0))
    for y in range(ny):
        for x in range(nx):
            value = distance[y, x]
            if y > 0:
                value = min(value, distance[y - 1, x] + 1.0)
                if x > 0:
                    value = min(value, distance[y - 1, x - 1] + diagonal)
                if x + 1 < nx:
                    value = min(value, distance[y - 1, x + 1] + diagonal)
            if x > 0:
                value = min(value, distance[y, x - 1] + 1.0)
            distance[y, x] = value

    for y in range(ny - 1, -1, -1):
        for x in range(nx - 1, -1, -1):
            value = distance[y, x]
            if y + 1 < ny:
                value = min(value, distance[y + 1, x] + 1.0)
                if x > 0:
                    value = min(value, distance[y + 1, x - 1] + diagonal)
                if x + 1 < nx:
                    value = min(value, distance[y + 1, x + 1] + diagonal)
            if x + 1 < nx:
                value = min(value, distance[y, x + 1] + 1.0)
            distance[y, x] = value
    return distance


def _wall_cost_from_distance(cfg: SimulationConfig, walkable: np.ndarray, distance: np.ndarray) -> np.ndarray:
    cost = np.ones(walkable.shape, dtype=float)
    weight = float(cfg.wall_avoidance_weight)
    if weight <= 0.0:
        return cost
    sigma = float(cfg.wall_avoidance_sigma_cells)
    radius = float(cfg.wall_avoidance_radius_cells)
    if sigma <= 0.0:
        raise ValueError("wall_avoidance_sigma_cells must be positive when wall avoidance is enabled")
    if radius < 0.0:
        raise ValueError("wall_avoidance_radius_cells must be non-negative")
    active = walkable & np.isfinite(distance) & (distance <= radius)
    cost[active] = 1.0 + weight * np.exp(-distance[active] / sigma)
    return cost


def _direction_shift_slices(dy: int, dx: int) -> tuple[slice, slice, slice, slice]:
    src_y = slice(None, -dy) if dy > 0 else slice(-dy, None) if dy < 0 else slice(None)
    dst_y = slice(dy, None) if dy > 0 else slice(None, dy) if dy < 0 else slice(None)
    src_x = slice(None, -dx) if dx > 0 else slice(-dx, None) if dx < 0 else slice(None)
    dst_x = slice(dx, None) if dx > 0 else slice(None, dx) if dx < 0 else slice(None)
    return src_y, src_x, dst_y, dst_x


def _corner_block_mask(walkable: np.ndarray) -> np.ndarray:
    blocked = np.zeros(walkable.shape, dtype=np.uint16)
    for bit, dy, dx in zip(DIRECTIONS.bits, DIRECTIONS.dy, DIRECTIONS.dx):
        dy_int = int(dy)
        dx_int = int(dx)
        if abs(dy_int) != 1 or abs(dx_int) != 1:
            continue
        src_y, src_x, dst_y, dst_x = _direction_shift_slices(dy_int, dx_int)
        blocked_source = (
            walkable[src_y, src_x]
            & walkable[dst_y, dst_x]
            & ((~walkable[dst_y, src_x]) | (~walkable[src_y, dst_x]))
        )
        target = blocked[src_y, src_x]
        target[blocked_source] = np.uint16(target[blocked_source] | np.uint16(bit))
    return blocked


def _normalize_axis(axis: tuple[float, float], *, region_name: str) -> tuple[float, float]:
    norm = float(np.hypot(axis[0], axis[1]))
    if norm <= 1.0e-12:
        raise ValueError(f"Region {region_name} axis must be non-zero")
    return float(axis[0] / norm), float(axis[1] / norm)


def _longest_edge_axis(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    best_axis = (1.0, 0.0)
    best_length2 = 0.0
    for start, end in zip(points, points[1:] + points[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length2 = dx * dx + dy * dy
        if length2 > best_length2:
            best_length2 = length2
            best_axis = (dx, dy)
    return best_axis


def _region_axis(region: object) -> tuple[float, float]:
    explicit_axis = getattr(region, "axis", None)
    if explicit_axis is not None:
        return _normalize_axis(explicit_axis, region_name=str(getattr(region, "name", "")))

    shape = str(getattr(region, "shape", "rect")).lower()
    if shape == "polygon":
        return _normalize_axis(
            _longest_edge_axis(getattr(region, "points", ())),
            region_name=str(getattr(region, "name", "")),
        )
    return (1.0, 0.0)


def _region_mask(cfg: SimulationConfig, region: object) -> np.ndarray:
    shape = str(getattr(region, "shape", "rect")).lower()
    if shape == "rect":
        x0 = getattr(region, "x0", None)
        x1 = getattr(region, "x1", None)
        y0 = getattr(region, "y0", None)
        y1 = getattr(region, "y1", None)
        if x0 is None or x1 is None or y0 is None or y1 is None:
            raise ValueError(f"Rect region {getattr(region, 'name', '')} is missing x0/x1/y0/y1")
        return _rect_mask(cfg, int(x0), int(x1), int(y0), int(y1))
    if shape == "polygon":
        return _polygon_mask(cfg, getattr(region, "points", ()))
    raise ValueError(f"Unsupported region shape: {shape}")


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


def _channel_axis(channel: object, region_axes: dict[str, tuple[float, float]]) -> tuple[float, float]:
    axes = [region_axes[name] for name in getattr(channel, "regions", ()) if name in region_axes]
    if not axes:
        return (1.0, 0.0)

    axis_x = float(sum(axis[0] for axis in axes))
    axis_y = float(sum(axis[1] for axis in axes))
    norm = float(np.hypot(axis_x, axis_y))
    if norm <= 1.0e-12:
        return axes[0]
    return axis_x / norm, axis_y / norm


def compile_scene(scene_spec: SceneSpec, cfg: SimulationConfig) -> CompiledSceneBundle:
    walkable = np.ones((cfg.ny, cfg.nx), dtype=bool)
    if scene_spec.block_boundaries and cfg.nx >= 2 and cfg.ny >= 2:
        walkable[0, :] = False
        walkable[-1, :] = False
        walkable[:, 0] = False
        walkable[:, -1] = False

    region_masks: dict[str, np.ndarray] = {}
    region_axes: dict[str, tuple[float, float]] = {}
    for region in scene_spec.regions:
        if region.name in region_masks:
            raise ValueError(f"Duplicate region name: {region.name}")
        region_masks[region.name] = _region_mask(cfg, region)
        region_axes[region.name] = _region_axis(region)

    wall_names: set[str] = set()
    for wall in scene_spec.walls:
        if wall.name in region_masks:
            raise ValueError(f"Duplicate region name: {wall.name}")
        wall_mask = _polyline_wall_mask(cfg, wall.points, wall.width)
        region_masks[wall.name] = wall_mask
        wall_names.add(wall.name)

    obstacle_union = np.zeros_like(walkable, dtype=bool)
    for obstacle_name in scene_spec.obstacles:
        obstacle_mask = region_masks.get(obstacle_name)
        if obstacle_mask is None:
            raise ValueError(f"Obstacle region not found: {obstacle_name}")
        obstacle_union |= obstacle_mask

    wall_union = np.zeros_like(walkable, dtype=bool)
    for wall_name in wall_names:
        wall_union |= region_masks[wall_name]

    solid_union = obstacle_union | wall_union
    walkable[solid_union] = False
    clearance_cells = int(cfg.wall_clearance_cells)
    if clearance_cells < 0:
        raise ValueError("wall_clearance_cells must be non-negative")
    if clearance_cells > 0 and np.any(solid_union):
        walkable[_dilate_mask(solid_union, clearance_cells)] = False

    wall_distance_cells = _distance_to_mask(~walkable)
    wall_cost = _wall_cost_from_distance(cfg, walkable, wall_distance_cells)
    corner_block_mask = _corner_block_mask(walkable) if cfg.block_diagonal_corner_cutting else np.zeros_like(walkable, dtype=np.uint16)


    exit_masks: dict[str, np.ndarray] = {}
    exit_union = np.zeros_like(walkable, dtype=bool)
    for exit_spec in scene_spec.exits:
        mask = _selector_mask(exit_spec.regions, region_masks, walkable)
        exit_masks[exit_spec.name] = mask
        exit_union |= mask

    channel_masks: dict[str, np.ndarray] = {}
    probe_x: dict[str, int] = {}
    channel_axes: dict[str, tuple[float, float]] = {}
    for channel in scene_spec.channels:
        mask = _selector_mask(channel.regions, region_masks, walkable)
        channel_masks[channel.name] = mask
        probe_x[channel.name] = _channel_probe_x(mask, channel.probe_x)
        channel_axes[channel.name] = _channel_axis(channel, region_axes)

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
        channel_axes=channel_axes,
        wall_mask=wall_union,
        wall_distance_cells=wall_distance_cells,
        wall_cost=wall_cost,
        corner_block_mask=corner_block_mask,
        wall_x0=0,
        wall_x1=0,
        tooth_x1=0,
        centers_y=(centers_y[0], centers_y[1], centers_y[2]),
        middle_entry=(0, 0),
    )
    return CompiledSceneBundle(
        scene=scene,
        region_masks=region_masks,
        exit_masks=exit_masks,
        region_axes=region_axes,
        channel_axes=channel_axes,
    )


def _direction_bitmask(direction_names: tuple[str, ...] | None) -> np.uint16:
    if direction_names is None or not direction_names:
        return DIRECTIONS.all_mask

    normalized = tuple(name.upper() for name in direction_names)
    if any(name in {"NONE", "CLOSED", "EMPTY"} for name in normalized):
        if len(normalized) != 1:
            raise ValueError("NONE/CLOSED direction cannot be combined with other directions")
        return np.uint16(0)
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


def _axis_direction(
    axis: tuple[float, float],
    direction_name: str | None,
) -> tuple[float, float, np.uint16 | None]:
    normalized = "both" if direction_name is None else direction_name.lower()
    if normalized in {"both", "bidirectional", "all", "free"}:
        return axis[0], axis[1], None
    if normalized in {"plus", "+", "forward", "positive"}:
        vector = axis
    elif normalized in {"minus", "-", "reverse", "negative"}:
        vector = (-axis[0], -axis[1])
    elif normalized in {"none", "closed"}:
        return axis[0], axis[1], np.uint16(0)
    else:
        raise ValueError(f"Unsupported region_axis direction: {direction_name!r}")

    dots = DIRECTIONS.ux * vector[0] + DIRECTIONS.uy * vector[1]
    nearest_index = int(np.argmax(dots))
    return vector[0], vector[1], np.uint16(DIRECTIONS.bits[nearest_index])


def _normalize_gate_side(side: str) -> str:
    normalized = str(side).lower()
    if normalized in {"plus", "+", "east", "eastbound", "e"}:
        return "plus"
    if normalized in {"minus", "-", "west", "westbound", "w"}:
        return "minus"
    if normalized in {"both", "bidirectional", "all"}:
        return "both"
    raise ValueError(f"Unsupported capacity-control side: {side!r}")


def _oriented_gate_faces(
    *,
    channel_mask: np.ndarray,
    walkable: np.ndarray,
    normal: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inside = channel_mask & walkable
    if not np.any(inside):
        empty_i = np.empty(0, dtype=int)
        empty_f = np.empty(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_i, empty_i, empty_f

    yy, xx = np.indices(inside.shape)
    projection = (xx.astype(float) + 0.5) * normal[0] + (yy.astype(float) + 0.5) * normal[1]

    x_rows: list[np.ndarray] = []
    x_cols: list[np.ndarray] = []
    x_signs: list[np.ndarray] = []
    x_projections: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    y_cols: list[np.ndarray] = []
    y_signs: list[np.ndarray] = []
    y_projections: list[np.ndarray] = []

    eps = 1.0e-9

    def add_x(mask: np.ndarray, sign: float, inside_projection: np.ndarray) -> None:
        rows, cols = np.where(mask)
        if rows.size == 0:
            return
        x_rows.append(rows.astype(int))
        x_cols.append(cols.astype(int))
        x_signs.append(np.full(rows.shape, sign, dtype=float))
        x_projections.append(inside_projection[mask].astype(float))

    def add_y(mask: np.ndarray, sign: float, inside_projection: np.ndarray) -> None:
        rows, cols = np.where(mask)
        if rows.size == 0:
            return
        y_rows.append(rows.astype(int))
        y_cols.append(cols.astype(int))
        y_signs.append(np.full(rows.shape, sign, dtype=float))
        y_projections.append(inside_projection[mask].astype(float))

    if normal[0] > eps:
        add_x((~inside[:, :-1]) & walkable[:, :-1] & inside[:, 1:], 1.0, projection[:, 1:])
    if normal[0] < -eps:
        add_x(inside[:, :-1] & walkable[:, 1:] & (~inside[:, 1:]), -1.0, projection[:, :-1])
    if normal[1] > eps:
        add_y((~inside[:-1, :]) & walkable[:-1, :] & inside[1:, :], 1.0, projection[1:, :])
    if normal[1] < -eps:
        add_y(inside[:-1, :] & walkable[1:, :] & (~inside[1:, :]), -1.0, projection[:-1, :])

    x_row_values = np.concatenate(x_rows) if x_rows else np.empty(0, dtype=int)
    x_col_values = np.concatenate(x_cols) if x_cols else np.empty(0, dtype=int)
    x_sign_values = np.concatenate(x_signs) if x_signs else np.empty(0, dtype=float)
    x_projection_values = np.concatenate(x_projections) if x_projections else np.empty(0, dtype=float)
    y_row_values = np.concatenate(y_rows) if y_rows else np.empty(0, dtype=int)
    y_col_values = np.concatenate(y_cols) if y_cols else np.empty(0, dtype=int)
    y_sign_values = np.concatenate(y_signs) if y_signs else np.empty(0, dtype=float)
    y_projection_values = np.concatenate(y_projections) if y_projections else np.empty(0, dtype=float)

    all_projections = np.concatenate([x_projection_values, y_projection_values])
    if all_projections.size == 0:
        empty_i = np.empty(0, dtype=int)
        empty_f = np.empty(0, dtype=float)
        return empty_i, empty_i, empty_f, empty_i, empty_i, empty_f

    entry_projection = float(np.min(all_projections))
    entry_limit = entry_projection + 1.5
    keep_x = x_projection_values <= entry_limit
    keep_y = y_projection_values <= entry_limit

    if not np.any(keep_x) and not np.any(keep_y):
        keep_x = x_projection_values <= entry_projection + 1.0e-9
        keep_y = y_projection_values <= entry_projection + 1.0e-9

    return (
        x_row_values[keep_x],
        x_col_values[keep_x],
        x_sign_values[keep_x],
        y_row_values[keep_y],
        y_col_values[keep_y],
        y_sign_values[keep_y],
    )


def _oriented_waiting_mask(
    *,
    channel_mask: np.ndarray,
    walkable: np.ndarray,
    normal: tuple[float, float],
    waiting_width: int,
) -> np.ndarray:
    inside = channel_mask & walkable
    waiting_mask = np.zeros_like(walkable, dtype=bool)
    if not np.any(inside):
        return waiting_mask

    yy, xx = np.indices(inside.shape)
    x_center = xx.astype(float) + 0.5
    y_center = yy.astype(float) + 0.5
    projection = x_center * normal[0] + y_center * normal[1]
    tangent_x = -normal[1]
    tangent_y = normal[0]
    lateral = x_center * tangent_x + y_center * tangent_y

    width = max(int(waiting_width), 1)
    entry_projection = float(np.min(projection[inside]))
    lateral_values = lateral[inside]
    lateral_min = float(np.min(lateral_values)) - width
    lateral_max = float(np.max(lateral_values)) + width
    waiting_mask = (
        walkable
        & (~inside)
        & (projection >= entry_projection - width)
        & (projection < entry_projection)
        & (lateral >= lateral_min)
        & (lateral <= lateral_max)
    )
    return waiting_mask


def _build_channel_gate(
    *,
    channel_name: str,
    side: str,
    channel_mask: np.ndarray,
    walkable: np.ndarray,
    axis: tuple[float, float],
    waiting_width: int,
) -> ChannelGateModel:
    points = np.argwhere(channel_mask & walkable)
    if points.size == 0:
        raise ValueError(f"Channel {channel_name!r} is empty; cannot build capacity gate")

    axis_norm = float(np.hypot(axis[0], axis[1]))
    if axis_norm <= 1.0e-12:
        raise ValueError(f"Channel {channel_name!r} axis must be non-zero")
    axis_unit = (float(axis[0] / axis_norm), float(axis[1] / axis_norm))
    normal = axis_unit if side == "plus" else (-axis_unit[0], -axis_unit[1])
    face_x_rows, face_x_cols, face_x_signs, face_y_rows, face_y_cols, face_y_signs = _oriented_gate_faces(
        channel_mask=channel_mask,
        walkable=walkable,
        normal=normal,
    )
    if face_x_rows.size == 0 and face_y_rows.size == 0:
        raise ValueError(f"Capacity gate {channel_name}:{side} has no walkable entrance faces")

    rows = np.unique(points[:, 0]).astype(int)
    x_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 1]))
    nx = int(walkable.shape[1])

    if side == "plus":
        face_index = x_min - 1
    elif side == "minus":
        face_index = x_max
    else:
        raise ValueError(f"Gate side must be plus/minus, got {side!r}")

    if face_index < 0 or face_index >= nx - 1:
        fallback_index = int(np.median(face_x_cols)) if face_x_cols.size > 0 else int(round(float(np.mean(points[:, 1]))))
        face_index = min(max(fallback_index, 0), nx - 2)
    if face_index < 0 or face_index >= nx - 1:
        raise ValueError(
            f"Capacity gate {channel_name}:{side} has invalid face_index={face_index} for nx={nx}"
        )

    waiting_mask = _oriented_waiting_mask(
        channel_mask=channel_mask,
        walkable=walkable,
        normal=normal,
        waiting_width=waiting_width,
    )

    gate_id = f"{channel_name}:{side}"
    return ChannelGateModel(
        gate_id=gate_id,
        channel=channel_name,
        side=side,
        face_axis="oriented",
        face_index=face_index,
        face_rows=rows,
        waiting_mask=waiting_mask,
        face_x_rows=face_x_rows,
        face_x_cols=face_x_cols,
        face_x_signs=face_x_signs,
        face_y_rows=face_y_rows,
        face_y_cols=face_y_cols,
        face_y_signs=face_y_signs,
        normal=normal,
    )


def _compile_capacity_controls(
    *,
    controls: tuple[CapacityControlSpec, ...],
    channel_masks: dict[str, np.ndarray],
    channel_axes: dict[str, tuple[float, float]],
    walkable: np.ndarray,
) -> tuple[dict[str, ChannelGateModel], tuple[GateCapacitySchedule, ...]]:
    gates: dict[str, ChannelGateModel] = {}
    schedules: list[GateCapacitySchedule] = []

    for control in controls:
        if control.channel not in channel_masks:
            raise ValueError(f"Capacity control references unknown channel: {control.channel}")
        if control.rate < 0.0:
            raise ValueError(f"Capacity-control rate must be non-negative: {control.channel}:{control.side}")
        if control.time_end is not None and control.time_end <= control.time_start:
            raise ValueError(f"Capacity-control time_end must be greater than time_start: {control.channel}:{control.side}")

        sides = ("plus", "minus") if _normalize_gate_side(control.side) == "both" else (_normalize_gate_side(control.side),)
        for side in sides:
            gate_id = f"{control.channel}:{side}"
            if gate_id not in gates:
                gates[gate_id] = _build_channel_gate(
                    channel_name=control.channel,
                    side=side,
                    channel_mask=channel_masks[control.channel],
                    walkable=walkable,
                    axis=channel_axes.get(control.channel, (1.0, 0.0)),
                    waiting_width=control.waiting_width,
                )
            schedules.append(
                GateCapacitySchedule(
                    gate_id=gate_id,
                    channel=control.channel,
                    side=side,
                    rate=float(control.rate),
                    time_start=float(control.time_start),
                    time_end=None if control.time_end is None else float(control.time_end),
                )
            )

    return gates, tuple(schedules)


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
    region_axes: dict[str, tuple[float, float]],
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
    elif mode == "closed":
        allowed_mask[control_region] = np.uint16(0)
        return
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
    elif mode == "region_axis":
        if control.region is None or control.alpha is None or control.beta is None:
            raise ValueError(f"region_axis control is missing region/alpha/beta in stage {stage.stage_id}")
        axis_region = control.axis_region or control.region
        axis = region_axes.get(axis_region)
        if axis is None:
            raise ValueError(f"region_axis control references region without axis: {axis_region}")
        tau_x_value, tau_y_value, direction_bit = _axis_direction(axis, control.direction)
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
        if direction_bit is not None:
            allowed_mask[control_region] = direction_bit
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
                region_axes=bundle.region_axes,
                goal_mask=goal_mask,
                m11=m11,
                m12=m12,
                m22=m22,
                allowed_mask=allowed_mask,
            )

        if np.any(bundle.scene.corner_block_mask):
            allowed_mask = np.bitwise_and(allowed_mask, np.bitwise_not(bundle.scene.corner_block_mask))

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

    inflows: list[InflowModel] = []
    for inflow_group in population_spec.inflow_groups:
        stage = stage_by_id.get(inflow_group.stage_id)
        if stage is None:
            raise ValueError(f"Inflow references unknown stage: {inflow_group.stage_id}")
        region_mask = _selector_mask((inflow_group.region,), region_masks, walkable)
        if not np.any(region_mask):
            raise ValueError(f"Inflow region is empty: {inflow_group.region}")
        if inflow_group.rate < 0.0:
            raise ValueError(f"Inflow rate must be non-negative: {inflow_group.group_id}")
        if inflow_group.time_end is not None and inflow_group.time_end <= inflow_group.time_start:
            raise ValueError(f"Inflow time_end must be greater than time_start: {inflow_group.group_id}")
        if inflow_group.rho_cap is not None and inflow_group.rho_cap <= 0.0:
            raise ValueError(f"Inflow rho_cap must be positive: {inflow_group.group_id}")
        inflows.append(
            InflowModel(
                key=stage.group_key,
                name=inflow_group.group_id,
                region_mask=region_mask,
                rate=float(inflow_group.rate),
                time_start=float(inflow_group.time_start),
                time_end=None if inflow_group.time_end is None else float(inflow_group.time_end),
                rho_cap=None if inflow_group.rho_cap is None else float(inflow_group.rho_cap),
            )
        )

    transitions: list[TransitionRule] = []
    handoff_rules: list[DirectionHandoffRule] = []
    for stage in route_spec.stages:
        has_next = stage.next_stage is not None
        has_targets = bool(stage.targets)
        if has_next and has_targets:
            raise ValueError(f"Stage {stage.stage_id} cannot define both next_stage and targets")
        if not has_next and not has_targets:
            continue

        transition_direction = stage.transition_direction.lower()
        if transition_direction not in {"stop", "inherit_target"}:
            raise ValueError(
                f"Unsupported transition_direction {stage.transition_direction!r} in stage {stage.stage_id}"
            )

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
            if transition_direction == "inherit_target":
                handoff_rules.append(
                    DirectionHandoffRule(
                        source=stage.group_key,
                        target=next_stage.group_key,
                        handoff_mask=decision_mask,
                    )
                )
            continue

        if transition_direction != "stop":
            raise ValueError(
                f"transition_direction={stage.transition_direction!r} requires a single next_stage in stage {stage.stage_id}"
            )

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

    gates, gate_capacity_schedules = _compile_capacity_controls(
        controls=tuple(route_spec.capacity_controls),
        channel_masks=bundle.scene.channel_masks,
        channel_axes=bundle.channel_axes,
        walkable=walkable,
    )

    scene = replace(bundle.scene, initial_rho=total_initial)
    case = CaseModel(
        case_id=route_spec.case_id,
        title=route_spec.title,
        walkable=walkable,
        exit_mask=case_exit_mask,
        channel_masks=scene.channel_masks,
        probe_x=scene.probe_x,
        channel_axes=scene.channel_axes,
        m11=groups[first_stage.group_key].m11,
        m12=groups[first_stage.group_key].m12,
        m22=groups[first_stage.group_key].m22,
        allowed_mask=groups[first_stage.group_key].allowed_mask,
        groups=groups,
        transitions=tuple(transitions),
        handoff_rules=tuple(handoff_rules),
        initial_group_density=initial_group_density,
        inflows=tuple(inflows),
        gates=gates,
        gate_capacity_schedules=gate_capacity_schedules,
    )
    return scene, case
