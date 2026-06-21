from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle


DEFAULT_NX = 512
DEFAULT_NY = 224
DEFAULT_DX = 0.125
DEFAULT_INITIAL_DENSITY = 2.0
DEFAULT_INFLOW_RATE = 0.08
DEFAULT_INFLOW_RHO_CAP = 4.8
DEFAULT_CHANNEL_ROUTE_MODE = "pass_through"
DEFAULT_ROUTE_SEQUENCE = (
    "goal_region2",
    "channel_3",
    "goal_region3",
    "goal_region4",
    "channel_4",
    "goal_region5",
    "exits",
)

COLORS = {
    "channel": "#1f77b4",
    "initial": "#2ca02c",
    "goal": "#d62728",
    "exit": "#9467bd",
    "route": "#ff7f0e",
    "wall": "#111111",
    "other": "#7f7f7f",
}


@dataclass(frozen=True)
class ImageFrame:
    path: str | None
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class RegionRecord:
    name: str
    kind: str
    x: float
    y: float
    width: float
    height: float
    rotation: float
    anylogic_corners: tuple[tuple[float, float], ...]
    grid_bbox: tuple[int, int, int, int]
    grid_corners: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ChannelRouteRegion:
    channel_name: str
    suffix: str
    post_name: str
    pre_name: str | None = None
    waypoint_name: str | None = None


@dataclass(frozen=True)
class WallPathRecord:
    name: str
    kind: str
    point_mode: str
    raw_point_count: int
    line_width: float
    grid_line_width: float
    anylogic_points: tuple[tuple[float, float], ...]
    grid_points: tuple[tuple[float, float], ...]
    grid_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class WallObstacleRegion:
    name: str
    source_path: str
    x0: int
    x1: int
    y0: int
    y1: int


def _text(node: ET.Element, child_name: str, default: str = "") -> str:
    child = node.find(child_name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _float_text(node: ET.Element, child_name: str, default: float = 0.0) -> float:
    text = _text(node, child_name)
    if not text:
        return default
    return float(text)


def _classify(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("channel_"):
        return "channel"
    if lowered.startswith("initial"):
        return "initial"
    if lowered.startswith("goal_region"):
        return "goal"
    if lowered.startswith(("decision_", "post_channel_", "pre_channel_", "merge_", "waypoint_")):
        return "route"
    if lowered in {"exit", "exits"} or lowered.startswith("exit_"):
        return "exit"
    return "other"


def _natural_key(name: str) -> tuple[str, int]:
    match = re.match(r"([A-Za-z_]+?)(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def _rotated_corners(x: float, y: float, width: float, height: float, rotation: float) -> tuple[tuple[float, float], ...]:
    cx = x
    cy = y
    radians = math.radians(rotation)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    corners = (
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    )
    rotated: list[tuple[float, float]] = []
    for px, py in corners:
        dx = px - cx
        dy = py - cy
        rotated.append((cx + dx * cos_v - dy * sin_v, cy + dx * sin_v + dy * cos_v))
    return tuple(rotated)


def _extract_image_frame(root: ET.Element, alp_path: Path) -> ImageFrame | None:
    image = next(root.iter("Image"), None)
    if image is None:
        return None

    image_name: str | None = None
    for ref in image.iter("ImageResourceReference"):
        class_name = ref.find("ClassName")
        if class_name is not None and class_name.text:
            image_name = class_name.text.strip()
            break

    image_path = alp_path.parent / image_name if image_name else None
    return ImageFrame(
        path=str(image_path) if image_path is not None and image_path.exists() else None,
        x=_float_text(image, "X"),
        y=_float_text(image, "Y"),
        width=_float_text(image, "Width"),
        height=_float_text(image, "Height"),
    )


def _to_grid(point: tuple[float, float], frame: ImageFrame, nx: int, ny: int) -> tuple[float, float]:
    x, y = point
    gx = (x - frame.x) / frame.width * nx
    gy = (y - frame.y) / frame.height * ny
    return gx, gy


def _grid_bbox(grid_corners: tuple[tuple[float, float], ...], nx: int, ny: int) -> tuple[int, int, int, int]:
    xs = [point[0] for point in grid_corners]
    ys = [point[1] for point in grid_corners]
    x0 = max(0, min(nx - 1, int(math.floor(min(xs)))))
    y0 = max(0, min(ny - 1, int(math.floor(min(ys)))))
    x1 = max(x0 + 1, min(nx, int(math.ceil(max(xs)))))
    y1 = max(y0 + 1, min(ny, int(math.ceil(max(ys)))))
    return x0, x1, y0, y1


def _grid_bbox_from_points(grid_points: tuple[tuple[float, float], ...], nx: int, ny: int) -> tuple[int, int, int, int]:
    if not grid_points:
        return 0, 1, 0, 1
    return _grid_bbox(grid_points, nx, ny)


def _format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _format_points(points: tuple[tuple[float, float], ...]) -> str:
    return "[" + ", ".join(f"[{_format_number(x)}, {_format_number(y)}]" for x, y in points) + "]"


def _format_string_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(f'"{value}"' for value in values) + "]"


def _longest_edge_axis(points: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    best_axis = (1.0, 0.0)
    best_length2 = 0.0
    for start, end in zip(points, points[1:] + points[:1]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length2 = dx * dx + dy * dy
        if length2 > best_length2:
            best_axis = (dx, dy)
            best_length2 = length2
    norm = math.hypot(best_axis[0], best_axis[1])
    if norm <= 1.0e-12:
        return 1.0, 0.0
    return best_axis[0] / norm, best_axis[1] / norm


def _format_axis(points: tuple[tuple[float, float], ...]) -> str:
    axis_x, axis_y = _longest_edge_axis(points)
    return f"[{_format_number(axis_x)}, {_format_number(axis_y)}]"


def _channel_suffix(name: str) -> str | None:
    match = re.fullmatch(r"channel_(\d+)", name.lower())
    return match.group(1) if match else None


def _region_center(record: RegionRecord) -> tuple[float, float]:
    xs = [point[0] for point in record.grid_corners]
    ys = [point[1] for point in record.grid_corners]
    return float(np.mean(xs)), float(np.mean(ys))


def _region_center_distance(left: RegionRecord, right: RegionRecord) -> float:
    left_x, left_y = _region_center(left)
    right_x, right_y = _region_center(right)
    return math.hypot(left_x - right_x, left_y - right_y)


def _pre_channel_inference_radius(channel: RegionRecord) -> float:
    x0, x1, y0, y1 = channel.grid_bbox
    diagonal = math.hypot(max(1, x1 - x0), max(1, y1 - y0))
    return max(8.0, diagonal * 1.8)


def _channel_waypoint_name(suffix: str, available_region_names: set[str]) -> str | None:
    prefixes = (
        f"waypoint_channel{suffix}_",
        f"waypoint_channel_{suffix}_",
    )
    candidates = [
        name
        for name in available_region_names
        if any(name.lower().startswith(prefix) for prefix in prefixes)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_natural_key)[0]


def _channel_route_regions(records: list[RegionRecord], available_region_names: set[str]) -> list[ChannelRouteRegion]:
    routes: list[ChannelRouteRegion] = []
    goal_candidates = sorted((record for record in records if record.kind == "goal"), key=lambda item: _natural_key(item.name))
    used_inferred_pre_regions: set[str] = set()
    channels = sorted((record for record in records if record.kind == "channel"), key=lambda item: _natural_key(item.name))
    for channel in channels:
        suffix = _channel_suffix(channel.name)
        if suffix is None:
            continue
        post_name = f"post_channel_{suffix}"
        if post_name not in available_region_names:
            continue

        pre_name: str | None = None
        explicit_pre_name = f"pre_channel_{suffix}"
        if explicit_pre_name in available_region_names:
            pre_name = explicit_pre_name
        else:
            best_candidate: RegionRecord | None = None
            best_distance = math.inf
            for candidate in goal_candidates:
                if candidate.name in used_inferred_pre_regions:
                    continue
                distance = _region_center_distance(channel, candidate)
                if distance < best_distance:
                    best_candidate = candidate
                    best_distance = distance
            if best_candidate is not None and best_distance <= _pre_channel_inference_radius(channel):
                pre_name = best_candidate.name
                used_inferred_pre_regions.add(pre_name)

        routes.append(
            ChannelRouteRegion(
                channel_name=channel.name,
                suffix=suffix,
                post_name=post_name,
                pre_name=pre_name,
                waypoint_name=_channel_waypoint_name(suffix, available_region_names),
            )
        )
    return routes


def _append_channel_split_routes(
    routes_lines: list[str],
    *,
    channel_routes: list[ChannelRouteRegion],
    exit_region: str,
    channel_route_mode: str = DEFAULT_CHANNEL_ROUTE_MODE,
) -> None:
    if channel_route_mode not in {"pass_through", "direct_merge", "post_stage"}:
        raise ValueError(f"Unsupported channel route mode: {channel_route_mode}")

    probability = 1.0 / len(channel_routes)
    routes_lines.extend(
        [
            "[[stages]]",
            'stage_id = "to_decision_channel_split"',
            "group_key = [1, 1]",
            'goal_region = "decision_channel_split"',
            'decision_region = "decision_channel_split"',
            "kappa = 2.0",
            "",
        ]
    )
    for channel_route in channel_routes:
        target_stage_id = f"to_pre_channel_{channel_route.suffix}" if channel_route.pre_name else f"to_channel_{channel_route.suffix}"
        routes_lines.extend(
            [
                "[[stages.targets]]",
                f'stage_id = "{target_stage_id}"',
                f"probability = {_format_number(probability)}",
                "",
            ]
        )

    group_index = 2
    for channel_route in channel_routes:
        if channel_route.pre_name is None:
            continue
        if channel_route_mode == "pass_through":
            goal_line = f'goal_region = "{channel_route.channel_name}"'
            decision_line = f"decision_regions = {_format_string_list((channel_route.pre_name, channel_route.channel_name))}"
        else:
            goal_line = f'goal_region = "{channel_route.pre_name}"'
            decision_line = f'decision_region = "{channel_route.pre_name}"'
        routes_lines.extend(
            [
                "[[stages]]",
                f'stage_id = "to_pre_channel_{channel_route.suffix}"',
                f"group_key = [1, {group_index}]",
                goal_line,
                decision_line,
                f'next_stage = "to_channel_{channel_route.suffix}"',
                'transition_direction = "inherit_target"',
                "",
            ]
        )
        group_index += 1

    for channel_route in channel_routes:
        if channel_route_mode == "post_stage":
            goal_line = f'goal_region = "{channel_route.channel_name}"'
            decision_line = f'decision_region = "{channel_route.channel_name}"'
            next_stage = f"post_channel_{channel_route.suffix}"
        elif channel_route_mode == "direct_merge":
            goal_line = f'goal_region = "{channel_route.channel_name}"'
            decision_line = f'decision_region = "{channel_route.channel_name}"'
            next_stage = "to_merge_after_channels"
        else:
            if channel_route.waypoint_name is not None:
                goal_line = f'goal_region = "{channel_route.waypoint_name}"'
                decision_line = f'decision_region = "{channel_route.waypoint_name}"'
            else:
                goal_line = 'goal_region = "merge_after_channels"'
                decision_line = (
                    "decision_regions = "
                    f"{_format_string_list((channel_route.channel_name, channel_route.post_name))}"
                )
            next_stage = "to_merge_after_channels"
        routes_lines.extend(
            [
                "[[stages]]",
                f'stage_id = "to_channel_{channel_route.suffix}"',
                f"group_key = [1, {group_index}]",
                goal_line,
                decision_line,
                f'next_stage = "{next_stage}"',
                'transition_direction = "inherit_target"',
                "",
            ]
        )
        group_index += 1

    if channel_route_mode == "post_stage":
        for channel_route in channel_routes:
            routes_lines.extend(
                [
                    "[[stages]]",
                    f'stage_id = "post_channel_{channel_route.suffix}"',
                    f"group_key = [1, {group_index}]",
                    f'goal_region = "{channel_route.post_name}"',
                    f'decision_region = "{channel_route.post_name}"',
                    'next_stage = "to_merge_after_channels"',
                    'transition_direction = "inherit_target"',
                    "",
                ]
            )
            group_index += 1

    if channel_route_mode == "pass_through":
        merge_goal_line = f'goal_region = "{exit_region}"'
        merge_decision_line = f"decision_regions = {_format_string_list(('merge_after_channels', exit_region))}"
    else:
        merge_goal_line = 'goal_region = "merge_after_channels"'
        merge_decision_line = 'decision_region = "merge_after_channels"'

    routes_lines.extend(
        [
            "[[stages]]",
            'stage_id = "to_merge_after_channels"',
            f"group_key = [1, {group_index}]",
            merge_goal_line,
            merge_decision_line,
            'next_stage = "to_exits"',
            'transition_direction = "inherit_target"',
            "",
            "[[stages]]",
            'stage_id = "to_exits"',
            f"group_key = [1, {group_index + 1}]",
            f'goal_region = "{exit_region}"',
            f'sink_region = "{exit_region}"',
            "",
        ]
    )


def extract_regions(alp_path: Path, *, nx: int, ny: int) -> tuple[ImageFrame, list[RegionRecord]]:
    root = ET.parse(alp_path).getroot()
    frame = _extract_image_frame(root, alp_path)
    if frame is None:
        raise ValueError(f"No Image element found in {alp_path}")

    records: list[RegionRecord] = []
    for node in root.iter("RectangleNode"):
        name = _text(node, "Name")
        if not name:
            continue
        kind = _classify(name)
        if kind == "other":
            continue

        x = _float_text(node, "X")
        y = _float_text(node, "Y")
        width = _float_text(node, "Width")
        height = _float_text(node, "Height")
        rotation = _float_text(node, "Rotation")
        if width <= 0.0 or height <= 0.0:
            continue

        anylogic_corners = _rotated_corners(x, y, width, height, rotation)
        grid_corners = tuple(_to_grid(point, frame, nx, ny) for point in anylogic_corners)
        records.append(
            RegionRecord(
                name=name,
                kind=kind,
                x=x,
                y=y,
                width=width,
                height=height,
                rotation=rotation,
                anylogic_corners=anylogic_corners,
                grid_bbox=_grid_bbox(grid_corners, nx, ny),
                grid_corners=grid_corners,
            )
        )

    records.sort(key=lambda item: ({"channel": 0, "initial": 1, "goal": 2, "exit": 3}.get(item.kind, 9), _natural_key(item.name)))
    return frame, records


def _main_wall_points(points: list[tuple[float, float]], *, point_mode: str) -> tuple[tuple[float, float], ...]:
    if point_mode == "all":
        return tuple(points)
    if point_mode != "anchors":
        raise ValueError(f"Unsupported wall point mode: {point_mode}")
    if len(points) <= 3:
        return tuple(points)
    anchors = points[::3]
    if len(anchors) < 2:
        return tuple(points)
    return tuple(anchors)


def extract_wall_paths(alp_path: Path, *, nx: int, ny: int, wall_point_mode: str = "anchors") -> tuple[ImageFrame, list[WallPathRecord]]:
    root = ET.parse(alp_path).getroot()
    frame = _extract_image_frame(root, alp_path)
    if frame is None:
        raise ValueError(f"No Image element found in {alp_path}")

    walls: list[WallPathRecord] = []
    for node in root.iter("Path"):
        name = _text(node, "Name")
        if not name:
            continue
        points_node = node.find("Points")
        if points_node is None:
            continue

        origin_x = _float_text(node, "X")
        origin_y = _float_text(node, "Y")
        line_width = max(_float_text(node, "LineWidth", 1.0), 1.0)
        anylogic_points: list[tuple[float, float]] = []
        for point in points_node.findall("Point"):
            anylogic_points.append(
                (
                    origin_x + _float_text(point, "X"),
                    origin_y + _float_text(point, "Y"),
                )
            )
        if len(anylogic_points) < 2:
            continue

        wall_points = _main_wall_points(anylogic_points, point_mode=wall_point_mode)
        if len(wall_points) < 2:
            continue

        grid_points = tuple(_to_grid(point, frame, nx, ny) for point in wall_points)
        walls.append(
            WallPathRecord(
                name=name,
                kind="wall",
                point_mode=wall_point_mode,
                raw_point_count=len(anylogic_points),
                line_width=line_width,
                grid_line_width=max(1.0, line_width * nx / frame.width),
                anylogic_points=wall_points,
                grid_points=grid_points,
                grid_bbox=_grid_bbox_from_points(grid_points, nx, ny),
            )
        )

    walls.sort(key=lambda item: _natural_key(item.name))
    return frame, walls


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
        return math.hypot(x - sx, y - sy)
    t = max(0.0, min(1.0, ((x - sx) * dx + (y - sy) * dy) / length2))
    px = sx + t * dx
    py = sy + t * dy
    return math.hypot(x - px, y - py)


def _mark_segment(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], *, width: float) -> None:
    radius = width / 2.0
    ny, nx = mask.shape
    x0 = max(0, int(math.floor(min(start[0], end[0]) - radius)))
    x1 = min(nx, int(math.ceil(max(start[0], end[0]) + radius)) + 1)
    y0 = max(0, int(math.floor(min(start[1], end[1]) - radius)))
    y1 = min(ny, int(math.ceil(max(start[1], end[1]) + radius)) + 1)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if _distance_to_segment(float(xx), float(yy), start, end) <= radius + 1.0e-9:
                mask[yy, xx] = True


def _mask_to_obstacle_regions(mask: np.ndarray, *, source_path: str) -> list[WallObstacleRegion]:
    records: list[WallObstacleRegion] = []
    safe_source = re.sub(r"[^0-9A-Za-z_]+", "_", source_path).strip("_") or "path"
    ny, nx = mask.shape
    counter = 0
    for y in range(ny):
        x = 0
        while x < nx:
            while x < nx and not mask[y, x]:
                x += 1
            if x >= nx:
                break
            x0 = x
            while x < nx and mask[y, x]:
                x += 1
            x1 = x
            records.append(
                WallObstacleRegion(
                    name=f"wall_{safe_source}_{counter:04d}",
                    source_path=source_path,
                    x0=x0,
                    x1=x1,
                    y0=y,
                    y1=y + 1,
                )
            )
            counter += 1
    return records


def build_wall_obstacle_regions(walls: list[WallPathRecord], *, nx: int, ny: int) -> list[WallObstacleRegion]:
    obstacles: list[WallObstacleRegion] = []
    for wall in walls:
        mask = np.zeros((ny, nx), dtype=bool)
        for start, end in zip(wall.grid_points[:-1], wall.grid_points[1:]):
            _mark_segment(mask, start, end, width=wall.grid_line_width)
        obstacles.extend(_mask_to_obstacle_regions(mask, source_path=wall.name))
    return obstacles


def write_records(
    output_dir: Path,
    frame: ImageFrame,
    records: list[RegionRecord],
    *,
    wall_paths: list[WallPathRecord] | None = None,
    wall_obstacles: list[WallObstacleRegion] | None = None,
    nx: int,
    ny: int,
    dx: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_paths = wall_paths or []
    wall_obstacles = wall_obstacles or []
    payload = {
        "source_image": asdict(frame),
        "grid": {"nx": nx, "ny": ny, "dx": dx},
        "regions": [asdict(record) for record in records],
        "wall_paths": [asdict(record) for record in wall_paths],
        "wall_obstacles": [asdict(record) for record in wall_obstacles],
        "notes": [
            "AnyLogic rectangles are rendered as rotated polygons in preview images.",
            "Draft TOML writes AnyLogic RectangleNode objects as polygon regions using their transformed grid corners.",
            "AnyLogic Path objects are written as [[walls]] polyline definitions in scene.toml.",
            "wall_obstacles.csv is a preview-only raster diagnostic and is not required by scene.toml.",
            "Y coordinates are not flipped; both AnyLogic drawing space and numpy grid preview use downward-positive screen coordinates.",
        ],
    }
    (output_dir / "regions_extracted.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "regions_extracted.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "kind", "x", "y", "width", "height", "rotation", "x0", "x1", "y0", "y1"])
        for record in records:
            x0, x1, y0, y1 = record.grid_bbox
            writer.writerow([record.name, record.kind, record.x, record.y, record.width, record.height, record.rotation, x0, x1, y0, y1])

    with (output_dir / "wall_paths_extracted.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "point_mode", "raw_point_count", "point_count", "line_width", "grid_line_width", "x0", "x1", "y0", "y1"])
        for wall in wall_paths:
            x0, x1, y0, y1 = wall.grid_bbox
            writer.writerow([wall.name, wall.point_mode, wall.raw_point_count, len(wall.grid_points), wall.line_width, wall.grid_line_width, x0, x1, y0, y1])

    with (output_dir / "wall_obstacles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "source_path", "x0", "x1", "y0", "y1"])
        for obstacle in wall_obstacles:
            writer.writerow([obstacle.name, obstacle.source_path, obstacle.x0, obstacle.x1, obstacle.y0, obstacle.y1])


def write_draft_toml(
    output_dir: Path,
    records: list[RegionRecord],
    *,
    wall_paths: list[WallPathRecord] | None = None,
    wall_obstacles: list[WallObstacleRegion] | None = None,
    nx: int,
    ny: int,
    dx: float,
    channel_route_mode: str = DEFAULT_CHANNEL_ROUTE_MODE,
) -> None:
    wall_paths = wall_paths or []
    wall_obstacles = wall_obstacles or []
    obstacle_names = [obstacle.name for obstacle in wall_obstacles]
    scene_lines = [
        "# Auto-generated from AnyLogic RectangleNode and Path objects.",
        "# RectangleNode regions are written as polygons; Path walls remain polylines.",
        "block_boundaries = true",
    ]
    if obstacle_names:
        scene_lines.append("obstacles = [")
        scene_lines.extend(f'  "{name}",' for name in obstacle_names)
        scene_lines.append("]")
    else:
        scene_lines.append("obstacles = []")
    scene_lines.append("")
    for record in records:
        scene_lines.extend(
            [
                "[[regions]]",
                f'name = "{record.name}"',
                'shape = "polygon"',
                f"points = {_format_points(record.grid_corners)}",
                f"axis = {_format_axis(record.grid_corners)}",
                "",
            ]
        )
    for obstacle in wall_obstacles:
        scene_lines.extend(
            [
                "[[regions]]",
                f'name = "{obstacle.name}"',
                f"x0 = {obstacle.x0}",
                f"x1 = {obstacle.x1}",
                f"y0 = {obstacle.y0}",
                f"y1 = {obstacle.y1}",
                "",
            ]
        )

    for wall in wall_paths:
        scene_lines.extend(
            [
                "[[walls]]",
                f'name = "{wall.name}"',
                'shape = "polyline"',
                f"points = {_format_points(wall.grid_points)}",
                f"width = {_format_number(wall.grid_line_width)}",
                "",
            ]
        )

    exits = [record for record in records if record.kind == "exit"]
    for record in exits:
        scene_lines.extend(["[[exits]]", f'name = "{record.name}"', f'region = "{record.name}"', ""])

    channels = [record for record in records if record.kind == "channel"]
    for record in channels:
        x0, x1, _y0, _y1 = record.grid_bbox
        scene_lines.extend(
            [
                "[[channels]]",
                f'name = "{record.name}"',
                f'region = "{record.name}"',
                f"probe_x = {(x0 + x1) // 2}",
                "",
            ]
        )

    initial = next((record for record in records if record.kind == "initial"), None)
    goals = [record for record in records if record.kind == "goal"]
    exit_region = exits[0].name if exits else (goals[-1].name if goals else "")
    goal_names = [record.name for record in goals] or ([exit_region] if exit_region else [])
    available_region_names = {record.name for record in records}
    channel_routes = _channel_route_regions(records, available_region_names)
    use_channel_split_route = (
        bool(exit_region)
        and "decision_channel_split" in available_region_names
        and "merge_after_channels" in available_region_names
        and bool(channel_routes)
    )
    route_sequence = [
        region_name for region_name in DEFAULT_ROUTE_SEQUENCE if region_name in available_region_names
    ]
    use_default_route_sequence = (
        len(route_sequence) == len(DEFAULT_ROUTE_SEQUENCE)
        and route_sequence[-1] == exit_region
    )
    if use_channel_split_route:
        first_stage_id = "to_decision_channel_split"
    elif use_default_route_sequence:
        first_stage_id = f"to_{route_sequence[0]}"
    else:
        first_stage_id = "main_exit"

    population_lines = [
        "[[initial_groups]]",
        'group_id = "bund_anylogic_initial"',
        f'stage_id = "{first_stage_id}"',
        f'region = "{initial.name if initial else ""}"',
        f"density = {DEFAULT_INITIAL_DENSITY}",
        "",
    ]
    if initial is not None:
        population_lines.extend(
            [
                "[[inflow_groups]]",
                'group_id = "bund_anylogic_inflow"',
                f'stage_id = "{first_stage_id}"',
                f'region = "{initial.name}"',
                f"rate = {DEFAULT_INFLOW_RATE}",
                "time_start = 0.0",
                f"rho_cap = {DEFAULT_INFLOW_RHO_CAP}",
                "",
            ]
        )

    routes_lines = [
        "[case]",
        'case_id = "bund_anylogic_simple_route"',
        'title = "Bund AnyLogic simple route"',
        "",
    ]

    if use_channel_split_route:
        _append_channel_split_routes(
            routes_lines,
            channel_routes=channel_routes,
            exit_region=exit_region,
            channel_route_mode=channel_route_mode,
        )
    elif use_default_route_sequence:
        for index, region_name in enumerate(route_sequence):
            stage_id = f"to_{region_name}"
            routes_lines.extend(
                [
                    "[[stages]]",
                    f'stage_id = "{stage_id}"',
                    f"group_key = [1, {index + 1}]",
                    f'goal_region = "{region_name}"',
                ]
            )
            if index < len(route_sequence) - 1:
                routes_lines.extend(
                    [
                        f'next_stage = "to_{route_sequence[index + 1]}"',
                        'transition_direction = "inherit_target"',
                    ]
                )
            else:
                routes_lines.append(f'sink_region = "{region_name}"')
            routes_lines.append("")
    elif len(goal_names) == 1:
        routes_lines.extend(
            [
                "[[stages]]",
                'stage_id = "main_exit"',
                "group_key = [1, 1]",
                f'goal_region = "{goal_names[0]}"',
            ]
        )
        if exit_region:
            routes_lines.append(f'sink_region = "{exit_region}"')
        routes_lines.append("")
    else:
        routes_lines.extend(
            [
                "[[stages]]",
                'stage_id = "main_exit"',
                "group_key = [1, 1]",
                "goal_regions = [" + ", ".join(f'"{name}"' for name in goal_names) + "]",
            ]
        )
        if exit_region:
            routes_lines.append(f'sink_region = "{exit_region}"')
        routes_lines.append("")

    run_lines = [
        "[simulation]",
        f"nx = {nx}",
        f"ny = {ny}",
        f"dx = {dx}",
        "steps = 200",
        "time_horizon = 40.0",
        "vmax = 1.5",
        "rho_max = 5.0",
        "rho_init = 2.2",
        "bellman_every = 1",
        "bellman_f_eps = 0.08",
        "cfl = 0.9",
        "dt_cap = 0.18",
        "save_every = 40",
        "density_contour_levels = 0",
        "",
        "[objective]",
        'name = "bund_anylogic_baseline"',
        "lambda_j1 = 1.0",
        "lambda_j2 = 1.0",
        "lambda_j5 = 1.0",
        "rho_safe = 3.5",
        "use_normalized_terms = true",
        "j1_scale = 1.0",
        "j2_scale = 0.001",
        "j5_scale = 1.0",
        "",
        "[scene]",
        'file = "scene.toml"',
        "",
        "[population]",
        'file = "population.toml"',
        "",
        "[routes]",
        'file = "routes.toml"',
        "",
        "[outputs]",
        'output_root = "../../../results/bund_anylogic"',
        "",
    ]

    (output_dir / "scene.toml").write_text("\n".join(scene_lines), encoding="utf-8")
    (output_dir / "population.toml").write_text("\n".join(population_lines), encoding="utf-8")
    (output_dir / "routes.toml").write_text("\n".join(routes_lines), encoding="utf-8")
    (output_dir / "run.toml").write_text("\n".join(run_lines), encoding="utf-8")


def _draw_label(ax, text: str, points: tuple[tuple[float, float], ...], color: str, *, fontsize: int = 7) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.text(float(np.mean(xs)), float(np.mean(ys)), text, color=color, fontsize=fontsize, ha="center", va="center")


def draw_anylogic_overlay(
    output_path: Path,
    frame: ImageFrame,
    records: list[RegionRecord],
    *,
    wall_paths: list[WallPathRecord] | None = None,
) -> None:
    wall_paths = wall_paths or []
    fig, ax = plt.subplots(figsize=(18, 8), dpi=150)
    if frame.path:
        image = plt.imread(frame.path)
        ax.imshow(image, extent=[frame.x, frame.x + frame.width, frame.y + frame.height, frame.y], alpha=0.72)
    else:
        ax.add_patch(Rectangle((frame.x, frame.y), frame.width, frame.height, facecolor="#f7f7f7", edgecolor="#dddddd"))

    for record in records:
        color = COLORS[record.kind]
        ax.add_patch(Polygon(record.anylogic_corners, closed=True, fill=False, edgecolor=color, linewidth=2.0))
        _draw_label(ax, record.name, record.anylogic_corners, color)

    for wall in wall_paths:
        xs = [point[0] for point in wall.anylogic_points]
        ys = [point[1] for point in wall.anylogic_points]
        ax.plot(xs, ys, color=COLORS["wall"], linewidth=max(1.5, wall.line_width), alpha=0.9)
        ax.text(xs[0], ys[0], wall.name, color=COLORS["wall"], fontsize=7, ha="left", va="bottom")

    ax.set_title("AnyLogic extracted regions and Path walls over source image")
    ax.set_xlim(frame.x, frame.x + frame.width)
    ax.set_ylim(frame.y + frame.height, frame.y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#ffffff", linewidth=0.25, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def draw_grid_overlay(
    output_path: Path,
    records: list[RegionRecord],
    *,
    wall_paths: list[WallPathRecord] | None = None,
    wall_obstacles: list[WallObstacleRegion] | None = None,
    nx: int,
    ny: int,
) -> None:
    wall_paths = wall_paths or []
    wall_obstacles = wall_obstacles or []
    dpi = 200
    fig, ax = plt.subplots(figsize=(max(nx / dpi, 1.0), max(ny / dpi, 1.0)), dpi=dpi)
    ax.set_position([0, 0, 1, 1])
    ax.imshow(np.ones((ny, nx, 3)), extent=[0, nx, ny, 0])
    if wall_obstacles:
        wall_mask = np.zeros((ny, nx), dtype=float)
        for obstacle in wall_obstacles:
            wall_mask[obstacle.y0 : obstacle.y1, obstacle.x0 : obstacle.x1] = 1.0
        wall_mask = np.ma.masked_where(wall_mask == 0.0, wall_mask)
        ax.imshow(wall_mask, cmap="gray_r", alpha=0.34, extent=[0, nx, ny, 0], vmin=0.0, vmax=1.0)
    for record in records:
        color = COLORS[record.kind]
        ax.add_patch(Polygon(record.grid_corners, closed=True, fill=True, facecolor=color, alpha=0.12, edgecolor=color, linewidth=1.2))
    for wall in wall_paths:
        xs = [point[0] for point in wall.grid_points]
        ys = [point[1] for point in wall.grid_points]
        ax.plot(xs, ys, color=COLORS["wall"], linewidth=1.0, alpha=0.92)
    ax.set_xlim(0, nx)
    ax.set_ylim(ny, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(output_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert AnyLogic RectangleNode regions into draft CrowdModels scene files and previews.")
    parser.add_argument("--alp", type=Path, default=script_dir / "BundScene.alp")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "converted")
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--dx", type=float, default=DEFAULT_DX)
    parser.add_argument(
        "--wall-point-mode",
        choices=("anchors", "all"),
        default="anchors",
        help="anchors uses AnyLogic Path anchor points as solid walls; all preserves every XML point, including control points.",
    )
    parser.add_argument(
        "--channel-route-mode",
        choices=("pass_through", "direct_merge", "post_stage"),
        default=DEFAULT_CHANNEL_ROUTE_MODE,
        help="pass_through decouples navigation goals from handoff regions; direct_merge skips post_channel_i stages; post_stage restores the legacy channel_i -> post_channel_i route chain.",
    )
    args = parser.parse_args(argv)

    alp_path = args.alp.resolve()
    output_dir = args.output_dir.resolve()
    if not alp_path.exists():
        raise FileNotFoundError(alp_path)
    if args.nx <= 0 or args.ny <= 0:
        raise ValueError("--nx and --ny must be positive")

    frame, records = extract_regions(alp_path, nx=args.nx, ny=args.ny)
    _wall_frame, wall_paths = extract_wall_paths(alp_path, nx=args.nx, ny=args.ny, wall_point_mode=args.wall_point_mode)
    wall_obstacles = build_wall_obstacle_regions(wall_paths, nx=args.nx, ny=args.ny)
    if not records:
        raise ValueError(f"No named AnyLogic regions found in {alp_path}")

    write_records(
        output_dir,
        frame,
        records,
        wall_paths=wall_paths,
        wall_obstacles=wall_obstacles,
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
    )
    write_draft_toml(
        output_dir,
        records,
        wall_paths=wall_paths,
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        channel_route_mode=args.channel_route_mode,
    )
    draw_anylogic_overlay(output_dir / "anylogic_overlay.png", frame, records, wall_paths=wall_paths)
    draw_grid_overlay(
        output_dir / "grid_overlay.png",
        records,
        wall_paths=wall_paths,
        wall_obstacles=wall_obstacles,
        nx=args.nx,
        ny=args.ny,
    )

    manifest = {
        "alp": str(alp_path),
        "output_dir": str(output_dir),
        "region_count": len(records),
        "wall_path_count": len(wall_paths),
        "wall_point_mode": args.wall_point_mode,
        "channel_route_mode": args.channel_route_mode,
        "wall_obstacle_region_count": len(wall_obstacles),
        "counts_by_kind": {kind: sum(1 for record in records if record.kind == kind) for kind in sorted(COLORS)},
        "outputs": {
            "regions_json": str(output_dir / "regions_extracted.json"),
            "regions_csv": str(output_dir / "regions_extracted.csv"),
            "wall_paths_csv": str(output_dir / "wall_paths_extracted.csv"),
            "wall_obstacles_csv": str(output_dir / "wall_obstacles.csv"),
            "anylogic_overlay": str(output_dir / "anylogic_overlay.png"),
            "grid_overlay": str(output_dir / "grid_overlay.png"),
            "draft_scene": str(output_dir / "scene.toml"),
            "draft_population": str(output_dir / "population.toml"),
            "draft_routes": str(output_dir / "routes.toml"),
            "draft_run": str(output_dir / "run.toml"),
        },
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
