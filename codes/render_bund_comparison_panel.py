from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from render_refined_density_heatmap import (
    _draw_vector_wall_overlay,
    _load_manifest,
    _sorted_entries,
    load_density_frame,
    load_wall_polylines,
    load_walkable_mask,
    make_density_cmap,
    make_density_norm,
    resolve_fields_dir,
    resolve_scene_path,
    smooth_density,
    upsample_density,
    upsample_mask,
)
from crowd_bellman.spec.scene_spec import WallSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_NAME = "bund_comparison_density_panel.png"
CASE_ORDER = ("uncontrolled", "controlled")
CASE_LABELS = {
    "uncontrolled": "Uncontrolled",
    "controlled": "Controlled",
}


@dataclass(frozen=True)
class CaseData:
    key: str
    label: str
    case_dir: Path
    fields_dir: Path
    entries: list[dict[str, object]]
    walkable: np.ndarray | None
    walls: tuple[WallSpec, ...]


def parse_steps(raw_values: Iterable[str]) -> list[int]:
    steps: list[int] = []
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            value = part.strip()
            if not value:
                continue
            steps.append(int(value))
    return steps


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_fields(path: Path) -> bool:
    try:
        resolve_fields_dir(path)
    except (FileNotFoundError, ValueError):
        return False
    return True


def resolve_case_dir(comparison_dir: Path, case_key: str) -> Path:
    manifest_path = comparison_dir / "comparison_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        cases = manifest.get("cases", {})
        if isinstance(cases, dict) and case_key in cases:
            candidate = Path(str(cases[case_key]))
            if candidate.exists() and _has_fields(candidate):
                return candidate

    candidates = (
        comparison_dir / case_key / case_key,
        comparison_dir / case_key,
    )
    for candidate in candidates:
        if candidate.exists() and _has_fields(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {case_key!r} case with fields under {comparison_dir}")


def load_case_data(comparison_dir: Path, case_key: str) -> CaseData:
    case_dir = resolve_case_dir(comparison_dir, case_key)
    fields_dir = resolve_fields_dir(case_dir)
    manifest = _load_manifest(fields_dir)
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Manifest files must be a list: {fields_dir / 'fields_manifest.json'}")
    entries = _sorted_entries(entry for entry in raw_entries if isinstance(entry, dict))
    walkable = load_walkable_mask(fields_dir, manifest)
    scene_path = resolve_scene_path(None, fields_dir)
    walls = load_wall_polylines(scene_path)
    return CaseData(
        key=case_key,
        label=CASE_LABELS[case_key],
        case_dir=case_dir,
        fields_dir=fields_dir,
        entries=entries,
        walkable=walkable,
        walls=walls,
    )


def _entry_step(entry: dict[str, object]) -> int:
    if "step" in entry:
        return int(entry["step"])
    return int(Path(str(entry["file"])).stem.rsplit("_", 1)[-1])


def select_entry_for_step(entries: list[dict[str, object]], step: int, *, nearest: bool) -> dict[str, object]:
    if not entries:
        raise ValueError("No field entries available")
    exact = [entry for entry in entries if _entry_step(entry) == int(step)]
    if exact:
        return exact[0]
    available = [_entry_step(entry) for entry in entries]
    if nearest:
        return min(entries, key=lambda entry: abs(_entry_step(entry) - int(step)))
    preview = ", ".join(str(item) for item in available[:20])
    if len(available) > 20:
        preview += ", ..."
    raise ValueError(f"Step {step} not found. Available steps: {preview}")


def crop_bounds_from_walls(
    walls: Iterable[WallSpec],
    *,
    shape: tuple[int, int],
    padding: int,
) -> tuple[int, int, int, int] | None:
    points: list[tuple[float, float]] = []
    for wall in walls:
        points.extend((float(x), float(y)) for x, y in wall.points)
    if not points:
        return None
    values = np.asarray(points, dtype=float)
    ny, nx = shape
    x0 = max(0, int(math.floor(float(np.nanmin(values[:, 0])) - int(padding))))
    x1 = min(nx, int(math.ceil(float(np.nanmax(values[:, 0])) + int(padding))) + 1)
    y0 = max(0, int(math.floor(float(np.nanmin(values[:, 1])) - int(padding))))
    y1 = min(ny, int(math.ceil(float(np.nanmax(values[:, 1])) + int(padding))) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (y0, y1, x0, x1)


def crop_bounds_from_mask(mask: np.ndarray | None, *, padding: int) -> tuple[int, int, int, int] | None:
    if mask is None:
        return None
    active = np.asarray(mask, dtype=bool)
    if not np.any(active):
        return None
    rows, cols = np.where(active)
    y0 = max(0, int(np.min(rows)) - int(padding))
    y1 = min(active.shape[0], int(np.max(rows)) + int(padding) + 1)
    x0 = max(0, int(np.min(cols)) - int(padding))
    x1 = min(active.shape[1], int(np.max(cols)) + int(padding) + 1)
    return (y0, y1, x0, x1)


def union_bounds(bounds_list: Iterable[tuple[int, int, int, int] | None], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    valid = [item for item in bounds_list if item is not None]
    if not valid:
        return (0, shape[0], 0, shape[1])
    y0 = min(item[0] for item in valid)
    y1 = max(item[1] for item in valid)
    x0 = min(item[2] for item in valid)
    x1 = max(item[3] for item in valid)
    return (max(0, y0), min(shape[0], y1), max(0, x0), min(shape[1], x1))


def render_panel_array(
    density: np.ndarray,
    walkable: np.ndarray | None,
    *,
    bounds: tuple[int, int, int, int],
    scale: int,
    smooth_sigma: float,
) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    raw_density = density[y0:y1, x0:x1]
    raw_walkable = walkable[y0:y1, x0:x1] if walkable is not None else None
    upsampled = upsample_density(raw_density, scale=scale)
    upsampled_walkable = upsample_mask(raw_walkable, scale=scale) if raw_walkable is not None else None
    smoothed = smooth_density(upsampled, sigma=smooth_sigma, walkable=upsampled_walkable)
    return np.where(np.isfinite(smoothed), smoothed, 0.0)


def render_comparison_panel(
    comparison_dir: Path,
    *,
    steps: list[int],
    output: Path,
    scale: int,
    smooth_sigma: float,
    vmax: float,
    dpi: int,
    crop_padding: int,
    nearest: bool,
    gamma: float,
    wall_color: str,
    wall_alpha: float,
    wall_linewidth: float | None,
    title_fontsize: float,
    row_label_fontsize: float,
    seconds_per_step: float = 0.0,
) -> Path:
    if len(steps) != 4:
        raise ValueError(f"Expected exactly 4 steps, got {len(steps)}")
    cases = [load_case_data(comparison_dir, key) for key in CASE_ORDER]
    first_shape = None
    for case in cases:
        if case.walkable is not None:
            first_shape = case.walkable.shape
            break
    if first_shape is None:
        first_entry = select_entry_for_step(cases[0].entries, steps[0], nearest=nearest)
        first_shape = load_density_frame(cases[0].fields_dir, first_entry).density.shape

    bounds = union_bounds(
        [
            crop_bounds_from_walls(case.walls, shape=first_shape, padding=crop_padding)
            for case in cases
        ],
        first_shape,
    )
    if bounds == (0, first_shape[0], 0, first_shape[1]):
        bounds = union_bounds(
            [crop_bounds_from_mask(case.walkable, padding=crop_padding) for case in cases],
            first_shape,
        )

    y0, y1, x0, x1 = bounds
    crop_height = max(y1 - y0, 1)
    crop_width = max(x1 - x0, 1)
    aspect = crop_width / crop_height
    tile_height = 2.0
    fig_width = max(10.0, 4 * tile_height * aspect + 1.2)
    fig_height = 2 * tile_height + 0.95
    fig, axes = plt.subplots(2, 4, figsize=(fig_width, fig_height), dpi=dpi)
    cmap = make_density_cmap("low-density", transparent_bad=False)
    norm = make_density_norm(vmin=0.0, vmax=float(vmax), norm_mode="power", gamma=float(gamma))
    last_image = None

    for row_index, case in enumerate(cases):
        for col_index, requested_step in enumerate(steps):
            entry = select_entry_for_step(case.entries, requested_step, nearest=nearest)
            frame = load_density_frame(case.fields_dir, entry, walkable=case.walkable)
            panel_density = render_panel_array(
                frame.density,
                frame.walkable,
                bounds=bounds,
                scale=scale,
                smooth_sigma=smooth_sigma,
            )
            ax = axes[row_index, col_index]
            last_image = ax.imshow(
                panel_density,
                origin="upper",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            _draw_vector_wall_overlay(
                ax,
                case.walls,
                bounds=bounds,
                display_shape=panel_density.shape,
                scale=scale,
                dpi=dpi,
                color=wall_color,
                alpha=wall_alpha,
                linewidth=wall_linewidth,
            )
            if seconds_per_step > 0.0:
                title_text = f"t = {frame.step * seconds_per_step:.0f} s"
            else:
                title_text = f"Step={frame.step}"
            ax.set_title(title_text, fontsize=title_fontsize, pad=5)
            ax.set_axis_off()

    fig.subplots_adjust(left=0.075, right=0.945, bottom=0.06, top=0.91, wspace=0.035, hspace=0.28)
    for row_index, case in enumerate(cases):
        bbox = axes[row_index, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.055,
            y_center,
            case.label,
            ha="right",
            va="center",
            fontsize=row_label_fontsize,
            fontweight="bold",
        )
    if last_image is not None:
        cbar_ax = fig.add_axes([0.958, 0.18, 0.012, 0.62])
        cbar = fig.colorbar(last_image, cax=cbar_ax)
        cbar.set_label("density (ped/m$^2$)", fontsize=max(8.0, row_label_fontsize - 1.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a 2x4 controlled/uncontrolled Bund density comparison panel from saved field data."
    )
    parser.add_argument("comparison_dir", help="bund_control_comparison_* output directory.")
    parser.add_argument("steps", nargs="+", help="Exactly four steps, either space-separated or comma-separated.")
    parser.add_argument("--output", default=None, help=f"Output PNG. Defaults to <comparison_dir>/figures/{DEFAULT_OUTPUT_NAME}.")
    parser.add_argument("--scale", type=int, default=8, help="Density upsampling scale.")
    parser.add_argument("--smooth-sigma", type=float, default=5.0, help="Wall-aware smoothing sigma after upsampling.")
    parser.add_argument("--vmax", type=float, default=6.0, help="Absolute density colorbar maximum.")
    parser.add_argument("--dpi", type=int, default=240, help="Output DPI.")
    parser.add_argument("--crop-padding", type=int, default=10, help="Padding in original grid cells around wall polyline bounds.")
    parser.add_argument("--nearest", action="store_true", help="Use nearest saved field frame when a requested step is absent.")
    parser.add_argument("--gamma", type=float, default=0.42, help="PowerNorm gamma for stronger low-density colors.")
    parser.add_argument("--wall-color", default="#111111", help="Vector wall overlay color.")
    parser.add_argument("--wall-alpha", type=float, default=0.95, help="Vector wall overlay alpha.")
    parser.add_argument("--wall-linewidth", type=float, default=None, help="Fixed wall linewidth in points. Defaults to scene widths.")
    parser.add_argument("--title-fontsize", type=float, default=11.0)
    parser.add_argument("--row-label-fontsize", type=float, default=11.0)
    parser.add_argument(
        "--seconds-per-step",
        type=float,
        default=0.43625,
        help="Physical seconds per simulation step; panel titles show physical time. "
        "Default 0.43625 (Bund: 698 s horizon / 1600 steps). Pass 0 to label by step index.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    comparison_dir = Path(args.comparison_dir)
    if not comparison_dir.is_absolute():
        comparison_dir = (REPO_ROOT / comparison_dir).resolve()
    steps = parse_steps(args.steps)
    output = Path(args.output) if args.output else comparison_dir / "figures" / DEFAULT_OUTPUT_NAME
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()
    written = render_comparison_panel(
        comparison_dir,
        steps=steps,
        output=output,
        scale=args.scale,
        smooth_sigma=args.smooth_sigma,
        vmax=args.vmax,
        dpi=args.dpi,
        crop_padding=args.crop_padding,
        nearest=args.nearest,
        gamma=args.gamma,
        wall_color=args.wall_color,
        wall_alpha=args.wall_alpha,
        wall_linewidth=args.wall_linewidth,
        title_fontsize=args.title_fontsize,
        row_label_fontsize=args.row_label_fontsize,
        seconds_per_step=args.seconds_per_step,
    )
    print(json.dumps({"output": str(written), "steps": steps}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
