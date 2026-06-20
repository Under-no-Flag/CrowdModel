from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from PIL import Image

from crowd_bellman.plotting import DENSITY_CMAP


TRANSPARENT_DENSITY_CMAP = DENSITY_CMAP.copy()
TRANSPARENT_DENSITY_CMAP.set_bad((1.0, 1.0, 1.0, 0.0))


@dataclass(frozen=True)
class DensityFrame:
    step: int
    time: float
    density: np.ndarray
    source_path: Path


def resolve_fields_dir(result_dir: Path | str) -> Path:
    root = Path(result_dir)
    if (root / "fields_manifest.json").exists():
        return root
    if (root / "fields" / "fields_manifest.json").exists():
        return root / "fields"

    candidates = sorted(root.glob("*/fields/fields_manifest.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if len(candidates) > 1:
        candidate_text = ", ".join(str(path.parent) for path in candidates)
        raise ValueError(f"Multiple field-data directories found under {root}: {candidate_text}")
    raise FileNotFoundError(f"Could not find fields_manifest.json under {root}")


def resolve_background_path(explicit_background: Path | str | None, fields_dir: Path) -> Path | None:
    if explicit_background is not None:
        path = Path(explicit_background)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    case_dir = fields_dir.parent
    direct_candidates = (
        case_dir / "grid_overlay.png",
        case_dir.parent / "grid_overlay.png",
    )
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    summary_path = case_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if isinstance(summary, dict):
            scene_path = summary.get("scene_path")
            if scene_path:
                candidate = Path(str(scene_path)).parent / "grid_overlay.png"
                if candidate.exists():
                    return candidate

            snapshot = summary.get("config_snapshot", {})
            if isinstance(snapshot, dict):
                files = snapshot.get("files", {})
                if isinstance(files, dict) and files.get("scene"):
                    candidate = Path(str(files["scene"])).parent / "grid_overlay.png"
                    if candidate.exists():
                        return candidate

    return None


def _load_manifest(fields_dir: Path) -> dict[str, object]:
    manifest_path = fields_dir / "fields_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
    return payload


def _entry_step(entry: dict[str, object]) -> int:
    if "step" in entry:
        return int(entry["step"])
    stem = Path(str(entry["file"])).stem
    return int(stem.rsplit("_", 1)[-1])


def _sorted_entries(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return sorted((dict(entry) for entry in entries), key=_entry_step)


def select_manifest_entries(
    entries: list[dict[str, object]],
    *,
    step: int | None,
    render_all: bool,
) -> list[dict[str, object]]:
    ordered = _sorted_entries(entries)
    if not ordered:
        raise ValueError("Manifest does not contain any field files")
    if render_all:
        return ordered
    if step is None:
        return [ordered[-1]]

    matches = [entry for entry in ordered if _entry_step(entry) == step]
    if not matches:
        available = ", ".join(str(_entry_step(entry)) for entry in ordered)
        raise ValueError(f"Step {step} not found. Available steps: {available}")
    return matches


def load_density_frame(fields_dir: Path, entry: dict[str, object]) -> DensityFrame:
    path = fields_dir / str(entry["file"])
    with np.load(path) as payload:
        if "rho" not in payload.files:
            raise ValueError(f"Field file does not contain rho: {path}")
        density = np.asarray(payload["rho"], dtype=float)
        step = int(payload["step"][0]) if "step" in payload.files else _entry_step(entry)
        time = float(payload["time"][0]) if "time" in payload.files else float(entry.get("time", math.nan))
    if density.ndim != 2:
        raise ValueError(f"rho must be a 2D array, got shape {density.shape} from {path}")
    return DensityFrame(step=step, time=time, density=density, source_path=path)


def parse_percentile_pair(raw: str) -> tuple[float, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("percentiles must use lower,upper form, e.g. 2,98")
    lower = float(parts[0])
    upper = float(parts[1])
    if lower < 0.0 or upper > 100.0 or lower >= upper:
        raise argparse.ArgumentTypeError("percentiles must satisfy 0 <= lower < upper <= 100")
    return (lower, upper)


def density_percentile_limits(
    densities: Iterable[np.ndarray],
    *,
    percentiles: tuple[float, float],
    threshold: float = 1.0e-8,
) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    for density in densities:
        values = np.asarray(density, dtype=float)
        finite = values[np.isfinite(values) & (values > float(threshold))]
        if finite.size:
            samples.append(finite)
    if not samples:
        return (0.0, 1.0)

    merged = np.concatenate(samples)
    vmin, vmax = np.percentile(merged, percentiles)
    vmin = float(max(vmin, 0.0))
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return (vmin, vmax)


def make_density_norm(*, vmin: float, vmax: float, norm_mode: str, gamma: float) -> Normalize:
    if vmax <= vmin:
        vmax = vmin + 1.0
    if norm_mode == "linear":
        return Normalize(vmin=vmin, vmax=vmax, clip=True)
    if norm_mode == "power":
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    raise ValueError(f"Unsupported norm mode: {norm_mode!r}")


def upsample_density(density: np.ndarray, *, scale: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    values = np.asarray(density, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"density must be 2D, got shape {values.shape}")
    if scale == 1:
        return values.copy()

    ny, nx = values.shape
    x_old = np.arange(nx, dtype=float)
    x_new = np.linspace(0.0, float(nx - 1), nx * scale)
    row_interp = np.empty((ny, nx * scale), dtype=float)
    for row_index in range(ny):
        row_interp[row_index, :] = np.interp(x_new, x_old, values[row_index, :])

    y_old = np.arange(ny, dtype=float)
    y_new = np.linspace(0.0, float(ny - 1), ny * scale)
    output = np.empty((ny * scale, nx * scale), dtype=float)
    for col_index in range(row_interp.shape[1]):
        output[:, col_index] = np.interp(y_new, y_old, row_interp[:, col_index])
    return output


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / float(np.sum(kernel))


def _convolve_along_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    if kernel.size == 1:
        return values.copy()
    pad = kernel.size // 2
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    mode = "reflect" if values.shape[axis] > 1 else "edge"
    padded = np.pad(values, pad_width, mode=mode)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), axis, padded)


def smooth_density(density: np.ndarray, *, sigma: float) -> np.ndarray:
    values = np.asarray(density, dtype=float)
    kernel = _gaussian_kernel1d(float(sigma))
    smoothed = _convolve_along_axis(values, kernel, axis=1)
    smoothed = _convolve_along_axis(smoothed, kernel, axis=0)
    smoothed[smoothed < 0.0] = 0.0
    return smoothed


def density_crop_bounds(
    density: np.ndarray,
    *,
    threshold: float = 1.0e-8,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    values = np.asarray(density, dtype=float)
    active = np.isfinite(values) & (values > float(threshold))
    if not np.any(active):
        return (0, values.shape[0], 0, values.shape[1])

    rows, cols = np.where(active)
    y0 = max(0, int(np.min(rows)) - int(padding))
    y1 = min(values.shape[0], int(np.max(rows)) + int(padding) + 1)
    x0 = max(0, int(np.min(cols)) - int(padding))
    x1 = min(values.shape[1], int(np.max(cols)) + int(padding) + 1)
    return (y0, y1, x0, x1)


def crop_density(density: np.ndarray, *, threshold: float = 1.0e-8, padding: int = 8) -> np.ndarray:
    values = np.asarray(density, dtype=float)
    y0, y1, x0, x1 = density_crop_bounds(values, threshold=threshold, padding=padding)
    return values[y0:y1, x0:x1].copy()


def _crop_background_to_density_bounds(
    background: np.ndarray,
    *,
    density_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    bg = np.asarray(background)
    bg_height, bg_width = bg.shape[:2]
    density_height, density_width = density_shape
    y0, y1, x0, x1 = bounds
    by0 = max(0, int(math.floor(y0 / max(density_height, 1) * bg_height)))
    by1 = min(bg_height, int(math.ceil(y1 / max(density_height, 1) * bg_height)))
    bx0 = max(0, int(math.floor(x0 / max(density_width, 1) * bg_width)))
    bx1 = min(bg_width, int(math.ceil(x1 / max(density_width, 1) * bg_width)))
    return bg[by0:by1, bx0:bx1].copy()


def upsample_image(image: np.ndarray, *, scale: int) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return values.copy()

    clipped = np.clip(values, 0.0, 1.0)
    image_u8 = np.asarray(np.rint(clipped * 255.0), dtype=np.uint8)
    pil_image = Image.fromarray(image_u8)
    resampling = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
    resized = pil_image.resize((pil_image.width * scale, pil_image.height * scale), resampling)
    return np.asarray(resized, dtype=float) / 255.0


def resize_image_to_shape(image: np.ndarray, *, target_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    target_height, target_width = target_shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_shape must contain positive dimensions")
    if values.shape[0] == target_height and values.shape[1] == target_width:
        return values.copy()

    clipped = np.clip(values, 0.0, 1.0)
    image_u8 = np.asarray(np.rint(clipped * 255.0), dtype=np.uint8)
    pil_image = Image.fromarray(image_u8)
    resampling = getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)
    resized = pil_image.resize((target_width, target_height), resampling)
    return np.asarray(resized, dtype=float) / 255.0


def _density_overlay_alpha(
    density: np.ndarray,
    *,
    vmax: float | None,
    density_alpha: float,
    overlay_threshold: float,
) -> np.ndarray:
    values = np.asarray(density, dtype=float)
    alpha = np.zeros_like(values, dtype=float)
    visible = np.isfinite(values) & (values > overlay_threshold)
    if not np.any(visible):
        return alpha
    scale = max((float(vmax) if vmax is not None else float(np.nanmax(values))) * 0.35, 1.0e-9)
    alpha[visible] = float(density_alpha) * np.clip((values[visible] - overlay_threshold) / scale, 0.0, 1.0)
    return alpha


def render_density_heatmap(
    frame: DensityFrame,
    output_path: Path,
    *,
    scale: int = 4,
    smooth_sigma: float = 1.0,
    vmax: float | None = 5.0,
    color_limits: tuple[float, float] | None = None,
    norm_mode: str = "power",
    gamma: float = 0.55,
    dpi: int = 220,
    origin: str = "upper",
    colorbar: bool = True,
    crop: bool = True,
    crop_threshold: float = 1.0e-8,
    crop_padding: int = 8,
    background_path: Path | None = None,
    background_alpha: float = 1.0,
    density_alpha: float = 0.82,
    overlay_threshold: float = 0.05,
) -> None:
    bounds = (
        density_crop_bounds(frame.density, threshold=crop_threshold, padding=crop_padding)
        if crop
        else (0, frame.density.shape[0], 0, frame.density.shape[1])
    )
    y0, y1, x0, x1 = bounds
    raw_density = frame.density[y0:y1, x0:x1]
    density = upsample_density(raw_density, scale=scale)
    density = smooth_density(density, sigma=smooth_sigma)
    display_vmin, display_vmax = color_limits if color_limits is not None else (0.0, float(vmax or np.nanmax(density)))
    density_norm = make_density_norm(vmin=display_vmin, vmax=display_vmax, norm_mode=norm_mode, gamma=gamma)
    background = None
    if background_path is not None:
        background_raw = plt.imread(background_path)
        background_crop = _crop_background_to_density_bounds(
            background_raw,
            density_shape=frame.density.shape,
            bounds=bounds,
        )
        background = resize_image_to_shape(background_crop, target_shape=density.shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = density.shape
    aspect = float(width) / max(float(height), 1.0)
    fig_width = min(max(7.0, 5.0 * aspect), 15.0)
    fig_height = max(fig_width / max(aspect, 1.0e-9), 3.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)

    if background is not None:
        ax.imshow(background, origin=origin, interpolation="nearest", aspect="equal", alpha=background_alpha)
        density_for_plot = density.copy()
        density_for_plot[density_for_plot <= overlay_threshold] = np.nan
        cmap = TRANSPARENT_DENSITY_CMAP
        alpha = _density_overlay_alpha(
            density,
            vmax=display_vmax,
            density_alpha=density_alpha,
            overlay_threshold=overlay_threshold,
        )
    else:
        density_for_plot = density
        cmap = DENSITY_CMAP
        alpha = 1.0

    image = ax.imshow(
        density_for_plot,
        origin=origin,
        cmap=cmap,
        norm=density_norm,
        interpolation="bicubic",
        aspect="equal",
        alpha=alpha,
    )
    ax.set_axis_off()
    if colorbar:
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
        cbar.set_label("density")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def render_from_result_dir(
    result_dir: Path,
    *,
    output_dir: Path | None = None,
    output: Path | None = None,
    step: int | None = None,
    render_all: bool = False,
    scale: int = 4,
    smooth_sigma: float = 1.0,
    vmax: float | None = 5.0,
    color_scale: str = "sequence-percentile",
    color_percentiles: tuple[float, float] = (2.0, 98.0),
    norm_mode: str = "power",
    gamma: float = 0.55,
    dpi: int = 220,
    origin: str = "upper",
    colorbar: bool = True,
    crop: bool = True,
    crop_threshold: float = 1.0e-8,
    crop_padding: int = 8,
    background: Path | None = None,
    no_background: bool = False,
    background_alpha: float = 1.0,
    density_alpha: float = 0.82,
    overlay_threshold: float = 0.05,
) -> list[Path]:
    if output is not None and render_all:
        raise ValueError("--output can only be used for single-frame rendering")

    fields_dir = resolve_fields_dir(result_dir)
    background_path = None if no_background else resolve_background_path(background, fields_dir)
    manifest = _load_manifest(fields_dir)
    raw_entries = manifest.get("files")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Manifest files must be a list: {fields_dir / 'fields_manifest.json'}")
    entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    selected_entries = select_manifest_entries(entries, step=step, render_all=render_all)
    sequence_color_limits = None
    if color_scale == "sequence-percentile":
        sequence_color_limits = density_percentile_limits(
            (load_density_frame(fields_dir, entry).density for entry in _sorted_entries(entries)),
            percentiles=color_percentiles,
            threshold=overlay_threshold,
        )
    elif color_scale not in {"absolute", "frame-percentile"}:
        raise ValueError(f"Unsupported color scale: {color_scale!r}")

    target_dir = output_dir if output_dir is not None else fields_dir.parent / "refined_density"
    written: list[Path] = []
    for entry in selected_entries:
        frame = load_density_frame(fields_dir, entry)
        if color_scale == "absolute":
            frame_color_limits = (0.0, float(vmax if vmax is not None else np.nanmax(frame.density)))
        elif color_scale == "frame-percentile":
            frame_color_limits = density_percentile_limits(
                [frame.density],
                percentiles=color_percentiles,
                threshold=overlay_threshold,
            )
        else:
            assert sequence_color_limits is not None
            frame_color_limits = sequence_color_limits
        output_path = output if output is not None else target_dir / f"density_step_{frame.step:04d}.png"
        assert output_path is not None
        render_density_heatmap(
            frame,
            output_path,
            scale=scale,
            smooth_sigma=smooth_sigma,
            vmax=vmax,
            color_limits=frame_color_limits,
            norm_mode=norm_mode,
            gamma=gamma,
            dpi=dpi,
            origin=origin,
            colorbar=colorbar,
            crop=crop,
            crop_threshold=crop_threshold,
            crop_padding=crop_padding,
            background_path=background_path,
            background_alpha=background_alpha,
            density_alpha=density_alpha,
            overlay_threshold=overlay_threshold,
        )
        written.append(output_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render refined density heatmaps from saved field_step_XXXX.npz files with optional scene structure background."
    )
    parser.add_argument("result_dir", help="Case output dir, fields dir, or output root containing one case with fields data.")
    parser.add_argument("--output-dir", default=None, help="Directory for rendered PNG files. Defaults to <case>/refined_density.")
    parser.add_argument("--output", default=None, help="PNG path for a single rendered frame.")
    parser.add_argument("--step", type=int, default=None, help="Exact saved step to render. Defaults to the latest saved frame.")
    parser.add_argument("--all", dest="render_all", action="store_true", help="Render every saved density frame.")
    parser.add_argument("--scale", type=int, default=4, help="Integer upsampling scale applied before rendering.")
    parser.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian smoothing sigma in upsampled pixels.")
    parser.add_argument("--vmax", type=float, default=5.0, help="Density color scale maximum when --color-scale absolute.")
    parser.add_argument(
        "--color-scale",
        choices=("absolute", "frame-percentile", "sequence-percentile"),
        default="sequence-percentile",
        help="Density color scaling. sequence-percentile keeps all selected frames comparable while using relative limits.",
    )
    parser.add_argument("--color-percentiles", default="2,98", help="Percentile limits for relative color scaling, e.g. 2,98.")
    parser.add_argument("--norm", choices=("linear", "power"), default="power", help="Color normalization mode.")
    parser.add_argument("--gamma", type=float, default=0.55, help="PowerNorm gamma. Values below 1 brighten low densities.")
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI.")
    parser.add_argument("--origin", choices=("upper", "lower"), default="upper", help="Image origin orientation.")
    parser.add_argument("--no-colorbar", action="store_true", help="Do not draw the density colorbar.")
    parser.add_argument("--no-crop", action="store_true", help="Render the full grid instead of cropping around active density.")
    parser.add_argument("--crop-threshold", type=float, default=1.0e-8, help="Density threshold used by active-region cropping.")
    parser.add_argument("--crop-padding", type=int, default=8, help="Raw-grid cells retained around the active density crop.")
    parser.add_argument("--background", default=None, help="Optional grid_overlay.png path. Defaults to auto-detect from case summary.")
    parser.add_argument("--no-background", action="store_true", help="Disable scene structure background overlay.")
    parser.add_argument("--background-alpha", type=float, default=1.0, help="Scene background opacity.")
    parser.add_argument("--density-alpha", type=float, default=0.82, help="Density overlay opacity when a background is drawn.")
    parser.add_argument("--overlay-threshold", type=float, default=0.05, help="Density values below this threshold are transparent over the background.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    output = Path(args.output) if args.output else None
    paths = render_from_result_dir(
        Path(args.result_dir),
        output_dir=output_dir,
        output=output,
        step=args.step,
        render_all=args.render_all,
        scale=args.scale,
        smooth_sigma=args.smooth_sigma,
        vmax=args.vmax,
        color_scale=args.color_scale,
        color_percentiles=parse_percentile_pair(args.color_percentiles),
        norm_mode=args.norm,
        gamma=args.gamma,
        dpi=args.dpi,
        origin=args.origin,
        colorbar=not args.no_colorbar,
        crop=not args.no_crop,
        crop_threshold=args.crop_threshold,
        crop_padding=args.crop_padding,
        background=Path(args.background) if args.background else None,
        no_background=args.no_background,
        background_alpha=args.background_alpha,
        density_alpha=args.density_alpha,
        overlay_threshold=args.overlay_threshold,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
