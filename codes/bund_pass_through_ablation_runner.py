from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bund_capacity_response_runner import _dump_routes_toml, _dump_run_toml, _format_scalar
from crowd_bellman.compilers.config_compiler import compile_scene
from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.field_recorder import make_field_data_observer_factory
from crowd_bellman.loaders.config_loader import load_scene_spec, load_run_config
from crowd_bellman.metrics import save_json
from render_refined_density_heatmap import render_from_result_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("anylogic-scene/BundScene/converted")
CENTER_PREFIX = "center_"


@dataclass(frozen=True)
class StagePatch:
    stage_id: str
    full_goal_region: str
    downstream_goal_region: str
    decision_regions: tuple[str, ...]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    label: str
    mode: str
    description: str


STAGE_PATCHES: tuple[StagePatch, ...] = (
    StagePatch("to_pre_channel_1", "goal_region11", "channel_1", ("goal_region11", "channel_1")),
    StagePatch("to_pre_channel_2", "goal_region", "channel_2", ("goal_region", "channel_2")),
    StagePatch("to_pre_channel_3", "goal_region2", "channel_3", ("goal_region2", "channel_3")),
    StagePatch("to_pre_channel_4", "goal_region5", "channel_4", ("goal_region5", "channel_4")),
    StagePatch("to_channel_1", "post_channel_1", "merge_after_channels", ("channel_1", "post_channel_1")),
    StagePatch("to_channel_2", "waypoint_channel2_to_channel3", "merge_after_channels", ("waypoint_channel2_to_channel3",)),
    StagePatch("to_channel_3", "post_channel_3", "merge_after_channels", ("channel_3", "post_channel_3")),
    StagePatch("to_channel_4", "post_channel_4", "merge_after_channels", ("channel_4", "post_channel_4")),
    StagePatch("to_merge_after_channels", "merge_after_channels", "exits", ("merge_after_channels", "exits")),
)
MONITORED_REGIONS = tuple(dict.fromkeys(patch.full_goal_region for patch in STAGE_PATCHES))

VARIANT_SPECS: dict[str, VariantSpec] = {
    "a_full_goal": VariantSpec(
        variant_id="a_full_goal",
        label="A full rectangle goal",
        mode="full_goal",
        description="Use each handoff rectangle as a full-area Bellman goal and decision region.",
    ),
    "b_center_goal": VariantSpec(
        variant_id="b_center_goal",
        label="B center goal",
        mode="center_goal",
        description="Use a shrunken center polygon as Bellman goal; keep the full rectangle as decision region.",
    ),
    "c_pass_through": VariantSpec(
        variant_id="c_pass_through",
        label="C pass-through",
        mode="pass_through",
        description="Use downstream goal regions; keep rectangles only as decision/handoff regions.",
    ),
}


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"TOML payload must be a table: {path}")
    return payload


def _copy_jsonable(value: object) -> object:
    return json.loads(json.dumps(value))


def shrink_polygon(points: Iterable[Iterable[float]], *, factor: float) -> list[list[float]]:
    raw_points = [(float(point[0]), float(point[1])) for point in points]
    if len(raw_points) < 3:
        raise ValueError("polygon must contain at least three points")
    if not (0.0 < float(factor) <= 1.0):
        raise ValueError("factor must be in (0, 1]")
    center_x = sum(x for x, _ in raw_points) / float(len(raw_points))
    center_y = sum(y for _, y in raw_points) / float(len(raw_points))
    return [
        [center_x + (x - center_x) * float(factor), center_y + (y - center_y) * float(factor)]
        for x, y in raw_points
    ]


def _region_by_name(scene: dict[str, object]) -> dict[str, dict[str, object]]:
    regions = scene.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("scene.regions must be a list")
    return {str(region["name"]): region for region in regions if isinstance(region, dict) and "name" in region}


def add_center_goal_regions(
    scene: dict[str, object],
    source_region_names: Iterable[str],
    *,
    factor: float = 0.35,
) -> dict[str, object]:
    updated = deepcopy(scene)
    regions = updated.setdefault("regions", [])
    if not isinstance(regions, list):
        raise ValueError("scene.regions must be a list")
    by_name = _region_by_name(updated)
    for source_name in dict.fromkeys(source_region_names):
        center_name = f"{CENTER_PREFIX}{source_name}"
        if center_name in by_name:
            continue
        source = by_name.get(source_name)
        if source is None:
            raise ValueError(f"region not found for center goal: {source_name}")
        if str(source.get("shape", "rect")).lower() != "polygon":
            raise ValueError(f"center goal currently expects polygon source region: {source_name}")
        center_region: dict[str, object] = {
            "name": center_name,
            "shape": "polygon",
            "points": shrink_polygon(source["points"], factor=factor),
        }
        if source.get("axis") is not None:
            center_region["axis"] = _copy_jsonable(source["axis"])
        regions.append(center_region)
        by_name[center_name] = center_region
    return updated


def _patch_for_stage(stage_id: str) -> StagePatch | None:
    for patch in STAGE_PATCHES:
        if patch.stage_id == stage_id:
            return patch
    return None


def make_route_variant(
    base_routes: dict[str, object],
    variant: VariantSpec,
    *,
    case_id: str,
    kappa: float,
) -> dict[str, object]:
    routes = deepcopy(base_routes)
    case_table = dict(routes.get("case", {}))
    case_table["case_id"] = case_id
    case_table["title"] = variant.label
    routes["case"] = case_table
    routes["capacity_controls"] = []

    stages = routes.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("routes.stages must be a list")
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        patch = _patch_for_stage(str(stage.get("stage_id", "")))
        if patch is None:
            continue
        if variant.mode == "full_goal":
            stage["goal_region"] = patch.full_goal_region
        elif variant.mode == "center_goal":
            stage["goal_region"] = f"{CENTER_PREFIX}{patch.full_goal_region}"
        elif variant.mode == "pass_through":
            stage["goal_region"] = patch.downstream_goal_region
        else:
            raise ValueError(f"Unsupported variant mode: {variant.mode}")
        stage.pop("goal_regions", None)
        stage["decision_regions"] = list(patch.decision_regions)
        stage.pop("decision_region", None)
        if stage.get("next_stage") is not None or stage.get("targets"):
            stage["kappa"] = float(kappa)
            stage["transition_direction"] = "inherit_target" if stage.get("next_stage") is not None else str(stage.get("transition_direction", "stop"))
        stage["controls"] = []
    return routes


def _dump_scene_toml(scene: dict[str, object]) -> str:
    lines: list[str] = []
    for key in ("block_boundaries", "obstacles"):
        if key in scene:
            lines.append(f"{key} = {_format_scalar(scene[key])}")
    if lines:
        lines.append("")

    def append_array_table(table_name: str, items: object, keys: tuple[str, ...]) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[[{table_name}]]")
            for key in keys:
                if key in item and item[key] is not None:
                    lines.append(f"{key} = {_format_scalar(item[key])}")

    append_array_table("regions", scene.get("regions", []), ("name", "shape", "x0", "x1", "y0", "y1", "points", "axis"))
    append_array_table("walls", scene.get("walls", []), ("name", "shape", "points", "width"))
    append_array_table("exits", scene.get("exits", []), ("name", "region", "regions"))
    append_array_table("channels", scene.get("channels", []), ("name", "region", "regions", "probe_x"))
    return "\n".join(lines).rstrip() + "\n"


def _dump_population_toml(population: dict[str, object], *, inflow_rate_scale: float) -> str:
    scaled = deepcopy(population)
    for group in scaled.get("inflow_groups", []):
        if isinstance(group, dict) and "rate" in group:
            group["rate"] = float(group["rate"]) * float(inflow_rate_scale)
    lines: list[str] = []
    for table_name in ("initial_groups", "inflow_groups"):
        groups = scaled.get(table_name, [])
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            if lines:
                lines.append("")
            lines.append(f"[[{table_name}]]")
            for key, value in group.items():
                if value is not None:
                    lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines).rstrip() + "\n"


def _write_case_config(
    *,
    base_dir: Path,
    output_root: Path,
    variant: VariantSpec,
    steps: int,
    time_horizon: float,
    save_every: int,
    rho_max: float,
    inflow_rate_scale: float,
    kappa: float,
) -> Path:
    base_run = _load_toml(base_dir / "run.toml")
    base_scene = _load_toml(base_dir / str(base_run["scene"]["file"]))
    base_population = _load_toml(base_dir / str(base_run["population"]["file"]))
    base_routes = _load_toml(base_dir / str(base_run["routes"]["file"]))

    generated_dir = output_root / variant.variant_id / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    scene_path = generated_dir / f"scene_{variant.variant_id}.toml"
    routes_path = generated_dir / f"routes_{variant.variant_id}.toml"
    population_path = generated_dir / f"population_{variant.variant_id}.toml"
    run_path = generated_dir / f"run_{variant.variant_id}.toml"

    scene = base_scene
    if variant.mode == "center_goal":
        scene = add_center_goal_regions(scene, MONITORED_REGIONS)

    routes = make_route_variant(base_routes, variant, case_id=variant.variant_id, kappa=kappa)
    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": dict(base_run.get("objective", {})),
        "scene": {"file": str(scene_path.resolve())},
        "population": {"file": str(population_path.resolve())},
        "routes": {"file": str(routes_path.resolve())},
        "outputs": {"output_root": str((output_root / variant.variant_id).resolve())},
    }
    generated_run["simulation"].update(
        {
            "steps": int(steps),
            "time_horizon": float(time_horizon),
            "save_every": int(save_every),
            "rho_max": float(rho_max),
            "density_contour_levels": 0,
        }
    )
    generated_run["objective"]["name"] = variant.variant_id

    scene_path.write_text(_dump_scene_toml(scene), encoding="utf-8")
    routes_path.write_text(_dump_routes_toml(routes), encoding="utf-8")
    population_path.write_text(_dump_population_toml(base_population, inflow_rate_scale=inflow_rate_scale), encoding="utf-8")
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def _dilate4(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    out = values.copy()
    out[1:, :] |= values[:-1, :]
    out[:-1, :] |= values[1:, :]
    out[:, 1:] |= values[:, :-1]
    out[:, :-1] |= values[:, 1:]
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    out = values.copy()
    out[1:, :] &= values[:-1, :]
    out[:-1, :] &= values[1:, :]
    out[:, 1:] &= values[:, :-1]
    out[:, :-1] &= values[:, 1:]
    return out


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def _safe_max(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.max(values[mask]))


def build_region_edge_metrics(
    density: np.ndarray,
    *,
    region_masks: dict[str, np.ndarray],
    walkable: np.ndarray,
    regions: Iterable[str],
    case_id: str,
    step: int,
    time_value: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region_name in regions:
        region_mask = region_masks.get(region_name)
        if region_mask is None:
            continue
        inside = np.asarray(region_mask, dtype=bool) & walkable
        if not np.any(inside):
            continue
        boundary = inside & (~_erode4(inside))
        interior = inside & _erode4(inside)
        outside_ring = _dilate4(inside) & (~inside) & walkable
        boundary_mean = _safe_mean(density, boundary)
        interior_mean = _safe_mean(density, interior)
        outside_mean = _safe_mean(density, outside_ring)
        rows.append(
            {
                "case_id": case_id,
                "step": int(step),
                "time": float(time_value),
                "region": region_name,
                "inside_mean": _safe_mean(density, inside),
                "inside_max": _safe_max(density, inside),
                "boundary_mean": boundary_mean,
                "boundary_max": _safe_max(density, boundary),
                "interior_mean": interior_mean,
                "interior_max": _safe_max(density, interior),
                "outside_ring_mean": outside_mean,
                "outside_ring_max": _safe_max(density, outside_ring),
                "boundary_to_interior": boundary_mean / max(interior_mean, 1.0e-12) if np.isfinite(interior_mean) else float("nan"),
                "boundary_to_outside": boundary_mean / max(outside_mean, 1.0e-12) if np.isfinite(outside_mean) else float("nan"),
                "inside_cells": int(np.count_nonzero(inside)),
                "boundary_cells": int(np.count_nonzero(boundary)),
                "interior_cells": int(np.count_nonzero(interior)),
                "outside_ring_cells": int(np.count_nonzero(outside_ring)),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_field_manifest(fields_dir: Path) -> dict[str, object]:
    with (fields_dir / "fields_manifest.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid fields manifest: {fields_dir}")
    return payload


def collect_region_edge_metrics(case_dir: Path, *, case_id: str, config_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_spec = load_run_config(config_path)
    scene_spec = load_scene_spec(run_spec.scene_path)
    bundle = compile_scene(scene_spec, run_spec.simulation)
    fields_dir = case_dir / "fields"
    manifest = _load_field_manifest(fields_dir)
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError(f"Manifest files must be a list: {fields_dir}")
    timeseries_rows: list[dict[str, object]] = []
    for entry in sorted((item for item in files if isinstance(item, dict)), key=lambda item: int(item["step"])):
        with np.load(fields_dir / str(entry["file"])) as payload:
            density = np.asarray(payload["rho"], dtype=float)
            step = int(payload["step"][0]) if "step" in payload.files else int(entry["step"])
            time_value = float(payload["time"][0]) if "time" in payload.files else float(entry.get("time", 0.0))
        timeseries_rows.extend(
            build_region_edge_metrics(
                density,
                region_masks=bundle.region_masks,
                walkable=bundle.scene.walkable,
                regions=MONITORED_REGIONS,
                case_id=case_id,
                step=step,
                time_value=time_value,
            )
        )
    last_step = max((int(row["step"]) for row in timeseries_rows), default=-1)
    final_rows = [row for row in timeseries_rows if int(row["step"]) == last_step]
    return timeseries_rows, final_rows


def _plot_final_boundary(summary_rows: list[dict[str, object]], output: Path) -> None:
    regions = list(MONITORED_REGIONS)
    cases = [spec.variant_id for spec in VARIANT_SPECS.values()]
    grouped = {(str(row["case_id"]), str(row["region"])): row for row in summary_rows}
    x = np.arange(len(regions), dtype=float)
    width = 0.8 / max(len(cases), 1)
    fig, ax = plt.subplots(figsize=(13.0, 5.0), dpi=180)
    for index, case_id in enumerate(cases):
        values = [float(grouped.get((case_id, region), {}).get("boundary_mean", 0.0)) for region in regions]
        ax.bar(x + (index - (len(cases) - 1) / 2.0) * width, values, width=width, label=case_id)
    ax.set_xticks(x, regions, rotation=35, ha="right")
    ax.set_ylabel("boundary density mean")
    ax.set_title("Final Boundary Density by Handoff Region")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_average_boundary_timeseries(timeseries_rows: list[dict[str, object]], output: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = {}
    times: dict[tuple[str, int], float] = {}
    for row in timeseries_rows:
        key = (str(row["case_id"]), int(row["step"]))
        grouped.setdefault(key, []).append(float(row["boundary_mean"]))
        times[key] = float(row["time"])
    by_case: dict[str, list[tuple[float, float]]] = {}
    for (case_id, step), values in grouped.items():
        finite = [value for value in values if np.isfinite(value)]
        if finite:
            by_case.setdefault(case_id, []).append((times[(case_id, step)], float(np.mean(finite))))
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    for case_id, items in sorted(by_case.items()):
        ordered = sorted(items)
        ax.plot([time for time, _ in ordered], [value for _, value in ordered], label=case_id, linewidth=1.8)
    ax.set_title("Average Handoff-Region Boundary Density")
    ax.set_xlabel("time")
    ax.set_ylabel("mean boundary density")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_merge_boundary_timeseries(timeseries_rows: list[dict[str, object]], output: Path) -> None:
    rows = [row for row in timeseries_rows if row["region"] == "merge_after_channels"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        items = sorted((row for row in rows if row["case_id"] == case_id), key=lambda row: int(row["step"]))
        ax.plot([float(row["time"]) for row in items], [float(row["boundary_mean"]) for row in items], label=case_id, linewidth=1.8)
    ax.set_title("merge_after_channels Boundary Density")
    ax.set_xlabel("time")
    ax.set_ylabel("boundary density mean")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _write_variant_plan(path: Path) -> None:
    lines = [
        "# Bund Pass-Through Region Ablation",
        "",
        "A: full rectangle goal. The handoff rectangles are full-area Bellman goals and decision regions.",
        "B: center goal. The original rectangles remain decision regions, but Bellman goals are shrunken center polygons.",
        "C: pass-through. The rectangles remain decision/handoff triggers only; Bellman goals point downstream.",
        "",
        "Primary diagnostic: region boundary density, interior density, outside-ring density, and boundary ratios.",
        "The target symptom is reduced boundary_mean and boundary_to_interior around the handoff rectangles.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_variant(
    *,
    config_path: Path,
    output_root: Path,
    save_every: int,
    field_dtype: str,
) -> dict[str, object]:
    summary = run_from_config(
        config_path=config_path,
        output_root=output_root,
        write_root_summary=False,
        step_observer_factory=make_field_data_observer_factory(
            save_every=save_every,
            dtype=field_dtype,
            output_dir_name="fields",
        ),
    )
    return summary


def run_ablation(args: argparse.Namespace) -> dict[str, object]:
    base_dir = _resolve_path(args.base_dir)
    output_root = _resolve_path(args.output_root) if args.output_root else (
        REPO_ROOT / "codes" / "results" / f"bund_pass_through_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_variant_plan(output_root / "ablation_plan.md")

    config_paths: dict[str, Path] = {}
    summaries: list[dict[str, object]] = []
    for variant in VARIANT_SPECS.values():
        config_path = _write_case_config(
            base_dir=base_dir,
            output_root=output_root,
            variant=variant,
            steps=args.steps,
            time_horizon=args.time_horizon,
            save_every=args.save_every,
            rho_max=args.rho_max,
            inflow_rate_scale=args.inflow_rate_scale,
            kappa=args.kappa,
        )
        config_paths[variant.variant_id] = config_path
        if not args.skip_runs:
            summary = run_variant(
                config_path=config_path,
                output_root=output_root / variant.variant_id,
                save_every=args.save_every,
                field_dtype=args.field_dtype,
            )
            summaries.append(
                {
                    "case_id": str(summary.get("case_id", variant.variant_id)),
                    "objective_value": float(summary.get("objective_value", 0.0)),
                    "final_sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
                    "final_mass": float(summary.get("final_mass", 0.0)),
                    "peak_density_max": float(summary.get("peak_density_max", 0.0)),
                    "case_dir": str((output_root / variant.variant_id / variant.variant_id).resolve()),
                }
            )

    all_timeseries: list[dict[str, object]] = []
    all_final: list[dict[str, object]] = []
    for variant in VARIANT_SPECS.values():
        case_dir = output_root / variant.variant_id / variant.variant_id
        timeseries, final_rows = collect_region_edge_metrics(
            case_dir,
            case_id=variant.variant_id,
            config_path=config_paths[variant.variant_id],
        )
        all_timeseries.extend(timeseries)
        all_final.extend(final_rows)

    _write_csv(output_root / "case_summary.csv", summaries)
    _write_csv(output_root / "region_edge_timeseries.csv", all_timeseries)
    _write_csv(output_root / "region_edge_final.csv", all_final)
    _plot_final_boundary(all_final, output_root / "figures" / "final_boundary_density.png")
    _plot_average_boundary_timeseries(all_timeseries, output_root / "figures" / "average_boundary_density_timeseries.png")
    _plot_merge_boundary_timeseries(all_timeseries, output_root / "figures" / "merge_boundary_density_timeseries.png")

    if not args.skip_hq:
        for variant in VARIANT_SPECS.values():
            case_dir = output_root / variant.variant_id / variant.variant_id
            render_from_result_dir(
                case_dir,
                output_dir=case_dir / "refined_density_vector_walls_vmax6_full_every10",
                render_all=True,
                scale=args.hq_scale,
                smooth_sigma=args.hq_smooth_sigma,
                vmax=args.hq_vmax,
                color_scale="absolute",
                cmap_name="low-density",
                norm_mode="power",
                gamma=0.42,
                dpi=args.hq_dpi,
                crop=False,
                background=None,
                no_background=True,
                density_alpha=1.0,
                overlay_threshold=0.005,
                alpha_gamma=0.32,
                fusion_mode="wall-preserve",
                nonwalkable_fill="zero",
                wall_overlay="vector",
                vector_wall_color="#111111",
            )

    manifest = {
        "output_root": str(output_root.resolve()),
        "variants": {variant.variant_id: variant.__dict__ for variant in VARIANT_SPECS.values()},
        "generated_configs": {case_id: str(path.resolve()) for case_id, path in config_paths.items()},
        "case_summary_csv": str((output_root / "case_summary.csv").resolve()),
        "region_edge_timeseries_csv": str((output_root / "region_edge_timeseries.csv").resolve()),
        "region_edge_final_csv": str((output_root / "region_edge_final.csv").resolve()),
        "figures_dir": str((output_root / "figures").resolve()),
        "high_resolution_density_dirs": {
            variant.variant_id: str((output_root / variant.variant_id / variant.variant_id / "refined_density_vector_walls_vmax6_full_every10").resolve())
            for variant in VARIANT_SPECS.values()
        },
    }
    save_json(output_root / "ablation_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run A/B/C ablation for Bund handoff rectangles as full goals, center goals, or pass-through decision regions.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Converted BundScene directory.")
    parser.add_argument("--output-root", default=None, help="Output root for generated configs, runs, metrics, and figures.")
    parser.add_argument("--steps", type=int, default=800, help="Short-run ablation steps.")
    parser.add_argument("--time-horizon", type=float, default=25.0, help="Physical time horizon.")
    parser.add_argument("--save-every", type=int, default=10, help="Snapshot and field-data interval.")
    parser.add_argument("--rho-max", type=float, default=4.0, help="Density cap.")
    parser.add_argument("--inflow-rate-scale", type=float, default=6.0, help="Continuous inflow multiplier.")
    parser.add_argument("--kappa", type=float, default=32.0, help="Transition kappa for all patched transition stages.")
    parser.add_argument("--field-dtype", choices=("float32", "float64"), default="float32", help="Saved field array dtype.")
    parser.add_argument("--skip-runs", action="store_true", help="Only post-process existing variant outputs.")
    parser.add_argument("--skip-hq", action="store_true", help="Do not render high-resolution density frames.")
    parser.add_argument("--hq-scale", type=int, default=8, help="High-resolution density upsampling scale.")
    parser.add_argument("--hq-smooth-sigma", type=float, default=5.0, help="High-resolution density smoothing sigma.")
    parser.add_argument("--hq-vmax", type=float, default=6.0, help="High-resolution density vmax.")
    parser.add_argument("--hq-dpi", type=int, default=240, help="High-resolution PNG dpi.")
    return parser


def main() -> None:
    manifest = run_ablation(build_arg_parser().parse_args())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
