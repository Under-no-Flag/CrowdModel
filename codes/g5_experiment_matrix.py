from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable

import numpy as np

from crowd_bellman.g5_hcmbo import (
    ALL_GATE_IDS,
    CHANNEL_NAMES,
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_PRIOR_DIRECTIONS,
    G5EvaluationCache,
    HCMBOConfig,
    V2ControlVector,
    V2EvaluationRecord,
    build_method_comparison,
    control_from_capacity_mode,
    generate_direction_candidates,
    make_no_cap_control,
    optimize_fixed_direction,
    qbar_from_reference,
    run_hcmbo_experiment,
    run_random_search,
    save_g5_plots,
    write_report,
)
from crowd_bellman.metrics import save_json


REQUIRED_OUTPUTS = (
    "G5_evaluation_log.csv",
    "G5_top_candidates.csv",
    "G5_method_comparison.csv",
    "G5_config_summary.json",
)

DEFAULT_G5_MATRIX_CONFIG = Path("codes/scenes/examples/g5_hcmbo_v2_full/g5.toml")

WEIGHT_SETS = {
    "default": (1.0, 1.0, 1.0, 1.0, 0.1),
    "efficiency_first": (2.0, 1.0, 1.0, 0.5, 0.1),
    "safety_first": (1.0, 2.0, 1.0, 1.0, 0.1),
    "waiting_first": (1.0, 1.0, 1.0, 2.0, 0.1),
    "smoothness_first": (1.0, 1.0, 1.0, 1.0, 1.0),
}


@dataclass(frozen=True)
class FidelityBudget:
    steps: int
    time_horizon: float
    bellman_every: int
    save_every: int = 100000
    density_contour_levels: int | str = 0

    def to_overrides(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "time_horizon": float(self.time_horizon),
            "bellman_every": int(self.bellman_every),
            "save_every": int(self.save_every),
            "density_contour_levels": self.density_contour_levels,
        }


@dataclass(frozen=True)
class MatrixProfile:
    name: str
    config: HCMBOConfig
    screen: FidelityBudget
    optimization: FidelityBudget
    high_fidelity: FidelityBudget


@dataclass(frozen=True)
class MatrixExperiment:
    name: str
    runner: Callable[[Path, MatrixProfile, Path, bool], dict[str, object]]
    description: str


@dataclass(frozen=True)
class LoadedMatrixConfig:
    profile_name: str | None = None
    output_root: Path | None = None
    baseline_config: Path | None = None
    seed: int | None = None
    force: bool | None = None
    fail_fast: bool | None = None
    experiments: str | None = None
    workers: int | None = None
    profile_overrides: dict[str, object] | None = None


def profile_from_name(name: str, seed: int) -> MatrixProfile:
    if name == "full":
        return MatrixProfile(
            name=name,
            config=HCMBOConfig(
                time_segments=4,
                direction_candidate_limit=50,
                shortlist_size=20,
                screen_capacity_modes=("high", "medium", "low"),
                initial_samples=30,
                bo_iterations=80,
                dfo_evaluations=50,
                high_fidelity_top_k=20,
                random_search_evaluations=400,
                random_seed=seed,
            ),
            screen=FidelityBudget(steps=120, time_horizon=6.0, bellman_every=4),
            optimization=FidelityBudget(steps=300, time_horizon=20.0, bellman_every=4),
            high_fidelity=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
        )
    if name == "smoke":
        return MatrixProfile(
            name=name,
            config=HCMBOConfig(
                time_segments=2,
                direction_candidate_limit=3,
                shortlist_size=1,
                screen_capacity_modes=("high", "medium"),
                initial_samples=2,
                bo_iterations=0,
                dfo_evaluations=0,
                high_fidelity_top_k=1,
                random_search_evaluations=1,
                random_seed=seed,
            ),
            screen=FidelityBudget(steps=8, time_horizon=1.0, bellman_every=4),
            optimization=FidelityBudget(steps=8, time_horizon=1.0, bellman_every=4),
            high_fidelity=FidelityBudget(steps=8, time_horizon=1.0, bellman_every=4),
        )
    raise ValueError(f"Unsupported profile: {name!r}")


def load_matrix_config(path: Path) -> LoadedMatrixConfig:
    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a TOML document")
    base_dir = path.parent

    g5_table = raw.get("g5", {})
    if g5_table and not isinstance(g5_table, dict):
        raise ValueError("[g5] must be a table")
    g5_table = dict(g5_table)

    profile_overrides: dict[str, object] = {}
    hcmbo_table = raw.get("hcmbo", {})
    if hcmbo_table:
        if not isinstance(hcmbo_table, dict):
            raise ValueError("[hcmbo] must be a table")
        profile_overrides["hcmbo"] = dict(hcmbo_table)

    weights_table = raw.get("weights", {})
    if weights_table:
        if not isinstance(weights_table, dict):
            raise ValueError("[weights] must be a table")
        hcmbo = dict(profile_overrides.get("hcmbo", {}))
        hcmbo.update(weights_table)
        profile_overrides["hcmbo"] = hcmbo

    for table_name in ("screen", "optimization", "high_fidelity"):
        table = raw.get(table_name, {})
        if table:
            if not isinstance(table, dict):
                raise ValueError(f"[{table_name}] must be a table")
            profile_overrides[table_name] = dict(table)

    experiments = g5_table.get("experiments")
    return LoadedMatrixConfig(
        profile_name=str(g5_table["profile"]) if "profile" in g5_table else None,
        output_root=resolve_config_path(base_dir, str(g5_table["output_root"])) if "output_root" in g5_table else None,
        baseline_config=resolve_config_path(base_dir, str(g5_table["baseline_config"])) if "baseline_config" in g5_table else None,
        seed=int(g5_table["seed"]) if "seed" in g5_table else None,
        force=bool(g5_table["force"]) if "force" in g5_table else None,
        fail_fast=bool(g5_table["fail_fast"]) if "fail_fast" in g5_table else None,
        experiments=parse_experiments_config_value(experiments) if experiments is not None else None,
        workers=int(g5_table["workers"]) if "workers" in g5_table else None,
        profile_overrides=profile_overrides,
    )


def resolve_config_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_experiments_config_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    raise ValueError("g5.experiments must be a comma-separated string or a list")


def should_apply_profile_overrides(cli_profile: str | None, loaded_config: LoadedMatrixConfig) -> bool:
    if not loaded_config.profile_overrides:
        return False
    if cli_profile is None:
        return True
    return loaded_config.profile_name is None or cli_profile == loaded_config.profile_name


def apply_profile_overrides(profile: MatrixProfile, overrides: dict[str, object]) -> MatrixProfile:
    hcmbo_overrides = overrides.get("hcmbo", {})
    hcmbo_config = profile.config
    if hcmbo_overrides:
        if not isinstance(hcmbo_overrides, dict):
            raise ValueError("hcmbo overrides must be a table")
        valid_fields = {item.name for item in fields(HCMBOConfig)}
        replace_kwargs: dict[str, object] = {}
        for key, value in hcmbo_overrides.items():
            if key not in valid_fields:
                raise ValueError(f"Unsupported hcmbo config key: {key}")
            replace_kwargs[key] = coerce_like(getattr(hcmbo_config, key), value)
        hcmbo_config = replace(hcmbo_config, **replace_kwargs)

    return MatrixProfile(
        name=profile.name,
        config=hcmbo_config,
        screen=apply_budget_overrides(profile.screen, overrides.get("screen", {}), "screen"),
        optimization=apply_budget_overrides(profile.optimization, overrides.get("optimization", {}), "optimization"),
        high_fidelity=apply_budget_overrides(profile.high_fidelity, overrides.get("high_fidelity", {}), "high_fidelity"),
    )


def apply_budget_overrides(budget: FidelityBudget, raw_table: object, table_name: str) -> FidelityBudget:
    if not raw_table:
        return budget
    if not isinstance(raw_table, dict):
        raise ValueError(f"{table_name} overrides must be a table")
    allowed = {"steps", "time_horizon", "bellman_every", "save_every", "density_contour_levels"}
    for key in raw_table:
        if key not in allowed:
            raise ValueError(f"Unsupported {table_name} config key: {key}")
    return replace(
        budget,
        steps=int(raw_table.get("steps", budget.steps)),
        time_horizon=float(raw_table.get("time_horizon", budget.time_horizon)),
        bellman_every=int(raw_table.get("bellman_every", budget.bellman_every)),
        save_every=int(raw_table.get("save_every", budget.save_every)),
        density_contour_levels=raw_table.get("density_contour_levels", budget.density_contour_levels),
    )


def coerce_like(current: object, value: object) -> object:
    if isinstance(current, tuple):
        if isinstance(value, str):
            raw_values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list | tuple):
            raw_values = list(value)
        else:
            raise ValueError(f"Expected tuple-like value, got {value!r}")
        if not current:
            return tuple(raw_values)
        sample = current[0]
        if isinstance(sample, float):
            return tuple(float(item) for item in raw_values)
        if isinstance(sample, bool):
            return tuple(bool(item) for item in raw_values)
        if isinstance(sample, int):
            return tuple(int(item) for item in raw_values)
        if isinstance(sample, str):
            return tuple(str(item) for item in raw_values)
        return tuple(raw_values)
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize the G5 V2 experiment matrix.")
    parser.add_argument(
        "--config",
        help=f"Path to a G5 matrix TOML config file, e.g. {DEFAULT_G5_MATRIX_CONFIG}",
    )
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker process count for parallel G5 matrix runs. Defaults to the number of selected experiments.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run experiments even when complete outputs exist.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated experiment names to run. Defaults to the full matrix.",
    )
    args = parser.parse_args()

    loaded_config = load_matrix_config(Path(args.config)) if args.config else LoadedMatrixConfig()
    profile_name = args.profile or loaded_config.profile_name or "full"
    seed = int(args.seed if args.seed is not None else loaded_config.seed if loaded_config.seed is not None else 23)
    profile = profile_from_name(profile_name, seed)
    if should_apply_profile_overrides(args.profile, loaded_config):
        profile = apply_profile_overrides(profile, loaded_config.profile_overrides or {})

    output_root = Path(
        args.output_root
        or loaded_config.output_root
        or "codes/results/g5_full_parallel"
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_config = Path(args.baseline_config or loaded_config.baseline_config or DEFAULT_BASELINE_CONFIG).resolve()
    experiments = build_experiments()
    selected = parse_experiment_selection(args.experiments if args.experiments is not None else loaded_config.experiments or "", experiments)
    force = bool(args.force or loaded_config.force)
    fail_fast = bool(args.fail_fast or loaded_config.fail_fast)
    workers = resolve_worker_count(args.workers if args.workers is not None else loaded_config.workers, len(selected))

    manifest: dict[str, object] = {
        "profile": profile.name,
        "output_root": str(output_root),
        "baseline_config": str(baseline_config),
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "seed": seed,
        "workers": workers,
        "argv": sys.argv,
        "experiments": [],
    }

    failures = run_selected_experiments(
        selected=selected,
        manifest=manifest,
        output_root=output_root,
        profile=profile,
        baseline_config=baseline_config,
        force=force,
        fail_fast=fail_fast,
        workers=workers,
    )
    if failures:
        raise SystemExit(f"G5 matrix failed for: {', '.join(failures)}")


def resolve_worker_count(raw_workers: int | None, selected_count: int) -> int:
    if selected_count <= 0:
        return 1
    if raw_workers is None:
        return selected_count
    return max(1, min(int(raw_workers), selected_count))


def run_selected_experiments(
    *,
    selected: list[MatrixExperiment],
    manifest: dict[str, object],
    output_root: Path,
    profile: MatrixProfile,
    baseline_config: Path,
    force: bool,
    fail_fast: bool,
    workers: int,
) -> list[str]:
    entries_by_name: dict[str, dict[str, object]] = {}
    run_payloads: list[dict[str, object]] = []
    failures: list[str] = []

    for experiment in selected:
        subdir = output_root / experiment.name
        entry = {
            "name": experiment.name,
            "description": experiment.description,
            "output_dir": str(subdir),
            "status": "pending",
        }
        cast_entries(manifest).append(entry)
        entries_by_name[experiment.name] = entry
        if not force and experiment_complete(subdir):
            payload = load_json(subdir / "G5_config_summary.json")
            entry.update(
                {
                    "status": "skipped_complete",
                    "best_objective": nested_get(payload, "best_high_fidelity", "objective_value"),
                    "best_case_id": nested_get(payload, "best_high_fidelity", "case_id"),
                }
            )
        else:
            entry["status"] = "running"
            run_payloads.append(
                {
                    "name": experiment.name,
                    "output_dir": str(subdir),
                    "profile": profile,
                    "baseline_config": str(baseline_config),
                    "force": force,
                }
            )

    write_manifest(output_root, manifest)
    write_matrix_outputs(output_root=output_root, manifest=manifest)

    if not run_payloads:
        return failures

    if workers == 1:
        for payload in run_payloads:
            result = run_experiment_payload(payload)
            apply_experiment_result(
                result=result,
                entries_by_name=entries_by_name,
                failures=failures,
                output_root=output_root,
                manifest=manifest,
            )
            if result.get("status") == "failed" and fail_fast:
                raise RuntimeError(result.get("error", "G5 experiment failed"))
        write_matrix_outputs(output_root=output_root, manifest=manifest)
        return failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_experiment_payload, payload): str(payload["name"]) for payload in run_payloads}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "name": name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            apply_experiment_result(
                result=result,
                entries_by_name=entries_by_name,
                failures=failures,
                output_root=output_root,
                manifest=manifest,
            )
            write_matrix_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(result.get("error", "G5 experiment failed"))

    write_matrix_outputs(output_root=output_root, manifest=manifest)
    return failures


def run_experiment_payload(payload: dict[str, object]) -> dict[str, object]:
    name = str(payload["name"])
    try:
        by_name = {experiment.name: experiment for experiment in build_experiments()}
        experiment = by_name[name]
        output_dir = Path(str(payload["output_dir"]))
        profile = payload["profile"]
        if not isinstance(profile, MatrixProfile):
            raise TypeError("profile payload must be a MatrixProfile")
        result_payload = experiment.runner(
            output_dir,
            profile,
            Path(str(payload["baseline_config"])),
            bool(payload["force"]),
        )
        return {
            "name": name,
            "status": "completed",
            "best_objective": nested_get(result_payload, "best_high_fidelity", "objective_value"),
            "best_case_id": nested_get(result_payload, "best_high_fidelity", "case_id"),
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def apply_experiment_result(
    *,
    result: dict[str, object],
    entries_by_name: dict[str, dict[str, object]],
    failures: list[str],
    output_root: Path,
    manifest: dict[str, object],
) -> None:
    name = str(result["name"])
    entry = entries_by_name[name]
    entry.update(result)
    if result.get("status") == "failed" and name not in failures:
        failures.append(name)
    write_manifest(output_root, manifest)


def build_experiments() -> list[MatrixExperiment]:
    return [
        MatrixExperiment("main_hcmbo_full", run_main_hcmbo, "Full HCMBO search."),
        MatrixExperiment("no_dfo", run_no_dfo, "Full HCMBO without DFO polishing."),
        MatrixExperiment("no_lf_selection", run_no_lf_selection, "Do not reduce directions by LF screening."),
        MatrixExperiment("no_jb", run_no_jb, "Full HCMBO with lambda_jb=0."),
        MatrixExperiment("only_s_high", run_only_s_high, "Search directions with q fixed to high capacity."),
        MatrixExperiment("only_q_prior", run_only_q_prior, "Optimize q with directions fixed to the prior setting."),
        MatrixExperiment("random_search", run_random_search_only, "Mixed-variable random search baseline."),
    ]


def parse_experiment_selection(raw: str, experiments: list[MatrixExperiment]) -> list[MatrixExperiment]:
    if not raw.strip():
        return experiments
    by_name = {experiment.name: experiment for experiment in experiments}
    selected: list[MatrixExperiment] = []
    for name in [part.strip() for part in raw.split(",") if part.strip()]:
        if name not in by_name:
            raise ValueError(f"Unknown experiment {name!r}; choices are {', '.join(by_name)}")
        selected.append(by_name[name])
    return selected


def run_main_hcmbo(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    return run_standard_hcmbo(output_dir, profile, baseline_config, profile.config)


def run_no_dfo(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    return run_standard_hcmbo(output_dir, profile, baseline_config, replace(profile.config, dfo_evaluations=0))


def run_no_lf_selection(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    config = replace(profile.config, shortlist_size=profile.config.direction_candidate_limit)
    return run_standard_hcmbo(output_dir, profile, baseline_config, config)


def run_no_jb(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    return run_standard_hcmbo(output_dir, profile, baseline_config, replace(profile.config, lambda_jb=0.0))


def run_standard_hcmbo(
    output_dir: Path,
    profile: MatrixProfile,
    baseline_config: Path,
    config: HCMBOConfig,
) -> dict[str, object]:
    return run_hcmbo_experiment(
        baseline_config=baseline_config,
        output_root=output_dir,
        config=config,
        screen_overrides=profile.screen.to_overrides(),
        optimization_overrides=profile.optimization.to_overrides(),
        high_fidelity_overrides=profile.high_fidelity.to_overrides(),
    )


def run_only_s_high(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = profile.config
    reference_record, qbar_by_gate = evaluate_reference(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(config.random_seed)
    directions_list = generate_direction_candidates(config=config, rng=rng)
    opt_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    records = [
        opt_evaluator.evaluate(
            control_from_capacity_mode(
                directions=directions,
                mode="high",
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            ),
            source="only_s_high",
            phase="only_s_high",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        for directions in directions_list
    ]
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(records, config.high_fidelity_top_k),
    )
    return write_custom_experiment_outputs(
        output_dir=output_dir,
        experiment_name="only_s_high",
        profile=profile,
        config=config,
        reference_record=reference_record,
        mid_records=records,
        hf_records=hf_records,
        method_records={"only_s_high_mf": records, "high_fidelity_recheck": hf_records},
        qbar_by_gate=qbar_by_gate,
        extra_payload={
            "direction_candidates": [directions_dict(directions) for directions in directions_list],
            "fixed_capacity_mode": "high",
        },
    )


def run_only_q_prior(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(profile.config, direction_candidate_limit=1, shortlist_size=1, random_search_evaluations=0)
    reference_record, qbar_by_gate = evaluate_reference(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    opt_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    rng = np.random.default_rng(config.random_seed)
    records, bo_traces = optimize_fixed_direction(
        evaluator=opt_evaluator,
        directions=DEFAULT_PRIOR_DIRECTIONS,
        qbar_by_gate=qbar_by_gate,
        config=config,
        rng=rng,
        source_prefix="only_q_prior",
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(records, config.high_fidelity_top_k),
    )
    return write_custom_experiment_outputs(
        output_dir=output_dir,
        experiment_name="only_q_prior",
        profile=profile,
        config=config,
        reference_record=reference_record,
        mid_records=records,
        hf_records=hf_records,
        method_records={"only_q_prior_mf": records, "high_fidelity_recheck": hf_records},
        qbar_by_gate=qbar_by_gate,
        extra_payload={
            "fixed_directions": directions_dict(DEFAULT_PRIOR_DIRECTIONS),
            "bo_traces": bo_traces,
        },
    )


def run_random_search_only(output_dir: Path, profile: MatrixProfile, baseline_config: Path, force: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = profile.config
    reference_record, qbar_by_gate = evaluate_reference(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(config.random_seed)
    directions_list = generate_direction_candidates(config=config, rng=rng)
    opt_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    records = run_random_search(
        evaluator=opt_evaluator,
        directions_list=directions_list,
        qbar_by_gate=qbar_by_gate,
        config=config,
        rng=rng,
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=baseline_config,
        output_dir=output_dir,
        config=config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(records, config.high_fidelity_top_k),
    )
    return write_custom_experiment_outputs(
        output_dir=output_dir,
        experiment_name="random_search",
        profile=profile,
        config=config,
        reference_record=reference_record,
        mid_records=records,
        hf_records=hf_records,
        method_records={"random_search_mf": records, "high_fidelity_recheck": hf_records},
        qbar_by_gate=qbar_by_gate,
        extra_payload={
            "direction_candidates": [directions_dict(directions) for directions in directions_list],
        },
    )


def evaluate_reference(
    *,
    baseline_config: Path,
    output_dir: Path,
    config: HCMBOConfig,
    optimization_overrides: dict[str, object],
) -> tuple[V2EvaluationRecord, dict[str, float]]:
    evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_reference",
        objective_config=config,
        simulation_overrides=optimization_overrides,
        fidelity="reference",
    )
    record = evaluator.evaluate(
        make_no_cap_control(tuple("FREE" for _ in CHANNEL_NAMES), config.time_segments),
        source="qbar_reference_all_free",
        phase="reference",
        qbar_by_gate={gate_id: math.inf for gate_id in ALL_GATE_IDS},
    )
    return record, qbar_from_reference(record.summary, config=config)


def evaluate_high_fidelity_controls(
    *,
    baseline_config: Path,
    output_dir: Path,
    config: HCMBOConfig,
    qbar_by_gate: dict[str, float],
    overrides: dict[str, object],
    controls: list[V2ControlVector],
) -> list[V2EvaluationRecord]:
    evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_high_fidelity",
        objective_config=config,
        simulation_overrides=overrides,
        fidelity="hf",
    )
    return [
        evaluator.evaluate(
            control,
            source="high_fidelity_recheck",
            phase="high_fidelity",
            qbar_by_gate=qbar_by_gate,
        )
        for control in controls
    ]


def select_unique_controls(records: list[V2EvaluationRecord], limit: int) -> list[V2ControlVector]:
    selected: list[V2ControlVector] = []
    seen: set[V2ControlVector] = set()
    for record in sorted(records, key=lambda item: item.objective_value):
        if record.control in seen:
            continue
        selected.append(record.control)
        seen.add(record.control)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def write_custom_experiment_outputs(
    *,
    output_dir: Path,
    experiment_name: str,
    profile: MatrixProfile,
    config: HCMBOConfig,
    reference_record: V2EvaluationRecord,
    mid_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
    method_records: dict[str, list[V2EvaluationRecord]],
    qbar_by_gate: dict[str, float],
    extra_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = min(hf_records, key=lambda item: item.objective_value) if hf_records else min(mid_records, key=lambda item: item.objective_value)
    all_rows = [reference_record.to_row()] + [record.to_row() for record in mid_records] + [record.to_row() for record in hf_records]
    top_rows = [record.to_row() for record in sorted(hf_records or mid_records, key=lambda item: item.objective_value)]
    method_comparison = method_comparison_from_groups(method_records)
    write_csv(output_dir / "G5_evaluation_log.csv", all_rows)
    write_csv(output_dir / "G5_top_candidates.csv", top_rows)
    write_csv(output_dir / "G5_method_comparison.csv", method_comparison)
    save_json(output_dir / "G5_best_control.json", best.control.to_dict())
    payload: dict[str, object] = {
        "experiment_group": "G5",
        "matrix_experiment": experiment_name,
        "design_version": "v2_hcmbo_s_q",
        "profile": profile.name,
        "output_root": str(output_dir),
        "config": config.__dict__,
        "screen_overrides": profile.screen.to_overrides(),
        "optimization_overrides": profile.optimization.to_overrides(),
        "high_fidelity_overrides": profile.high_fidelity.to_overrides(),
        "qbar_by_gate": qbar_by_gate,
        "reference": reference_record.to_row(),
        "method_comparison": method_comparison,
        "best_high_fidelity": best.to_row(),
        "outputs": {
            "evaluation_log": str(output_dir / "G5_evaluation_log.csv"),
            "top_candidates": str(output_dir / "G5_top_candidates.csv"),
            "method_comparison": str(output_dir / "G5_method_comparison.csv"),
            "best_control": str(output_dir / "G5_best_control.json"),
            "capacity_profiles": str(output_dir / "G5_capacity_profiles.png"),
            "flux_share": str(output_dir / "G5_flux_share.png"),
            "objective_trace": str(output_dir / "G5_objective_trace.png"),
            "pareto_j1_j2": str(output_dir / "G5_pareto_j1_j2.png"),
            "report": str(output_dir / "G5_report.md"),
        },
    }
    if extra_payload:
        payload.update(extra_payload)
    save_json(output_dir / "G5_config_summary.json", payload)
    save_g5_plots(output_root=output_dir, records=hf_records or mid_records, all_records=mid_records, best=best)
    write_report(output_root=output_dir, payload=payload, records=hf_records or mid_records, best=best)
    return payload


def method_comparison_from_groups(groups: dict[str, list[V2EvaluationRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, records in groups.items():
        if not records:
            continue
        best = min(records, key=lambda item: item.objective_value)
        rows.append(
            {
                "method": method,
                "evaluation_count": len(records),
                "best_objective": float(best.objective_value),
                "best_case_id": best.summary.get("case_id"),
                "directions": directions_string(best.control.directions),
                "j1": best.metrics.get("j1"),
                "j2_eval": best.metrics.get("j2_eval"),
                "j5": best.metrics.get("j5"),
                "jb_normalized": best.metrics.get("jb_normalized"),
                "jr_normalized": best.metrics.get("jr_normalized"),
                "gate_rejected": best.metrics.get("gate_rejected"),
                "feasible": best.metrics.get("feasible"),
            }
        )
    return rows


def write_matrix_outputs(*, output_root: Path, manifest: dict[str, object]) -> None:
    experiment_entries = [entry for entry in cast_entries(manifest) if str(entry.get("status")) in {"completed", "skipped_complete"}]
    summary_rows = build_matrix_summary(output_root, experiment_entries)
    write_csv(output_root / "G5_matrix_summary.csv", summary_rows)
    candidate_rows = build_candidate_union(output_root, experiment_entries)
    write_csv(output_root / "G5_candidate_union.csv", candidate_rows)
    write_csv(output_root / "G5_weight_sensitivity.csv", build_weight_sensitivity(candidate_rows))
    write_csv(output_root / "G5_ablation_comparison.csv", build_ablation_comparison(summary_rows))
    write_full_report(output_root=output_root, manifest=manifest, summary_rows=summary_rows, candidate_rows=candidate_rows)


def build_matrix_summary(output_root: Path, entries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        name = str(entry["name"])
        subdir = output_root / name
        payload_path = subdir / "G5_config_summary.json"
        if not payload_path.exists():
            continue
        payload = load_json(payload_path)
        best = payload.get("best_high_fidelity", {})
        if not isinstance(best, dict):
            best = {}
        rows.append(
            {
                "experiment": name,
                "status": entry.get("status"),
                "best_objective": best.get("objective_value"),
                "best_case_id": best.get("case_id"),
                "directions": best_directions_from_row(best),
                "j1": best.get("j1"),
                "j1_eval": best.get("j1_eval"),
                "j2_eval": best.get("j2_eval"),
                "j5": best.get("j5"),
                "j5_eval": best.get("j5_eval"),
                "jb_normalized": best.get("jb_normalized"),
                "jr_normalized": best.get("jr_normalized"),
                "gate_rejected": best.get("gate_rejected"),
                "feasible": best.get("feasible"),
                "evaluation_rows": count_csv_rows(subdir / "G5_evaluation_log.csv"),
                "hf_candidates": count_csv_rows(subdir / "G5_top_candidates.csv"),
                "output_dir": str(subdir),
            }
        )
    return rows


def build_candidate_union(output_root: Path, entries: list[dict[str, object]]) -> list[dict[str, object]]:
    best_by_signature: dict[str, dict[str, object]] = {}
    for entry in entries:
        name = str(entry["name"])
        path = output_root / name / "G5_top_candidates.csv"
        for row in read_csv(path):
            row["experiment"] = name
            signature = candidate_signature(row)
            previous = best_by_signature.get(signature)
            if previous is None or to_float(row.get("objective_value")) < to_float(previous.get("objective_value")):
                best_by_signature[signature] = row
    rows = list(best_by_signature.values())
    rows.sort(key=lambda item: to_float(item.get("objective_value")))
    for rank, row in enumerate(rows, start=1):
        row["union_rank_default_objective"] = rank
    return rows


def build_weight_sensitivity(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for weight_name, weights in WEIGHT_SETS.items():
        scored = []
        for row in candidate_rows:
            score = weighted_score(row, weights)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0])
        for rank, (score, row) in enumerate(scored, start=1):
            rows.append(
                {
                    "weight_set": weight_name,
                    "rank": rank,
                    "reranked_objective": score,
                    "experiment": row.get("experiment"),
                    "case_id": row.get("case_id"),
                    "directions": best_directions_from_row(row),
                    "j1_eval": row.get("j1_eval"),
                    "j2_eval": row.get("j2_eval"),
                    "j5_eval": row.get("j5_eval"),
                    "jb_normalized": row.get("jb_normalized"),
                    "jr_normalized": row.get("jr_normalized"),
                    "candidate_library_note": "candidate-library reranking",
                }
            )
    return rows


def build_ablation_comparison(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    main = next((row for row in summary_rows if row.get("experiment") == "main_hcmbo_full"), None)
    main_objective = to_float(main.get("best_objective")) if main else math.nan
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        objective = to_float(row.get("best_objective"))
        rows.append(
            {
                "experiment": row.get("experiment"),
                "best_objective": objective,
                "delta_vs_main": objective - main_objective if math.isfinite(main_objective) else "",
                "directions": row.get("directions"),
                "j1": row.get("j1"),
                "j2_eval": row.get("j2_eval"),
                "j5": row.get("j5"),
                "jb_normalized": row.get("jb_normalized"),
                "jr_normalized": row.get("jr_normalized"),
                "gate_rejected": row.get("gate_rejected"),
            }
        )
    return rows


def write_full_report(
    *,
    output_root: Path,
    manifest: dict[str, object],
    summary_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> None:
    completed = [row for row in summary_rows if row.get("best_objective") not in (None, "")]
    completed.sort(key=lambda item: to_float(item.get("best_objective")))
    best = completed[0] if completed else {}
    lines = [
        "# G5 V2 Full Experiment Matrix",
        "",
        "## Scope",
        "",
        "- This run expands experiment budget and ablations for the existing G5 V2 implementation.",
        "- Algorithmic limitations are intentionally left unchanged: no GP/RF/TPE BO replacement, no NOMAD/MADS, no FREE-semantics change.",
        "- Weight sensitivity is candidate-library reranking unless explicitly supplemented by additional optimization runs.",
        "",
        "## Matrix Status",
        "",
    ]
    for entry in cast_entries(manifest):
        lines.append(f"- `{entry.get('name')}`: `{entry.get('status')}`")
    lines.extend(["", "## Best Available HF Candidate", ""])
    if best:
        lines.extend(
            [
                f"- experiment: `{best.get('experiment')}`",
                f"- case: `{best.get('best_case_id')}`",
                f"- objective: `{to_float(best.get('best_objective')):.6f}`",
                f"- directions: `{best.get('directions')}`",
                f"- J1: `{to_float(best.get('j1')):.6f}`",
                f"- J2_eval: `{to_float(best.get('j2_eval')):.6f}`",
                f"- J5: `{to_float(best.get('j5')):.6f}`",
                f"- J_B normalized: `{to_float(best.get('jb_normalized')):.6f}`",
                f"- J_R normalized: `{to_float(best.get('jr_normalized')):.6f}`",
            ]
        )
    else:
        lines.append("- No completed high-fidelity candidate was found.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- `G5_matrix_manifest.json`: `{output_root / 'G5_matrix_manifest.json'}`",
            f"- `G5_matrix_summary.csv`: `{output_root / 'G5_matrix_summary.csv'}`",
            f"- `G5_ablation_comparison.csv`: `{output_root / 'G5_ablation_comparison.csv'}`",
            f"- `G5_weight_sensitivity.csv`: `{output_root / 'G5_weight_sensitivity.csv'}`",
            f"- `G5_candidate_union.csv`: `{output_root / 'G5_candidate_union.csv'}`",
            "",
            "## Interpretation Guardrails",
            "",
            "- Final recommendations should use high-fidelity rankings, not LF/MF rankings.",
            "- A lower objective than no-cap or prior baselines supports optimization benefit; otherwise report the observed target tradeoff.",
            f"- Candidate union size: `{len(candidate_rows)}`.",
        ]
    )
    output_root.joinpath("G5_full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def weighted_score(row: dict[str, object], weights: tuple[float, float, float, float, float]) -> float:
    lambda_j1, lambda_j2, lambda_j5, lambda_jb, lambda_jr = weights
    return (
        lambda_j1 * to_float(row.get("j1_eval", row.get("j1")))
        + lambda_j2 * to_float(row.get("j2_eval"))
        + lambda_j5 * to_float(row.get("j5_eval", row.get("j5")))
        + lambda_jb * to_float(row.get("jb_normalized"))
        + lambda_jr * to_float(row.get("jr_normalized"))
        + to_float(row.get("penalty"))
    )


def candidate_signature(row: dict[str, object]) -> str:
    parts = [str(row.get(f"direction_{channel}", "")) for channel in CHANNEL_NAMES]
    for gate_id in ALL_GATE_IDS:
        safe_gate = gate_id.replace(":", "_")
        parts.append(str(row.get(f"q_{safe_gate}", "")))
    return "|".join(parts)


def best_directions_from_row(row: dict[str, object]) -> str:
    return ",".join(f"{channel}:{row.get(f'direction_{channel}', '')}" for channel in CHANNEL_NAMES)


def directions_dict(directions: tuple[str, ...]) -> dict[str, str]:
    return {channel: state for channel, state in zip(CHANNEL_NAMES, directions)}


def directions_string(directions: tuple[str, ...]) -> str:
    return ",".join(f"{channel}:{state}" for channel, state in zip(CHANNEL_NAMES, directions))


def experiment_complete(path: Path) -> bool:
    return all(path.joinpath(name).exists() for name in REQUIRED_OUTPUTS)


def write_manifest(output_root: Path, manifest: dict[str, object]) -> None:
    save_json(output_root / "G5_matrix_manifest.json", manifest)


def cast_entries(manifest: dict[str, object]) -> list[dict[str, object]]:
    entries = manifest.setdefault("experiments", [])
    if not isinstance(entries, list):
        raise TypeError("manifest['experiments'] must be a list")
    return entries


def nested_get(payload: dict[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_csv_rows(path: Path) -> int:
    return len(read_csv(path))


def to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
