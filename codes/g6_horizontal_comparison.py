from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable

import numpy as np

from crowd_bellman.g5_hcmbo import (
    CHANNEL_NAMES,
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_PRIOR_DIRECTIONS,
    G5EvaluationCache,
    HCMBOConfig,
    V2ControlVector,
    V2EvaluationRecord,
    control_from_capacity_mode,
    control_from_x,
    evaluate_baselines,
    free_dimension,
    generate_direction_candidates,
    optimize_fixed_direction,
    run_random_search,
)
from crowd_bellman.metrics import save_json
from g5_experiment_matrix import (
    FidelityBudget,
    apply_budget_overrides,
    coerce_like,
    evaluate_high_fidelity_controls,
    evaluate_reference,
    method_comparison_from_groups,
    select_unique_controls,
)


DEFAULT_G6_CONFIG = Path("codes/scenes/examples/g6_horizontal_comparison/g6.toml")
DEFAULT_METHODS = (
    "baseline_prior_best",
    "random_search",
    "pure_sa",
    "tpe_mixed_bo",
    "enum_de",
    "hcmbo_proposed",
)
DEFAULT_VISUALIZATION = {
    "enabled": True,
    "top_n": 12,
    "exclude_methods": ("tpe_mixed_bo",),
    "figure_dir_name": "paper_figures_no_tpe",
}


@dataclass(frozen=True)
class G6Profile:
    name: str
    config: HCMBOConfig
    screen: FidelityBudget
    optimization: FidelityBudget
    high_fidelity: FidelityBudget
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    output_root: Path
    baseline_config: Path


@dataclass(frozen=True)
class G6Method:
    name: str
    runner: Callable[[Path, G6Profile, int], dict[str, object]]
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G6 horizontal comparison for z=(s,q) optimizers.")
    parser.add_argument("--config", default=str(DEFAULT_G6_CONFIG))
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated method names.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel worker count for method/seed runs.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--visualize-only", action="store_true", help="Only draw figures from existing G6 CSV/JSON outputs.")
    parser.add_argument("--no-visualization", action="store_true", help="Skip chart generation after writing G6 result tables.")
    parser.add_argument("--visual-top-n", type=int, default=None, help="Number of top HF candidates to export.")
    parser.add_argument("--visual-exclude-methods", default=None, help="Comma-separated methods to omit from figures.")
    parser.add_argument("--visual-figure-dir-name", default=None, help="Output subdirectory name for G6 paper figures.")
    args = parser.parse_args()

    loaded = load_g6_config(Path(args.config)) if args.config else {}
    loaded_profile = str(loaded.get("profile") or "full")
    profile_name = args.profile or loaded_profile
    seed_values = parse_seeds(args.seeds) if args.seeds else tuple(int(item) for item in loaded.get("seeds", (23,)))
    methods = parse_methods(args.methods or loaded.get("methods") or ",".join(DEFAULT_METHODS))
    output_root = Path(args.output_root or loaded.get("output_root") or "codes/results/g6_horizontal_comparison").resolve()
    baseline_config = Path(args.baseline_config or loaded.get("baseline_config") or DEFAULT_BASELINE_CONFIG).resolve()
    force = bool(args.force or loaded.get("force", False))
    fail_fast = bool(args.fail_fast or loaded.get("fail_fast", False))
    workers = resolve_worker_count(args.workers if args.workers is not None else loaded.get("workers"), len(methods) * len(seed_values))
    visualization = dict(loaded.get("visualization") or DEFAULT_VISUALIZATION)
    if args.no_visualization:
        visualization["enabled"] = False
    if args.visual_top_n is not None:
        visualization["top_n"] = args.visual_top_n
    if args.visual_exclude_methods is not None:
        visualization["exclude_methods"] = parse_visual_exclude_methods(args.visual_exclude_methods)
    if args.visual_figure_dir_name is not None:
        visualization["figure_dir_name"] = args.visual_figure_dir_name

    if args.visualize_only:
        run_g6_visualization(output_root, visualization)
        return

    profile = profile_from_name(
        profile_name,
        output_root=output_root,
        baseline_config=baseline_config,
        seeds=seed_values,
        methods=methods,
    )
    if loaded.get("overrides") and (args.profile is None or args.profile == loaded_profile):
        profile = apply_profile_overrides(profile, loaded["overrides"])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "experiment_group": "G6",
        "design_version": "horizontal_comparison_budget_s",
        "profile": profile.name,
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "baseline_config": str(profile.baseline_config),
        "output_root": str(profile.output_root),
        "methods": list(profile.methods),
        "seeds": list(profile.seeds),
        "workers": workers,
        "visualization": {
            "enabled": bool(visualization.get("enabled", True)),
            "top_n": int(visualization.get("top_n", 12)),
            "exclude_methods": list(parse_visual_exclude_methods(visualization.get("exclude_methods", ()))),
            "figure_dir_name": str(visualization.get("figure_dir_name", "paper_figures_no_tpe")),
        },
        "argv": sys.argv,
        "runs": [],
    }
    save_json(output_root / "G6_manifest.json", manifest)

    failures = run_selected_runs(
        profile=profile,
        manifest=manifest,
        output_root=output_root,
        force=force,
        fail_fast=fail_fast,
        workers=workers,
    )

    save_g6_outputs(output_root=output_root, manifest=manifest)
    if bool(visualization.get("enabled", True)):
        run_g6_visualization(output_root, visualization)
    if failures:
        raise RuntimeError(f"G6 failed runs: {', '.join(failures)}")
    print(f"G6 summary: {output_root / 'G6_method_summary.csv'}")


def profile_from_name(
    name: str,
    *,
    output_root: Path,
    baseline_config: Path,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
) -> G6Profile:
    if name == "smoke":
        return G6Profile(
            name=name,
            config=HCMBOConfig(
                time_segments=2,
                direction_candidate_limit=3,
                shortlist_size=3,
                screen_capacity_modes=("high",),
                initial_samples=1,
                bo_iterations=0,
                dfo_evaluations=0,
                high_fidelity_top_k=1,
                random_search_evaluations=1,
                random_seed=seeds[0],
            ),
            screen=FidelityBudget(steps=6, time_horizon=0.6, bellman_every=3),
            optimization=FidelityBudget(steps=6, time_horizon=0.6, bellman_every=3),
            high_fidelity=FidelityBudget(steps=6, time_horizon=0.6, bellman_every=3),
            methods=methods,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    if name == "full":
        return G6Profile(
            name=name,
            config=HCMBOConfig(
                time_segments=4,
                direction_candidate_limit=12,
                shortlist_size=12,
                screen_capacity_modes=("high", "medium"),
                initial_samples=8,
                bo_iterations=12,
                bo_candidate_pool=48,
                dfo_top_k=1,
                dfo_evaluations=5,
                high_fidelity_top_k=10,
                random_search_evaluations=400,
                random_seed=seeds[0],
            ),
            screen=FidelityBudget(steps=60, time_horizon=4.0, bellman_every=6),
            optimization=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            high_fidelity=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            methods=methods,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    raise ValueError(f"Unsupported G6 profile: {name!r}")


def load_g6_config(path: Path) -> dict[str, object]:
    base_dir = path.resolve().parent
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    g6 = dict(raw.get("g6", {}))
    overrides: dict[str, object] = {}
    for table_name in ("hcmbo", "screen", "optimization", "high_fidelity"):
        table = raw.get(table_name, {})
        if table:
            if not isinstance(table, dict):
                raise ValueError(f"[{table_name}] must be a table")
            overrides[table_name] = dict(table)
    weights = raw.get("weights", {})
    if weights:
        hcmbo = dict(overrides.get("hcmbo", {}))
        hcmbo.update(dict(weights))
        overrides["hcmbo"] = hcmbo
    visualization = dict(DEFAULT_VISUALIZATION)
    visualization_table = raw.get("visualization", {})
    if visualization_table:
        if not isinstance(visualization_table, dict):
            raise ValueError("[visualization] must be a table")
        visualization.update(dict(visualization_table))
    visualization["enabled"] = bool(visualization.get("enabled", True))
    visualization["top_n"] = int(visualization.get("top_n", 12))
    visualization["exclude_methods"] = parse_visual_exclude_methods(visualization.get("exclude_methods", ()))
    visualization["figure_dir_name"] = str(visualization.get("figure_dir_name", "paper_figures_no_tpe"))
    return {
        "profile": g6.get("profile"),
        "output_root": resolve_config_path(base_dir, str(g6["output_root"])) if "output_root" in g6 else None,
        "baseline_config": resolve_config_path(base_dir, str(g6["baseline_config"])) if "baseline_config" in g6 else None,
        "methods": parse_methods(g6.get("methods", ",".join(DEFAULT_METHODS))),
        "seeds": parse_seeds(g6.get("seeds", "23")),
        "workers": int(g6["workers"]) if "workers" in g6 else None,
        "force": bool(g6["force"]) if "force" in g6 else False,
        "fail_fast": bool(g6["fail_fast"]) if "fail_fast" in g6 else False,
        "overrides": overrides,
        "visualization": visualization,
    }


def apply_profile_overrides(profile: G6Profile, overrides: object) -> G6Profile:
    if not isinstance(overrides, dict):
        return profile
    config = profile.config
    hcmbo = overrides.get("hcmbo", {})
    if hcmbo:
        if not isinstance(hcmbo, dict):
            raise ValueError("[hcmbo] overrides must be a table")
        valid = {field.name for field in fields(HCMBOConfig)}
        replace_kwargs: dict[str, object] = {}
        for key, value in hcmbo.items():
            if key not in valid:
                raise ValueError(f"Unsupported hcmbo config key: {key}")
            replace_kwargs[key] = coerce_like(getattr(config, key), value)
        config = replace(config, **replace_kwargs)
    return replace(
        profile,
        config=config,
        screen=apply_budget_overrides(profile.screen, overrides.get("screen", {}), "screen"),
        optimization=apply_budget_overrides(profile.optimization, overrides.get("optimization", {}), "optimization"),
        high_fidelity=apply_budget_overrides(profile.high_fidelity, overrides.get("high_fidelity", {}), "high_fidelity"),
    )


def build_methods() -> list[G6Method]:
    return [
        G6Method("baseline_prior_best", run_baseline_prior_best, "Best HF candidate among no-cap and prior-capacity policies."),
        G6Method("random_search", run_g6_random_search, "Mixed-variable random search under the shared budget."),
        G6Method("pure_sa", run_pure_sa, "Mixed simulated annealing over direction and capacity variables."),
        G6Method("tpe_mixed_bo", run_tpe_mixed_bo, "Dependency-light TPE-style mixed-variable Bayesian optimization baseline."),
        G6Method("enum_de", run_enum_de, "Direction-wise differential evolution continuous black-box baseline."),
        G6Method("hcmbo_proposed", run_hcmbo_proposed, "Proposed HCMBO with no LF hard shortlist."),
    ]


def resolve_worker_count(raw_workers: object, selected_count: int) -> int:
    if selected_count <= 0:
        return 1
    if raw_workers is None:
        return selected_count
    return max(1, min(int(raw_workers), selected_count))


def run_selected_runs(
    *,
    profile: G6Profile,
    manifest: dict[str, object],
    output_root: Path,
    force: bool,
    fail_fast: bool,
    workers: int,
) -> list[str]:
    entries_by_key: dict[str, dict[str, object]] = {}
    run_payloads: list[dict[str, object]] = []
    failures: list[str] = []
    by_name = {method.name: method for method in build_methods()}

    for seed in profile.seeds:
        for method_name in profile.methods:
            method = by_name[method_name]
            key = run_key(method_name, seed)
            run_dir = output_root / method_name / f"seed_{seed}"
            entry = {
                "key": key,
                "method": method_name,
                "seed": seed,
                "description": method.description,
                "output_dir": str(run_dir),
                "status": "pending",
            }
            cast_runs(manifest).append(entry)
            entries_by_key[key] = entry

            required = run_dir / "method_summary.json"
            if required.exists() and not force:
                payload = load_json(required)
                entry.update({"status": "skipped_complete", **summary_entry(payload)})
            else:
                entry["status"] = "running"
                run_payloads.append(
                    {
                        "key": key,
                        "method": method_name,
                        "seed": seed,
                        "output_dir": str(run_dir),
                        "profile": profile,
                    }
                )

    save_json(output_root / "G6_manifest.json", manifest)
    save_g6_outputs(output_root=output_root, manifest=manifest)

    if not run_payloads:
        return failures

    if workers == 1:
        for payload in run_payloads:
            result = run_g6_payload(payload)
            apply_g6_result(
                result=result,
                entries_by_key=entries_by_key,
                failures=failures,
                output_root=output_root,
                manifest=manifest,
            )
            save_g6_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                raise RuntimeError(result.get("error", "G6 run failed"))
        return failures

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_g6_payload, payload): str(payload["key"]) for payload in run_payloads}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "key": key,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            apply_g6_result(
                result=result,
                entries_by_key=entries_by_key,
                failures=failures,
                output_root=output_root,
                manifest=manifest,
            )
            save_g6_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(result.get("error", "G6 run failed"))
    return failures


def run_g6_payload(payload: dict[str, object]) -> dict[str, object]:
    key = str(payload["key"])
    method_name = str(payload["method"])
    seed = int(payload["seed"])
    try:
        by_name = {method.name: method for method in build_methods()}
        method = by_name[method_name]
        profile = payload["profile"]
        if not isinstance(profile, G6Profile):
            raise TypeError("profile payload must be a G6Profile")
        start = time.perf_counter()
        result_payload = method.runner(
            Path(str(payload["output_dir"])),
            replace(profile, config=replace(profile.config, random_seed=seed)),
            seed,
        )
        result_payload["runtime_seconds"] = time.perf_counter() - start
        save_json(Path(str(payload["output_dir"])) / "method_summary.json", result_payload)
        return {
            "key": key,
            "method": method_name,
            "seed": seed,
            "status": "completed",
            **summary_entry(result_payload),
        }
    except Exception as exc:
        return {
            "key": key,
            "method": method_name,
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def apply_g6_result(
    *,
    result: dict[str, object],
    entries_by_key: dict[str, dict[str, object]],
    failures: list[str],
    output_root: Path,
    manifest: dict[str, object],
) -> None:
    key = str(result["key"])
    entry = entries_by_key[key]
    entry.update(result)
    if result.get("status") == "failed" and key not in failures:
        failures.append(key)
    save_json(output_root / "G6_manifest.json", manifest)


def run_baseline_prior_best(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=profile.config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    mid_records = evaluate_baselines(evaluator=evaluator, qbar_by_gate=qbar_by_gate, config=profile.config)
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, profile.config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="baseline_prior_best",
        profile=profile,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={},
    )


def run_g6_random_search(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(seed)
    directions = generate_direction_candidates(config=profile.config, rng=rng)
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=profile.config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    mid_records = run_random_search(
        evaluator=evaluator,
        directions_list=directions,
        qbar_by_gate=qbar_by_gate,
        config=profile.config,
        rng=rng,
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, profile.config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="random_search",
        profile=profile,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={"direction_candidate_count": len(directions)},
    )


def run_pure_sa(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(seed)
    directions_list = generate_direction_candidates(config=profile.config, rng=rng)
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=profile.config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    mid_records = run_sa_search(
        evaluator=evaluator,
        directions_list=directions_list,
        qbar_by_gate=qbar_by_gate,
        config=profile.config,
        rng=rng,
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, profile.config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="pure_sa",
        profile=profile,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={"direction_candidate_count": len(directions_list)},
    )


def run_tpe_mixed_bo(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(seed)
    directions_list = generate_direction_candidates(config=profile.config, rng=rng)
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=profile.config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    mid_records = run_tpe_search(
        evaluator=evaluator,
        directions_list=directions_list,
        qbar_by_gate=qbar_by_gate,
        config=profile.config,
        rng=rng,
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, profile.config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="tpe_mixed_bo",
        profile=profile,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={"direction_candidate_count": len(directions_list), "tpe_variant": "local_tree_parzen_style"},
    )


def run_enum_de(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(seed)
    directions_list = generate_direction_candidates(config=profile.config, rng=rng)
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=profile.config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    mid_records = run_enum_de_search(
        evaluator=evaluator,
        directions_list=directions_list,
        qbar_by_gate=qbar_by_gate,
        config=profile.config,
        rng=rng,
    )
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=profile.config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, profile.config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="enum_de",
        profile=profile,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={"direction_candidate_count": len(directions_list), "de_variant": "equal_direction_budget"},
    )


def run_hcmbo_proposed(output_dir: Path, profile: G6Profile, seed: int) -> dict[str, object]:
    config = replace(
        profile.config,
        random_seed=seed,
        shortlist_size=profile.config.direction_candidate_limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, qbar_by_gate = evaluate_reference(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=config,
        optimization_overrides=profile.optimization.to_overrides(),
    )
    rng = np.random.default_rng(seed)
    directions_list = generate_direction_candidates(config=config, rng=rng)
    evaluator = G5EvaluationCache(
        baseline_config=profile.baseline_config,
        output_root=output_dir / "_optimization",
        objective_config=config,
        simulation_overrides=profile.optimization.to_overrides(),
        fidelity="mf",
    )
    hcmbo_records: list[V2EvaluationRecord] = []
    for directions in directions_list:
        records, _trace = optimize_fixed_direction(
            evaluator=evaluator,
            directions=directions,
            qbar_by_gate=qbar_by_gate,
            config=config,
            rng=rng,
            source_prefix="hcmbo",
        )
        hcmbo_records.extend(records)
    remaining_budget = max(0, int(config.random_search_evaluations) - len(hcmbo_records))
    random_records = run_random_search(
        evaluator=evaluator,
        directions_list=directions_list,
        qbar_by_gate=qbar_by_gate,
        config=replace(config, random_search_evaluations=remaining_budget),
        rng=rng,
    )
    mid_records = random_records + hcmbo_records
    hf_records = evaluate_high_fidelity_controls(
        baseline_config=profile.baseline_config,
        output_dir=output_dir,
        config=config,
        qbar_by_gate=qbar_by_gate,
        overrides=profile.high_fidelity.to_overrides(),
        controls=select_unique_controls(mid_records, config.high_fidelity_top_k),
    )
    return write_method_outputs(
        output_dir=output_dir,
        method_name="hcmbo_proposed",
        profile=replace(profile, config=config),
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        extra={
            "direction_candidate_count": len(directions_list),
            "hcmbo_structured_evaluations": len(hcmbo_records),
            "hcmbo_internal_random_evaluations": len(random_records),
        },
    )


def run_tpe_search(
    *,
    evaluator: G5EvaluationCache,
    directions_list: list[tuple[str, ...]],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
) -> list[V2EvaluationRecord]:
    budget = max(1, int(config.random_search_evaluations))
    warmup = min(budget, max(4, min(len(directions_list), budget // 4 if budget >= 8 else budget)))
    records: list[V2EvaluationRecord] = []
    history: list[tuple[tuple[str, ...], np.ndarray, V2EvaluationRecord]] = []
    for _ in range(warmup):
        directions = directions_list[int(rng.integers(0, len(directions_list)))]
        x = rng.random(free_dimension(directions, config.time_segments))
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=x, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source="tpe_mixed_bo",
            phase="tpe_warmup",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        history.append((directions, x, record))

    for step in range(warmup, budget):
        elite_count = max(1, int(math.ceil(len(history) * 0.25)))
        elite = sorted(history, key=lambda item: item[2].objective_value)[:elite_count]
        if rng.random() < 0.8:
            directions, parent_x, _parent = elite[int(rng.integers(0, len(elite)))]
            sigma = max(0.04, 0.30 * (1.0 - step / max(budget, 1)))
            x = np.clip(parent_x + rng.normal(0.0, sigma, size=parent_x.size), 0.0, 1.0)
        else:
            directions = directions_list[int(rng.integers(0, len(directions_list)))]
            x = rng.random(free_dimension(directions, config.time_segments))
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=x, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source="tpe_mixed_bo",
            phase="tpe_suggest",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        history.append((directions, x, record))
    return records


def run_enum_de_search(
    *,
    evaluator: G5EvaluationCache,
    directions_list: list[tuple[str, ...]],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
) -> list[V2EvaluationRecord]:
    budget = max(1, int(config.random_search_evaluations))
    records: list[V2EvaluationRecord] = []
    direction_count = max(1, len(directions_list))
    base_budget = budget // direction_count
    remainder = budget % direction_count
    for direction_index, directions in enumerate(directions_list):
        local_budget = base_budget + (1 if direction_index < remainder else 0)
        if local_budget <= 0:
            continue
        records.extend(
            run_de_for_direction(
                evaluator=evaluator,
                directions=directions,
                qbar_by_gate=qbar_by_gate,
                config=config,
                rng=rng,
                budget=local_budget,
            )
        )
    return records


def run_de_for_direction(
    *,
    evaluator: G5EvaluationCache,
    directions: tuple[str, ...],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
    budget: int,
) -> list[V2EvaluationRecord]:
    dim = free_dimension(directions, config.time_segments)
    if dim <= 0:
        return []
    pop_size = min(max(4, 2 * dim), max(1, budget))
    population = [rng.random(dim) for _ in range(pop_size)]
    records: list[V2EvaluationRecord] = []
    scored: list[tuple[np.ndarray, V2EvaluationRecord]] = []
    for x in population:
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=x, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source="enum_de",
            phase="de_init",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        scored.append((x, record))
        if len(records) >= budget:
            return records

    cursor = 0
    while len(records) < budget:
        target_x, target_record = scored[cursor % len(scored)]
        choices = rng.choice(len(scored), size=3, replace=len(scored) < 3)
        a, b, c = (scored[int(index)][0] for index in choices)
        mutant = np.clip(a + 0.65 * (b - c), 0.0, 1.0)
        mask = rng.random(dim) < 0.7
        if not np.any(mask):
            mask[int(rng.integers(0, dim))] = True
        trial = np.where(mask, mutant, target_x)
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=trial, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source="enum_de",
            phase="de",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        if record.objective_value < target_record.objective_value:
            scored[cursor % len(scored)] = (trial, record)
        cursor += 1
    return records


def run_sa_search(
    *,
    evaluator: G5EvaluationCache,
    directions_list: list[tuple[str, ...]],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
) -> list[V2EvaluationRecord]:
    budget = max(1, int(config.random_search_evaluations))
    directions = directions_list[int(rng.integers(0, len(directions_list)))]
    x = rng.random(free_dimension(directions, config.time_segments))
    current = evaluator.evaluate(
        control_from_x(directions=directions, x=x, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
        source="pure_sa",
        phase="sa",
        qbar_by_gate=qbar_by_gate,
        record_cached=True,
    )
    records = [current]
    t0 = max(0.05, abs(current.objective_value) * 0.15)
    for step in range(1, budget):
        trial_directions = directions
        trial_x = np.array(x, copy=True)
        if rng.random() < 0.35 or trial_x.size == 0:
            trial_directions = directions_list[int(rng.integers(0, len(directions_list)))]
            trial_x = rng.random(free_dimension(trial_directions, config.time_segments))
        else:
            sigma = max(0.04, 0.25 * (1.0 - step / max(budget, 1)))
            trial_x = np.clip(trial_x + rng.normal(0.0, sigma, size=trial_x.size), 0.0, 1.0)
        record = evaluator.evaluate(
            control_from_x(
                directions=trial_directions,
                x=trial_x,
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            ),
            source="pure_sa",
            phase="sa",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        temp = max(1.0e-6, t0 * (1.0 - step / max(budget, 1)))
        delta = record.objective_value - current.objective_value
        if delta <= 0.0 or rng.random() < math.exp(-delta / temp):
            directions, x, current = trial_directions, trial_x, record
    return records


def write_method_outputs(
    *,
    output_dir: Path,
    method_name: str,
    profile: G6Profile,
    seed: int,
    reference: V2EvaluationRecord,
    qbar_by_gate: dict[str, float],
    mid_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
    extra: dict[str, object],
) -> dict[str, object]:
    best = min(hf_records or mid_records, key=lambda item: item.objective_value)
    all_rows = [reference.to_row()] + [record.to_row() for record in mid_records] + [record.to_row() for record in hf_records]
    candidate_rows = [prefix_row(record.to_row(), method_name, seed) for record in sorted(hf_records or mid_records, key=lambda item: item.objective_value)]
    write_csv(output_dir / "G6_evaluation_log.csv", [prefix_row(row, method_name, seed) for row in all_rows])
    write_csv(output_dir / "G6_hf_candidates.csv", candidate_rows)
    write_csv(output_dir / "G6_convergence_curves.csv", build_convergence_rows([record.to_row() for record in mid_records], method_name, seed))
    write_csv(output_dir / "G6_method_comparison.csv", method_comparison_from_groups({method_name: hf_records or mid_records}))
    save_json(output_dir / "G6_best_control.json", best.control.to_dict())
    payload: dict[str, object] = {
        "method": method_name,
        "seed": seed,
        "profile": profile.name,
        "best_high_fidelity": best.to_row(),
        "candidate_count": len(candidate_rows),
        "optimization_evaluation_count": len(mid_records),
        "hf_candidate_count": len(hf_records),
        "qbar_by_gate": qbar_by_gate,
        "config": profile.config.__dict__,
        "optimization_overrides": profile.optimization.to_overrides(),
        "high_fidelity_overrides": profile.high_fidelity.to_overrides(),
        "outputs": {
            "evaluation_log": str(output_dir / "G6_evaluation_log.csv"),
            "hf_candidates": str(output_dir / "G6_hf_candidates.csv"),
            "convergence_curves": str(output_dir / "G6_convergence_curves.csv"),
            "best_control": str(output_dir / "G6_best_control.json"),
        },
    }
    payload.update(extra)
    return payload


def save_g6_outputs(*, output_root: Path, manifest: dict[str, object]) -> None:
    save_json(output_root / "G6_manifest.json", manifest)
    summaries: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    curves: list[dict[str, object]] = []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict) or run.get("status") not in {"completed", "skipped_complete"}:
            continue
        method = str(run["method"])
        seed = int(run["seed"])
        run_dir = Path(str(run["output_dir"]))
        payload_path = run_dir / "method_summary.json"
        payload = load_json(payload_path) if payload_path.exists() else {}
        best = payload.get("best_high_fidelity", {}) if isinstance(payload, dict) else {}
        if not isinstance(best, dict):
            best = {}
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "best_hf_objective_default": best.get("objective_value"),
                "best_case_id": best.get("case_id"),
                "feasible_best": best.get("feasible"),
                "J1_eval": best.get("j1_eval"),
                "J2_eval": best.get("j2_eval"),
                "J5_eval": best.get("j5_eval"),
                "JB_normalized": best.get("jb_normalized"),
                "JR_normalized": best.get("jr_normalized"),
                "gate_rejected": best.get("gate_rejected"),
                "runtime_seconds": payload.get("runtime_seconds") if isinstance(payload, dict) else None,
                "output_dir": str(run_dir),
            }
        )
        candidates.extend(read_csv(run_dir / "G6_hf_candidates.csv"))
        curves.extend(read_csv(run_dir / "G6_convergence_curves.csv"))
    write_csv(output_root / "G6_method_summary.csv", build_method_summary_rows(summaries))
    write_csv(output_root / "G6_seed_summary.csv", summaries)
    write_csv(output_root / "G6_hf_candidates.csv", candidates)
    write_csv(output_root / "G6_convergence_curves.csv", curves)
    write_csv(output_root / "G6_statistical_tests.csv", build_pairwise_delta_rows(summaries))
    write_report(output_root, summaries)


def build_method_summary_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted({str(row["method"]) for row in seed_rows})
    for method in methods:
        values = [
            to_float(row.get("best_hf_objective_default"))
            for row in seed_rows
            if str(row.get("method")) == method and math.isfinite(to_float(row.get("best_hf_objective_default")))
        ]
        feasible_flags = [str(row.get("feasible_best")).lower() == "true" for row in seed_rows if str(row.get("method")) == method]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        method_seed_rows = [row for row in seed_rows if str(row.get("method")) == method]
        best_row = min(method_seed_rows, key=lambda row: to_float(row.get("best_hf_objective_default")))
        rows.append(
            {
                "method": method,
                "seed_count": len(values),
                "mean_best_hf_objective": float(np.mean(arr)),
                "std_best_hf_objective": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median_best_hf_objective": float(np.median(arr)),
                "iqr_best_hf_objective": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) > 1 else 0.0,
                "best_hf_objective": float(np.min(arr)),
                "worst_hf_objective": float(np.max(arr)),
                "feasible_rate": sum(1 for item in feasible_flags if item) / max(len(feasible_flags), 1),
                "best_seed": best_row.get("seed"),
                "best_case_id": best_row.get("best_case_id"),
            }
        )
    return rows


def build_convergence_rows(rows: list[dict[str, object]], method: str, seed: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    best = math.inf
    eval_index = 0
    for row in rows:
        if str(row.get("fidelity")) == "reference":
            continue
        value = to_float(row.get("objective_value"))
        if not math.isfinite(value):
            continue
        eval_index += 1
        best = min(best, value)
        result.append(
            {
                "method": method,
                "seed": seed,
                "evaluation_index": eval_index,
                "objective_value": value,
                "best_so_far": best,
                "source": row.get("source"),
                "phase": row.get("phase"),
                "fidelity": row.get("fidelity"),
            }
        )
    return result


def build_pairwise_delta_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in summaries}
    methods = sorted({str(row["method"]) for row in summaries})
    seeds = sorted({int(row["seed"]) for row in summaries})
    rows: list[dict[str, object]] = []
    for left in methods:
        for right in methods:
            if left >= right:
                continue
            deltas = []
            for seed in seeds:
                a = by_seed_method.get((seed, left))
                b = by_seed_method.get((seed, right))
                if not a or not b:
                    continue
                deltas.append(to_float(a.get("best_hf_objective_default")) - to_float(b.get("best_hf_objective_default")))
            if deltas:
                wins = sum(1 for delta in deltas if delta < 0.0)
                losses = sum(1 for delta in deltas if delta > 0.0)
                ties = len(deltas) - wins - losses
                p_value = exact_two_sided_sign_test_p(wins, losses)
                rows.append(
                    {
                        "method_a": left,
                        "method_b": right,
                        "paired_seed_count": len(deltas),
                        "mean_delta_a_minus_b": sum(deltas) / len(deltas),
                        "median_delta_a_minus_b": float(np.median(np.array(deltas, dtype=float))),
                        "method_a_wins": wins,
                        "method_b_wins": losses,
                        "ties": ties,
                        "sign_test_p": p_value,
                        "vargha_delaney_a12_a_better": (wins + 0.5 * ties) / max(len(deltas), 1),
                        "cliffs_delta_a_better": (wins - losses) / max(len(deltas), 1),
                    }
                )
    apply_holm_bonferroni(rows)
    return rows


def exact_two_sided_sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    probability = sum(math.comb(n, item) for item in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * probability)


def apply_holm_bonferroni(rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: to_float(row.get("sign_test_p", 1.0)))
    m = len(ordered)
    previous = 0.0
    for rank, row in enumerate(ordered, start=1):
        adjusted = min(1.0, max(previous, (m - rank + 1) * to_float(row.get("sign_test_p", 1.0))))
        row["holm_bonferroni_p"] = adjusted
        previous = adjusted


def write_report(output_root: Path, summaries: list[dict[str, object]]) -> None:
    lines = ["# G6 Horizontal Comparison Report", "", "## Method Summary", ""]
    for row in build_method_summary_rows(summaries):
        lines.append(
            f"- `{row.get('method')}`: mean `{row.get('mean_best_hf_objective')}`, "
            f"median `{row.get('median_best_hf_objective')}`, best `{row.get('best_hf_objective')}`, "
            f"feasible rate `{row.get('feasible_rate')}`"
        )
    lines.extend(["", "## Seed Results", ""])
    for row in sorted(summaries, key=lambda item: (str(item.get("method")), int(item.get("seed", 0)))):
        lines.append(
            f"- `{row.get('method')}` seed `{row.get('seed')}`: objective `{row.get('best_hf_objective_default')}`, "
            f"feasible `{row.get('feasible_best')}`, case `{row.get('best_case_id')}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `G6_manifest.json`: `{output_root / 'G6_manifest.json'}`",
            f"- `G6_method_summary.csv`: `{output_root / 'G6_method_summary.csv'}`",
            f"- `G6_hf_candidates.csv`: `{output_root / 'G6_hf_candidates.csv'}`",
            f"- `G6_convergence_curves.csv`: `{output_root / 'G6_convergence_curves.csv'}`",
        ]
    )
    output_root.joinpath("G6_full_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_g6_visualization(output_root: Path, visualization: dict[str, object]) -> None:
    from crowd_bellman.g6_visualization import build_g6_visual_report

    report = build_g6_visual_report(
        output_root,
        exclude_methods=parse_visual_exclude_methods(visualization.get("exclude_methods", ())),
        top_n=int(visualization.get("top_n", 12)),
        figure_dir_name=str(visualization.get("figure_dir_name", "paper_figures_no_tpe")),
    )
    print(f"Wrote G6 visual report to {report['output_root']}")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")


def summary_entry(payload: dict[str, object]) -> dict[str, object]:
    best = payload.get("best_high_fidelity", {})
    if not isinstance(best, dict):
        best = {}
    return {
        "best_objective": best.get("objective_value"),
        "best_case_id": best.get("case_id"),
        "feasible": best.get("feasible"),
    }


def parse_methods(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        methods = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, list | tuple):
        methods = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise ValueError("methods must be a comma-separated string or list")
    valid = {method.name for method in build_methods()}
    unknown = [method for method in methods if method not in valid]
    if unknown:
        raise ValueError(f"Unknown G6 methods: {', '.join(unknown)}")
    return methods


def parse_seeds(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    elif isinstance(value, list | tuple):
        seeds = tuple(int(item) for item in value)
    else:
        seeds = (int(value),)
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def parse_visual_exclude_methods(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def resolve_config_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def cast_runs(manifest: dict[str, object]) -> list[dict[str, object]]:
    runs = manifest.setdefault("runs", [])
    if not isinstance(runs, list):
        raise TypeError("manifest['runs'] must be a list")
    return runs


def run_key(method: str, seed: int) -> str:
    return f"{method}:seed_{seed}"


def prefix_row(row: dict[str, object], method: str, seed: int) -> dict[str, object]:
    output = dict(row)
    output["method"] = method
    output["seed"] = seed
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


if __name__ == "__main__":
    main()
