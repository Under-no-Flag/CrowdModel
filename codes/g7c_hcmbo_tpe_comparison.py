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
    DEFAULT_BASELINE_CONFIG,
    G5EvaluationCache,
    HCMBOConfig,
    V2EvaluationRecord,
    generate_direction_candidates,
    run_random_search,
)
from crowd_bellman.metrics import save_json
from g5_experiment_matrix import (
    FidelityBudget,
    apply_budget_overrides,
    coerce_like,
    evaluate_reference,
    method_comparison_from_groups,
)
from g6_horizontal_comparison import run_tpe_search
from g7_hcmbo_variant_ablation import (
    RunContext,
    evaluate_high_fidelity_with_origin,
    objective_score,
    queue_aware_score,
    run_equal_direction_lcb_budget,
    select_top_unique_records,
    with_source_family,
)


DEFAULT_G7C_CONFIG = Path("codes/scenes/examples/g7c_hcmbo_tpe_comparison/g7c.toml")
DEFAULT_METHODS = (
    "hcmbo_structured_only",
    "hcmbo_queue_aware_lcb",
    "hcmbo_structured_queue_aware",
    "tpe_mixed_bo",
)


@dataclass(frozen=True)
class G7CProfile:
    name: str
    config: HCMBOConfig
    optimization: FidelityBudget
    high_fidelity: FidelityBudget
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    output_root: Path
    baseline_config: Path


@dataclass(frozen=True)
class G7CMethod:
    name: str
    runner: Callable[[Path, G7CProfile, int], dict[str, object]]
    description: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G7-C HCMBO/TPE focused comparison.")
    parser.add_argument("--config", default=str(DEFAULT_G7C_CONFIG))
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated method names.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    loaded = load_g7c_config(Path(args.config)) if args.config else {}
    loaded_profile = str(loaded.get("profile") or "full")
    profile_name = args.profile or loaded_profile
    seeds = parse_seeds(args.seeds) if args.seeds else tuple(int(item) for item in loaded.get("seeds", (23,)))
    methods = parse_methods(args.methods or loaded.get("methods") or ",".join(DEFAULT_METHODS))
    output_root = Path(args.output_root or loaded.get("output_root") or "codes/results/g7c_hcmbo_tpe_comparison").resolve()
    baseline_config = Path(args.baseline_config or loaded.get("baseline_config") or DEFAULT_BASELINE_CONFIG).resolve()
    force = bool(args.force or loaded.get("force", False))
    fail_fast = bool(args.fail_fast or loaded.get("fail_fast", False))
    workers = resolve_worker_count(args.workers if args.workers is not None else loaded.get("workers"), len(methods) * len(seeds))

    profile = profile_from_name(
        profile_name,
        output_root=output_root,
        baseline_config=baseline_config,
        seeds=seeds,
        methods=methods,
    )
    if loaded.get("overrides") and (args.profile is None or args.profile == loaded_profile):
        profile = apply_profile_overrides(profile, loaded["overrides"])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "experiment_group": "G7-C",
        "design_version": "focused_hcmbo_tpe_comparison",
        "profile": profile.name,
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "baseline_config": str(profile.baseline_config),
        "output_root": str(profile.output_root),
        "methods": list(profile.methods),
        "seeds": list(profile.seeds),
        "workers": workers,
        "argv": sys.argv,
        "runs": [],
    }
    save_json(output_root / "G7C_manifest.json", manifest)
    failures = run_selected_runs(
        profile=profile,
        manifest=manifest,
        output_root=output_root,
        force=force,
        fail_fast=fail_fast,
        workers=workers,
    )
    write_g7c_outputs(output_root=output_root, manifest=manifest)
    if failures:
        raise RuntimeError(f"G7-C failed runs: {', '.join(failures)}")
    print(f"G7-C summary: {output_root / 'G7C_method_summary.csv'}")


def profile_from_name(
    name: str,
    *,
    output_root: Path,
    baseline_config: Path,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
) -> G7CProfile:
    if name == "smoke":
        return G7CProfile(
            name=name,
            config=HCMBOConfig(
                time_segments=2,
                direction_candidate_limit=3,
                shortlist_size=3,
                initial_samples=1,
                bo_iterations=0,
                bo_candidate_pool=8,
                dfo_top_k=1,
                dfo_evaluations=0,
                high_fidelity_top_k=1,
                random_search_evaluations=6,
                random_seed=seeds[0],
            ),
            optimization=FidelityBudget(steps=6, time_horizon=0.6, bellman_every=3),
            high_fidelity=FidelityBudget(steps=6, time_horizon=0.6, bellman_every=3),
            methods=methods,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    if name == "full":
        return G7CProfile(
            name=name,
            config=HCMBOConfig(
                time_segments=4,
                direction_candidate_limit=12,
                shortlist_size=12,
                initial_samples=8,
                bo_iterations=12,
                bo_candidate_pool=48,
                dfo_top_k=1,
                dfo_evaluations=5,
                high_fidelity_top_k=10,
                random_search_evaluations=400,
                random_seed=seeds[0],
            ),
            optimization=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            high_fidelity=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            methods=methods,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    raise ValueError(f"Unsupported G7-C profile: {name!r}")


def load_g7c_config(path: Path) -> dict[str, object]:
    base_dir = path.resolve().parent
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    g7c = dict(raw.get("g7c", {}))
    overrides: dict[str, object] = {}
    for table_name in ("hcmbo", "optimization", "high_fidelity"):
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
    return {
        "profile": g7c.get("profile"),
        "output_root": resolve_config_path(base_dir, str(g7c["output_root"])) if "output_root" in g7c else None,
        "baseline_config": resolve_config_path(base_dir, str(g7c["baseline_config"])) if "baseline_config" in g7c else None,
        "methods": parse_methods(g7c.get("methods", ",".join(DEFAULT_METHODS))),
        "seeds": parse_seeds(g7c.get("seeds", "23")),
        "workers": int(g7c["workers"]) if "workers" in g7c else None,
        "force": bool(g7c["force"]) if "force" in g7c else False,
        "fail_fast": bool(g7c["fail_fast"]) if "fail_fast" in g7c else False,
        "overrides": overrides,
    }


def apply_profile_overrides(profile: G7CProfile, overrides: object) -> G7CProfile:
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
        optimization=apply_budget_overrides(profile.optimization, overrides.get("optimization", {}), "optimization"),
        high_fidelity=apply_budget_overrides(profile.high_fidelity, overrides.get("high_fidelity", {}), "high_fidelity"),
    )


def build_methods() -> list[G7CMethod]:
    return [
        G7CMethod("hcmbo_structured_only", run_hcmbo_structured_only, "All B=400 evaluations go to objective-ranked structured LCB."),
        G7CMethod(
            "hcmbo_queue_aware_lcb",
            run_hcmbo_queue_aware_lcb,
            "Current-style 300 structured + 100 random budget, selected by queue-aware LCB score.",
        ),
        G7CMethod(
            "hcmbo_structured_queue_aware",
            run_hcmbo_structured_queue_aware,
            "All B=400 evaluations go to structured LCB ranked by queue-aware score.",
        ),
        G7CMethod("tpe_mixed_bo", run_tpe_mixed_bo, "G6 TPE-style mixed-variable BO baseline under B=400."),
    ]


def run_selected_runs(
    *,
    profile: G7CProfile,
    manifest: dict[str, object],
    output_root: Path,
    force: bool,
    fail_fast: bool,
    workers: int,
) -> list[str]:
    entries_by_key: dict[str, dict[str, object]] = {}
    payloads: list[dict[str, object]] = []
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
                payloads.append({"key": key, "method": method_name, "seed": seed, "output_dir": str(run_dir), "profile": profile})
    save_json(output_root / "G7C_manifest.json", manifest)
    write_g7c_outputs(output_root=output_root, manifest=manifest)
    if not payloads:
        return failures
    if workers == 1:
        for payload in payloads:
            result = run_g7c_payload(payload)
            apply_g7c_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7c_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                raise RuntimeError(str(result.get("error") or "G7-C run failed"))
        return failures
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_g7c_payload, payload): str(payload["key"]) for payload in payloads}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"key": key, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            apply_g7c_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7c_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(str(result.get("error") or "G7-C run failed"))
    return failures


def run_g7c_payload(payload: dict[str, object]) -> dict[str, object]:
    key = str(payload["key"])
    method_name = str(payload["method"])
    seed = int(payload["seed"])
    try:
        by_name = {method.name: method for method in build_methods()}
        method = by_name[method_name]
        profile = payload["profile"]
        if not isinstance(profile, G7CProfile):
            raise TypeError("profile payload must be a G7CProfile")
        start = time.perf_counter()
        result_payload = method.runner(
            Path(str(payload["output_dir"])),
            replace(profile, config=replace(profile.config, random_seed=seed)),
            seed,
        )
        result_payload["runtime_seconds"] = time.perf_counter() - start
        save_json(Path(str(payload["output_dir"])) / "method_summary.json", result_payload)
        return {"key": key, "method": method_name, "seed": seed, "status": "completed", **summary_entry(result_payload)}
    except Exception as exc:
        return {
            "key": key,
            "method": method_name,
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def apply_g7c_result(
    *,
    result: dict[str, object],
    entries_by_key: dict[str, dict[str, object]],
    failures: list[str],
    output_root: Path,
    manifest: dict[str, object],
) -> None:
    key = str(result["key"])
    entries_by_key[key].update(result)
    if result.get("status") == "failed" and key not in failures:
        failures.append(key)
    save_json(output_root / "G7C_manifest.json", manifest)


def prepare_context(output_dir: Path, profile: G7CProfile, seed: int) -> RunContext:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = replace(profile.config, random_seed=seed, shortlist_size=profile.config.direction_candidate_limit)
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
    return RunContext(
        output_dir=output_dir,
        profile=replace(profile, config=config),
        config=config,
        seed=seed,
        reference=reference,
        qbar_by_gate=qbar_by_gate,
        directions_list=directions_list,
        evaluator=evaluator,
        rng=rng,
    )


def run_hcmbo_structured_only(output_dir: Path, profile: G7CProfile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed)
    mid_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="g7c_structured_only",
        score_fn=objective_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_g7c_run(ctx=ctx, method_name="hcmbo_structured_only", mid_records=mid_records, selected_records=selected, extra={})


def run_hcmbo_queue_aware_lcb(output_dir: Path, profile: G7CProfile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed)
    structured_budget = min(int(ctx.config.random_search_evaluations), len(ctx.directions_list) * (
        int(ctx.config.initial_samples) + int(ctx.config.bo_iterations) + int(ctx.config.dfo_evaluations)
    ))
    structured_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="g7c_queue_aware_lcb",
        score_fn=queue_aware_score,
        total_budget=structured_budget,
    )
    remaining_budget = max(0, int(ctx.config.random_search_evaluations) - len(structured_records))
    random_records = run_random_search(
        evaluator=ctx.evaluator,
        directions_list=ctx.directions_list,
        qbar_by_gate=ctx.qbar_by_gate,
        config=replace(ctx.config, random_search_evaluations=remaining_budget),
        rng=ctx.rng,
    )
    mid_records = structured_records + random_records
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, queue_aware_score)
    return finalize_g7c_run(
        ctx=ctx,
        method_name="hcmbo_queue_aware_lcb",
        mid_records=mid_records,
        selected_records=selected,
        extra={"structured_evaluations": len(structured_records), "internal_random_evaluations": len(random_records)},
    )


def run_hcmbo_structured_queue_aware(output_dir: Path, profile: G7CProfile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed)
    mid_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="g7c_structured_queue",
        score_fn=queue_aware_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, queue_aware_score)
    return finalize_g7c_run(
        ctx=ctx,
        method_name="hcmbo_structured_queue_aware",
        mid_records=mid_records,
        selected_records=selected,
        extra={},
    )


def run_tpe_mixed_bo(output_dir: Path, profile: G7CProfile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed)
    mid_records = run_tpe_search(
        evaluator=ctx.evaluator,
        directions_list=ctx.directions_list,
        qbar_by_gate=ctx.qbar_by_gate,
        config=ctx.config,
        rng=ctx.rng,
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_g7c_run(ctx=ctx, method_name="tpe_mixed_bo", mid_records=mid_records, selected_records=selected, extra={})


def finalize_g7c_run(
    *,
    ctx: RunContext,
    method_name: str,
    mid_records: list[V2EvaluationRecord],
    selected_records: list[V2EvaluationRecord],
    extra: dict[str, object],
) -> dict[str, object]:
    hf_records, hf_rows = evaluate_high_fidelity_with_origin(
        baseline_config=ctx.profile.baseline_config,
        output_dir=ctx.output_dir,
        config=ctx.config,
        qbar_by_gate=ctx.qbar_by_gate,
        overrides=ctx.profile.high_fidelity.to_overrides(),
        selected_records=selected_records,
        variant_name=method_name,
    )
    return write_method_outputs(
        output_dir=ctx.output_dir,
        method_name=method_name,
        profile=ctx.profile,
        seed=ctx.seed,
        reference=ctx.reference,
        qbar_by_gate=ctx.qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        hf_rows=hf_rows,
        extra=extra,
    )


def write_method_outputs(
    *,
    output_dir: Path,
    method_name: str,
    profile: G7CProfile,
    seed: int,
    reference: V2EvaluationRecord,
    qbar_by_gate: dict[str, float],
    mid_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
    hf_rows: list[dict[str, object]],
    extra: dict[str, object],
) -> dict[str, object]:
    best_row = min(hf_rows, key=lambda item: to_float(item.get("objective_value"))) if hf_rows else {}
    all_rows = [prefix_row(reference.to_row(), method_name, seed)]
    all_rows.extend(prefix_row(with_source_family(record.to_row()), method_name, seed) for record in mid_records)
    all_rows.extend(prefix_row(row, method_name, seed) for row in hf_rows)
    candidate_rows = [prefix_row(row, method_name, seed) for row in sorted(hf_rows, key=lambda item: to_float(item.get("objective_value")))]
    write_csv(output_dir / "G7C_evaluation_log.csv", all_rows)
    write_csv(output_dir / "G7C_hf_candidates.csv", candidate_rows)
    write_csv(output_dir / "G7C_method_comparison.csv", method_comparison_from_groups({method_name: hf_records or mid_records}))
    if hf_records:
        best_index = min(range(len(hf_rows)), key=lambda index: to_float(hf_rows[index].get("objective_value")))
        save_json(output_dir / "G7C_best_control.json", hf_records[best_index].control.to_dict())
    payload: dict[str, object] = {
        "method": method_name,
        "seed": seed,
        "profile": profile.name,
        "best_high_fidelity": prefix_row(best_row, method_name, seed) if best_row else {},
        "candidate_count": len(candidate_rows),
        "optimization_evaluation_count": len(mid_records),
        "hf_candidate_count": len(hf_records),
        "qbar_by_gate": qbar_by_gate,
        "direction_candidate_count": len(generate_direction_candidates(config=profile.config, rng=np.random.default_rng(seed))),
        "config": profile.config.__dict__,
        "optimization_overrides": profile.optimization.to_overrides(),
        "high_fidelity_overrides": profile.high_fidelity.to_overrides(),
        "outputs": {
            "evaluation_log": str(output_dir / "G7C_evaluation_log.csv"),
            "hf_candidates": str(output_dir / "G7C_hf_candidates.csv"),
            "best_control": str(output_dir / "G7C_best_control.json"),
        },
    }
    payload.update(extra)
    return payload


def write_g7c_outputs(*, output_root: Path, manifest: dict[str, object]) -> None:
    save_json(output_root / "G7C_manifest.json", manifest)
    summaries: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
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
                "best_hf_objective": best.get("objective_value"),
                "best_case_id": best.get("case_id"),
                "feasible_best": best.get("feasible"),
                "J1_eval": best.get("j1_eval"),
                "J2_eval": best.get("j2_eval"),
                "J5_eval": best.get("j5_eval"),
                "JB_normalized": best.get("jb_normalized"),
                "JR_normalized": best.get("jr_normalized"),
                "gate_rejected": best.get("gate_rejected"),
                "origin_source_family": best.get("original_source_family"),
                "origin_case_id": best.get("original_case_id"),
                "runtime_seconds": payload.get("runtime_seconds") if isinstance(payload, dict) else None,
                "output_dir": str(run_dir),
            }
        )
        candidates.extend(read_csv(run_dir / "G7C_hf_candidates.csv"))
    write_csv(output_root / "G7C_seed_summary.csv", summaries)
    write_csv(output_root / "G7C_hf_candidates.csv", candidates)
    write_csv(output_root / "G7C_method_summary.csv", build_method_summary_rows(summaries))
    write_csv(output_root / "G7C_pairwise_deltas_vs_tpe.csv", build_pairwise_rows(summaries, baseline="tpe_mixed_bo"))
    write_report(output_root, summaries, candidates)


def build_method_summary_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted({str(row.get("method")) for row in seed_rows})
    for method in methods:
        items = [row for row in seed_rows if str(row.get("method")) == method]
        values = finite_values(row.get("best_hf_objective") for row in items)
        feasible = [parse_bool(row.get("feasible_best")) for row in items if parse_bool(row.get("feasible_best")) is not None]
        if not values:
            continue
        best = min(items, key=lambda row: to_float(row.get("best_hf_objective")))
        rows.append(
            {
                "method": method,
                "seed_count": len(values),
                "mean_best_hf_objective": float(np.mean(values)),
                "std_best_hf_objective": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median_best_hf_objective": float(np.median(values)),
                "best_hf_objective": min(values),
                "worst_hf_objective": max(values),
                "feasible_rate": sum(1 for item in feasible if item) / len(feasible) if feasible else "",
                "mean_j2_eval": mean_or_blank(row.get("J2_eval") for row in items),
                "mean_gate_rejected": mean_or_blank(row.get("gate_rejected") for row in items),
                "best_seed": best.get("seed"),
                "best_case_id": best.get("best_case_id"),
                "best_origin_source_family": best.get("origin_source_family"),
            }
        )
    rows.sort(key=lambda row: to_float(row.get("mean_best_hf_objective")))
    return rows


def build_pairwise_rows(seed_rows: list[dict[str, object]], *, baseline: str) -> list[dict[str, object]]:
    by_seed_method = {(int(row["seed"]), str(row["method"])): row for row in seed_rows}
    methods = sorted({str(row.get("method")) for row in seed_rows if str(row.get("method")) != baseline})
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    rows = []
    for method in methods:
        deltas = []
        wins = 0
        losses = 0
        for seed in seeds:
            base = by_seed_method.get((seed, baseline))
            other = by_seed_method.get((seed, method))
            if not base or not other:
                continue
            delta = to_float(other.get("best_hf_objective")) - to_float(base.get("best_hf_objective"))
            if not math.isfinite(delta):
                continue
            deltas.append(delta)
            if delta < 0:
                wins += 1
            elif delta > 0:
                losses += 1
        rows.append(
            {
                "method": method,
                "baseline": baseline,
                "paired_seed_count": len(deltas),
                "mean_delta_method_minus_tpe": float(np.mean(deltas)) if deltas else "",
                "median_delta_method_minus_tpe": float(np.median(deltas)) if deltas else "",
                "method_wins": wins,
                "tpe_wins": losses,
            }
        )
    return rows


def write_report(output_root: Path, summaries: list[dict[str, object]], candidates: list[dict[str, object]]) -> None:
    ordered = build_method_summary_rows(summaries)
    lines = [
        "# G7-C Focused HCMBO/TPE Comparison",
        "",
        "## Scope",
        "",
        "- Methods: hcmbo_structured_only, hcmbo_queue_aware_lcb, hcmbo_structured_queue_aware, tpe_mixed_bo.",
        "- Configuration: seed 23, B=400, HF top_k=10, optimization/high-fidelity steps=1600, time_horizon=160.0.",
        "- Final ranking uses high-fidelity recheck objective values only.",
        "",
        "## Method Summary",
        "",
    ]
    for row in ordered:
        lines.append(
            f"- `{row.get('method')}`: mean `{row.get('mean_best_hf_objective')}`, "
            f"best `{row.get('best_hf_objective')}`, feasible rate `{row.get('feasible_rate')}`, "
            f"mean J2 `{row.get('mean_j2_eval')}`, mean gate rejected `{row.get('mean_gate_rejected')}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `G7C_manifest.json`: `{output_root / 'G7C_manifest.json'}`",
            f"- `G7C_seed_summary.csv`: `{output_root / 'G7C_seed_summary.csv'}`",
            f"- `G7C_method_summary.csv`: `{output_root / 'G7C_method_summary.csv'}`",
            f"- `G7C_pairwise_deltas_vs_tpe.csv`: `{output_root / 'G7C_pairwise_deltas_vs_tpe.csv'}`",
            f"- `G7C_hf_candidates.csv`: `{output_root / 'G7C_hf_candidates.csv'}`",
            f"- HF candidate count: `{len(candidates)}`",
        ]
    )
    output_root.joinpath("G7C_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_entry(payload: dict[str, object]) -> dict[str, object]:
    best = payload.get("best_high_fidelity", {})
    if not isinstance(best, dict):
        best = {}
    return {"best_objective": best.get("objective_value"), "best_case_id": best.get("case_id"), "feasible": best.get("feasible")}


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
        raise ValueError(f"Unknown G7-C methods: {', '.join(unknown)}")
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


def resolve_worker_count(raw_workers: object, selected_count: int) -> int:
    if raw_workers is None:
        return max(1, selected_count)
    return max(1, int(raw_workers))


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


def finite_values(values: object) -> list[float]:
    result = []
    for value in values:
        number = to_float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def mean_or_blank(values: object) -> float | str:
    finite = finite_values(values)
    return float(np.mean(finite)) if finite else ""


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


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
        return math.nan


if __name__ == "__main__":
    main()
