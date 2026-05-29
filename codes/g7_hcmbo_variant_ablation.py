from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import tomllib
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Callable

import numpy as np

from crowd_bellman.g5_hcmbo import (
    ALL_GATE_IDS,
    CHANNEL_NAMES,
    DEFAULT_BASELINE_CONFIG,
    G5EvaluationCache,
    HCMBOConfig,
    V2ControlVector,
    V2EvaluationRecord,
    control_from_x,
    free_dimension,
    generate_direction_candidates,
    initial_design,
    optimize_fixed_direction,
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


DEFAULT_G7_CONFIG = Path("codes/scenes/examples/g7_hcmbo_variant_ablation/g7.toml")
DEFAULT_VARIANTS = (
    "hcmbo_current",
    "hcmbo_structured_only",
    "hcmbo_adaptive_racing",
    "hcmbo_queue_aware_lcb",
    "hcmbo_trust_region",
    "hcmbo_rf_constrained_bo",
    "hcmbo_diverse_hf_topk",
    "hcmbo_adaptive_racing_queue_aware",
)


ScoreFn = Callable[[V2EvaluationRecord], float]


@dataclass(frozen=True)
class G7Profile:
    name: str
    config: HCMBOConfig
    optimization: FidelityBudget
    high_fidelity: FidelityBudget
    variants: tuple[str, ...]
    seeds: tuple[int, ...]
    output_root: Path
    baseline_config: Path


@dataclass(frozen=True)
class G7Variant:
    name: str
    runner: Callable[[Path, G7Profile, int], dict[str, object]]
    description: str


@dataclass
class RunContext:
    output_dir: Path
    profile: G7Profile
    config: HCMBOConfig
    seed: int
    reference: V2EvaluationRecord
    qbar_by_gate: dict[str, float]
    directions_list: list[tuple[str, ...]]
    evaluator: G5EvaluationCache
    rng: np.random.Generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G7-B HCMBO-v2 variant ablation experiments.")
    parser.add_argument("--config", default=str(DEFAULT_G7_CONFIG))
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--variants", default=None, help="Comma-separated variant names.")
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    loaded = load_g7_config(Path(args.config)) if args.config else {}
    loaded_profile = str(loaded.get("profile") or "full")
    profile_name = args.profile or loaded_profile
    seeds = parse_seeds(args.seeds) if args.seeds else tuple(int(item) for item in loaded.get("seeds", (23,)))
    variants = parse_variants(args.variants or loaded.get("variants") or ",".join(DEFAULT_VARIANTS))
    output_root = Path(args.output_root or loaded.get("output_root") or "codes/results/g7_b_variant_ablation").resolve()
    baseline_config = Path(args.baseline_config or loaded.get("baseline_config") or DEFAULT_BASELINE_CONFIG).resolve()
    force = bool(args.force or loaded.get("force", False))
    fail_fast = bool(args.fail_fast or loaded.get("fail_fast", False))
    workers = resolve_worker_count(args.workers if args.workers is not None else loaded.get("workers"), len(variants) * len(seeds))

    profile = profile_from_name(
        profile_name,
        output_root=output_root,
        baseline_config=baseline_config,
        seeds=seeds,
        variants=variants,
    )
    if loaded.get("overrides") and (args.profile is None or args.profile == loaded_profile):
        profile = apply_profile_overrides(profile, loaded["overrides"])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "experiment_group": "G7-B",
        "design_version": "hcmbo_v2_variant_ablation",
        "profile": profile.name,
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "baseline_config": str(profile.baseline_config),
        "output_root": str(profile.output_root),
        "variants": list(profile.variants),
        "seeds": list(profile.seeds),
        "workers": workers,
        "argv": sys.argv,
        "runs": [],
    }
    save_json(output_root / "G7B_manifest.json", manifest)
    failures = run_selected_runs(
        profile=profile,
        manifest=manifest,
        output_root=output_root,
        force=force,
        fail_fast=fail_fast,
        workers=workers,
    )
    write_g7_outputs(output_root=output_root, manifest=manifest)
    if failures:
        raise RuntimeError(f"G7-B failed runs: {', '.join(failures)}")
    print(f"G7-B summary: {output_root / 'G7B_method_summary.csv'}")


def profile_from_name(
    name: str,
    *,
    output_root: Path,
    baseline_config: Path,
    seeds: tuple[int, ...],
    variants: tuple[str, ...],
) -> G7Profile:
    if name == "smoke":
        return G7Profile(
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
            variants=variants,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    if name == "full":
        return G7Profile(
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
            variants=variants,
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    raise ValueError(f"Unsupported G7 profile: {name!r}")


def load_g7_config(path: Path) -> dict[str, object]:
    base_dir = path.resolve().parent
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    g7 = dict(raw.get("g7", {}))
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
        "profile": g7.get("profile"),
        "output_root": resolve_config_path(base_dir, str(g7["output_root"])) if "output_root" in g7 else None,
        "baseline_config": resolve_config_path(base_dir, str(g7["baseline_config"])) if "baseline_config" in g7 else None,
        "variants": parse_variants(g7.get("variants", ",".join(DEFAULT_VARIANTS))),
        "seeds": parse_seeds(g7.get("seeds", "23")),
        "workers": int(g7["workers"]) if "workers" in g7 else None,
        "force": bool(g7["force"]) if "force" in g7 else False,
        "fail_fast": bool(g7["fail_fast"]) if "fail_fast" in g7 else False,
        "overrides": overrides,
    }


def apply_profile_overrides(profile: G7Profile, overrides: object) -> G7Profile:
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


def build_variants() -> list[G7Variant]:
    return [
        G7Variant("hcmbo_current", run_hcmbo_current, "Current G6 HCMBO: equal direction BO/DFO plus internal random search."),
        G7Variant("hcmbo_structured_only", run_hcmbo_structured_only, "Direction-wise LCB budget without internal random candidates."),
        G7Variant("hcmbo_adaptive_racing", run_hcmbo_adaptive_racing, "Adaptive direction racing with staged budget concentration."),
        G7Variant("hcmbo_queue_aware_lcb", run_hcmbo_queue_aware_lcb, "LCB search ranked by queue-aware constrained score."),
        G7Variant("hcmbo_trust_region", run_hcmbo_trust_region, "Direction-wise local trust-region capacity search."),
        G7Variant("hcmbo_rf_constrained_bo", run_hcmbo_rf_constrained_bo, "ExtraTrees/RF-style constrained surrogate search."),
        G7Variant("hcmbo_diverse_hf_topk", run_hcmbo_diverse_hf_topk, "Current candidate generation with diverse HF recheck policy."),
        G7Variant(
            "hcmbo_adaptive_racing_queue_aware",
            run_hcmbo_adaptive_racing_queue_aware,
            "Adaptive racing using the queue-aware constrained score.",
        ),
    ]


def run_selected_runs(
    *,
    profile: G7Profile,
    manifest: dict[str, object],
    output_root: Path,
    force: bool,
    fail_fast: bool,
    workers: int,
) -> list[str]:
    entries_by_key: dict[str, dict[str, object]] = {}
    payloads: list[dict[str, object]] = []
    failures: list[str] = []
    by_name = {variant.name: variant for variant in build_variants()}
    for seed in profile.seeds:
        for variant_name in profile.variants:
            variant = by_name[variant_name]
            key = run_key(variant_name, seed)
            run_dir = output_root / variant_name / f"seed_{seed}"
            entry = {
                "key": key,
                "variant": variant_name,
                "seed": seed,
                "description": variant.description,
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
                payloads.append({"key": key, "variant": variant_name, "seed": seed, "output_dir": str(run_dir), "profile": profile})
    save_json(output_root / "G7B_manifest.json", manifest)
    write_g7_outputs(output_root=output_root, manifest=manifest)
    if not payloads:
        return failures
    if workers == 1:
        for payload in payloads:
            result = run_g7_payload(payload)
            apply_g7_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                raise RuntimeError(str(result.get("error") or "G7-B run failed"))
        return failures
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_g7_payload, payload): str(payload["key"]) for payload in payloads}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"key": key, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            apply_g7_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(str(result.get("error") or "G7-B run failed"))
    return failures


def run_g7_payload(payload: dict[str, object]) -> dict[str, object]:
    key = str(payload["key"])
    variant_name = str(payload["variant"])
    seed = int(payload["seed"])
    try:
        by_name = {variant.name: variant for variant in build_variants()}
        variant = by_name[variant_name]
        profile = payload["profile"]
        if not isinstance(profile, G7Profile):
            raise TypeError("profile payload must be a G7Profile")
        start = time.perf_counter()
        result_payload = variant.runner(
            Path(str(payload["output_dir"])),
            replace(profile, config=replace(profile.config, random_seed=seed)),
            seed,
        )
        result_payload["runtime_seconds"] = time.perf_counter() - start
        save_json(Path(str(payload["output_dir"])) / "method_summary.json", result_payload)
        return {"key": key, "variant": variant_name, "seed": seed, "status": "completed", **summary_entry(result_payload)}
    except Exception as exc:
        return {
            "key": key,
            "variant": variant_name,
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def apply_g7_result(
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
    save_json(output_root / "G7B_manifest.json", manifest)


def run_hcmbo_current(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_current")
    hcmbo_records: list[V2EvaluationRecord] = []
    for directions in ctx.directions_list:
        records, _trace = optimize_fixed_direction(
            evaluator=ctx.evaluator,
            directions=directions,
            qbar_by_gate=ctx.qbar_by_gate,
            config=ctx.config,
            rng=ctx.rng,
            source_prefix="hcmbo_current",
        )
        hcmbo_records.extend(records)
    remaining_budget = max(0, int(ctx.config.random_search_evaluations) - len(hcmbo_records))
    random_records = run_random_search(
        evaluator=ctx.evaluator,
        directions_list=ctx.directions_list,
        qbar_by_gate=ctx.qbar_by_gate,
        config=replace(ctx.config, random_search_evaluations=remaining_budget),
        rng=ctx.rng,
    )
    mid_records = hcmbo_records + random_records
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_variant_run(
        ctx=ctx,
        variant_name="hcmbo_current",
        mid_records=mid_records,
        selected_records=selected,
        extra={"structured_evaluations": len(hcmbo_records), "internal_random_evaluations": len(random_records)},
    )


def run_hcmbo_structured_only(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_structured_only")
    mid_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="hcmbo_structured_only",
        score_fn=objective_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_variant_run(ctx=ctx, variant_name="hcmbo_structured_only", mid_records=mid_records, selected_records=selected, extra={})


def run_hcmbo_adaptive_racing(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_adaptive_racing")
    mid_records = run_adaptive_racing_records(ctx=ctx, source_prefix="hcmbo_adaptive_racing", score_fn=objective_score)
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_variant_run(ctx=ctx, variant_name="hcmbo_adaptive_racing", mid_records=mid_records, selected_records=selected, extra={})


def run_hcmbo_queue_aware_lcb(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_queue_aware_lcb")
    mid_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="hcmbo_queue_aware_lcb",
        score_fn=queue_aware_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, queue_aware_score)
    return finalize_variant_run(ctx=ctx, variant_name="hcmbo_queue_aware_lcb", mid_records=mid_records, selected_records=selected, extra={})


def run_hcmbo_trust_region(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_trust_region")
    mid_records = run_equal_direction_trust_region_budget(ctx=ctx, source_prefix="hcmbo_trust_region")
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    return finalize_variant_run(ctx=ctx, variant_name="hcmbo_trust_region", mid_records=mid_records, selected_records=selected, extra={})


def run_hcmbo_rf_constrained_bo(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_rf_constrained_bo")
    mid_records, used_backend = run_equal_direction_rf_budget(
        ctx=ctx,
        source_prefix="hcmbo_rf_constrained_bo",
        score_fn=queue_aware_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, queue_aware_score)
    return finalize_variant_run(
        ctx=ctx,
        variant_name="hcmbo_rf_constrained_bo",
        mid_records=mid_records,
        selected_records=selected,
        extra={"surrogate_backend": used_backend},
    )


def run_hcmbo_diverse_hf_topk(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_diverse_hf_topk")
    hcmbo_records: list[V2EvaluationRecord] = []
    for directions in ctx.directions_list:
        records, _trace = optimize_fixed_direction(
            evaluator=ctx.evaluator,
            directions=directions,
            qbar_by_gate=ctx.qbar_by_gate,
            config=ctx.config,
            rng=ctx.rng,
            source_prefix="hcmbo_diverse_hf_topk",
        )
        hcmbo_records.extend(records)
    remaining_budget = max(0, int(ctx.config.random_search_evaluations) - len(hcmbo_records))
    random_records = run_random_search(
        evaluator=ctx.evaluator,
        directions_list=ctx.directions_list,
        qbar_by_gate=ctx.qbar_by_gate,
        config=replace(ctx.config, random_search_evaluations=remaining_budget),
        rng=ctx.rng,
    )
    mid_records = hcmbo_records + random_records
    selected = select_diverse_hf_records(mid_records, ctx.config.high_fidelity_top_k)
    return finalize_variant_run(
        ctx=ctx,
        variant_name="hcmbo_diverse_hf_topk",
        mid_records=mid_records,
        selected_records=selected,
        extra={"structured_evaluations": len(hcmbo_records), "internal_random_evaluations": len(random_records)},
    )


def run_hcmbo_adaptive_racing_queue_aware(output_dir: Path, profile: G7Profile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed, variant_name="hcmbo_adaptive_racing_queue_aware")
    mid_records = run_adaptive_racing_records(ctx=ctx, source_prefix="hcmbo_adaptive_racing_queue_aware", score_fn=queue_aware_score)
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, queue_aware_score)
    return finalize_variant_run(
        ctx=ctx,
        variant_name="hcmbo_adaptive_racing_queue_aware",
        mid_records=mid_records,
        selected_records=selected,
        extra={},
    )


def prepare_context(output_dir: Path, profile: G7Profile, seed: int, *, variant_name: str) -> RunContext:
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


def run_equal_direction_lcb_budget(
    *,
    ctx: RunContext,
    source_prefix: str,
    score_fn: ScoreFn,
    total_budget: int,
) -> list[V2EvaluationRecord]:
    records: list[V2EvaluationRecord] = []
    direction_count = max(1, len(ctx.directions_list))
    base = max(1, total_budget // direction_count)
    remainder = max(0, total_budget % direction_count)
    for index, directions in enumerate(ctx.directions_list):
        budget = base + (1 if index < remainder else 0)
        records.extend(
            run_lcb_for_direction(
                ctx=ctx,
                directions=directions,
                budget=budget,
                source_prefix=source_prefix,
                score_fn=score_fn,
            )
        )
    return records[: max(0, total_budget)]


def run_lcb_for_direction(
    *,
    ctx: RunContext,
    directions: tuple[str, ...],
    budget: int,
    source_prefix: str,
    score_fn: ScoreFn,
) -> list[V2EvaluationRecord]:
    dim = free_dimension(directions, ctx.config.time_segments)
    xs = initial_design(dim=dim, sample_count=min(max(1, ctx.config.initial_samples), max(1, budget)), rng=ctx.rng)
    records: list[V2EvaluationRecord] = []
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]] = []
    for x in xs:
        if len(records) >= budget:
            break
        record = evaluate_x(ctx, directions, x, source=f"{source_prefix}_init", phase="bo_init")
        records.append(record)
        x_records.append((x, record))
    while len(records) < budget:
        x_next = propose_lcb_candidate_with_score(
            x_records=x_records,
            dim=dim,
            candidate_pool=max(4, int(ctx.config.bo_candidate_pool)),
            kappa=float(ctx.config.lcb_kappa),
            rng=ctx.rng,
            score_fn=score_fn,
        )
        record = evaluate_x(ctx, directions, x_next, source=f"{source_prefix}_bo", phase="bo")
        records.append(record)
        x_records.append((x_next, record))
    return records


def run_adaptive_racing_records(*, ctx: RunContext, source_prefix: str, score_fn: ScoreFn) -> list[V2EvaluationRecord]:
    total_budget = int(ctx.config.random_search_evaluations)
    state: dict[tuple[str, ...], list[tuple[np.ndarray, V2EvaluationRecord]]] = {directions: [] for directions in ctx.directions_list}
    records: list[V2EvaluationRecord] = []

    if total_budget < 100 or len(ctx.directions_list) != 12:
        for directions in ctx.directions_list:
            if len(records) >= total_budget:
                break
            records.extend(evaluate_more_for_direction(ctx, directions, state[directions], 1, source_prefix, score_fn, "race_init"))
        while len(records) < total_budget:
            ranked = rank_directions(state, score_fn)
            selected = [item[0] for item in ranked[: max(1, min(len(ranked), max(1, len(ranked) // 2)))]]
            for directions in selected:
                if len(records) >= total_budget:
                    break
                records.extend(evaluate_more_for_direction(ctx, directions, state[directions], 1, source_prefix, score_fn, "race"))
        return records

    plan = [(12, 4), (6, 12), (4, 25), (3, 40), (2, 30)]
    active = list(ctx.directions_list)
    for stage_index, (keep_count, add_budget) in enumerate(plan, start=1):
        if stage_index > 1:
            ranked = rank_directions(state, score_fn)
            active = [directions for directions, _score in ranked[: min(keep_count, len(ranked))]]
        else:
            active = active[: min(keep_count, len(active))]
        for directions in active:
            if len(records) >= total_budget:
                break
            remaining = total_budget - len(records)
            new_records = evaluate_more_for_direction(
                ctx,
                directions,
                state[directions],
                min(add_budget, remaining),
                source_prefix,
                score_fn,
                f"race_stage_{stage_index}",
            )
            records.extend(new_records)
    return records[:total_budget]


def evaluate_more_for_direction(
    ctx: RunContext,
    directions: tuple[str, ...],
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]],
    count: int,
    source_prefix: str,
    score_fn: ScoreFn,
    phase: str,
) -> list[V2EvaluationRecord]:
    dim = free_dimension(directions, ctx.config.time_segments)
    records: list[V2EvaluationRecord] = []
    while len(records) < count:
        if not x_records:
            x = initial_design(dim=dim, sample_count=1, rng=ctx.rng)[0]
            source = f"{source_prefix}_init"
            row_phase = "bo_init"
        else:
            x = propose_lcb_candidate_with_score(
                x_records=x_records,
                dim=dim,
                candidate_pool=max(4, int(ctx.config.bo_candidate_pool)),
                kappa=float(ctx.config.lcb_kappa),
                rng=ctx.rng,
                score_fn=score_fn,
            )
            source = f"{source_prefix}_bo"
            row_phase = phase
        record = evaluate_x(ctx, directions, x, source=source, phase=row_phase)
        records.append(record)
        x_records.append((x, record))
    return records


def rank_directions(
    state: dict[tuple[str, ...], list[tuple[np.ndarray, V2EvaluationRecord]]],
    score_fn: ScoreFn,
) -> list[tuple[tuple[str, ...], float]]:
    ranked = []
    for directions, x_records in state.items():
        if not x_records:
            ranked.append((directions, math.inf))
        else:
            ranked.append((directions, min(score_fn(record) for _x, record in x_records)))
    ranked.sort(key=lambda item: item[1])
    return ranked


def run_equal_direction_trust_region_budget(*, ctx: RunContext, source_prefix: str) -> list[V2EvaluationRecord]:
    total_budget = int(ctx.config.random_search_evaluations)
    records: list[V2EvaluationRecord] = []
    direction_count = max(1, len(ctx.directions_list))
    base = max(1, total_budget // direction_count)
    remainder = max(0, total_budget % direction_count)
    for index, directions in enumerate(ctx.directions_list):
        budget = base + (1 if index < remainder else 0)
        records.extend(run_trust_region_for_direction(ctx=ctx, directions=directions, budget=budget, source_prefix=source_prefix))
    return records[:total_budget]


def run_trust_region_for_direction(
    *,
    ctx: RunContext,
    directions: tuple[str, ...],
    budget: int,
    source_prefix: str,
) -> list[V2EvaluationRecord]:
    dim = free_dimension(directions, ctx.config.time_segments)
    xs = initial_design(dim=dim, sample_count=min(3, max(1, budget)), rng=ctx.rng)
    records: list[V2EvaluationRecord] = []
    scored: list[tuple[np.ndarray, V2EvaluationRecord]] = []
    for x in xs:
        if len(records) >= budget:
            break
        record = evaluate_x(ctx, directions, x, source=f"{source_prefix}_init", phase="tr_init")
        records.append(record)
        scored.append((x, record))
    radius = 0.28
    stagnant = 0
    while len(records) < budget:
        best_x, best_record = min(scored, key=lambda item: item[1].objective_value)
        x_trial = np.clip(best_x + ctx.rng.normal(0.0, radius, size=dim), 0.0, 1.0) if dim > 0 else np.zeros(0, dtype=float)
        record = evaluate_x(ctx, directions, x_trial, source=f"{source_prefix}_local", phase="trust_region")
        records.append(record)
        scored.append((x_trial, record))
        if record.objective_value < best_record.objective_value:
            radius = min(0.35, radius * 1.15)
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 4:
                radius = max(0.035, radius * 0.65)
                stagnant = 0
    return records


def run_equal_direction_rf_budget(
    *,
    ctx: RunContext,
    source_prefix: str,
    score_fn: ScoreFn,
    total_budget: int,
) -> tuple[list[V2EvaluationRecord], str]:
    records: list[V2EvaluationRecord] = []
    backend = "extra_trees"
    try:
        from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
    except Exception:
        ExtraTreesRegressor = None  # type: ignore
        backend = "lcb_fallback_no_sklearn"

    direction_count = max(1, len(ctx.directions_list))
    base = max(1, total_budget // direction_count)
    remainder = max(0, total_budget % direction_count)
    for index, directions in enumerate(ctx.directions_list):
        budget = base + (1 if index < remainder else 0)
        if ExtraTreesRegressor is None:
            records.extend(run_lcb_for_direction(ctx=ctx, directions=directions, budget=budget, source_prefix=source_prefix, score_fn=score_fn))
        else:
            records.extend(
                run_rf_for_direction(
                    ctx=ctx,
                    directions=directions,
                    budget=budget,
                    source_prefix=source_prefix,
                    score_fn=score_fn,
                    regressor_cls=ExtraTreesRegressor,
                )
            )
    return records[:total_budget], backend


def run_rf_for_direction(
    *,
    ctx: RunContext,
    directions: tuple[str, ...],
    budget: int,
    source_prefix: str,
    score_fn: ScoreFn,
    regressor_cls: object,
) -> list[V2EvaluationRecord]:
    dim = free_dimension(directions, ctx.config.time_segments)
    xs = initial_design(dim=dim, sample_count=min(max(4, ctx.config.initial_samples), max(1, budget)), rng=ctx.rng)
    records: list[V2EvaluationRecord] = []
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]] = []
    for x in xs:
        if len(records) >= budget:
            break
        record = evaluate_x(ctx, directions, x, source=f"{source_prefix}_init", phase="rf_init")
        records.append(record)
        x_records.append((x, record))
    while len(records) < budget:
        x_next = propose_rf_candidate(
            x_records=x_records,
            dim=dim,
            candidate_pool=max(12, int(ctx.config.bo_candidate_pool)),
            rng=ctx.rng,
            score_fn=score_fn,
            regressor_cls=regressor_cls,
        )
        record = evaluate_x(ctx, directions, x_next, source=f"{source_prefix}_rf", phase="rf_bo")
        records.append(record)
        x_records.append((x_next, record))
    return records


def propose_lcb_candidate_with_score(
    *,
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]],
    dim: int,
    candidate_pool: int,
    kappa: float,
    rng: np.random.Generator,
    score_fn: ScoreFn,
) -> np.ndarray:
    if dim <= 0:
        return np.zeros(0, dtype=float)
    xs = np.vstack([item[0] for item in x_records])
    ys = np.array([score_fn(item[1]) for item in x_records], dtype=float)
    best_x = xs[int(np.argmin(ys))]
    pool = [rng.random(dim) for _ in range(candidate_pool)]
    for _ in range(max(4, candidate_pool // 4)):
        pool.append(np.clip(best_x + rng.normal(0.0, 0.18, size=dim), 0.0, 1.0))
    y_std = max(float(np.std(ys)), 1.0e-6)
    length_scale = max(0.18, 1.0 / math.sqrt(max(dim, 1)))
    best_score = math.inf
    best_candidate = pool[0]
    for candidate in pool:
        distances = np.linalg.norm(xs - candidate, axis=1)
        weights = np.exp(-(distances * distances) / (2.0 * length_scale * length_scale))
        mean = float(np.mean(ys)) if float(np.sum(weights)) <= 1.0e-12 else float(np.sum(weights * ys) / np.sum(weights))
        uncertainty = min(1.0, float(np.min(distances)) / math.sqrt(max(dim, 1)))
        acquisition = mean - float(kappa) * y_std * uncertainty
        if acquisition < best_score:
            best_score = acquisition
            best_candidate = candidate
    return np.array(best_candidate, dtype=float)


def propose_rf_candidate(
    *,
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]],
    dim: int,
    candidate_pool: int,
    rng: np.random.Generator,
    score_fn: ScoreFn,
    regressor_cls: object,
) -> np.ndarray:
    if dim <= 0:
        return np.zeros(0, dtype=float)
    xs = np.vstack([item[0] for item in x_records])
    ys = np.array([score_fn(item[1]) for item in x_records], dtype=float)
    best_x = xs[int(np.argmin(ys))]
    pool = [rng.random(dim) for _ in range(candidate_pool)]
    for scale in (0.20, 0.10, 0.05):
        for _ in range(max(2, candidate_pool // 8)):
            pool.append(np.clip(best_x + rng.normal(0.0, scale, size=dim), 0.0, 1.0))
    candidates = np.vstack(pool)
    if len(x_records) < max(5, min(2 * dim, 16)):
        return propose_lcb_candidate_with_score(
            x_records=x_records,
            dim=dim,
            candidate_pool=candidate_pool,
            kappa=1.2,
            rng=rng,
            score_fn=score_fn,
        )
    model = regressor_cls(n_estimators=48, random_state=int(rng.integers(0, 2**31 - 1)), min_samples_leaf=2)
    model.fit(xs, ys)
    means = np.asarray(model.predict(candidates), dtype=float)
    estimators = getattr(model, "estimators_", [])
    if estimators:
        predictions = np.vstack([estimator.predict(candidates) for estimator in estimators])
        uncertainty = np.std(predictions, axis=0)
    else:
        uncertainty = np.zeros(len(candidates), dtype=float)
    scores = means - 0.5 * uncertainty
    return np.array(candidates[int(np.argmin(scores))], dtype=float)


def evaluate_x(ctx: RunContext, directions: tuple[str, ...], x: np.ndarray, *, source: str, phase: str) -> V2EvaluationRecord:
    return ctx.evaluator.evaluate(
        control_from_x(directions=directions, x=x, qbar_by_gate=ctx.qbar_by_gate, segment_count=ctx.config.time_segments),
        source=source,
        phase=phase,
        qbar_by_gate=ctx.qbar_by_gate,
        record_cached=True,
    )


def finalize_variant_run(
    *,
    ctx: RunContext,
    variant_name: str,
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
        variant_name=variant_name,
    )
    return write_variant_outputs(
        output_dir=ctx.output_dir,
        variant_name=variant_name,
        profile=ctx.profile,
        seed=ctx.seed,
        reference=ctx.reference,
        qbar_by_gate=ctx.qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        hf_rows=hf_rows,
        extra=extra,
    )


def evaluate_high_fidelity_with_origin(
    *,
    baseline_config: Path,
    output_dir: Path,
    config: HCMBOConfig,
    qbar_by_gate: dict[str, float],
    overrides: dict[str, object],
    selected_records: list[V2EvaluationRecord],
    variant_name: str,
) -> tuple[list[V2EvaluationRecord], list[dict[str, object]]]:
    evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_dir / "_high_fidelity",
        objective_config=config,
        simulation_overrides=overrides,
        fidelity="hf",
    )
    hf_records: list[V2EvaluationRecord] = []
    hf_rows: list[dict[str, object]] = []
    for rank, origin in enumerate(selected_records, start=1):
        record = evaluator.evaluate(
            origin.control,
            source=f"{variant_name}_hf",
            phase="high_fidelity",
            qbar_by_gate=qbar_by_gate,
        )
        hf_records.append(record)
        row = record.to_row()
        row.update(origin_fields(origin, rank))
        hf_rows.append(row)
    return hf_records, hf_rows


def write_variant_outputs(
    *,
    output_dir: Path,
    variant_name: str,
    profile: G7Profile,
    seed: int,
    reference: V2EvaluationRecord,
    qbar_by_gate: dict[str, float],
    mid_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
    hf_rows: list[dict[str, object]],
    extra: dict[str, object],
) -> dict[str, object]:
    best_row = min(hf_rows, key=lambda item: to_float(item.get("objective_value"))) if hf_rows else {}
    all_rows = [prefix_row(reference.to_row(), variant_name, seed)]
    all_rows.extend(prefix_row(with_source_family(record.to_row()), variant_name, seed) for record in mid_records)
    all_rows.extend(prefix_row(row, variant_name, seed) for row in hf_rows)
    candidate_rows = [prefix_row(row, variant_name, seed) for row in sorted(hf_rows, key=lambda item: to_float(item.get("objective_value")))]
    write_csv(output_dir / "G7B_evaluation_log.csv", all_rows)
    write_csv(output_dir / "G7B_hf_candidates.csv", candidate_rows)
    write_csv(output_dir / "G7B_method_comparison.csv", method_comparison_from_groups({variant_name: hf_records or mid_records}))
    if hf_records:
        best_index = min(range(len(hf_rows)), key=lambda index: to_float(hf_rows[index].get("objective_value")))
        save_json(output_dir / "G7B_best_control.json", hf_records[best_index].control.to_dict())
    payload: dict[str, object] = {
        "variant": variant_name,
        "seed": seed,
        "profile": profile.name,
        "best_high_fidelity": prefix_row(best_row, variant_name, seed) if best_row else {},
        "candidate_count": len(candidate_rows),
        "optimization_evaluation_count": len(mid_records),
        "hf_candidate_count": len(hf_records),
        "qbar_by_gate": qbar_by_gate,
        "direction_candidate_count": len(generate_direction_candidates(config=profile.config, rng=np.random.default_rng(seed))),
        "config": profile.config.__dict__,
        "optimization_overrides": profile.optimization.to_overrides(),
        "high_fidelity_overrides": profile.high_fidelity.to_overrides(),
        "outputs": {
            "evaluation_log": str(output_dir / "G7B_evaluation_log.csv"),
            "hf_candidates": str(output_dir / "G7B_hf_candidates.csv"),
            "best_control": str(output_dir / "G7B_best_control.json"),
        },
    }
    payload.update(extra)
    return payload


def select_top_unique_records(records: list[V2EvaluationRecord], limit: int, score_fn: ScoreFn) -> list[V2EvaluationRecord]:
    selected: list[V2EvaluationRecord] = []
    seen: set[V2ControlVector] = set()
    for record in sorted(records, key=score_fn):
        if record.control in seen:
            continue
        selected.append(record)
        seen.add(record.control)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def select_diverse_hf_records(records: list[V2EvaluationRecord], limit: int) -> list[V2EvaluationRecord]:
    limit = max(1, int(limit))
    selected: list[V2EvaluationRecord] = []
    seen: set[V2ControlVector] = set()

    def add(record: V2EvaluationRecord | None) -> None:
        if record is None or len(selected) >= limit or record.control in seen:
            return
        selected.append(record)
        seen.add(record.control)

    feasible_records = [record for record in records if parse_bool(record.metrics.get("feasible")) is True]
    add(min(records, key=objective_score) if records else None)
    add(min(feasible_records, key=objective_score) if feasible_records else None)
    add(min(records, key=lambda record: metric(record, "j2_eval")) if records else None)
    add(min(records, key=lambda record: metric(record, "gate_rejected")) if records else None)
    add(min(records, key=lambda record: metric(record, "j5_eval")) if records else None)

    by_direction: dict[tuple[str, ...], list[V2EvaluationRecord]] = defaultdict(list)
    for record in records:
        by_direction[record.control.directions].append(record)
    for _directions, items in sorted(by_direction.items(), key=lambda item: min(objective_score(record) for record in item[1])):
        add(min(items, key=objective_score))
    for record in sorted(records, key=objective_score):
        add(record)
    return selected


def objective_score(record: V2EvaluationRecord) -> float:
    return float(record.objective_value)


def queue_aware_score(record: V2EvaluationRecord) -> float:
    value = objective_score(record)
    rejected = max(0.0, metric(record, "gate_rejected"))
    waiting = max(0.0, metric(record, "waiting_mass_peak"))
    binding = max(0.0, metric(record, "binding_time_ratio_max"))
    cap_removed = max(0.0, metric(record, "cap_removed_relative"))
    feasibility_penalty = 0.0 if parse_bool(record.metrics.get("feasible")) is True else 0.5
    cap_penalty = max(0.0, cap_removed - 0.02) * 30.0
    return value + 0.0001 * rejected + 0.005 * waiting + 0.5 * binding + cap_penalty + feasibility_penalty


def metric(record: V2EvaluationRecord, key: str) -> float:
    return to_float(record.metrics.get(key))


def origin_fields(origin: V2EvaluationRecord, rank: int) -> dict[str, object]:
    row = origin.to_row()
    return {
        "hf_selection_rank": rank,
        "original_source": origin.source,
        "original_phase": origin.phase,
        "original_fidelity": origin.fidelity,
        "original_eval_id": origin.eval_id,
        "original_case_id": row.get("case_id"),
        "original_objective_value": origin.objective_value,
        "original_source_family": source_family(origin.source),
        "selection_score_objective": origin.objective_value,
    }


def with_source_family(row: dict[str, object]) -> dict[str, object]:
    output = dict(row)
    output["source_family"] = source_family(str(row.get("source", "")))
    return output


def source_family(source: str) -> str:
    if source.startswith("hcmbo") and source.endswith("_hf"):
        return "high_fidelity_recheck"
    if source.startswith("hcmbo") and ("random" not in source):
        return "hcmbo_structured"
    if source == "random_search":
        return "hcmbo_internal_random"
    return source or "unknown"


def write_g7_outputs(*, output_root: Path, manifest: dict[str, object]) -> None:
    save_json(output_root / "G7B_manifest.json", manifest)
    summaries: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict) or run.get("status") not in {"completed", "skipped_complete"}:
            continue
        variant = str(run["variant"])
        seed = int(run["seed"])
        run_dir = Path(str(run["output_dir"]))
        payload_path = run_dir / "method_summary.json"
        payload = load_json(payload_path) if payload_path.exists() else {}
        best = payload.get("best_high_fidelity", {}) if isinstance(payload, dict) else {}
        if not isinstance(best, dict):
            best = {}
        summaries.append(
            {
                "variant": variant,
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
        candidates.extend(read_csv(run_dir / "G7B_hf_candidates.csv"))
    write_csv(output_root / "G7B_seed_summary.csv", summaries)
    write_csv(output_root / "G7B_hf_candidates.csv", candidates)
    write_csv(output_root / "G7B_method_summary.csv", build_method_summary_rows(summaries))
    write_csv(output_root / "G7B_pairwise_deltas_vs_current.csv", build_pairwise_rows(summaries, baseline="hcmbo_current"))
    write_report(output_root, summaries, candidates)


def build_method_summary_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = sorted({str(row.get("variant")) for row in seed_rows})
    for variant in variants:
        items = [row for row in seed_rows if str(row.get("variant")) == variant]
        values = finite_values(row.get("best_hf_objective") for row in items)
        feasible = [parse_bool(row.get("feasible_best")) for row in items if parse_bool(row.get("feasible_best")) is not None]
        if not values:
            continue
        best = min(items, key=lambda row: to_float(row.get("best_hf_objective")))
        rows.append(
            {
                "variant": variant,
                "seed_count": len(values),
                "mean_best_hf_objective": float(np.mean(values)),
                "std_best_hf_objective": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median_best_hf_objective": float(np.median(values)),
                "best_hf_objective": min(values),
                "worst_hf_objective": max(values),
                "feasible_rate": sum(1 for item in feasible if item) / len(feasible) if feasible else "",
                "mean_j2_eval": float(np.mean(finite_values(row.get("J2_eval") for row in items))),
                "mean_gate_rejected": float(np.mean(finite_values(row.get("gate_rejected") for row in items))),
                "best_seed": best.get("seed"),
                "best_case_id": best.get("best_case_id"),
                "best_origin_source_family": best.get("origin_source_family"),
            }
        )
    return rows


def build_pairwise_rows(seed_rows: list[dict[str, object]], *, baseline: str) -> list[dict[str, object]]:
    by_seed_variant = {(int(row["seed"]), str(row["variant"])): row for row in seed_rows}
    variants = sorted({str(row.get("variant")) for row in seed_rows if str(row.get("variant")) != baseline})
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    rows = []
    for variant in variants:
        deltas = []
        wins = 0
        losses = 0
        for seed in seeds:
            current = by_seed_variant.get((seed, baseline))
            other = by_seed_variant.get((seed, variant))
            if not current or not other:
                continue
            delta = to_float(other.get("best_hf_objective")) - to_float(current.get("best_hf_objective"))
            if not math.isfinite(delta):
                continue
            deltas.append(delta)
            if delta < 0:
                wins += 1
            elif delta > 0:
                losses += 1
        rows.append(
            {
                "variant": variant,
                "baseline": baseline,
                "paired_seed_count": len(deltas),
                "mean_delta_variant_minus_current": float(np.mean(deltas)) if deltas else "",
                "median_delta_variant_minus_current": float(np.median(deltas)) if deltas else "",
                "variant_wins": wins,
                "current_wins": losses,
            }
        )
    return rows


def write_report(output_root: Path, summaries: list[dict[str, object]], candidates: list[dict[str, object]]) -> None:
    method_rows = build_method_summary_rows(summaries)
    ordered = sorted(method_rows, key=lambda row: to_float(row.get("mean_best_hf_objective")))
    lines = [
        "# G7-B HCMBO Variant Ablation Report",
        "",
        "## Scope",
        "",
        "- This experiment compares HCMBO-v2 implementation variants under a shared mixed-variable budget.",
        "- Final ranking uses high-fidelity recheck objective values only.",
        "- HF candidate rows preserve the original MF source, phase, case id, and objective for source audit.",
        "",
        "## Method Summary",
        "",
    ]
    for row in ordered:
        lines.append(
            f"- `{row.get('variant')}`: mean `{row.get('mean_best_hf_objective')}`, "
            f"best `{row.get('best_hf_objective')}`, feasible rate `{row.get('feasible_rate')}`, "
            f"mean J2 `{row.get('mean_j2_eval')}`, mean gate rejected `{row.get('mean_gate_rejected')}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `G7B_manifest.json`: `{output_root / 'G7B_manifest.json'}`",
            f"- `G7B_seed_summary.csv`: `{output_root / 'G7B_seed_summary.csv'}`",
            f"- `G7B_method_summary.csv`: `{output_root / 'G7B_method_summary.csv'}`",
            f"- `G7B_pairwise_deltas_vs_current.csv`: `{output_root / 'G7B_pairwise_deltas_vs_current.csv'}`",
            f"- `G7B_hf_candidates.csv`: `{output_root / 'G7B_hf_candidates.csv'}`",
            f"- HF candidate count: `{len(candidates)}`",
        ]
    )
    output_root.joinpath("G7B_variant_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_entry(payload: dict[str, object]) -> dict[str, object]:
    best = payload.get("best_high_fidelity", {})
    if not isinstance(best, dict):
        best = {}
    return {"best_objective": best.get("objective_value"), "best_case_id": best.get("case_id"), "feasible": best.get("feasible")}


def parse_variants(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        variants = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, list | tuple):
        variants = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise ValueError("variants must be a comma-separated string or list")
    valid = {variant.name for variant in build_variants()}
    unknown = [variant for variant in variants if variant not in valid]
    if unknown:
        raise ValueError(f"Unknown G7-B variants: {', '.join(unknown)}")
    return variants


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
    if selected_count <= 0:
        return 1
    if raw_workers is None:
        return min(selected_count, 4)
    return max(1, min(int(raw_workers), selected_count))


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


def run_key(variant: str, seed: int) -> str:
    return f"{variant}:seed_{seed}"


def prefix_row(row: dict[str, object], variant: str, seed: int) -> dict[str, object]:
    output = dict(row)
    output["variant"] = variant
    output["seed"] = seed
    return output


def finite_values(values: object) -> list[float]:
    result = []
    for value in values:
        number = to_float(value)
        if math.isfinite(number):
            result.append(number)
    return result


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
