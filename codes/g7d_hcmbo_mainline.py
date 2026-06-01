from __future__ import annotations

import argparse
import sys
import time
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from crowd_bellman.g5_hcmbo import (
    DEFAULT_BASELINE_CONFIG,
    G5EvaluationCache,
    HCMBOConfig,
    V2EvaluationRecord,
    generate_direction_candidates,
)
from crowd_bellman.metrics import save_json
from g5_experiment_matrix import (
    FidelityBudget,
    apply_budget_overrides,
    coerce_like,
    evaluate_reference,
    method_comparison_from_groups,
)
from g7_hcmbo_variant_ablation import (
    RunContext,
    evaluate_high_fidelity_with_origin,
    objective_score,
    run_equal_direction_lcb_budget,
    select_top_unique_records,
    with_source_family,
)
from g7c_hcmbo_tpe_comparison import (
    cast_runs,
    finite_values,
    load_json,
    mean_or_blank,
    parse_bool,
    parse_seeds,
    prefix_row,
    read_csv,
    resolve_config_path,
    resolve_worker_count,
    run_key,
    to_float,
    write_csv,
)


DEFAULT_G7D_CONFIG = Path("codes/scenes/examples/g7d_hcmbo_mainline/g7d.toml")
METHOD_NAME = "hcmbo"


@dataclass(frozen=True)
class G7DProfile:
    name: str
    config: HCMBOConfig
    optimization: FidelityBudget
    high_fidelity: FidelityBudget
    seeds: tuple[int, ...]
    output_root: Path
    baseline_config: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G7-D HCMBO mainline multi-seed experiment.")
    parser.add_argument("--config", default=str(DEFAULT_G7D_CONFIG))
    parser.add_argument("--profile", choices=("full", "smoke"), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    loaded = load_g7d_config(Path(args.config)) if args.config else {}
    loaded_profile = str(loaded.get("profile") or "full")
    profile_name = args.profile or loaded_profile
    seeds = parse_seeds(args.seeds) if args.seeds else tuple(int(item) for item in loaded.get("seeds", (11, 23, 37, 51, 73)))
    output_root = Path(args.output_root or loaded.get("output_root") or "codes/results/g7d_hcmbo_mainline").resolve()
    baseline_config = Path(args.baseline_config or loaded.get("baseline_config") or DEFAULT_BASELINE_CONFIG).resolve()
    force = bool(args.force or loaded.get("force", False))
    fail_fast = bool(args.fail_fast or loaded.get("fail_fast", False))
    workers = resolve_worker_count(args.workers if args.workers is not None else loaded.get("workers"), len(seeds))

    profile = profile_from_name(
        profile_name,
        output_root=output_root,
        baseline_config=baseline_config,
        seeds=seeds,
    )
    if loaded.get("overrides") and (args.profile is None or args.profile == loaded_profile):
        profile = apply_profile_overrides(profile, loaded["overrides"])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "experiment_group": "G7-D",
        "design_version": "hcmbo_mainline_structured_capacity_search",
        "profile": profile.name,
        "config_path": str(Path(args.config).resolve()) if args.config else None,
        "baseline_config": str(profile.baseline_config),
        "output_root": str(profile.output_root),
        "method": METHOD_NAME,
        "method_description": "HCMBO mainline: direction-wise structured capacity search, no internal random search.",
        "seeds": list(profile.seeds),
        "workers": workers,
        "argv": sys.argv,
        "runs": [],
    }
    save_json(output_root / "G7D_manifest.json", manifest)
    failures = run_selected_runs(
        profile=profile,
        manifest=manifest,
        output_root=output_root,
        force=force,
        fail_fast=fail_fast,
        workers=workers,
    )
    write_g7d_outputs(output_root=output_root, manifest=manifest)
    if failures:
        raise RuntimeError(f"G7-D failed runs: {', '.join(failures)}")
    print(f"G7-D summary: {output_root / 'G7D_method_summary.csv'}")


def profile_from_name(
    name: str,
    *,
    output_root: Path,
    baseline_config: Path,
    seeds: tuple[int, ...],
) -> G7DProfile:
    if name == "smoke":
        return G7DProfile(
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
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    if name == "full":
        return G7DProfile(
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
            optimization=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            high_fidelity=FidelityBudget(steps=1600, time_horizon=160.0, bellman_every=5),
            seeds=seeds,
            output_root=output_root,
            baseline_config=baseline_config,
        )
    raise ValueError(f"Unsupported G7-D profile: {name!r}")


def load_g7d_config(path: Path) -> dict[str, object]:
    base_dir = path.resolve().parent
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    g7d = dict(raw.get("g7d", {}))
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
        "profile": g7d.get("profile"),
        "output_root": resolve_config_path(base_dir, str(g7d["output_root"])) if "output_root" in g7d else None,
        "baseline_config": resolve_config_path(base_dir, str(g7d["baseline_config"])) if "baseline_config" in g7d else None,
        "seeds": parse_seeds(g7d.get("seeds", "11,23,37,51,73")),
        "workers": int(g7d["workers"]) if "workers" in g7d else None,
        "force": bool(g7d["force"]) if "force" in g7d else False,
        "fail_fast": bool(g7d["fail_fast"]) if "fail_fast" in g7d else False,
        "overrides": overrides,
    }


def apply_profile_overrides(profile: G7DProfile, overrides: object) -> G7DProfile:
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


def run_selected_runs(
    *,
    profile: G7DProfile,
    manifest: dict[str, object],
    output_root: Path,
    force: bool,
    fail_fast: bool,
    workers: int,
) -> list[str]:
    entries_by_key: dict[str, dict[str, object]] = {}
    payloads: list[dict[str, object]] = []
    failures: list[str] = []
    for seed in profile.seeds:
        key = run_key(METHOD_NAME, seed)
        run_dir = output_root / METHOD_NAME / f"seed_{seed}"
        entry = {
            "key": key,
            "method": METHOD_NAME,
            "seed": seed,
            "description": str(manifest["method_description"]),
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
            payloads.append({"key": key, "seed": seed, "output_dir": str(run_dir), "profile": profile})
    save_json(output_root / "G7D_manifest.json", manifest)
    write_g7d_outputs(output_root=output_root, manifest=manifest)
    if not payloads:
        return failures
    if workers == 1:
        for payload in payloads:
            result = run_g7d_payload(payload)
            apply_g7d_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7d_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                raise RuntimeError(str(result.get("error") or "G7-D run failed"))
        return failures
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_g7d_payload, payload): str(payload["key"]) for payload in payloads}
        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"key": key, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
            apply_g7d_result(result=result, entries_by_key=entries_by_key, failures=failures, output_root=output_root, manifest=manifest)
            write_g7d_outputs(output_root=output_root, manifest=manifest)
            if result.get("status") == "failed" and fail_fast:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(str(result.get("error") or "G7-D run failed"))
    return failures


def run_g7d_payload(payload: dict[str, object]) -> dict[str, object]:
    key = str(payload["key"])
    seed = int(payload["seed"])
    try:
        profile = payload["profile"]
        if not isinstance(profile, G7DProfile):
            raise TypeError("profile payload must be a G7DProfile")
        start = time.perf_counter()
        result_payload = run_hcmbo_mainline(
            Path(str(payload["output_dir"])),
            replace(profile, config=replace(profile.config, random_seed=seed)),
            seed,
        )
        result_payload["runtime_seconds"] = time.perf_counter() - start
        save_json(Path(str(payload["output_dir"])) / "method_summary.json", result_payload)
        return {"key": key, "method": METHOD_NAME, "seed": seed, "status": "completed", **summary_entry(result_payload)}
    except Exception as exc:
        return {
            "key": key,
            "method": METHOD_NAME,
            "seed": seed,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def apply_g7d_result(
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
    save_json(output_root / "G7D_manifest.json", manifest)


def prepare_context(output_dir: Path, profile: G7DProfile, seed: int) -> RunContext:
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


def run_hcmbo_mainline(output_dir: Path, profile: G7DProfile, seed: int) -> dict[str, object]:
    ctx = prepare_context(output_dir, profile, seed)
    mid_records = run_equal_direction_lcb_budget(
        ctx=ctx,
        source_prefix="g7d_hcmbo",
        score_fn=objective_score,
        total_budget=int(ctx.config.random_search_evaluations),
    )
    selected = select_top_unique_records(mid_records, ctx.config.high_fidelity_top_k, objective_score)
    hf_records, hf_rows = evaluate_high_fidelity_with_origin(
        baseline_config=ctx.profile.baseline_config,
        output_dir=ctx.output_dir,
        config=ctx.config,
        qbar_by_gate=ctx.qbar_by_gate,
        overrides=ctx.profile.high_fidelity.to_overrides(),
        selected_records=selected,
        variant_name=METHOD_NAME,
    )
    return write_method_outputs(
        output_dir=ctx.output_dir,
        profile=ctx.profile,
        seed=ctx.seed,
        reference=ctx.reference,
        qbar_by_gate=ctx.qbar_by_gate,
        mid_records=mid_records,
        hf_records=hf_records,
        hf_rows=hf_rows,
    )


def write_method_outputs(
    *,
    output_dir: Path,
    profile: G7DProfile,
    seed: int,
    reference: V2EvaluationRecord,
    qbar_by_gate: dict[str, float],
    mid_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
    hf_rows: list[dict[str, object]],
) -> dict[str, object]:
    best_row = min(hf_rows, key=lambda item: to_float(item.get("objective_value"))) if hf_rows else {}
    all_rows = [prefix_row(reference.to_row(), METHOD_NAME, seed)]
    all_rows.extend(prefix_row(with_source_family(record.to_row()), METHOD_NAME, seed) for record in mid_records)
    all_rows.extend(prefix_row(row, METHOD_NAME, seed) for row in hf_rows)
    candidate_rows = [prefix_row(row, METHOD_NAME, seed) for row in sorted(hf_rows, key=lambda item: to_float(item.get("objective_value")))]
    write_csv(output_dir / "G7D_evaluation_log.csv", all_rows)
    write_csv(output_dir / "G7D_hf_candidates.csv", candidate_rows)
    write_csv(output_dir / "G7D_method_comparison.csv", method_comparison_from_groups({METHOD_NAME: hf_records or mid_records}))
    if hf_records:
        best_index = min(range(len(hf_rows)), key=lambda index: to_float(hf_rows[index].get("objective_value")))
        save_json(output_dir / "G7D_best_control.json", hf_records[best_index].control.to_dict())
    payload: dict[str, object] = {
        "method": METHOD_NAME,
        "seed": seed,
        "profile": profile.name,
        "best_high_fidelity": prefix_row(best_row, METHOD_NAME, seed) if best_row else {},
        "candidate_count": len(candidate_rows),
        "optimization_evaluation_count": len(mid_records),
        "hf_candidate_count": len(hf_records),
        "structured_evaluations": len(mid_records),
        "internal_random_evaluations": 0,
        "qbar_by_gate": qbar_by_gate,
        "direction_candidate_count": len(generate_direction_candidates(config=profile.config, rng=np.random.default_rng(seed))),
        "config": profile.config.__dict__,
        "optimization_overrides": profile.optimization.to_overrides(),
        "high_fidelity_overrides": profile.high_fidelity.to_overrides(),
        "outputs": {
            "evaluation_log": str(output_dir / "G7D_evaluation_log.csv"),
            "hf_candidates": str(output_dir / "G7D_hf_candidates.csv"),
            "best_control": str(output_dir / "G7D_best_control.json"),
        },
    }
    return payload


def write_g7d_outputs(*, output_root: Path, manifest: dict[str, object]) -> None:
    save_json(output_root / "G7D_manifest.json", manifest)
    summaries: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for run in manifest.get("runs", []):
        if not isinstance(run, dict) or run.get("status") not in {"completed", "skipped_complete"}:
            continue
        seed = int(run["seed"])
        run_dir = Path(str(run["output_dir"]))
        payload_path = run_dir / "method_summary.json"
        payload = load_json(payload_path) if payload_path.exists() else {}
        best = payload.get("best_high_fidelity", {}) if isinstance(payload, dict) else {}
        if not isinstance(best, dict):
            best = {}
        summaries.append(
            {
                "method": METHOD_NAME,
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
                "optimization_evaluation_count": payload.get("optimization_evaluation_count") if isinstance(payload, dict) else None,
                "hf_candidate_count": payload.get("hf_candidate_count") if isinstance(payload, dict) else None,
                "runtime_seconds": payload.get("runtime_seconds") if isinstance(payload, dict) else None,
                "output_dir": str(run_dir),
            }
        )
        candidates.extend(read_csv(run_dir / "G7D_hf_candidates.csv"))
    write_csv(output_root / "G7D_seed_summary.csv", summaries)
    write_csv(output_root / "G7D_hf_candidates.csv", candidates)
    write_csv(output_root / "G7D_method_summary.csv", build_method_summary_rows(summaries))
    write_report(output_root, summaries, candidates)


def build_method_summary_rows(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = finite_values(row.get("best_hf_objective") for row in seed_rows)
    feasible = [parse_bool(row.get("feasible_best")) for row in seed_rows if parse_bool(row.get("feasible_best")) is not None]
    if not values:
        return []
    best = min(seed_rows, key=lambda row: to_float(row.get("best_hf_objective")))
    return [
        {
            "method": METHOD_NAME,
            "seed_count": len(values),
            "mean_best_hf_objective": float(np.mean(values)),
            "std_best_hf_objective": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median_best_hf_objective": float(np.median(values)),
            "best_hf_objective": min(values),
            "worst_hf_objective": max(values),
            "feasible_rate": sum(1 for item in feasible if item) / len(feasible) if feasible else "",
            "mean_j2_eval": mean_or_blank(row.get("J2_eval") for row in seed_rows),
            "mean_gate_rejected": mean_or_blank(row.get("gate_rejected") for row in seed_rows),
            "best_seed": best.get("seed"),
            "best_case_id": best.get("best_case_id"),
            "best_origin_source_family": best.get("origin_source_family"),
        }
    ]


def write_report(output_root: Path, summaries: list[dict[str, object]], candidates: list[dict[str, object]]) -> None:
    ordered = build_method_summary_rows(summaries)
    lines = [
        "# G7-D HCMBO Mainline Multi-Seed Experiment",
        "",
        "## Scope",
        "",
        "- Method: HCMBO mainline.",
        "- Implementation: direction-wise structured capacity search plus high-fidelity recheck.",
        "- Internal random search is disabled; random search is only an external baseline in G6.",
        "- Seeds and budgets follow the G6 horizontal comparison configuration.",
        "- Final ranking uses high-fidelity recheck objective values only.",
        "",
        "## Method Summary",
        "",
    ]
    if not ordered:
        lines.append("- No completed seeds yet.")
    for row in ordered:
        lines.append(
            f"- `hcmbo`: mean `{row.get('mean_best_hf_objective')}`, "
            f"best `{row.get('best_hf_objective')}`, worst `{row.get('worst_hf_objective')}`, "
            f"feasible rate `{row.get('feasible_rate')}`, "
            f"mean J2 `{row.get('mean_j2_eval')}`, mean gate rejected `{row.get('mean_gate_rejected')}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `G7D_manifest.json`: `{output_root / 'G7D_manifest.json'}`",
            f"- `G7D_seed_summary.csv`: `{output_root / 'G7D_seed_summary.csv'}`",
            f"- `G7D_method_summary.csv`: `{output_root / 'G7D_method_summary.csv'}`",
            f"- `G7D_hf_candidates.csv`: `{output_root / 'G7D_hf_candidates.csv'}`",
            f"- HF candidate count: `{len(candidates)}`",
        ]
    )
    output_root.joinpath("G7D_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_entry(payload: dict[str, object]) -> dict[str, object]:
    best = payload.get("best_high_fidelity", {})
    if not isinstance(best, dict):
        best = {}
    return {"best_objective": best.get("objective_value"), "best_case_id": best.get("case_id"), "feasible": best.get("feasible")}


if __name__ == "__main__":
    main()
