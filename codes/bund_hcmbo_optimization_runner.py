from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from bund_capacity_response_runner import _dump_routes_toml, _dump_run_toml, _load_toml
from bund_control_comparison_runner import DEFAULT_BASE_DIR, prepare_route_variant_base_dir
from bund_hcmbo_transfer_runner import (
    DEFAULT_CHANNEL_MAP,
    _capacity_controls_from_q,
    _direction_controls_from_g6,
    _dump_population_toml,
    _population_with_scaled_inflows,
)
from crowd_bellman.config_workflow import run_from_config
from crowd_bellman.g5_hcmbo import (
    ALL_GATE_IDS,
    CHANNEL_NAMES,
    HCMBOConfig,
    V2ControlVector,
    V2EvaluationRecord,
    compute_v2_objective,
    control_from_capacity_mode,
    generate_direction_candidates,
    make_no_cap_control,
    optimize_fixed_direction,
    screen_directions,
    shortlist_directions,
)
from crowd_bellman.metrics import save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("codes/results/bund_hcmbo_optimization_small")


@dataclass(frozen=True)
class RunBudgets:
    screen_steps: int
    optimization_steps: int
    high_fidelity_steps: int
    screen_time_horizon: float
    optimization_time_horizon: float
    high_fidelity_time_horizon: float


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _duration_from_run(base_run: dict[str, object], overrides: dict[str, object]) -> float:
    simulation = base_run.get("simulation", {})
    duration = float(simulation.get("time_horizon", 1.0)) if isinstance(simulation, dict) else 1.0
    if "time_horizon" in overrides:
        duration = float(overrides["time_horizon"])
    return duration


def _simulation_overrides(
    *,
    steps: int,
    time_horizon: float,
    rho_max: float,
    save_every: int | None = None,
) -> dict[str, object]:
    effective_save_every = int(save_every) if save_every is not None else int(steps) + 1
    return {
        "steps": int(steps),
        "time_horizon": float(time_horizon),
        "rho_max": float(rho_max),
        "save_every": effective_save_every,
        "density_contour_levels": 0,
    }


def _objective_table(base_objective: object, config: HCMBOConfig, *, name: str) -> dict[str, object]:
    table = dict(base_objective) if isinstance(base_objective, dict) else {}
    table.update(
        {
            "name": name,
            "lambda_j1": float(config.lambda_j1),
            "lambda_j2": float(config.lambda_j2),
            "lambda_j5": float(config.lambda_j5),
            "j2_metric": "soft",
            "j2_gamma": 1.0,
            "j1_scale": float(config.j1_scale),
            "j2_scale": float(config.j2_scale),
            "j5_scale": float(config.j5_scale),
            "use_normalized_terms": True,
        }
    )
    return table


def _control_to_q_by_gate(control: V2ControlVector) -> dict[str, object]:
    return {
        gate_id: [float(value) for value in profile]
        for gate_id, profile in zip(ALL_GATE_IDS, control.q_by_gate)
    }


def build_bund_controlled_routes(
    base_routes: dict[str, object],
    control: V2ControlVector,
    *,
    case_id: str,
    duration: float,
    alpha: float,
    beta: float,
    waiting_width: int,
    channel_map: dict[str, str] | None = None,
    include_controls: bool = True,
) -> dict[str, object]:
    channel_map = dict(channel_map or DEFAULT_CHANNEL_MAP)
    routes = deepcopy(base_routes)
    case_table = dict(routes.get("case", {}))
    case_table["case_id"] = case_id
    case_table["title"] = f"Bund HCMBO control {control.digest}"
    routes["case"] = case_table

    if not include_controls:
        routes["capacity_controls"] = []
        return routes

    directions = {
        name: state
        for name, state in zip(CHANNEL_NAMES, control.directions)
    }
    direction_controls = _direction_controls_from_g6(
        directions,
        channel_map=channel_map,
        alpha=alpha,
        beta=beta,
    )
    stages: list[dict[str, object]] = []
    for stage in routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        existing = stage_copy.get("controls", [])
        copied_controls = [dict(item) for item in existing if isinstance(item, dict)] if isinstance(existing, list) else []
        copied_controls.extend(dict(item) for item in direction_controls)
        if copied_controls:
            stage_copy["controls"] = copied_controls
        stages.append(stage_copy)
    routes["stages"] = stages
    routes["capacity_controls"] = _capacity_controls_from_q(
        _control_to_q_by_gate(control),
        channel_map=channel_map,
        duration=duration,
        waiting_width=waiting_width,
        capacity_scale=1.0,
    )
    return routes


class BundHCMBOEvaluationCache:
    def __init__(
        self,
        *,
        baseline_config: Path,
        output_root: Path,
        objective_config: HCMBOConfig,
        simulation_overrides: dict[str, object],
        fidelity: str,
        alpha: float,
        beta: float,
        waiting_width: int,
    ) -> None:
        self.baseline_config = baseline_config.resolve()
        self.output_root = output_root.resolve()
        self.objective_config = objective_config
        self.simulation_overrides = dict(simulation_overrides)
        self.fidelity = fidelity
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.waiting_width = int(waiting_width)
        self.records: list[V2EvaluationRecord] = []
        self._cache: dict[V2ControlVector, V2EvaluationRecord] = {}
        self._generated_dir = self.output_root / "_generated_configs"
        self._generated_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def evaluation_count(self) -> int:
        return len(self.records)

    def evaluate(
        self,
        control: V2ControlVector,
        *,
        source: str,
        phase: str,
        qbar_by_gate: dict[str, float],
        record_cached: bool = False,
    ) -> V2EvaluationRecord:
        normalized_control = control.normalized()
        cached = self._cache.get(normalized_control)
        if cached is not None and not record_cached:
            return cached
        if cached is not None and record_cached:
            record = V2EvaluationRecord(
                eval_id=len(self.records) + 1,
                phase=phase,
                source=source,
                fidelity=self.fidelity,
                control=normalized_control,
                objective_value=cached.objective_value,
                metrics=dict(cached.metrics),
                summary=dict(cached.summary),
                config_path=cached.config_path,
            )
            self.records.append(record)
            return record

        run_path = self._write_config(normalized_control, source=source, eval_id=len(self.records) + 1)
        summary = run_from_config(
            config_path=run_path,
            output_root=self.output_root,
            simulation_overrides=self.simulation_overrides,
            write_root_summary=False,
            channel_flux_directions={
                DEFAULT_CHANNEL_MAP[name]: _state_to_bund_flux_direction(state)
                for name, state in zip(CHANNEL_NAMES, normalized_control.directions)
            },
        )
        metrics = compute_v2_objective(
            summary=summary,
            control=normalized_control,
            qbar_by_gate=qbar_by_gate,
            config=self.objective_config,
        )
        summary["bund_hcmbo_optimization"] = {
            "source": source,
            "phase": phase,
            "fidelity": self.fidelity,
            "eval_id": len(self.records) + 1,
            "control": normalized_control.to_dict(),
            "config_path": str(run_path.resolve()),
        }
        case_output_dir = self.output_root / str(summary["case_id"])
        save_json(case_output_dir / "summary.json", summary)
        record = V2EvaluationRecord(
            eval_id=len(self.records) + 1,
            phase=phase,
            source=source,
            fidelity=self.fidelity,
            control=normalized_control,
            objective_value=float(metrics["objective_value"]),
            metrics=metrics,
            summary=summary,
            config_path=str(run_path.resolve()),
        )
        self.records.append(record)
        self._cache[normalized_control] = record
        return record

    def _write_config(self, control: V2ControlVector, *, source: str, eval_id: int) -> Path:
        base_run = _load_toml(self.baseline_config)
        routes_path = (self.baseline_config.parent / str(base_run["routes"]["file"])).resolve()
        base_routes = _load_toml(routes_path)
        safe_source = source.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()[:18]
        case_id = f"bund_hcmbo_{self.fidelity}_{eval_id:04d}_{safe_source}_{control.digest}"
        duration = _duration_from_run(base_run, self.simulation_overrides)
        generated_routes = build_bund_controlled_routes(
            base_routes,
            control,
            case_id=case_id,
            duration=duration,
            alpha=self.alpha,
            beta=self.beta,
            waiting_width=self.waiting_width,
            include_controls=True,
        )
        routes_output = self._generated_dir / f"routes_{case_id}.toml"
        run_output = self._generated_dir / f"run_{case_id}.toml"
        generated_run = {
            "simulation": dict(base_run["simulation"]),
            "objective": _objective_table(base_run.get("objective", {}), self.objective_config, name=case_id),
            "scene": {"file": str((self.baseline_config.parent / str(base_run["scene"]["file"])).resolve())},
            "population": {"file": str((self.baseline_config.parent / str(base_run["population"]["file"])).resolve())},
            "routes": {"file": str(routes_output.resolve())},
            "outputs": {"output_root": str(self.output_root)},
        }
        generated_run["simulation"].update(self.simulation_overrides)
        routes_output.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
        run_output.write_text(_dump_run_toml(generated_run), encoding="utf-8")
        return run_output


def _state_to_bund_flux_direction(state: str) -> str:
    normalized = str(state).upper()
    if normalized == "E":
        return "plus"
    if normalized == "W":
        return "minus"
    if normalized == "FREE":
        return "both"
    if normalized == "CLOSED":
        return "closed"
    return normalized


def qbar_from_bund_reference(summary: dict[str, object], *, config: HCMBOConfig) -> dict[str, float]:
    final_time = max(float(summary.get("final_time", 0.0)), 1.0)
    attempted = summary.get("gate_attempted_cumulative", {})
    if not isinstance(attempted, dict):
        attempted = {}
    qbar: dict[str, float] = {}
    for gate_id in ALL_GATE_IDS:
        channel, side = gate_id.split(":", 1)
        bund_channel = DEFAULT_CHANNEL_MAP[channel]
        natural_rate = float(attempted.get(f"{bund_channel}:{side}", 0.0)) / final_time
        qbar[gate_id] = max(float(config.min_qbar), natural_rate * float(config.qbar_multiplier))
    return qbar


def prepare_bund_hcmbo_base_dir(
    *,
    source_base_dir: Path,
    output_root: Path,
    route_variant: str,
    transition_kappa: float,
    inflow_rate_scale: float,
) -> Path:
    base_dir = prepare_route_variant_base_dir(
        source_base_dir=source_base_dir,
        comparison_root=output_root,
        route_variant=route_variant,
        transition_kappa=transition_kappa,
    )
    if not math.isclose(float(inflow_rate_scale), 1.0):
        run = _load_toml(base_dir / "run.toml")
        population_file = base_dir / str(run["population"]["file"])
        scaled = _population_with_scaled_inflows(_load_toml(population_file), float(inflow_rate_scale))
        population_file.write_text(_dump_population_toml(scaled), encoding="utf-8")
    return base_dir


def _write_no_control_config(
    *,
    baseline_config: Path,
    output_root: Path,
    config: HCMBOConfig,
    simulation_overrides: dict[str, object],
) -> Path:
    base_run = _load_toml(baseline_config)
    case_id = "bund_hcmbo_no_control"
    run_path = output_root / "_generated_configs" / "run_no_control.toml"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    generated_run = {
        "simulation": dict(base_run["simulation"]),
        "objective": _objective_table(base_run.get("objective", {}), config, name=case_id),
        "scene": {"file": str((baseline_config.parent / str(base_run["scene"]["file"])).resolve())},
        "population": {"file": str((baseline_config.parent / str(base_run["population"]["file"])).resolve())},
        "routes": {"file": str((baseline_config.parent / str(base_run["routes"]["file"])).resolve())},
        "outputs": {"output_root": str(output_root.resolve())},
    }
    generated_run["simulation"].update(simulation_overrides)
    run_path.write_text(_dump_run_toml(generated_run), encoding="utf-8")
    return run_path


def run_no_control_reference(
    *,
    baseline_config: Path,
    output_root: Path,
    config: HCMBOConfig,
    simulation_overrides: dict[str, object],
    qbar_by_gate: dict[str, float],
) -> tuple[dict[str, object], dict[str, float | str | bool | None]]:
    run_path = _write_no_control_config(
        baseline_config=baseline_config,
        output_root=output_root,
        config=config,
        simulation_overrides=simulation_overrides,
    )
    summary = run_from_config(
        config_path=run_path,
        output_root=output_root,
        simulation_overrides=simulation_overrides,
        write_root_summary=False,
    )
    summary["bund_hcmbo_optimization"] = {
        "source": "no_control_reference",
        "phase": "comparison",
        "config_path": str(run_path.resolve()),
    }
    save_json(output_root / str(summary["case_id"]) / "summary.json", summary)
    no_cap = make_no_cap_control(tuple("FREE" for _ in CHANNEL_NAMES), config.time_segments)
    return summary, compute_v2_objective(summary=summary, control=no_cap, qbar_by_gate=qbar_by_gate, config=config)


def _record_rows(records: list[V2EvaluationRecord]) -> list[dict[str, object]]:
    return [record.to_row() for record in records]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
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


def _comparison_row(label: str, summary: dict[str, object], metrics: dict[str, object]) -> dict[str, object]:
    return {
        "label": label,
        "case_id": summary.get("case_id"),
        "objective_value": float(metrics.get("objective_value", summary.get("objective_value", 0.0))),
        "j1_eval": float(metrics.get("j1_eval", 0.0)),
        "j2_eval": float(metrics.get("j2_eval", 0.0)),
        "j5_eval": float(metrics.get("j5_eval", 0.0)),
        "gate_rejected": float(metrics.get("gate_rejected", 0.0)),
        "gate_actual": float(metrics.get("gate_actual", 0.0)),
        "final_sink_cumulative": float(summary.get("final_sink_cumulative", 0.0)),
        "final_inflow_cumulative": float(summary.get("final_inflow_cumulative", 0.0)),
        "final_mass": float(summary.get("final_mass", 0.0)),
        "peak_density_max": float(summary.get("peak_density_max", 0.0)),
        "mean_density_avg": float(summary.get("mean_density_avg", 0.0)),
        "case_dir": str(summary.get("output_dir", "")),
    }


def parse_direction_candidates(raw_values: list[str] | None) -> list[tuple[str, ...]]:
    if not raw_values:
        return []
    candidates: list[tuple[str, ...]] = []
    for raw in raw_values:
        parts = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
        if len(parts) != len(CHANNEL_NAMES):
            raise argparse.ArgumentTypeError(f"Expected {len(CHANNEL_NAMES)} comma-separated direction states: {raw!r}")
        candidates.append(parts)
    return candidates


def resolve_direction_candidates(
    raw_values: list[str] | None,
    *,
    config: HCMBOConfig,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    explicit = parse_direction_candidates(raw_values)
    if explicit:
        return explicit[: max(1, int(config.direction_candidate_limit))]
    return generate_direction_candidates(config=config, rng=rng)


def build_small_budget_config(args: argparse.Namespace) -> HCMBOConfig:
    return HCMBOConfig(
        time_segments=args.time_segments,
        min_open_channels=2,
        direction_candidate_limit=max(1, int(args.direction_candidate_limit)),
        shortlist_size=max(1, int(args.shortlist_size)),
        screen_capacity_modes=("high",),
        initial_samples=max(1, int(args.initial_samples)),
        bo_iterations=max(0, int(args.bo_iterations)),
        bo_candidate_pool=max(4, int(args.bo_candidate_pool)),
        lcb_kappa=float(args.lcb_kappa),
        dfo_top_k=0,
        dfo_evaluations=0,
        high_fidelity_top_k=max(1, int(args.high_fidelity_top_k)),
        random_search_evaluations=0,
        random_seed=int(args.random_seed),
        qbar_multiplier=float(args.qbar_multiplier),
        min_qbar=float(args.min_qbar),
        beta=float(args.beta),
        lambda_j1=float(args.lambda_j1),
        lambda_j2=float(args.lambda_j2),
        lambda_j5=float(args.lambda_j5),
        lambda_jb=float(args.lambda_jb),
        lambda_jr=float(args.lambda_jr),
    )


def run_bund_hcmbo_optimization(args: argparse.Namespace) -> dict[str, object]:
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = build_small_budget_config(args)
    budgets = RunBudgets(
        screen_steps=args.screen_steps,
        optimization_steps=args.optimization_steps,
        high_fidelity_steps=args.high_fidelity_steps,
        screen_time_horizon=args.screen_time_horizon,
        optimization_time_horizon=args.optimization_time_horizon,
        high_fidelity_time_horizon=args.high_fidelity_time_horizon,
    )
    base_dir = prepare_bund_hcmbo_base_dir(
        source_base_dir=_resolve_path(args.base_dir),
        output_root=output_root,
        route_variant=args.route_variant,
        transition_kappa=args.transition_kappa,
        inflow_rate_scale=args.inflow_rate_scale,
    )
    baseline_config = base_dir / "run.toml"
    optimization_overrides = _simulation_overrides(
        steps=budgets.optimization_steps,
        time_horizon=budgets.optimization_time_horizon,
        rho_max=args.rho_max,
    )
    screen_overrides = _simulation_overrides(
        steps=budgets.screen_steps,
        time_horizon=budgets.screen_time_horizon,
        rho_max=args.rho_max,
    )
    high_fidelity_overrides = _simulation_overrides(
        steps=budgets.high_fidelity_steps,
        time_horizon=budgets.high_fidelity_time_horizon,
        rho_max=args.rho_max,
    )
    rng = np.random.default_rng(config.random_seed)

    reference_evaluator = BundHCMBOEvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_reference",
        objective_config=config,
        simulation_overrides=optimization_overrides,
        fidelity="reference",
        alpha=args.alpha,
        beta=args.beta,
        waiting_width=args.waiting_width,
    )
    reference_control = make_no_cap_control(tuple("FREE" for _ in CHANNEL_NAMES), config.time_segments)
    reference_record = reference_evaluator.evaluate(
        reference_control,
        source="qbar_reference_all_free",
        phase="reference",
        qbar_by_gate={gate_id: math.inf for gate_id in ALL_GATE_IDS},
    )
    qbar_by_gate = qbar_from_bund_reference(reference_record.summary, config=config)

    direction_candidates = resolve_direction_candidates(args.direction_candidate, config=config, rng=rng)
    screen_evaluator = BundHCMBOEvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_screen",
        objective_config=config,
        simulation_overrides=screen_overrides,
        fidelity="lf",
        alpha=args.alpha,
        beta=args.beta,
        waiting_width=args.waiting_width,
    )
    screen_records = screen_directions(
        evaluator=screen_evaluator,
        directions_list=direction_candidates,
        qbar_by_gate=qbar_by_gate,
        config=config,
    )
    shortlisted = shortlist_directions(screen_records, config.shortlist_size)
    opt_evaluator = BundHCMBOEvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_optimization",
        objective_config=config,
        simulation_overrides=optimization_overrides,
        fidelity="mf",
        alpha=args.alpha,
        beta=args.beta,
        waiting_width=args.waiting_width,
    )
    hcmbo_records: list[V2EvaluationRecord] = []
    bo_traces: list[dict[str, object]] = []
    for directions in shortlisted:
        records, trace = optimize_fixed_direction(
            evaluator=opt_evaluator,
            directions=directions,
            qbar_by_gate=qbar_by_gate,
            config=config,
            rng=rng,
            source_prefix="bund_hcmbo",
        )
        hcmbo_records.extend(records)
        bo_traces.extend(trace)

    ranked_mid = sorted(screen_records + hcmbo_records, key=lambda item: item.objective_value)
    unique_controls: list[V2ControlVector] = []
    seen: set[V2ControlVector] = set()
    for record in ranked_mid:
        if record.control in seen:
            continue
        unique_controls.append(record.control)
        seen.add(record.control)
        if len(unique_controls) >= config.high_fidelity_top_k:
            break

    hf_evaluator = BundHCMBOEvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_high_fidelity",
        objective_config=config,
        simulation_overrides=high_fidelity_overrides,
        fidelity="hf",
        alpha=args.alpha,
        beta=args.beta,
        waiting_width=args.waiting_width,
    )
    hf_records = [
        hf_evaluator.evaluate(
            control,
            source="high_fidelity_recheck",
            phase="high_fidelity",
            qbar_by_gate=qbar_by_gate,
        )
        for control in unique_controls
    ]
    best = min(hf_records or hcmbo_records, key=lambda item: item.objective_value)
    no_control_summary, no_control_metrics = run_no_control_reference(
        baseline_config=baseline_config,
        output_root=output_root / "_no_control",
        config=config,
        simulation_overrides=high_fidelity_overrides,
        qbar_by_gate=qbar_by_gate,
    )

    all_records = [reference_record] + screen_records + hcmbo_records + hf_records
    _write_csv(output_root / "bund_hcmbo_evaluation_log.csv", _record_rows(all_records))
    _write_csv(output_root / "bund_hcmbo_top_candidates.csv", _record_rows(sorted(hf_records or hcmbo_records, key=lambda item: item.objective_value)))
    comparison_rows = [
        _comparison_row("hcmbo_controlled", best.summary, best.metrics),
        _comparison_row("no_control", no_control_summary, no_control_metrics),
    ]
    _write_csv(output_root / "bund_hcmbo_vs_no_control.csv", comparison_rows)
    save_json(output_root / "bund_hcmbo_best_control.json", best.control.to_dict())
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_root": str(output_root),
        "base_dir": str(base_dir),
        "baseline_config": str(baseline_config),
        "route_variant": args.route_variant,
        "inflow_rate_scale": float(args.inflow_rate_scale),
        "actual_inflow_rate": _actual_inflow_rate(base_dir),
        "rho_max": float(args.rho_max),
        "config": config.__dict__,
        "budgets": budgets.__dict__,
        "direction_candidates": [dict(zip(CHANNEL_NAMES, item)) for item in direction_candidates],
        "shortlisted_directions": [dict(zip(CHANNEL_NAMES, item)) for item in shortlisted],
        "qbar_by_gate": qbar_by_gate,
        "best_high_fidelity": best.to_row(),
        "no_control": {
            "summary": _comparison_row("no_control", no_control_summary, no_control_metrics),
            "metrics": no_control_metrics,
        },
        "comparison_rows": comparison_rows,
        "outputs": {
            "evaluation_log": str(output_root / "bund_hcmbo_evaluation_log.csv"),
            "top_candidates": str(output_root / "bund_hcmbo_top_candidates.csv"),
            "best_control": str(output_root / "bund_hcmbo_best_control.json"),
            "comparison": str(output_root / "bund_hcmbo_vs_no_control.csv"),
        },
        "intermediate_data_policy": "No field arrays or high-resolution visualizations are generated by default.",
    }
    save_json(output_root / "bund_hcmbo_optimization_summary.json", payload)
    return payload


def _actual_inflow_rate(base_dir: Path) -> float | None:
    try:
        population = _load_toml(base_dir / "population.toml")
    except FileNotFoundError:
        return None
    groups = population.get("inflow_groups", [])
    if not isinstance(groups, list) or not groups:
        return None
    first = groups[0]
    if not isinstance(first, dict) or "rate" not in first:
        return None
    return float(first["rate"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small-budget Bund HCMBO optimization for control inputs, compared with no control.")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Converted BundScene directory.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Optimization output directory.")
    parser.add_argument("--route-variant", default="b_center_goal", choices=("base", "a_full_goal", "b_center_goal", "c_pass_through"))
    parser.add_argument("--inflow-rate-scale", type=float, default=3.0)
    parser.add_argument("--rho-max", type=float, default=4.0)
    parser.add_argument("--transition-kappa", type=float, default=32.0)
    parser.add_argument("--alpha", type=float, default=2.8)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--waiting-width", type=int, default=8)
    parser.add_argument("--screen-steps", type=int, default=80)
    parser.add_argument("--optimization-steps", type=int, default=160)
    parser.add_argument("--high-fidelity-steps", type=int, default=240)
    parser.add_argument("--screen-time-horizon", type=float, default=2.5)
    parser.add_argument("--optimization-time-horizon", type=float, default=5.0)
    parser.add_argument("--high-fidelity-time-horizon", type=float, default=7.5)
    parser.add_argument("--time-segments", type=int, default=1)
    parser.add_argument("--direction-candidate-limit", type=int, default=12)
    parser.add_argument("--shortlist-size", type=int, default=1)
    parser.add_argument("--initial-samples", type=int, default=2)
    parser.add_argument("--bo-iterations", type=int, default=1)
    parser.add_argument("--bo-candidate-pool", type=int, default=12)
    parser.add_argument("--high-fidelity-top-k", type=int, default=1)
    parser.add_argument("--lcb-kappa", type=float, default=1.5)
    parser.add_argument("--qbar-multiplier", type=float, default=1.2)
    parser.add_argument("--min-qbar", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--lambda-j1", type=float, default=1.0)
    parser.add_argument("--lambda-j2", type=float, default=1.0)
    parser.add_argument("--lambda-j5", type=float, default=1.0)
    parser.add_argument("--lambda-jb", type=float, default=1.0)
    parser.add_argument("--lambda-jr", type=float, default=0.1)
    parser.add_argument(
        "--direction-candidate",
        action="append",
        default=None,
        help="Comma-separated HCMBO direction states for top,middle,lower_middle,bottom. Can be repeated.",
    )
    return parser


def main() -> None:
    payload = run_bund_hcmbo_optimization(build_arg_parser().parse_args())
    print(json.dumps(
        {
            "summary": payload["outputs"],
            "best_objective": payload["best_high_fidelity"]["objective_value"],
            "no_control_objective": payload["no_control"]["summary"]["objective_value"],
            "actual_inflow_rate": payload["actual_inflow_rate"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
