from __future__ import annotations

import csv
import hashlib
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .config_workflow import run_from_config
from .metrics import save_json


CHANNEL_NAMES = ("top", "middle", "lower_middle", "bottom")
CHANNEL_STATES = ("E", "W", "FREE", "CLOSED")
ALL_GATE_IDS = tuple(f"{channel}:{side}" for channel in CHANNEL_NAMES for side in ("plus", "minus"))
ACTIVE_SIDES_BY_STATE = {
    "E": ("plus",),
    "W": ("minus",),
    "FREE": ("plus", "minus"),
    "CLOSED": (),
}
DEFAULT_BASELINE_CONFIG = Path("codes/scenes/examples/g2_multistage_directional/run_baseline.toml")
DEFAULT_PRIOR_DIRECTIONS = ("FREE", "E", "W", "FREE")


@dataclass(frozen=True)
class V2ControlVector:
    directions: tuple[str, ...]
    q_by_gate: tuple[tuple[float, ...], ...]

    def normalized(self) -> "V2ControlVector":
        directions = tuple(_normalize_state(state) for state in self.directions)
        if len(directions) != len(CHANNEL_NAMES):
            raise ValueError(f"Expected {len(CHANNEL_NAMES)} channel directions, got {len(directions)}")
        if len(self.q_by_gate) != len(ALL_GATE_IDS):
            raise ValueError(f"Expected {len(ALL_GATE_IDS)} gate profiles, got {len(self.q_by_gate)}")
        segment_count = len(self.q_by_gate[0]) if self.q_by_gate else 0
        q_profiles: list[tuple[float, ...]] = []
        for profile in self.q_by_gate:
            if len(profile) != segment_count:
                raise ValueError("All gate profiles must have the same segment count")
            q_profiles.append(tuple(float(max(value, 0.0)) for value in profile))
        return V2ControlVector(directions=directions, q_by_gate=tuple(q_profiles))

    @property
    def segment_count(self) -> int:
        return len(self.q_by_gate[0]) if self.q_by_gate else 0

    @property
    def digest(self) -> str:
        parts = [";".join(self.directions)]
        for gate_id, profile in zip(ALL_GATE_IDS, self.q_by_gate):
            values = ",".join("inf" if math.isinf(value) else f"{value:.8g}" for value in profile)
            parts.append(f"{gate_id}={values}")
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]

    def to_dict(self) -> dict[str, object]:
        return {
            "directions": {name: state for name, state in zip(CHANNEL_NAMES, self.directions)},
            "q_by_gate": {
                gate_id: [float(value) for value in profile]
                for gate_id, profile in zip(ALL_GATE_IDS, self.q_by_gate)
            },
        }


@dataclass
class HCMBOConfig:
    time_segments: int = 2
    fixed_eta: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0)
    beta: float = 0.35
    qbar_multiplier: float = 1.2
    min_qbar: float = 0.2
    min_open_channels: int = 2
    direction_candidate_limit: int = 12
    shortlist_size: int = 3
    screen_capacity_modes: tuple[str, ...] = ("high", "medium")
    initial_samples: int = 6
    bo_iterations: int = 6
    bo_candidate_pool: int = 80
    lcb_kappa: float = 1.5
    dfo_top_k: int = 1
    dfo_evaluations: int = 6
    dfo_initial_step: float = 0.25
    dfo_min_step: float = 0.04
    high_fidelity_top_k: int = 5
    random_search_evaluations: int = 8
    random_seed: int = 23
    lambda_j1: float = 1.0
    lambda_j2: float = 1.0
    lambda_j5: float = 1.0
    lambda_jb: float = 1.0
    lambda_jr: float = 0.1
    j1_scale: float = 1.0
    j2_scale: float = 0.001
    j5_scale: float = 1.0
    jb_scale: float = 1.0
    jr_scale: float = 1.0
    cap_removed_relative_threshold: float = 0.02
    infeasible_penalty: float = 100.0


@dataclass
class V2EvaluationRecord:
    eval_id: int
    phase: str
    source: str
    fidelity: str
    control: V2ControlVector
    objective_value: float
    metrics: dict[str, float | str | bool | None]
    summary: dict[str, object]
    config_path: str

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "eval_id": self.eval_id,
            "phase": self.phase,
            "source": self.source,
            "fidelity": self.fidelity,
            "objective_value": float(self.objective_value),
            "case_id": self.summary.get("case_id"),
            "config_path": self.config_path,
        }
        row.update(self.metrics)
        for name, state in zip(CHANNEL_NAMES, self.control.directions):
            row[f"direction_{name}"] = state
        for gate_id, profile in zip(ALL_GATE_IDS, self.control.q_by_gate):
            safe_gate = gate_id.replace(":", "_")
            row[f"q_{safe_gate}"] = ";".join("inf" if math.isinf(value) else f"{value:.6g}" for value in profile)
            gate_actual = self.summary.get("gate_actual_cumulative", {})
            gate_rejected = self.summary.get("gate_rejected_cumulative", {})
            if isinstance(gate_actual, dict):
                row[f"actual_{safe_gate}"] = gate_actual.get(gate_id)
            if isinstance(gate_rejected, dict):
                row[f"rejected_{safe_gate}"] = gate_rejected.get(gate_id)
        return row


class G5EvaluationCache:
    def __init__(
        self,
        *,
        baseline_config: Path,
        output_root: Path,
        objective_config: HCMBOConfig,
        simulation_overrides: dict[str, object] | None = None,
        fidelity: str = "mf",
    ) -> None:
        self.baseline_config = baseline_config.resolve()
        self.output_root = output_root.resolve()
        self.objective_config = objective_config
        self.simulation_overrides = simulation_overrides
        self.fidelity = fidelity
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
                name: _state_to_flux_direction(state)
                for name, state in zip(CHANNEL_NAMES, normalized_control.directions)
            },
        )
        metrics = compute_v2_objective(
            summary=summary,
            control=normalized_control,
            qbar_by_gate=qbar_by_gate,
            config=self.objective_config,
        )
        summary["g5_control"] = normalized_control.to_dict()
        summary["g5_v2_metrics"] = metrics
        summary["g5_evaluation"] = {
            "source": source,
            "phase": phase,
            "fidelity": self.fidelity,
            "eval_id": len(self.records) + 1,
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
        routes_table = base_run.get("routes", {})
        if not isinstance(routes_table, dict):
            raise ValueError(f"{self.baseline_config} does not contain [routes]")
        routes_path = (self.baseline_config.parent / str(routes_table["file"])).resolve()
        base_routes = _load_toml(routes_path)

        safe_source = _short_label(source)
        case_id = f"g5_{self.fidelity}_{eval_id:04d}_{safe_source}_{control.digest}"
        generated_routes = _build_routes(
            base_routes=base_routes,
            control=control,
            case_id=case_id,
            duration=_duration_from_run(base_run, self.simulation_overrides),
            fixed_eta=self.objective_config.fixed_eta,
            beta=self.objective_config.beta,
        )
        generated_run = {
            "simulation": dict(base_run["simulation"]),
            "objective": dict(base_run["objective"]),
            "scene": {"file": str((self.baseline_config.parent / str(base_run["scene"]["file"])).resolve())},
            "population": {"file": str((self.baseline_config.parent / str(base_run["population"]["file"])).resolve())},
            "routes": {"file": str((self._generated_dir / f"routes_{case_id}.toml").resolve())},
            "outputs": {"output_root": str(self.output_root)},
        }
        generated_run["objective"].update(
            {
                "name": case_id,
                "lambda_j1": float(self.objective_config.lambda_j1),
                "lambda_j2": float(self.objective_config.lambda_j2),
                "lambda_j5": float(self.objective_config.lambda_j5),
                "lambda_jb": 0.0,
                "lambda_jr": 0.0,
                "j2_metric": "soft",
                "j2_gamma": 1.0,
                "j1_scale": float(self.objective_config.j1_scale),
                "j2_scale": float(self.objective_config.j2_scale),
                "j5_scale": float(self.objective_config.j5_scale),
                "use_normalized_terms": True,
            }
        )

        routes_output = self._generated_dir / f"routes_{case_id}.toml"
        run_output = self._generated_dir / f"run_{case_id}.toml"
        routes_output.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
        run_output.write_text(_dump_run_toml(generated_run), encoding="utf-8")
        return run_output


def run_hcmbo_experiment(
    *,
    baseline_config: Path = DEFAULT_BASELINE_CONFIG,
    output_root: Path,
    config: HCMBOConfig,
    screen_overrides: dict[str, object],
    optimization_overrides: dict[str, object],
    high_fidelity_overrides: dict[str, object],
    forced_direction_candidates: list[tuple[str, ...]] | None = None,
    forced_shortlist_directions: list[tuple[str, ...]] | None = None,
) -> dict[str, object]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.random_seed)

    reference_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_reference",
        objective_config=config,
        simulation_overrides=optimization_overrides,
        fidelity="reference",
    )
    reference_control = make_no_cap_control(
        directions=tuple("FREE" for _ in CHANNEL_NAMES),
        segment_count=config.time_segments,
    )
    reference_record = reference_evaluator.evaluate(
        reference_control,
        source="qbar_reference_all_free",
        phase="reference",
        qbar_by_gate={gate_id: math.inf for gate_id in ALL_GATE_IDS},
    )
    qbar_by_gate = qbar_from_reference(reference_record.summary, config=config)

    direction_candidates = (
        [tuple(_normalize_state(state) for state in directions) for directions in forced_direction_candidates]
        if forced_direction_candidates is not None
        else generate_direction_candidates(config=config, rng=rng)
    )
    screen_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_screen",
        objective_config=config,
        simulation_overrides=screen_overrides,
        fidelity="lf",
    )
    screen_records = screen_directions(
        evaluator=screen_evaluator,
        directions_list=direction_candidates,
        qbar_by_gate=qbar_by_gate,
        config=config,
    )
    shortlisted_directions = (
        [tuple(_normalize_state(state) for state in directions) for directions in forced_shortlist_directions]
        if forced_shortlist_directions is not None
        else shortlist_directions(screen_records, config.shortlist_size)
    )

    opt_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_optimization",
        objective_config=config,
        simulation_overrides=optimization_overrides,
        fidelity="mf",
    )
    baseline_records = evaluate_baselines(
        evaluator=opt_evaluator,
        qbar_by_gate=qbar_by_gate,
        config=config,
    )
    random_records = run_random_search(
        evaluator=opt_evaluator,
        directions_list=direction_candidates,
        qbar_by_gate=qbar_by_gate,
        config=config,
        rng=rng,
    )
    hcmbo_records: list[V2EvaluationRecord] = []
    bo_traces: list[dict[str, object]] = []
    for directions in shortlisted_directions:
        records, trace = optimize_fixed_direction(
            evaluator=opt_evaluator,
            directions=directions,
            qbar_by_gate=qbar_by_gate,
            config=config,
            rng=rng,
            source_prefix="hcmbo",
        )
        hcmbo_records.extend(records)
        bo_traces.extend(trace)

    mid_records = baseline_records + random_records + hcmbo_records
    hf_evaluator = G5EvaluationCache(
        baseline_config=baseline_config,
        output_root=output_root / "_high_fidelity",
        objective_config=config,
        simulation_overrides=high_fidelity_overrides,
        fidelity="hf",
    )
    unique_controls: list[V2ControlVector] = []
    seen: set[V2ControlVector] = set()
    for record in sorted(mid_records, key=lambda item: item.objective_value):
        if record.control in seen:
            continue
        unique_controls.append(record.control)
        seen.add(record.control)
        if len(unique_controls) >= max(1, config.high_fidelity_top_k):
            break
    hf_records = [
        hf_evaluator.evaluate(
            control,
            source="high_fidelity_recheck",
            phase="high_fidelity",
            qbar_by_gate=qbar_by_gate,
        )
        for control in unique_controls
    ]

    all_rows = (
        [reference_record.to_row()]
        + [record.to_row() for record in screen_records]
        + [record.to_row() for record in mid_records]
        + [record.to_row() for record in hf_records]
    )
    best_hf = min(hf_records, key=lambda item: item.objective_value) if hf_records else min(mid_records, key=lambda item: item.objective_value)
    method_comparison = build_method_comparison(
        baseline_records=baseline_records,
        random_records=random_records,
        hcmbo_records=hcmbo_records,
        hf_records=hf_records,
    )
    top_candidates = [record.to_row() for record in sorted(hf_records or mid_records, key=lambda item: item.objective_value)]

    _save_csv(output_root / "G5_evaluation_log.csv", all_rows)
    _save_csv(output_root / "G5_top_candidates.csv", top_candidates)
    _save_csv(output_root / "G5_method_comparison.csv", method_comparison)
    save_json(output_root / "G5_best_control.json", best_hf.control.to_dict())
    payload = {
        "experiment_group": "G5",
        "design_version": "v2_hcmbo_s_q",
        "baseline_config": str(baseline_config.resolve()),
        "output_root": str(output_root),
        "config": config.__dict__,
        "screen_overrides": screen_overrides,
        "optimization_overrides": optimization_overrides,
        "high_fidelity_overrides": high_fidelity_overrides,
        "qbar_by_gate": qbar_by_gate,
        "direction_candidates": [_directions_dict(directions) for directions in direction_candidates],
        "shortlisted_directions": [_directions_dict(directions) for directions in shortlisted_directions],
        "reference": reference_record.to_row(),
        "method_comparison": method_comparison,
        "best_high_fidelity": best_hf.to_row(),
        "bo_traces": bo_traces,
        "outputs": {
            "evaluation_log": str(output_root / "G5_evaluation_log.csv"),
            "top_candidates": str(output_root / "G5_top_candidates.csv"),
            "method_comparison": str(output_root / "G5_method_comparison.csv"),
            "best_control": str(output_root / "G5_best_control.json"),
            "capacity_profiles": str(output_root / "G5_capacity_profiles.png"),
            "flux_share": str(output_root / "G5_flux_share.png"),
            "objective_trace": str(output_root / "G5_objective_trace.png"),
            "pareto_j1_j2": str(output_root / "G5_pareto_j1_j2.png"),
            "report": str(output_root / "G5_report.md"),
        },
    }
    save_json(output_root / "G5_config_summary.json", payload)
    save_g5_plots(output_root=output_root, records=hf_records or mid_records, all_records=mid_records, best=best_hf)
    write_report(output_root=output_root, payload=payload, records=hf_records or mid_records, best=best_hf)
    return payload


def compute_v2_objective(
    *,
    summary: dict[str, object],
    control: V2ControlVector,
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
) -> dict[str, float | str | bool | None]:
    objective = summary.get("objective", {})
    if not isinstance(objective, dict):
        objective = {}
    normalization_context = summary.get("normalization_context", {})
    if not isinstance(normalization_context, dict):
        normalization_context = {}

    mass_reference = float(normalization_context.get("total_mass_reference", 0.0))
    final_time = float(summary.get("final_time", normalization_context.get("evaluation_time", 0.0)))
    mass_time = max(mass_reference * final_time, 1.0e-12)
    j1_eval = float(objective.get("j1_eval", float(summary.get("j1_normalized", 0.0)) / max(config.j1_scale, 1.0e-12)))
    # objective.j2_eval already includes j2_scale from the generated run.
    j2_eval = float(objective.get("j2_eval", float(summary.get("j2_normalized", 0.0)) / max(config.j2_scale, 1.0e-12)))
    j5_eval = float(objective.get("j5_eval", float(summary.get("j5_normalized", 0.0)) / max(config.j5_scale, 1.0e-12)))
    jb_raw = float(summary.get("jb_waiting_exposure", 0.0))
    jb_normalized = jb_raw / mass_time
    jb_eval = jb_normalized / max(config.jb_scale, 1.0e-12)
    jr_normalized = smoothness_index(control=control, qbar_by_gate=qbar_by_gate)
    jr_eval = jr_normalized / max(config.jr_scale, 1.0e-12)

    cap_removed = float(summary.get("final_cap_removed_cumulative", 0.0))
    cap_removed_relative = cap_removed / max(mass_reference, 1.0e-12)
    feasible = cap_removed_relative <= config.cap_removed_relative_threshold + 1.0e-12
    penalty = 0.0 if feasible else config.infeasible_penalty * (cap_removed_relative - config.cap_removed_relative_threshold)
    objective_value = (
        config.lambda_j1 * j1_eval
        + config.lambda_j2 * j2_eval
        + config.lambda_j5 * j5_eval
        + config.lambda_jb * jb_eval
        + config.lambda_jr * jr_eval
        + penalty
    )
    gate_rejected = _gate_total(summary, "gate_rejected_cumulative")
    gate_attempted = _gate_total(summary, "gate_attempted_cumulative")
    gate_actual = _gate_total(summary, "gate_actual_cumulative")
    return {
        "j1": float(summary.get("j1_normalized", 0.0)),
        "j2": float(summary.get("j2_normalized", 0.0)),
        "j2_eval": float(j2_eval),
        "j5": float(summary.get("j5_normalized", 0.0)),
        "j1_eval": float(j1_eval),
        "j5_eval": float(j5_eval),
        "jb": float(jb_raw),
        "jb_normalized": float(jb_normalized),
        "jr_normalized": float(jr_normalized),
        "gate_attempted": float(gate_attempted),
        "gate_actual": float(gate_actual),
        "gate_rejected": float(gate_rejected),
        "waiting_mass_peak": _max_dict_value(summary.get("gate_waiting_mass_peak", {})),
        "binding_time_ratio_max": _max_dict_value(summary.get("gate_binding_time_ratio", {})),
        "cap_removed_relative": float(cap_removed_relative),
        "objective_without_penalty": float(objective_value - penalty),
        "penalty": float(penalty),
        "objective_value": float(objective_value),
        "feasible": bool(feasible),
    }


def qbar_from_reference(summary: dict[str, object], *, config: HCMBOConfig) -> dict[str, float]:
    final_time = max(float(summary.get("final_time", 0.0)), 1.0)
    attempted = summary.get("gate_attempted_cumulative", {})
    if not isinstance(attempted, dict):
        attempted = {}
    qbar: dict[str, float] = {}
    for gate_id in ALL_GATE_IDS:
        natural_rate = float(attempted.get(gate_id, 0.0)) / final_time
        qbar[gate_id] = max(float(config.min_qbar), natural_rate * float(config.qbar_multiplier))
    return qbar


def generate_direction_candidates(*, config: HCMBOConfig, rng: np.random.Generator) -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = []
    for raw in np.array(np.meshgrid(*([CHANNEL_STATES] * len(CHANNEL_NAMES)))).T.reshape(-1, len(CHANNEL_NAMES)):
        directions = tuple(str(item) for item in raw)
        if not direction_is_feasible(directions, config=config):
            continue
        candidates.append(directions)
    candidates.sort(key=_direction_proxy_score)
    required = [
        tuple("FREE" for _ in CHANNEL_NAMES),
        DEFAULT_PRIOR_DIRECTIONS,
        ("E", "E", "W", "W"),
        ("E", "W", "E", "W"),
        ("W", "E", "W", "E"),
    ]
    selected: list[tuple[str, ...]] = []
    for directions in required + candidates:
        if directions in candidates and directions not in selected:
            selected.append(directions)
        if len(selected) >= config.direction_candidate_limit:
            break
    if len(selected) < config.direction_candidate_limit:
        remaining = [directions for directions in candidates if directions not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: config.direction_candidate_limit - len(selected)])
    return selected


def direction_is_feasible(directions: tuple[str, ...], *, config: HCMBOConfig) -> bool:
    states = tuple(_normalize_state(state) for state in directions)
    open_count = sum(1 for state in states if state != "CLOSED")
    if open_count < int(config.min_open_channels):
        return False
    has_plus = any(state in {"E", "FREE"} for state in states)
    has_minus = any(state in {"W", "FREE"} for state in states)
    return has_plus and has_minus


def screen_directions(
    *,
    evaluator: G5EvaluationCache,
    directions_list: list[tuple[str, ...]],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
) -> list[V2EvaluationRecord]:
    records: list[V2EvaluationRecord] = []
    for directions in directions_list:
        for mode in config.screen_capacity_modes:
            control = control_from_capacity_mode(
                directions=directions,
                mode=mode,
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            )
            records.append(
                evaluator.evaluate(
                    control,
                    source=f"screen_{mode}",
                    phase="screen",
                    qbar_by_gate=qbar_by_gate,
                )
            )
    return records


def shortlist_directions(records: list[V2EvaluationRecord], limit: int) -> list[tuple[str, ...]]:
    best_by_direction: dict[tuple[str, ...], V2EvaluationRecord] = {}
    for record in records:
        current = best_by_direction.get(record.control.directions)
        if current is None or record.objective_value < current.objective_value:
            best_by_direction[record.control.directions] = record
    ranked = sorted(best_by_direction.values(), key=lambda item: item.objective_value)
    return [record.control.directions for record in ranked[: max(1, int(limit))]]


def evaluate_baselines(
    *,
    evaluator: G5EvaluationCache,
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
) -> list[V2EvaluationRecord]:
    baseline_controls = [
        ("baseline_no_cap", make_no_cap_control(tuple("FREE" for _ in CHANNEL_NAMES), config.time_segments)),
        (
            "baseline_prior_high",
            control_from_capacity_mode(
                directions=DEFAULT_PRIOR_DIRECTIONS,
                mode="high",
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            ),
        ),
        (
            "baseline_prior_medium",
            control_from_capacity_mode(
                directions=DEFAULT_PRIOR_DIRECTIONS,
                mode="medium",
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            ),
        ),
        (
            "baseline_prior_low",
            control_from_capacity_mode(
                directions=DEFAULT_PRIOR_DIRECTIONS,
                mode="low",
                qbar_by_gate=qbar_by_gate,
                segment_count=config.time_segments,
            ),
        ),
    ]
    return [
        evaluator.evaluate(control, source=name, phase="baseline", qbar_by_gate=qbar_by_gate)
        for name, control in baseline_controls
    ]


def run_random_search(
    *,
    evaluator: G5EvaluationCache,
    directions_list: list[tuple[str, ...]],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
) -> list[V2EvaluationRecord]:
    records: list[V2EvaluationRecord] = []
    for _ in range(max(0, int(config.random_search_evaluations))):
        directions = directions_list[int(rng.integers(0, len(directions_list)))]
        dim = free_dimension(directions, config.time_segments)
        x = rng.random(dim)
        control = control_from_x(
            directions=directions,
            x=x,
            qbar_by_gate=qbar_by_gate,
            segment_count=config.time_segments,
        )
        records.append(
            evaluator.evaluate(
                control,
                source="random_search",
                phase="random_search",
                qbar_by_gate=qbar_by_gate,
                record_cached=True,
            )
        )
    return records


def optimize_fixed_direction(
    *,
    evaluator: G5EvaluationCache,
    directions: tuple[str, ...],
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
    rng: np.random.Generator,
    source_prefix: str,
) -> tuple[list[V2EvaluationRecord], list[dict[str, object]]]:
    dim = free_dimension(directions, config.time_segments)
    xs = initial_design(dim=dim, sample_count=config.initial_samples, rng=rng)
    records: list[V2EvaluationRecord] = []
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]] = []
    trace: list[dict[str, object]] = []
    for index, x in enumerate(xs):
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=x, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source=f"{source_prefix}_init",
            phase="bo_init",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        x_records.append((x, record))
        trace.append(_trace_row(directions, "bo_init", index, record))

    for iteration in range(max(0, int(config.bo_iterations))):
        x_next = propose_lcb_candidate(
            x_records=x_records,
            dim=dim,
            candidate_pool=max(4, int(config.bo_candidate_pool)),
            kappa=float(config.lcb_kappa),
            rng=rng,
        )
        record = evaluator.evaluate(
            control_from_x(directions=directions, x=x_next, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
            source=f"{source_prefix}_bo",
            phase="bo",
            qbar_by_gate=qbar_by_gate,
            record_cached=True,
        )
        records.append(record)
        x_records.append((x_next, record))
        trace.append(_trace_row(directions, "bo", iteration, record))

    for rank, (x0, record0) in enumerate(sorted(x_records, key=lambda item: item[1].objective_value)[: max(0, int(config.dfo_top_k))]):
        local_records, local_trace = dfo_polish(
            evaluator=evaluator,
            directions=directions,
            x0=x0,
            record0=record0,
            qbar_by_gate=qbar_by_gate,
            config=config,
        )
        records.extend(local_records)
        for row in local_trace:
            row["local_rank"] = rank + 1
        trace.extend(local_trace)
    return records, trace


def dfo_polish(
    *,
    evaluator: G5EvaluationCache,
    directions: tuple[str, ...],
    x0: np.ndarray,
    record0: V2EvaluationRecord,
    qbar_by_gate: dict[str, float],
    config: HCMBOConfig,
) -> tuple[list[V2EvaluationRecord], list[dict[str, object]]]:
    if x0.size == 0 or config.dfo_evaluations <= 0:
        return [], []
    x_current = np.array(x0, copy=True)
    current = record0
    step = float(config.dfo_initial_step)
    records: list[V2EvaluationRecord] = []
    trace: list[dict[str, object]] = []
    evaluations = 0
    dim_index = 0
    while evaluations < int(config.dfo_evaluations) and step >= float(config.dfo_min_step):
        dim = dim_index % x_current.size
        improved = False
        for sign in (1.0, -1.0):
            if evaluations >= int(config.dfo_evaluations):
                break
            x_trial = np.array(x_current, copy=True)
            x_trial[dim] = float(np.clip(x_trial[dim] + sign * step, 0.0, 1.0))
            record = evaluator.evaluate(
                control_from_x(directions=directions, x=x_trial, qbar_by_gate=qbar_by_gate, segment_count=config.time_segments),
                source="hcmbo_dfo",
                phase="dfo",
                qbar_by_gate=qbar_by_gate,
                record_cached=True,
            )
            evaluations += 1
            records.append(record)
            trace.append(_trace_row(directions, "dfo", evaluations, record))
            if record.objective_value < current.objective_value:
                current = record
                x_current = x_trial
                improved = True
                break
        if not improved:
            step *= 0.5
        dim_index += 1
    return records, trace


def initial_design(*, dim: int, sample_count: int, rng: np.random.Generator) -> list[np.ndarray]:
    if dim <= 0:
        return [np.zeros(0, dtype=float)]
    seeds = [
        np.full(dim, 0.9, dtype=float),
        np.full(dim, 0.6, dtype=float),
        np.full(dim, 0.3, dtype=float),
    ]
    while len(seeds) < max(1, int(sample_count)):
        seeds.append(rng.random(dim))
    return seeds[: max(1, int(sample_count))]


def propose_lcb_candidate(
    *,
    x_records: list[tuple[np.ndarray, V2EvaluationRecord]],
    dim: int,
    candidate_pool: int,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if dim <= 0:
        return np.zeros(0, dtype=float)
    xs = np.vstack([item[0] for item in x_records])
    ys = np.array([item[1].objective_value for item in x_records], dtype=float)
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
        if float(np.sum(weights)) <= 1.0e-12:
            mean = float(np.mean(ys))
        else:
            mean = float(np.sum(weights * ys) / np.sum(weights))
        uncertainty = min(1.0, float(np.min(distances)) / math.sqrt(max(dim, 1)))
        acquisition = mean - float(kappa) * y_std * uncertainty
        if acquisition < best_score:
            best_score = acquisition
            best_candidate = candidate
    return np.array(best_candidate, dtype=float)


def control_from_capacity_mode(
    *,
    directions: tuple[str, ...],
    mode: str,
    qbar_by_gate: dict[str, float],
    segment_count: int,
) -> V2ControlVector:
    multipliers = {
        "inf": math.inf,
        "no_cap": math.inf,
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
    }
    multiplier = multipliers.get(mode.lower())
    if multiplier is None:
        raise ValueError(f"Unsupported capacity mode: {mode!r}")
    q_profiles: list[tuple[float, ...]] = []
    active = set(active_gate_ids(directions))
    for gate_id in ALL_GATE_IDS:
        if gate_id not in active:
            q_profiles.append(tuple(0.0 for _ in range(segment_count)))
        elif math.isinf(multiplier):
            q_profiles.append(tuple(math.inf for _ in range(segment_count)))
        else:
            q_profiles.append(tuple(float(qbar_by_gate[gate_id] * multiplier) for _ in range(segment_count)))
    return V2ControlVector(directions=tuple(_normalize_state(state) for state in directions), q_by_gate=tuple(q_profiles)).normalized()


def make_no_cap_control(directions: tuple[str, ...], segment_count: int) -> V2ControlVector:
    q_profiles: list[tuple[float, ...]] = []
    active = set(active_gate_ids(directions))
    for gate_id in ALL_GATE_IDS:
        q_profiles.append(tuple(math.inf if gate_id in active else 0.0 for _ in range(segment_count)))
    return V2ControlVector(directions=tuple(_normalize_state(state) for state in directions), q_by_gate=tuple(q_profiles)).normalized()


def control_from_x(
    *,
    directions: tuple[str, ...],
    x: np.ndarray,
    qbar_by_gate: dict[str, float],
    segment_count: int,
) -> V2ControlVector:
    states = tuple(_normalize_state(state) for state in directions)
    cursor = 0
    q = {gate_id: [0.0 for _ in range(segment_count)] for gate_id in ALL_GATE_IDS}
    for channel, state in zip(CHANNEL_NAMES, states):
        for segment in range(segment_count):
            if state == "E":
                gate = f"{channel}:plus"
                q[gate][segment] = float(qbar_by_gate[gate] * x[cursor])
                cursor += 1
            elif state == "W":
                gate = f"{channel}:minus"
                q[gate][segment] = float(qbar_by_gate[gate] * x[cursor])
                cursor += 1
            elif state == "FREE":
                plus_gate = f"{channel}:plus"
                minus_gate = f"{channel}:minus"
                r_plus = float(max(x[cursor], 0.0))
                r_minus = float(max(x[cursor + 1], 0.0))
                cursor += 2
                denom = max(r_plus + r_minus, 1.0)
                qbar_total = max(float(qbar_by_gate[plus_gate]), float(qbar_by_gate[minus_gate]))
                q[plus_gate][segment] = qbar_total * r_plus / denom
                q[minus_gate][segment] = qbar_total * r_minus / denom
            elif state == "CLOSED":
                continue
    return V2ControlVector(
        directions=states,
        q_by_gate=tuple(tuple(q[gate_id]) for gate_id in ALL_GATE_IDS),
    ).normalized()


def free_dimension(directions: tuple[str, ...], segment_count: int) -> int:
    dim = 0
    for state in tuple(_normalize_state(item) for item in directions):
        if state in {"E", "W"}:
            dim += segment_count
        elif state == "FREE":
            dim += 2 * segment_count
    return dim


def smoothness_index(*, control: V2ControlVector, qbar_by_gate: dict[str, float]) -> float:
    segment_count = control.segment_count
    if segment_count <= 1:
        return 0.0
    total = 0.0
    count = 0
    for gate_id, profile in zip(ALL_GATE_IDS, control.q_by_gate):
        qbar = max(float(qbar_by_gate.get(gate_id, 0.0)), 1.0e-12)
        finite_profile = np.array([0.0 if math.isinf(value) else float(value) for value in profile], dtype=float)
        diffs = np.diff(finite_profile) / qbar
        total += float(np.sum(diffs * diffs))
        count += diffs.size
    return total / max(count, 1)


def build_method_comparison(
    *,
    baseline_records: list[V2EvaluationRecord],
    random_records: list[V2EvaluationRecord],
    hcmbo_records: list[V2EvaluationRecord],
    hf_records: list[V2EvaluationRecord],
) -> list[dict[str, object]]:
    groups = {
        "baselines_mf": baseline_records,
        "random_search_mf": random_records,
        "hcmbo_mf": hcmbo_records,
        "high_fidelity_recheck": hf_records,
    }
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
                "directions": _directions_string(best.control.directions),
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


def save_g5_plots(
    *,
    output_root: Path,
    records: list[V2EvaluationRecord],
    all_records: list[V2EvaluationRecord],
    best: V2EvaluationRecord,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    save_capacity_profiles(output_root / "G5_capacity_profiles.png", best)
    save_flux_share(output_root / "G5_flux_share.png", best)
    save_objective_trace(output_root / "G5_objective_trace.png", all_records)
    save_pareto(output_root / "G5_pareto_j1_j2.png", records)


def save_capacity_profiles(path: Path, best: V2EvaluationRecord) -> None:
    segment_count = best.control.segment_count
    x = np.arange(segment_count)
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 4.8), dpi=160)
    for gate_id, profile in zip(ALL_GATE_IDS, best.control.q_by_gate):
        finite_profile = [np.nan if math.isinf(value) else float(value) for value in profile]
        if not any(value > 0.0 for value in finite_profile if not np.isnan(value)):
            continue
        ax.step(x, finite_profile, where="mid", label=gate_id)
    ax.set_xlabel("time segment")
    ax.set_ylabel("capacity rate q")
    ax.set_title("Best G5 capacity profile")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=8, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "best candidate uses no finite capacity bound",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_flux_share(path: Path, best: V2EvaluationRecord) -> None:
    shares = best.summary.get("channel_flux_share", {})
    if not isinstance(shares, dict):
        shares = {}
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2), dpi=160)
    values = [float(shares.get(channel, 0.0)) for channel in CHANNEL_NAMES]
    ax.bar(CHANNEL_NAMES, values, color="#4C78A8")
    ax.set_ylim(0.0, max(1.0, max(values + [0.0]) * 1.1))
    ax.set_ylabel("actual flux share")
    ax.set_title("Best G5 channel load distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_objective_trace(path: Path, records: list[V2EvaluationRecord]) -> None:
    if not records:
        return
    values = np.array([record.objective_value for record in records], dtype=float)
    best_so_far = np.minimum.accumulate(values)
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2), dpi=160)
    ax.plot(np.arange(1, len(values) + 1), values, color="#B0B0B0", marker="o", markersize=3, label="evaluated")
    ax.plot(np.arange(1, len(best_so_far) + 1), best_so_far, color="#D95F02", linewidth=2.0, label="best so far")
    ax.set_xlabel("medium-fidelity evaluation")
    ax.set_ylabel("V2 objective")
    ax.set_title("G5 optimization trace")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pareto(path: Path, records: list[V2EvaluationRecord]) -> None:
    if not records:
        return
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=160)
    jb = np.array([float(record.metrics.get("jb_normalized") or 0.0) for record in records], dtype=float)
    scatter = ax.scatter(
        [float(record.metrics.get("j1") or 0.0) for record in records],
        [float(record.metrics.get("j2_eval") or 0.0) for record in records],
        c=jb,
        cmap="viridis",
        s=[70.0 + 180.0 * float(record.metrics.get("j5") or 0.0) for record in records],
        edgecolors="black",
        linewidths=0.6,
        alpha=0.85,
    )
    ax.set_xlabel("J1 normalized")
    ax.set_ylabel("J2 eval")
    ax.set_title("G5 high-fidelity candidate projection")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("J_B normalized")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(
    *,
    output_root: Path,
    payload: dict[str, object],
    records: list[V2EvaluationRecord],
    best: V2EvaluationRecord,
) -> None:
    lines = [
        "# G5 V2 HCMBO Report",
        "",
        "## Best High-Fidelity Candidate",
        "",
        f"- case: `{best.summary.get('case_id')}`",
        f"- objective: `{best.objective_value:.6f}`",
        f"- directions: `{_directions_string(best.control.directions)}`",
        f"- J1: `{float(best.metrics.get('j1') or 0.0):.6f}`",
        f"- J2_eval: `{float(best.metrics.get('j2_eval') or 0.0):.6f}`",
        f"- J5: `{float(best.metrics.get('j5') or 0.0):.6f}`",
        f"- J_B normalized: `{float(best.metrics.get('jb_normalized') or 0.0):.6f}`",
        f"- J_R normalized: `{float(best.metrics.get('jr_normalized') or 0.0):.6f}`",
        f"- gate rejected: `{float(best.metrics.get('gate_rejected') or 0.0):.6f}`",
        f"- feasible: `{best.metrics.get('feasible')}`",
        "",
        "## Candidate Count",
        "",
        f"- reported candidates: `{len(records)}`",
        f"- shortlisted directions: `{len(payload.get('shortlisted_directions', []))}`",
        "",
        "## Output Files",
        "",
    ]
    outputs = payload.get("outputs", {})
    if isinstance(outputs, dict):
        for name, path in outputs.items():
            lines.append(f"- `{name}`: `{path}`")
    output_root.joinpath("G5_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_routes(
    *,
    base_routes: dict[str, object],
    control: V2ControlVector,
    case_id: str,
    duration: float,
    fixed_eta: tuple[float, ...],
    beta: float,
) -> dict[str, object]:
    routes = {
        "case": dict(base_routes["case"]),
        "stages": [],
        "capacity_controls": [],
    }
    routes["case"]["case_id"] = case_id
    routes["case"]["title"] = f"G5 V2 control {control.digest}"
    channel_controls = _direction_controls(directions=control.directions, fixed_eta=fixed_eta, beta=beta)
    for stage in base_routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        copied_controls = [dict(item) for item in stage.get("controls", []) if isinstance(item, dict)]
        if str(stage_copy.get("stage_id")) in {"enter_platform", "return_left"}:
            copied_controls.extend(dict(item) for item in channel_controls)
        if copied_controls:
            stage_copy["controls"] = copied_controls
        routes["stages"].append(stage_copy)

    segment_count = control.segment_count
    segment_length = float(duration) / max(segment_count, 1)
    active = set(active_gate_ids(control.directions))
    for gate_id, profile in zip(ALL_GATE_IDS, control.q_by_gate):
        if gate_id not in active:
            continue
        channel, side = gate_id.split(":")
        for segment, rate in enumerate(profile):
            routes["capacity_controls"].append(
                {
                    "channel": channel,
                    "side": side,
                    "rate": float(rate),
                    "time_start": float(segment * segment_length),
                    "time_end": float((segment + 1) * segment_length),
                }
            )
    return routes


def _direction_controls(
    *,
    directions: tuple[str, ...],
    fixed_eta: tuple[float, ...],
    beta: float,
) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for channel_name, state, eta_value in zip(CHANNEL_NAMES, directions, fixed_eta):
        normalized_state = _normalize_state(state)
        if normalized_state == "CLOSED":
            controls.append({"mode": "closed", "region": f"{channel_name}_channel"})
            continue
        direction = "E" if normalized_state == "FREE" else normalized_state
        allowed = ["ALL"] if normalized_state == "FREE" else [normalized_state]
        controls.append(
            {
                "mode": "fixed_direction",
                "region": f"{channel_name}_channel",
                "direction": direction,
                "alpha": float(beta) * float(eta_value),
                "beta": float(beta),
                "allowed_directions": allowed,
            }
        )
    return controls


def _duration_from_run(base_run: dict[str, object], overrides: dict[str, object] | None) -> float:
    simulation = base_run.get("simulation", {})
    duration = float(simulation.get("time_horizon", 1.0)) if isinstance(simulation, dict) else 1.0
    if overrides and "time_horizon" in overrides:
        duration = float(overrides["time_horizon"])
    return duration


def active_gate_ids(directions: tuple[str, ...]) -> tuple[str, ...]:
    gate_ids: list[str] = []
    for channel, state in zip(CHANNEL_NAMES, tuple(_normalize_state(item) for item in directions)):
        for side in ACTIVE_SIDES_BY_STATE[state]:
            gate_ids.append(f"{channel}:{side}")
    return tuple(gate_ids)


def _gate_total(summary: dict[str, object], key: str) -> float:
    raw = summary.get(key, {})
    if not isinstance(raw, dict):
        return 0.0
    return float(sum(float(value) for value in raw.values()))


def _max_dict_value(raw: object) -> float:
    if not isinstance(raw, dict) or not raw:
        return 0.0
    return float(max(float(value) for value in raw.values()))


def _normalize_state(state: str) -> str:
    normalized = str(state).upper()
    if normalized in {"BOTH", "BIDIRECTIONAL", "ALL", "0"}:
        return "FREE"
    if normalized in {"C", "NONE", "EMPTY"}:
        return "CLOSED"
    if normalized not in CHANNEL_STATES:
        raise ValueError(f"Unsupported channel state: {state!r}")
    return normalized


def _state_to_flux_direction(state: str) -> str:
    normalized = _normalize_state(state)
    if normalized == "FREE":
        return "FREE"
    if normalized == "CLOSED":
        return "CLOSED"
    return normalized


def _direction_proxy_score(directions: tuple[str, ...]) -> float:
    states = tuple(_normalize_state(state) for state in directions)
    closed_count = sum(1 for state in states if state == "CLOSED")
    free_count = sum(1 for state in states if state == "FREE")
    plus_count = sum(1 for state in states if state in {"E", "FREE"})
    minus_count = sum(1 for state in states if state in {"W", "FREE"})
    edge_closed = sum(1 for index, state in enumerate(states) if index in {0, len(states) - 1} and state == "CLOSED")
    return 1.4 * closed_count + 0.4 * abs(plus_count - minus_count) + 0.08 * free_count + 0.6 * edge_closed


def _trace_row(directions: tuple[str, ...], phase: str, iteration: int, record: V2EvaluationRecord) -> dict[str, object]:
    return {
        "directions": _directions_dict(directions),
        "phase": phase,
        "iteration": int(iteration),
        "objective_value": float(record.objective_value),
        "case_id": record.summary.get("case_id"),
        "eval_id": record.eval_id,
    }


def _directions_dict(directions: tuple[str, ...]) -> dict[str, str]:
    return {name: state for name, state in zip(CHANNEL_NAMES, directions)}


def _directions_string(directions: tuple[str, ...]) -> str:
    return ",".join(f"{name}:{state}" for name, state in zip(CHANNEL_NAMES, directions))


def _short_label(source: str) -> str:
    normalized = source.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    aliases = {
        "qbar_reference_all_free": "qref",
        "high_fidelity_recheck": "hf",
        "baseline_no_cap": "baseinf",
        "baseline_prior_high": "basehi",
        "baseline_prior_medium": "basemed",
        "baseline_prior_low": "baselow",
        "random_search": "rnd",
    }
    return aliases.get(normalized, normalized[:18])


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, int | float):
        return repr(float(value)) if isinstance(value, float) else repr(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML scalar value: {value!r}")


def _dump_table(table_name: str, payload: dict[str, object]) -> list[str]:
    lines = [f"[{table_name}]"]
    for key, value in payload.items():
        lines.append(f"{key} = {_format_scalar(value)}")
    return lines


def _dump_run_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    for table_name in ("simulation", "objective", "scene", "population", "routes", "outputs"):
        table = payload.get(table_name, {})
        if not isinstance(table, dict):
            continue
        if lines:
            lines.append("")
        lines.extend(_dump_table(table_name, table))
    return "\n".join(lines) + "\n"


def _dump_routes_toml(payload: dict[str, object]) -> str:
    lines: list[str] = []
    case_table = payload.get("case", {})
    if isinstance(case_table, dict):
        lines.extend(_dump_table("case", case_table))

    stages = payload.get("stages", [])
    if isinstance(stages, list):
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            lines.append("")
            lines.append("[[stages]]")
            for key, value in stage.items():
                if key in {"controls", "targets"}:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
            controls = stage.get("controls", [])
            if isinstance(controls, list):
                for control in controls:
                    if not isinstance(control, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.controls]]")
                    for key, value in control.items():
                        lines.append(f"{key} = {_format_scalar(value)}")
            targets = stage.get("targets", [])
            if isinstance(targets, list):
                for target in targets:
                    if not isinstance(target, dict):
                        continue
                    lines.append("")
                    lines.append("[[stages.targets]]")
                    for key, value in target.items():
                        lines.append(f"{key} = {_format_scalar(value)}")

    capacity_controls = payload.get("capacity_controls", [])
    if isinstance(capacity_controls, list):
        for control in capacity_controls:
            if not isinstance(control, dict):
                continue
            lines.append("")
            lines.append("[[capacity_controls]]")
            for key, value in control.items():
                if value is None:
                    continue
                lines.append(f"{key} = {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
