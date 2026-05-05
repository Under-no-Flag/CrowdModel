from __future__ import annotations

import csv
import hashlib
import itertools
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_workflow import run_from_config
from .metrics import save_json


CHANNEL_NAMES = ("top", "middle", "lower_middle", "bottom")
CHANNEL_STATES = ("E", "W", "FREE", "CLOSED")
DEFAULT_BASELINE_CONFIG = Path("codes/scenes/examples/g2_multistage_directional/run_baseline.toml")


@dataclass(frozen=True)
class ControlVector:
    directions: tuple[str, ...]
    eta: tuple[float, ...]

    def normalized(self) -> "ControlVector":
        directions = tuple(_normalize_state(state) for state in self.directions)
        if len(directions) != len(CHANNEL_NAMES):
            raise ValueError(f"Expected {len(CHANNEL_NAMES)} channel directions, got {len(directions)}")
        eta = tuple(float(max(value, 1.0)) for value in self.eta)
        if len(eta) != len(CHANNEL_NAMES):
            raise ValueError(f"Expected {len(CHANNEL_NAMES)} eta values, got {len(eta)}")
        return ControlVector(directions=directions, eta=eta)

    @property
    def label(self) -> str:
        state_part = "_".join(f"{name}{state}" for name, state in zip(CHANNEL_NAMES, self.directions))
        eta_part = "_".join(f"{value:g}" for value in self.eta)
        return f"{state_part}_eta{eta_part}".replace(".", "p")

    def to_dict(self) -> dict[str, object]:
        return {
            "directions": {name: state for name, state in zip(CHANNEL_NAMES, self.directions)},
            "eta": {name: float(value) for name, value in zip(CHANNEL_NAMES, self.eta)},
        }


@dataclass(frozen=True)
class EvaluationRecord:
    eval_id: int
    source: str
    control: ControlVector
    objective_value: float
    summary: dict[str, object]
    config_path: str

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "eval_id": self.eval_id,
            "source": self.source,
            "objective_value": self.objective_value,
            "case_id": self.summary.get("case_id"),
            "j1": self.summary.get("j1_normalized", self.summary.get("j1_total_travel_time")),
            "j2": self.summary.get("j2_normalized", self.summary.get("j2_high_density_exposure")),
            "j5": self.summary.get("j5_normalized", self.summary.get("j5_channel_flux_variance")),
            "j1_raw": self.summary.get("j1_total_travel_time"),
            "j2_raw": self.summary.get("j2_high_density_exposure"),
            "j5_raw": self.summary.get("j5_channel_flux_variance"),
            "config_path": self.config_path,
        }
        for name, state, eta_value in zip(CHANNEL_NAMES, self.control.directions, self.control.eta):
            row[f"direction_{name}"] = state
            row[f"eta_{name}"] = float(eta_value)
            row[f"flux_{name}"] = self.summary.get("channel_flux_cumulative", {}).get(name)
            row[f"flux_share_{name}"] = self.summary.get("channel_flux_share", {}).get(name)
        return row


@dataclass
class SAHBOConfig:
    iterations: int = 4
    proxy_top_k: int = 3
    neighborhood_radius: int = 1
    initial_eta: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0)
    initial_directions: tuple[str, ...] = ("FREE", "FREE", "FREE", "FREE")
    eta_lower_bound: float = 1.0
    eta_upper_bound: float = 12.0
    eta_step_size: float = 1.2
    eta_perturbation: float = 0.8
    eta_backtrack_factor: float = 0.5
    eta_max_backtracks: int = 2
    acceptance_tolerance: float = 1.0e-9
    max_evaluations: int | None = None
    random_seed: int = 7
    beta: float = 0.35


@dataclass
class GridSearchConfig:
    direction_sets: tuple[tuple[str, ...], ...] = (
        ("FREE", "FREE", "FREE", "FREE"),
        ("E", "W", "W", "W"),
        ("W", "E", "W", "W"),
        ("W", "W", "E", "W"),
        ("W", "W", "W", "E"),
        ("W", "E", "E", "E"),
        ("E", "W", "E", "E"),
        ("E", "E", "W", "E"),
        ("E", "E", "E", "W"),
        ("CLOSED", "E", "W", "W"),
        ("E", "CLOSED", "W", "W"),
        ("E", "W", "CLOSED", "W"),
        ("E", "W", "W", "CLOSED"),
    )
    eta_values: tuple[float, ...] = (1.0, 4.0, 8.0, 12.0)
    max_evaluations: int | None = None
    beta: float = 0.35


class G4EvaluationCache:
    def __init__(
        self,
        *,
        baseline_config: Path,
        output_root: Path,
        simulation_overrides: dict[str, object] | None = None,
        beta: float = 0.35,
    ) -> None:
        self.baseline_config = baseline_config.resolve()
        self.output_root = output_root.resolve()
        self.simulation_overrides = simulation_overrides
        self.beta = float(beta)
        self.records: list[EvaluationRecord] = []
        self._cache: dict[ControlVector, EvaluationRecord] = {}
        self._generated_dir = self.output_root / "_generated_configs"
        self._generated_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def evaluation_count(self) -> int:
        return len(self.records)

    def evaluate(self, control: ControlVector, *, source: str) -> EvaluationRecord:
        normalized_control = control.normalized()
        cached = self._cache.get(normalized_control)
        if cached is not None:
            return cached

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
        summary["g4_control"] = normalized_control.to_dict()
        summary["g4_evaluation"] = {
            "source": source,
            "eval_id": len(self.records) + 1,
            "config_path": str(run_path.resolve()),
        }
        case_output_dir = self.output_root / str(summary["case_id"])
        save_json(case_output_dir / "summary.json", summary)

        record = EvaluationRecord(
            eval_id=len(self.records) + 1,
            source=source,
            control=normalized_control,
            objective_value=float(summary.get("objective_value", math.inf)),
            summary=summary,
            config_path=str(run_path.resolve()),
        )
        self.records.append(record)
        self._cache[normalized_control] = record
        return record

    def _write_config(self, control: ControlVector, *, source: str, eval_id: int) -> Path:
        base_run = _load_toml(self.baseline_config)
        routes_table = base_run.get("routes", {})
        if not isinstance(routes_table, dict):
            raise ValueError(f"{self.baseline_config} does not contain [routes]")
        routes_path = (self.baseline_config.parent / str(routes_table["file"])).resolve()
        base_routes = _load_toml(routes_path)

        safe_source = _short_source_label(source)
        case_id = f"g4_{eval_id:04d}_{safe_source}_{_control_digest(control)}"
        generated_routes = _build_routes(base_routes=base_routes, control=control, case_id=case_id, beta=self.beta)
        generated_run = {
            "simulation": dict(base_run["simulation"]),
            "objective": dict(base_run["objective"]),
            "scene": {"file": str((self.baseline_config.parent / str(base_run["scene"]["file"])).resolve())},
            "population": {"file": str((self.baseline_config.parent / str(base_run["population"]["file"])).resolve())},
            "routes": {"file": str((self._generated_dir / f"routes_{case_id}.toml").resolve())},
            "outputs": {"output_root": str(self.output_root)},
        }
        generated_run["objective"]["name"] = case_id

        routes_output = self._generated_dir / f"routes_{case_id}.toml"
        run_output = self._generated_dir / f"run_{case_id}.toml"
        routes_output.write_text(_dump_routes_toml(generated_routes), encoding="utf-8")
        run_output.write_text(_dump_run_toml(generated_run), encoding="utf-8")
        return run_output


def run_sahbo(
    *,
    evaluator: G4EvaluationCache,
    config: SAHBOConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(config.random_seed)
    current = ControlVector(config.initial_directions, config.initial_eta).normalized()
    current_record = evaluator.evaluate(current, source="sahbo_init")
    best_record = current_record
    iteration_logs: list[dict[str, object]] = []

    for iteration in range(config.iterations):
        if _budget_exhausted(evaluator, config.max_evaluations):
            break

        neighbors = generate_direction_neighbors(
            current.directions,
            radius=config.neighborhood_radius,
        )
        candidate_neighbors = [directions for directions in neighbors if directions != current.directions]
        proxy_rows = [
            {
                "directions": directions,
                "proxy_value": proxy_score(
                    directions=directions,
                    eta=current.eta,
                    incumbent_summary=current_record.summary,
                ),
            }
            for directions in candidate_neighbors
        ]
        proxy_rows.sort(key=lambda item: float(item["proxy_value"]))
        retained = proxy_rows[: max(1, config.proxy_top_k)]

        discrete_records: list[EvaluationRecord] = []
        for item in retained:
            if _budget_exhausted(evaluator, config.max_evaluations):
                break
            candidate = ControlVector(tuple(item["directions"]), current.eta).normalized()
            discrete_records.append(evaluator.evaluate(candidate, source=f"sahbo_iter{iteration}_discrete"))

        if discrete_records:
            current_record = min(discrete_records + [current_record], key=lambda record: record.objective_value)
            current = current_record.control
            if current_record.objective_value < best_record.objective_value:
                best_record = current_record

        proxy_report = _proxy_consistency_report(retained, discrete_records)

        continuous_log: dict[str, object] = {
            "accepted": False,
            "reason": "budget_exhausted",
        }
        if not _budget_exhausted(evaluator, config.max_evaluations):
            current, current_record, best_record, continuous_log = _continuous_block_update(
                evaluator=evaluator,
                current=current,
                current_record=current_record,
                best_record=best_record,
                config=config,
                rng=rng,
                iteration=iteration,
            )

        iteration_logs.append(
            {
                "iteration": iteration,
                "incumbent": current.to_dict(),
                "incumbent_objective": float(current_record.objective_value),
                "best": best_record.control.to_dict(),
                "best_objective": float(best_record.objective_value),
                "neighbor_count": len(neighbors),
                "proxy_retained": [
                    {
                        "directions": {name: state for name, state in zip(CHANNEL_NAMES, item["directions"])},
                        "proxy_value": float(item["proxy_value"]),
                    }
                    for item in retained
                ],
                "discrete_evaluations": [record.to_row() for record in discrete_records],
                "proxy_consistency": proxy_report,
                "continuous_update": continuous_log,
                "evaluation_count": evaluator.evaluation_count,
            }
        )

        if abs(float(current_record.objective_value) - float(best_record.objective_value)) <= config.acceptance_tolerance:
            pass

    return {
        "method": "SA-HBO",
        "config": {
            "iterations": config.iterations,
            "proxy_top_k": config.proxy_top_k,
            "neighborhood_radius": config.neighborhood_radius,
            "initial_control": ControlVector(config.initial_directions, config.initial_eta).normalized().to_dict(),
            "eta_lower_bound": config.eta_lower_bound,
            "eta_upper_bound": config.eta_upper_bound,
            "eta_step_size": config.eta_step_size,
            "eta_perturbation": config.eta_perturbation,
            "max_evaluations": config.max_evaluations,
            "random_seed": config.random_seed,
            "beta": config.beta,
        },
        "best": best_record.to_row(),
        "iterations": iteration_logs,
        "evaluation_count": evaluator.evaluation_count,
    }


def run_grid_search(
    *,
    evaluator: G4EvaluationCache,
    config: GridSearchConfig,
) -> dict[str, object]:
    records: list[EvaluationRecord] = []
    evaluated = 0
    for directions in config.direction_sets:
        for eta_value in config.eta_values:
            if config.max_evaluations is not None and evaluated >= config.max_evaluations:
                break
            control = ControlVector(
                directions=tuple(_normalize_state(state) for state in directions),
                eta=tuple(float(eta_value) for _ in CHANNEL_NAMES),
            )
            records.append(evaluator.evaluate(control, source="grid"))
            evaluated += 1
        if config.max_evaluations is not None and evaluated >= config.max_evaluations:
            break

    best = min(records, key=lambda record: record.objective_value) if records else None
    return {
        "method": "grid_search",
        "config": {
            "direction_sets": [
                {name: state for name, state in zip(CHANNEL_NAMES, directions)}
                for directions in config.direction_sets
            ],
            "eta_values": [float(value) for value in config.eta_values],
            "max_evaluations": config.max_evaluations,
            "beta": config.beta,
        },
        "best": None if best is None else best.to_row(),
        "candidate_count": len(records),
        "ranking": [record.to_row() for record in sorted(records, key=lambda item: item.objective_value)],
    }


def save_g4_outputs(
    *,
    output_root: Path,
    evaluator: G4EvaluationCache,
    sahbo_result: dict[str, object] | None,
    grid_result: dict[str, object] | None,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in evaluator.records]
    _save_csv(output_root / "g4_evaluation_log.csv", rows)

    comparison = _build_method_comparison(sahbo_result=sahbo_result, grid_result=grid_result)
    payload = {
        "experiment_group": "G4",
        "design_version": "sahbo_vs_grid_s_eta_fixed_p",
        "evaluator": {
            "baseline_config": str(evaluator.baseline_config),
            "evaluation_count": evaluator.evaluation_count,
            "fixed_behavior_parameter": "p_hat",
        },
        "sahbo": sahbo_result,
        "grid_search": grid_result,
        "method_comparison": comparison,
        "evaluations": rows,
    }
    save_json(output_root / "g4_sahbo_grid_summary.json", payload)
    _save_csv(output_root / "g4_method_comparison.csv", comparison)
    return payload


def generate_direction_neighbors(
    directions: tuple[str, ...],
    *,
    radius: int,
) -> list[tuple[str, ...]]:
    base = tuple(_normalize_state(state) for state in directions)
    neighbors: set[tuple[str, ...]] = {base}
    channel_count = len(base)
    max_radius = max(0, min(int(radius), channel_count))
    for change_count in range(1, max_radius + 1):
        for indices in itertools.combinations(range(channel_count), change_count):
            replacements = [
                [state for state in CHANNEL_STATES if state != base[index]]
                for index in indices
            ]
            for states in itertools.product(*replacements):
                candidate = list(base)
                for index, state in zip(indices, states):
                    candidate[index] = state
                neighbors.add(tuple(candidate))
    return sorted(neighbors)


def proxy_score(
    *,
    directions: tuple[str, ...],
    eta: tuple[float, ...],
    incumbent_summary: dict[str, object],
) -> float:
    states = tuple(_normalize_state(state) for state in directions)
    shares_raw = incumbent_summary.get("channel_flux_share", {})
    shares = np.array(
        [float(shares_raw.get(name, 1.0 / len(CHANNEL_NAMES))) if isinstance(shares_raw, dict) else 1.0 / len(CHANNEL_NAMES) for name in CHANNEL_NAMES],
        dtype=float,
    )
    eta_arr = np.array([max(float(value), 1.0) for value in eta], dtype=float)
    active = np.array([state != "CLOSED" for state in states], dtype=float)
    closed_penalty = float(np.sum((1.0 - active) * (1.0 + shares)))

    east_count = sum(1 for state in states if state == "E")
    west_count = sum(1 for state in states if state == "W")
    free_count = sum(1 for state in states if state == "FREE")
    direction_balance_penalty = abs(east_count - west_count) / max(len(CHANNEL_NAMES), 1)
    no_entry_or_return_penalty = 1.0 if east_count == 0 or west_count == 0 else 0.0
    free_penalty = 0.08 * free_count

    capacity = active * (1.0 + 0.05 * np.log1p(eta_arr))
    if float(np.sum(capacity)) <= 1.0e-12:
        balance_proxy = 1.0
    else:
        normalized_capacity = capacity / float(np.sum(capacity))
        balance_proxy = float(np.var(normalized_capacity))

    return (
        1.4 * closed_penalty
        + 0.9 * direction_balance_penalty
        + 1.2 * no_entry_or_return_penalty
        + 0.5 * balance_proxy
        + free_penalty
    )


def _continuous_block_update(
    *,
    evaluator: G4EvaluationCache,
    current: ControlVector,
    current_record: EvaluationRecord,
    best_record: EvaluationRecord,
    config: SAHBOConfig,
    rng: np.random.Generator,
    iteration: int,
) -> tuple[ControlVector, EvaluationRecord, EvaluationRecord, dict[str, object]]:
    delta = rng.choice(np.array([-1.0, 1.0]), size=len(CHANNEL_NAMES))
    eta_current = np.array(current.eta, dtype=float)
    c_k = config.eta_perturbation / math.sqrt(iteration + 1.0)
    alpha_k = config.eta_step_size / math.sqrt(iteration + 1.0)
    eta_plus = _project_eta(eta_current + c_k * delta, config=config)
    eta_minus = _project_eta(eta_current - c_k * delta, config=config)

    plus_record = evaluator.evaluate(
        ControlVector(current.directions, tuple(float(value) for value in eta_plus)),
        source=f"sahbo_iter{iteration}_eta_plus",
    )
    minus_record = evaluator.evaluate(
        ControlVector(current.directions, tuple(float(value) for value in eta_minus)),
        source=f"sahbo_iter{iteration}_eta_minus",
    )
    gradient = ((plus_record.objective_value - minus_record.objective_value) / max(2.0 * c_k, 1.0e-12)) * delta

    accepted = False
    candidate_record = current_record
    candidate_eta = eta_current.copy()
    backtracks = 0
    step_size = alpha_k
    while backtracks <= config.eta_max_backtracks:
        proposed_eta = _project_eta(eta_current - step_size * gradient, config=config)
        candidate_record = evaluator.evaluate(
            ControlVector(current.directions, tuple(float(value) for value in proposed_eta)),
            source=f"sahbo_iter{iteration}_eta_candidate_bt{backtracks}",
        )
        if candidate_record.objective_value <= current_record.objective_value - config.acceptance_tolerance:
            accepted = True
            candidate_eta = proposed_eta
            break
        step_size *= config.eta_backtrack_factor
        backtracks += 1

    if accepted:
        current = ControlVector(current.directions, tuple(float(value) for value in candidate_eta)).normalized()
        current_record = candidate_record
        if current_record.objective_value < best_record.objective_value:
            best_record = current_record

    for record in (plus_record, minus_record, candidate_record):
        if record.objective_value < best_record.objective_value:
            best_record = record

    log = {
        "accepted": accepted,
        "delta": [float(value) for value in delta],
        "c_k": float(c_k),
        "alpha_k_initial": float(alpha_k),
        "alpha_k_final": float(step_size),
        "gradient": [float(value) for value in gradient],
        "eta_plus": [float(value) for value in eta_plus],
        "eta_minus": [float(value) for value in eta_minus],
        "plus_objective": float(plus_record.objective_value),
        "minus_objective": float(minus_record.objective_value),
        "candidate_objective": float(candidate_record.objective_value),
        "current_objective_after_update": float(current_record.objective_value),
        "backtracks": int(backtracks),
        "reason": "accepted" if accepted else "rejected",
    }
    return current, current_record, best_record, log


def _project_eta(values: np.ndarray, *, config: SAHBOConfig) -> np.ndarray:
    return np.clip(values, float(config.eta_lower_bound), float(config.eta_upper_bound))


def _proxy_consistency_report(
    retained: list[dict[str, object]],
    records: list[EvaluationRecord],
) -> dict[str, object]:
    if not retained or not records:
        return {"count": 0, "spearman": None, "best_true_retained": None}

    proxy_by_direction = {
        tuple(item["directions"]): rank
        for rank, item in enumerate(sorted(retained, key=lambda row: float(row["proxy_value"])), start=1)
    }
    true_sorted = sorted(records, key=lambda record: record.objective_value)
    true_ranks = {record.control.directions: rank for rank, record in enumerate(true_sorted, start=1)}
    common = [directions for directions in proxy_by_direction if directions in true_ranks]
    if len(common) < 2:
        spearman = None
    else:
        proxy_rank = np.array([proxy_by_direction[directions] for directions in common], dtype=float)
        true_rank = np.array([true_ranks[directions] for directions in common], dtype=float)
        spearman = _spearman_from_ranks(proxy_rank, true_rank)
    return {
        "count": len(common),
        "spearman": spearman,
        "best_true_retained": true_sorted[0].control.to_dict() if true_sorted else None,
    }


def _spearman_from_ranks(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size != b.size or a.size < 2:
        return None
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a_centered * a_centered) * np.sum(b_centered * b_centered)))
    if denom <= 1.0e-12:
        return None
    return float(np.sum(a_centered * b_centered) / denom)


def _build_routes(
    *,
    base_routes: dict[str, object],
    control: ControlVector,
    case_id: str,
    beta: float,
) -> dict[str, object]:
    routes = {
        "case": dict(base_routes["case"]),
        "stages": [],
    }
    routes["case"]["case_id"] = case_id
    routes["case"]["title"] = f"G4 control {control.label}"

    channel_controls = _channel_controls(control=control, beta=beta)
    for stage in base_routes.get("stages", []):
        if not isinstance(stage, dict):
            continue
        stage_copy = dict(stage)
        copied_controls = [dict(control_item) for control_item in stage.get("controls", []) if isinstance(control_item, dict)]
        if str(stage_copy.get("stage_id")) in {"enter_platform", "return_left"}:
            copied_controls.extend(dict(item) for item in channel_controls)
        if copied_controls:
            stage_copy["controls"] = copied_controls
        routes["stages"].append(stage_copy)
    return routes


def _channel_controls(*, control: ControlVector, beta: float) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for channel_name, state, eta_value in zip(CHANNEL_NAMES, control.directions, control.eta):
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


def _short_source_label(source: str) -> str:
    normalized = source.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    aliases = {
        "sahbo_init": "si",
        "grid": "gr",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized.startswith("sahbo_iter"):
        compact = (
            normalized.replace("sahbo_iter", "s")
            .replace("_discrete", "d")
            .replace("_eta_plus", "ep")
            .replace("_eta_minus", "em")
            .replace("_eta_candidate", "ec")
            .replace("_bt", "b")
        )
        return compact[:18]
    return normalized[:18]


def _control_digest(control: ControlVector) -> str:
    payload = ";".join(control.directions) + "|" + ";".join(f"{value:.8g}" for value in control.eta)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _budget_exhausted(evaluator: G4EvaluationCache, max_evaluations: int | None) -> bool:
    return max_evaluations is not None and evaluator.evaluation_count >= max_evaluations


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
        return repr(value)
    if isinstance(value, list):
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
    return "\n".join(lines) + "\n"


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_method_comparison(
    *,
    sahbo_result: dict[str, object] | None,
    grid_result: dict[str, object] | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if sahbo_result is not None and isinstance(sahbo_result.get("best"), dict):
        best = sahbo_result["best"]
        rows.append(
            {
                "method": "SA-HBO",
                "objective_value": best.get("objective_value"),
                "evaluation_count": sahbo_result.get("evaluation_count"),
                "case_id": best.get("case_id"),
                "directions": _directions_string(best),
                "eta": _eta_string(best),
            }
        )
    if grid_result is not None and isinstance(grid_result.get("best"), dict):
        best = grid_result["best"]
        rows.append(
            {
                "method": "grid_search",
                "objective_value": best.get("objective_value"),
                "evaluation_count": grid_result.get("candidate_count"),
                "case_id": best.get("case_id"),
                "directions": _directions_string(best),
                "eta": _eta_string(best),
            }
        )
    return rows


def _directions_string(row: dict[str, object]) -> str:
    return ",".join(f"{name}:{row.get(f'direction_{name}')}" for name in CHANNEL_NAMES)


def _eta_string(row: dict[str, object]) -> str:
    return ",".join(f"{name}:{row.get(f'eta_{name}')}" for name in CHANNEL_NAMES)
