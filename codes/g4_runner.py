from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tomllib

from crowd_bellman.metrics import save_json
from crowd_bellman.g4_sahbo import (
    BaselineConfig,
    CHANNEL_NAMES,
    DEFAULT_BASELINE_CONFIG,
    G4EvaluationCache,
    GridSearchConfig,
    PureSAConfig,
    RandomSearchConfig,
    SAHBOConfig,
    run_baseline,
    run_grid_search,
    run_pure_sa,
    run_random_search,
    run_sahbo,
)

METHOD_ORDER = ("baseline", "random_search", "grid", "pure_sa", "sahbo_no_proxy", "sahbo")


DEFAULT_G4_RUN_CONFIG = {
    "mode": "matrix",
    "baseline_config": str(DEFAULT_BASELINE_CONFIG),
    "output_root": "codes/results/g4_minimal_matrix",
    "beta": 0.35,
    "workers": None,
    "simulation_overrides": {},
    "visualization": {
        "enabled": True,
        "top_n": 12,
    },
    "sahbo": {
        "iterations": 4,
        "proxy_top_k": 3,
        "neighborhood_radius": 1,
        "initial_directions": ("FREE", "FREE", "FREE", "FREE"),
        "initial_eta": (8.0, 8.0, 8.0, 8.0),
        "eta_lower_bound": 1.0,
        "eta_upper_bound": 12.0,
        "eta_step_size": 1.2,
        "eta_perturbation": 0.8,
        "max_evaluations": None,
        "random_seed": 7,
    },
    "sahbo_no_proxy": {
        "iterations": 4,
        "proxy_top_k": 3,
        "neighborhood_radius": 1,
        "initial_directions": ("FREE", "FREE", "FREE", "FREE"),
        "initial_eta": (8.0, 8.0, 8.0, 8.0),
        "eta_lower_bound": 1.0,
        "eta_upper_bound": 12.0,
        "eta_step_size": 1.2,
        "eta_perturbation": 0.8,
        "max_evaluations": 20,
        "random_seed": 7,
    },
    "baseline": {
        "directions": ("FREE", "FREE", "FREE", "FREE"),
        "eta": (8.0, 8.0, 8.0, 8.0),
    },
    "random_search": {
        "max_evaluations": 20,
        "eta_lower_bound": 1.0,
        "eta_upper_bound": 12.0,
        "random_seed": 17,
    },
    "pure_sa": {
        "max_evaluations": 20,
        "neighborhood_radius": 1,
        "initial_directions": ("FREE", "FREE", "FREE", "FREE"),
        "initial_eta": (8.0, 8.0, 8.0, 8.0),
        "eta_lower_bound": 1.0,
        "eta_upper_bound": 12.0,
        "eta_perturbation": 0.8,
        "initial_temperature": 0.08,
        "cooling_factor": 0.85,
        "random_seed": 11,
    },
    "grid": {
        "eta_values": (1.0, 4.0, 8.0, 12.0),
        "max_evaluations": None,
    },
}


def _parse_float_tuple(raw: str, *, expected_len: int | None = None) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated values, got {len(values)}")
    return values


def _parse_channel_float_tuple(raw: str) -> tuple[float, ...]:
    return _parse_float_tuple(raw, expected_len=len(CHANNEL_NAMES))


def _parse_state_tuple(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    if len(values) != len(CHANNEL_NAMES):
        raise ValueError(f"Expected exactly {len(CHANNEL_NAMES)} channel states, e.g. FREE,FREE,FREE,FREE")
    return values


def _as_float_tuple(value: object, *, field_name: str, expected_len: int | None = None) -> tuple[float, ...]:
    if isinstance(value, str):
        values = _parse_float_tuple(value, expected_len=expected_len)
    elif isinstance(value, list | tuple):
        values = tuple(float(item) for item in value)
        if expected_len is not None and len(values) != expected_len:
            raise ValueError(f"{field_name} must contain {expected_len} values")
    else:
        raise ValueError(f"{field_name} must be a comma-separated string or a list of numbers")
    return values


def _as_state_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return _parse_state_tuple(value)
    if isinstance(value, list | tuple):
        values = tuple(str(item).upper() for item in value)
        if len(values) != len(CHANNEL_NAMES):
            raise ValueError(f"{field_name} must contain exactly {len(CHANNEL_NAMES)} states")
        return values
    raise ValueError(f"{field_name} must be a comma-separated string or a list of {len(CHANNEL_NAMES)} states")


def _as_direction_sets(value: object) -> tuple[tuple[str, ...], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError(f"grid.direction_sets must be a list of {len(CHANNEL_NAMES)}-state lists")
    direction_sets: list[tuple[str, ...]] = []
    for index, item in enumerate(value):
        direction_sets.append(_as_state_tuple(item, field_name=f"grid.direction_sets[{index}]"))
    return tuple(direction_sets)


def _resolve_config_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_g4_config(path: Path | None) -> dict[str, object]:
    config: dict[str, object] = {
        "mode": DEFAULT_G4_RUN_CONFIG["mode"],
        "baseline_config": DEFAULT_G4_RUN_CONFIG["baseline_config"],
        "output_root": DEFAULT_G4_RUN_CONFIG["output_root"],
        "beta": DEFAULT_G4_RUN_CONFIG["beta"],
        "workers": DEFAULT_G4_RUN_CONFIG["workers"],
        "simulation_overrides": dict(DEFAULT_G4_RUN_CONFIG["simulation_overrides"]),
        "visualization": dict(DEFAULT_G4_RUN_CONFIG["visualization"]),
        "sahbo": dict(DEFAULT_G4_RUN_CONFIG["sahbo"]),
        "sahbo_no_proxy": dict(DEFAULT_G4_RUN_CONFIG["sahbo_no_proxy"]),
        "baseline": dict(DEFAULT_G4_RUN_CONFIG["baseline"]),
        "random_search": dict(DEFAULT_G4_RUN_CONFIG["random_search"]),
        "pure_sa": dict(DEFAULT_G4_RUN_CONFIG["pure_sa"]),
        "grid": dict(DEFAULT_G4_RUN_CONFIG["grid"]),
    }
    if path is None:
        return config

    path = path.resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    base_dir = path.parent

    g4_table = raw.get("g4", {})
    if not isinstance(g4_table, dict):
        raise ValueError("[g4] must be a table")
    if "mode" in g4_table:
        config["mode"] = str(g4_table["mode"])
    if "baseline_config" in g4_table:
        config["baseline_config"] = str(_resolve_config_path(base_dir, str(g4_table["baseline_config"])))
    if "output_root" in g4_table:
        config["output_root"] = str(_resolve_config_path(base_dir, str(g4_table["output_root"])))
    if "beta" in g4_table:
        config["beta"] = float(g4_table["beta"])
    if "workers" in g4_table:
        config["workers"] = int(g4_table["workers"])

    visualization_table = raw.get("visualization", {})
    if visualization_table:
        if not isinstance(visualization_table, dict):
            raise ValueError("[visualization] must be a table")
        visualization = dict(config["visualization"])
        for key, value in visualization_table.items():
            visualization[key] = value
        if "enabled" in visualization:
            visualization["enabled"] = bool(visualization["enabled"])
        if "top_n" in visualization:
            visualization["top_n"] = int(visualization["top_n"])
        config["visualization"] = visualization

    simulation_table = raw.get("simulation_overrides", raw.get("simulation", {}))
    if simulation_table:
        if not isinstance(simulation_table, dict):
            raise ValueError("[simulation_overrides] must be a table")
        config["simulation_overrides"] = dict(simulation_table)

    sahbo_table = raw.get("sahbo", {})
    if sahbo_table:
        if not isinstance(sahbo_table, dict):
            raise ValueError("[sahbo] must be a table")
        sahbo = dict(config["sahbo"])
        for key, value in sahbo_table.items():
            sahbo[key] = value
        if "initial_directions" in sahbo:
            sahbo["initial_directions"] = _as_state_tuple(sahbo["initial_directions"], field_name="sahbo.initial_directions")
        if "initial_eta" in sahbo:
            sahbo["initial_eta"] = _as_float_tuple(sahbo["initial_eta"], field_name="sahbo.initial_eta", expected_len=len(CHANNEL_NAMES))
        config["sahbo"] = sahbo

    sahbo_no_proxy_table = raw.get("sahbo_no_proxy", {})
    if sahbo_no_proxy_table:
        if not isinstance(sahbo_no_proxy_table, dict):
            raise ValueError("[sahbo_no_proxy] must be a table")
        no_proxy = dict(config["sahbo_no_proxy"])
        for key, value in sahbo_no_proxy_table.items():
            no_proxy[key] = value
        if "initial_directions" in no_proxy:
            no_proxy["initial_directions"] = _as_state_tuple(no_proxy["initial_directions"], field_name="sahbo_no_proxy.initial_directions")
        if "initial_eta" in no_proxy:
            no_proxy["initial_eta"] = _as_float_tuple(no_proxy["initial_eta"], field_name="sahbo_no_proxy.initial_eta", expected_len=len(CHANNEL_NAMES))
        config["sahbo_no_proxy"] = no_proxy

    baseline_table = raw.get("baseline", {})
    if baseline_table:
        if not isinstance(baseline_table, dict):
            raise ValueError("[baseline] must be a table")
        baseline = dict(config["baseline"])
        for key, value in baseline_table.items():
            baseline[key] = value
        if "directions" in baseline:
            baseline["directions"] = _as_state_tuple(baseline["directions"], field_name="baseline.directions")
        if "eta" in baseline:
            baseline["eta"] = _as_float_tuple(baseline["eta"], field_name="baseline.eta", expected_len=len(CHANNEL_NAMES))
        config["baseline"] = baseline

    random_table = raw.get("random_search", {})
    if random_table:
        if not isinstance(random_table, dict):
            raise ValueError("[random_search] must be a table")
        random_search = dict(config["random_search"])
        for key, value in random_table.items():
            random_search[key] = value
        config["random_search"] = random_search

    pure_sa_table = raw.get("pure_sa", {})
    if pure_sa_table:
        if not isinstance(pure_sa_table, dict):
            raise ValueError("[pure_sa] must be a table")
        pure_sa = dict(config["pure_sa"])
        for key, value in pure_sa_table.items():
            pure_sa[key] = value
        if "initial_directions" in pure_sa:
            pure_sa["initial_directions"] = _as_state_tuple(pure_sa["initial_directions"], field_name="pure_sa.initial_directions")
        if "initial_eta" in pure_sa:
            pure_sa["initial_eta"] = _as_float_tuple(pure_sa["initial_eta"], field_name="pure_sa.initial_eta", expected_len=len(CHANNEL_NAMES))
        config["pure_sa"] = pure_sa

    grid_table = raw.get("grid", {})
    if grid_table:
        if not isinstance(grid_table, dict):
            raise ValueError("[grid] must be a table")
        grid = dict(config["grid"])
        for key, value in grid_table.items():
            grid[key] = value
        if "eta_values" in grid:
            grid["eta_values"] = _as_float_tuple(grid["eta_values"], field_name="grid.eta_values")
        if "direction_sets" in grid:
            grid["direction_sets"] = _as_direction_sets(grid["direction_sets"])
        config["grid"] = grid

    return config


def _override_if_present(target: dict[str, object], key: str, value: object) -> None:
    if value is not None:
        target[key] = value


def _method_names_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "both":
        return ("sahbo", "grid")
    if mode == "matrix":
        return METHOD_ORDER
    return (mode,)


def _run_method_payload(payload: dict[str, object]) -> dict[str, object]:
    method = str(payload["method"])
    output_root = Path(str(payload["output_root"]))
    evaluator = G4EvaluationCache(
        baseline_config=Path(str(payload["baseline_config"])),
        output_root=output_root,
        simulation_overrides=payload.get("simulation_overrides"),  # type: ignore[arg-type]
        beta=float(payload["beta"]),
    )
    result = _run_method_with_evaluator(
        method=method,
        evaluator=evaluator,
        tables=payload["tables"],  # type: ignore[arg-type]
        beta=float(payload["beta"]),
    )
    result["method_output_root"] = str(output_root)
    return {
        "method": method,
        "result": result,
        "rows": [record.to_row() for record in evaluator.records],
        "output_root": str(output_root),
        "evaluation_count": evaluator.evaluation_count,
    }


def _run_method_with_evaluator(
    *,
    method: str,
    evaluator: G4EvaluationCache,
    tables: dict[str, dict[str, object]],
    beta: float,
) -> dict[str, object]:
    if method == "baseline":
        baseline_table = tables["baseline"]
        return run_baseline(
            evaluator=evaluator,
            config=BaselineConfig(
                directions=_as_state_tuple(baseline_table["directions"], field_name="baseline.directions"),
                eta=_as_float_tuple(baseline_table["eta"], field_name="baseline.eta", expected_len=len(CHANNEL_NAMES)),
                beta=beta,
            ),
        )

    if method == "random_search":
        random_table = tables["random_search"]
        return run_random_search(
            evaluator=evaluator,
            config=RandomSearchConfig(
                max_evaluations=int(random_table["max_evaluations"]),
                eta_lower_bound=float(random_table["eta_lower_bound"]),
                eta_upper_bound=float(random_table["eta_upper_bound"]),
                random_seed=int(random_table["random_seed"]),
                beta=beta,
            ),
        )

    if method == "grid":
        grid_table = tables["grid"]
        direction_sets = _as_direction_sets(grid_table.get("direction_sets"))
        grid_kwargs: dict[str, object] = {
            "eta_values": _as_float_tuple(grid_table["eta_values"], field_name="grid.eta_values"),
            "max_evaluations": None if grid_table.get("max_evaluations") is None else int(grid_table["max_evaluations"]),
            "beta": beta,
        }
        if direction_sets is not None:
            grid_kwargs["direction_sets"] = direction_sets
        return run_grid_search(
            evaluator=evaluator,
            config=GridSearchConfig(**grid_kwargs),
        )

    if method == "pure_sa":
        pure_sa_table = tables["pure_sa"]
        return run_pure_sa(
            evaluator=evaluator,
            config=PureSAConfig(
                max_evaluations=int(pure_sa_table["max_evaluations"]),
                neighborhood_radius=int(pure_sa_table["neighborhood_radius"]),
                initial_eta=_as_float_tuple(pure_sa_table["initial_eta"], field_name="pure_sa.initial_eta", expected_len=len(CHANNEL_NAMES)),  # type: ignore[arg-type]
                initial_directions=_as_state_tuple(pure_sa_table["initial_directions"], field_name="pure_sa.initial_directions"),
                eta_lower_bound=float(pure_sa_table["eta_lower_bound"]),
                eta_upper_bound=float(pure_sa_table["eta_upper_bound"]),
                eta_perturbation=float(pure_sa_table["eta_perturbation"]),
                initial_temperature=float(pure_sa_table["initial_temperature"]),
                cooling_factor=float(pure_sa_table["cooling_factor"]),
                random_seed=int(pure_sa_table["random_seed"]),
                beta=beta,
            ),
        )

    if method == "sahbo_no_proxy":
        no_proxy_table = tables["sahbo_no_proxy"]
        return run_sahbo(
            evaluator=evaluator,
            config=SAHBOConfig(
                iterations=int(no_proxy_table["iterations"]),
                proxy_top_k=int(no_proxy_table["proxy_top_k"]),
                neighborhood_radius=int(no_proxy_table["neighborhood_radius"]),
                initial_eta=_as_float_tuple(no_proxy_table["initial_eta"], field_name="sahbo_no_proxy.initial_eta", expected_len=len(CHANNEL_NAMES)),  # type: ignore[arg-type]
                initial_directions=_as_state_tuple(no_proxy_table["initial_directions"], field_name="sahbo_no_proxy.initial_directions"),
                eta_lower_bound=float(no_proxy_table["eta_lower_bound"]),
                eta_upper_bound=float(no_proxy_table["eta_upper_bound"]),
                eta_step_size=float(no_proxy_table["eta_step_size"]),
                eta_perturbation=float(no_proxy_table["eta_perturbation"]),
                max_evaluations=None if no_proxy_table.get("max_evaluations") is None else int(no_proxy_table["max_evaluations"]),
                random_seed=int(no_proxy_table["random_seed"]),
                beta=beta,
                use_proxy=False,
            ),
        )

    if method == "sahbo":
        sahbo_table = tables["sahbo"]
        return run_sahbo(
            evaluator=evaluator,
            config=SAHBOConfig(
                iterations=int(sahbo_table["iterations"]),
                proxy_top_k=int(sahbo_table["proxy_top_k"]),
                neighborhood_radius=int(sahbo_table["neighborhood_radius"]),
                initial_eta=_as_float_tuple(sahbo_table["initial_eta"], field_name="sahbo.initial_eta", expected_len=len(CHANNEL_NAMES)),  # type: ignore[arg-type]
                initial_directions=_as_state_tuple(sahbo_table["initial_directions"], field_name="sahbo.initial_directions"),
                eta_lower_bound=float(sahbo_table["eta_lower_bound"]),
                eta_upper_bound=float(sahbo_table["eta_upper_bound"]),
                eta_step_size=float(sahbo_table["eta_step_size"]),
                eta_perturbation=float(sahbo_table["eta_perturbation"]),
                max_evaluations=None if sahbo_table.get("max_evaluations") is None else int(sahbo_table["max_evaluations"]),
                random_seed=int(sahbo_table["random_seed"]),
                beta=beta,
                use_proxy=True,
            ),
        )

    raise ValueError(f"Unsupported G4 method: {method}")


def _run_methods_parallel(
    *,
    methods: tuple[str, ...],
    output_root: Path,
    baseline_config: Path,
    simulation_overrides: dict[str, object] | None,
    beta: float,
    tables: dict[str, dict[str, object]],
    workers: int | None,
) -> list[dict[str, object]]:
    method_outputs: dict[str, dict[str, object]] = {}
    max_workers = len(methods) if workers is None else max(1, min(int(workers), len(methods)))
    payloads = [
        {
            "method": method,
            "baseline_config": str(baseline_config),
            "output_root": str(output_root / "_method_runs" / method),
            "simulation_overrides": simulation_overrides,
            "beta": beta,
            "tables": tables,
        }
        for method in methods
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_method_payload, payload): str(payload["method"]) for payload in payloads}
        for future in as_completed(futures):
            method = futures[future]
            method_outputs[method] = future.result()
    return [method_outputs[method] for method in methods]


def _run_methods_sequential(
    *,
    methods: tuple[str, ...],
    output_root: Path,
    baseline_config: Path,
    simulation_overrides: dict[str, object] | None,
    beta: float,
    tables: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return [
        _run_method_payload(
            {
                "method": method,
                "baseline_config": str(baseline_config),
                "output_root": str(output_root / "_method_runs" / method),
                "simulation_overrides": simulation_overrides,
                "beta": beta,
                "tables": tables,
            }
        )
        for method in methods
    ]


def _save_aggregated_g4_outputs(
    *,
    output_root: Path,
    baseline_config: Path,
    method_outputs: list[dict[str, object]],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    method_runs: dict[str, dict[str, object]] = {}
    next_eval_id = 1

    for method_output in method_outputs:
        method = str(method_output["method"])
        result = method_output["result"]  # type: ignore[assignment]
        if not isinstance(result, dict):
            raise TypeError(f"Unexpected result payload for method {method}: {type(result)!r}")
        results[method] = result
        method_runs[method] = {
            "output_root": method_output.get("output_root"),
            "evaluation_count": method_output.get("evaluation_count"),
        }
        for raw_row in method_output.get("rows", []):  # type: ignore[union-attr]
            if not isinstance(raw_row, dict):
                continue
            row = dict(raw_row)
            row["method_key"] = method
            row["method_eval_id"] = row.get("eval_id")
            row["method_output_root"] = method_output.get("output_root")
            row["eval_id"] = next_eval_id
            rows.append(row)
            next_eval_id += 1

    comparison = _build_method_comparison_from_results(results)
    payload = {
        "experiment_group": "G4",
        "design_version": "minimal_matrix_s_eta_fixed_p_parallel",
        "evaluator": {
            "baseline_config": str(baseline_config),
            "evaluation_count": len(rows),
            "fixed_behavior_parameter": "p_hat",
            "parallelized_by_method": True,
        },
        "baseline": results.get("baseline"),
        "random_search": results.get("random_search"),
        "pure_sa": results.get("pure_sa"),
        "sahbo_no_proxy": results.get("sahbo_no_proxy"),
        "sahbo": results.get("sahbo"),
        "grid_search": results.get("grid"),
        "method_runs": method_runs,
        "method_comparison": comparison,
        "evaluations": rows,
    }
    _save_csv(output_root / "g4_evaluation_log.csv", rows)
    _save_csv(output_root / "g4_method_comparison.csv", comparison)
    save_json(output_root / "g4_sahbo_grid_summary.json", payload)
    return payload


def _build_method_comparison_from_results(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        result = results.get(method)
        if result is None or not isinstance(result.get("best"), dict):
            continue
        best = result["best"]
        rows.append(
            {
                "method": result.get("method"),
                "objective_value": best.get("objective_value"),
                "evaluation_count": result.get("evaluation_count", result.get("candidate_count")),
                "case_id": best.get("case_id"),
                "directions": _directions_string(best),
                "eta": _eta_string(best),
                "method_output_root": result.get("method_output_root"),
            }
        )
    return rows


def _directions_string(row: dict[str, object]) -> str:
    return ",".join(f"{name}:{row.get(f'direction_{name}')}" for name in CHANNEL_NAMES)


def _eta_string(row: dict[str, object]) -> str:
    return ",".join(f"{name}:{row.get(f'eta_{name}')}" for name in CHANNEL_NAMES)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 optimization baselines and SA-HBO comparisons.")
    parser.add_argument("--config", help="Path to a G4 TOML config file")
    parser.add_argument(
        "--mode",
        choices=("baseline", "random_search", "grid", "pure_sa", "sahbo_no_proxy", "sahbo", "both", "matrix"),
        default=None,
    )
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--time-horizon", type=float, default=None)

    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--proxy-top-k", type=int, default=None)
    parser.add_argument("--neighborhood-radius", type=int, default=None)
    parser.add_argument("--initial-directions", default=None)
    parser.add_argument("--initial-eta", default=None)
    parser.add_argument("--eta-lower", type=float, default=None)
    parser.add_argument("--eta-upper", type=float, default=None)
    parser.add_argument("--eta-step-size", type=float, default=None)
    parser.add_argument("--eta-perturbation", type=float, default=None)
    parser.add_argument("--sahbo-max-evals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--random-max-evals", type=int, default=None)
    parser.add_argument("--pure-sa-max-evals", type=int, default=None)
    parser.add_argument("--no-proxy-max-evals", type=int, default=None)
    parser.add_argument("--grid-eta-values", default=None)
    parser.add_argument("--grid-max-evals", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Worker process count for multi-method G4 runs. Defaults to the number of selected methods.")
    parser.add_argument("--no-visualization", action="store_true", help="Skip chart generation after writing G4 result tables.")
    parser.add_argument("--visual-top-n", type=int, default=None, help="Number of top candidates to include in the generated visual summary.")
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    run_config = _load_g4_config(Path(args.config) if args.config else None)
    _override_if_present(run_config, "mode", args.mode)
    _override_if_present(run_config, "baseline_config", args.baseline_config)
    _override_if_present(run_config, "output_root", args.output_root)
    _override_if_present(run_config, "beta", args.beta)
    _override_if_present(run_config, "workers", args.workers)

    simulation_overrides: dict[str, object] = dict(run_config["simulation_overrides"])
    if args.steps is not None:
        simulation_overrides["steps"] = args.steps
    if args.save_every is not None:
        simulation_overrides["save_every"] = args.save_every
    if args.time_horizon is not None:
        simulation_overrides["time_horizon"] = args.time_horizon

    sahbo_table = dict(run_config["sahbo"])
    _override_if_present(sahbo_table, "iterations", args.iterations)
    _override_if_present(sahbo_table, "proxy_top_k", args.proxy_top_k)
    _override_if_present(sahbo_table, "neighborhood_radius", args.neighborhood_radius)
    if args.initial_directions is not None:
        sahbo_table["initial_directions"] = _parse_state_tuple(args.initial_directions)
    if args.initial_eta is not None:
        sahbo_table["initial_eta"] = _parse_channel_float_tuple(args.initial_eta)
    _override_if_present(sahbo_table, "eta_lower_bound", args.eta_lower)
    _override_if_present(sahbo_table, "eta_upper_bound", args.eta_upper)
    _override_if_present(sahbo_table, "eta_step_size", args.eta_step_size)
    _override_if_present(sahbo_table, "eta_perturbation", args.eta_perturbation)
    _override_if_present(sahbo_table, "max_evaluations", args.sahbo_max_evals)
    _override_if_present(sahbo_table, "random_seed", args.seed)

    sahbo_no_proxy_table = dict(run_config["sahbo_no_proxy"])
    _override_if_present(sahbo_no_proxy_table, "max_evaluations", args.no_proxy_max_evals)
    _override_if_present(sahbo_no_proxy_table, "random_seed", args.seed)

    baseline_table = dict(run_config["baseline"])

    random_table = dict(run_config["random_search"])
    _override_if_present(random_table, "max_evaluations", args.random_max_evals)
    _override_if_present(random_table, "random_seed", args.seed)

    pure_sa_table = dict(run_config["pure_sa"])
    _override_if_present(pure_sa_table, "max_evaluations", args.pure_sa_max_evals)
    _override_if_present(pure_sa_table, "random_seed", args.seed)

    grid_table = dict(run_config["grid"])
    if args.grid_eta_values is not None:
        grid_table["eta_values"] = _parse_float_tuple(args.grid_eta_values)
    _override_if_present(grid_table, "max_evaluations", args.grid_max_evals)

    visualization_table = dict(run_config["visualization"])
    if args.no_visualization:
        visualization_table["enabled"] = False
    _override_if_present(visualization_table, "top_n", args.visual_top_n)

    mode = str(run_config["mode"])
    if mode not in {"baseline", "random_search", "grid", "pure_sa", "sahbo_no_proxy", "sahbo", "both", "matrix"}:
        raise ValueError("g4.mode must be one of: baseline, random_search, grid, pure_sa, sahbo_no_proxy, sahbo, both, matrix")
    beta = float(run_config["beta"])
    output_root = Path(str(run_config["output_root"]))
    baseline_config = Path(str(run_config["baseline_config"]))
    methods = _method_names_for_mode(mode)
    tables = {
        "baseline": baseline_table,
        "random_search": random_table,
        "grid": grid_table,
        "pure_sa": pure_sa_table,
        "sahbo_no_proxy": sahbo_no_proxy_table,
        "sahbo": sahbo_table,
    }

    workers = None if run_config.get("workers") is None else int(run_config["workers"])
    if len(methods) > 1 and workers != 1:
        method_outputs = _run_methods_parallel(
            methods=methods,
            output_root=output_root,
            baseline_config=baseline_config,
            simulation_overrides=simulation_overrides or None,
            beta=beta,
            tables=tables,
            workers=workers,
        )
    else:
        method_outputs = _run_methods_sequential(
            methods=methods,
            output_root=output_root,
            baseline_config=baseline_config,
            simulation_overrides=simulation_overrides or None,
            beta=beta,
            tables=tables,
        )

    _save_aggregated_g4_outputs(
        output_root=output_root,
        baseline_config=baseline_config,
        method_outputs=method_outputs,
    )
    if bool(visualization_table.get("enabled", True)):
        from crowd_bellman.g4_visualization import build_g4_visual_report

        report = build_g4_visual_report(output_root, top_n=int(visualization_table.get("top_n", 12)))
        print(f"Wrote G4 visual report to {report['output_root']}")
        for name, path in report["outputs"].items():
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
