from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from crowd_bellman.g4_sahbo import (
    DEFAULT_BASELINE_CONFIG,
    G4EvaluationCache,
    GridSearchConfig,
    SAHBOConfig,
    run_grid_search,
    run_sahbo,
    save_g4_outputs,
)


DEFAULT_G4_RUN_CONFIG = {
    "mode": "both",
    "baseline_config": str(DEFAULT_BASELINE_CONFIG),
    "output_root": "codes/results/g4_sahbo_vs_grid",
    "beta": 0.35,
    "simulation_overrides": {},
    "sahbo": {
        "iterations": 4,
        "proxy_top_k": 3,
        "neighborhood_radius": 1,
        "initial_directions": ("FREE", "FREE", "FREE"),
        "initial_eta": (8.0, 8.0, 8.0),
        "eta_lower_bound": 1.0,
        "eta_upper_bound": 12.0,
        "eta_step_size": 1.2,
        "eta_perturbation": 0.8,
        "max_evaluations": None,
        "random_seed": 7,
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


def _parse_three_float_tuple(raw: str) -> tuple[float, float, float]:
    values = _parse_float_tuple(raw, expected_len=3)
    return (values[0], values[1], values[2])


def _parse_state_tuple(raw: str) -> tuple[str, str, str]:
    values = tuple(item.strip().upper() for item in raw.split(",") if item.strip())
    if len(values) != 3:
        raise ValueError("Expected exactly three channel states, e.g. FREE,FREE,FREE")
    return (values[0], values[1], values[2])


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


def _as_state_tuple(value: object, *, field_name: str) -> tuple[str, str, str]:
    if isinstance(value, str):
        return _parse_state_tuple(value)
    if isinstance(value, list | tuple):
        values = tuple(str(item).upper() for item in value)
        if len(values) != 3:
            raise ValueError(f"{field_name} must contain exactly three states")
        return (values[0], values[1], values[2])
    raise ValueError(f"{field_name} must be a comma-separated string or a list of three states")


def _as_direction_sets(value: object) -> tuple[tuple[str, str, str], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError("grid.direction_sets must be a list of three-state lists")
    direction_sets: list[tuple[str, str, str]] = []
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
        "simulation_overrides": dict(DEFAULT_G4_RUN_CONFIG["simulation_overrides"]),
        "sahbo": dict(DEFAULT_G4_RUN_CONFIG["sahbo"]),
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
            sahbo["initial_eta"] = _as_float_tuple(sahbo["initial_eta"], field_name="sahbo.initial_eta", expected_len=3)
        config["sahbo"] = sahbo

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run G4 SA-HBO optimization and grid-search comparison.")
    parser.add_argument("--config", help="Path to a G4 TOML config file")
    parser.add_argument("--mode", choices=("sahbo", "grid", "both"), default=None)
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

    parser.add_argument("--grid-eta-values", default=None)
    parser.add_argument("--grid-max-evals", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    args = parser.parse_args()

    run_config = _load_g4_config(Path(args.config) if args.config else None)
    _override_if_present(run_config, "mode", args.mode)
    _override_if_present(run_config, "baseline_config", args.baseline_config)
    _override_if_present(run_config, "output_root", args.output_root)
    _override_if_present(run_config, "beta", args.beta)

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
        sahbo_table["initial_eta"] = _parse_three_float_tuple(args.initial_eta)
    _override_if_present(sahbo_table, "eta_lower_bound", args.eta_lower)
    _override_if_present(sahbo_table, "eta_upper_bound", args.eta_upper)
    _override_if_present(sahbo_table, "eta_step_size", args.eta_step_size)
    _override_if_present(sahbo_table, "eta_perturbation", args.eta_perturbation)
    _override_if_present(sahbo_table, "max_evaluations", args.sahbo_max_evals)
    _override_if_present(sahbo_table, "random_seed", args.seed)

    grid_table = dict(run_config["grid"])
    if args.grid_eta_values is not None:
        grid_table["eta_values"] = _parse_float_tuple(args.grid_eta_values)
    _override_if_present(grid_table, "max_evaluations", args.grid_max_evals)

    mode = str(run_config["mode"])
    if mode not in {"sahbo", "grid", "both"}:
        raise ValueError("g4.mode must be one of: sahbo, grid, both")
    beta = float(run_config["beta"])
    output_root = Path(str(run_config["output_root"]))
    evaluator = G4EvaluationCache(
        baseline_config=Path(str(run_config["baseline_config"])),
        output_root=output_root,
        simulation_overrides=simulation_overrides or None,
        beta=beta,
    )

    sahbo_result = None
    grid_result = None
    if mode in {"sahbo", "both"}:
        sahbo_result = run_sahbo(
            evaluator=evaluator,
            config=SAHBOConfig(
                iterations=int(sahbo_table["iterations"]),
                proxy_top_k=int(sahbo_table["proxy_top_k"]),
                neighborhood_radius=int(sahbo_table["neighborhood_radius"]),
                initial_eta=_as_float_tuple(sahbo_table["initial_eta"], field_name="sahbo.initial_eta", expected_len=3),  # type: ignore[arg-type]
                initial_directions=_as_state_tuple(sahbo_table["initial_directions"], field_name="sahbo.initial_directions"),
                eta_lower_bound=float(sahbo_table["eta_lower_bound"]),
                eta_upper_bound=float(sahbo_table["eta_upper_bound"]),
                eta_step_size=float(sahbo_table["eta_step_size"]),
                eta_perturbation=float(sahbo_table["eta_perturbation"]),
                max_evaluations=None if sahbo_table.get("max_evaluations") is None else int(sahbo_table["max_evaluations"]),
                random_seed=int(sahbo_table["random_seed"]),
                beta=beta,
            ),
        )

    if mode in {"grid", "both"}:
        direction_sets = _as_direction_sets(grid_table.get("direction_sets"))
        grid_kwargs: dict[str, object] = {
            "eta_values": _as_float_tuple(grid_table["eta_values"], field_name="grid.eta_values"),
            "max_evaluations": None if grid_table.get("max_evaluations") is None else int(grid_table["max_evaluations"]),
            "beta": beta,
        }
        if direction_sets is not None:
            grid_kwargs["direction_sets"] = direction_sets
        grid_result = run_grid_search(
            evaluator=evaluator,
            config=GridSearchConfig(**grid_kwargs),
        )

    save_g4_outputs(
        output_root=output_root,
        evaluator=evaluator,
        sahbo_result=sahbo_result,
        grid_result=grid_result,
    )


if __name__ == "__main__":
    main()
