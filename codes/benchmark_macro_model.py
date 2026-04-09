from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean

from crowd_bellman.compilers.config_compiler import compile_case, compile_scene
from crowd_bellman.core import (
    compute_total_density,
    greenshields_speed,
    precompute_step_factors,
    recover_optimal_direction,
    solve_bellman,
)
from crowd_bellman.loaders.config_loader import (
    load_population_spec,
    load_route_spec,
    load_run_config,
    load_scene_spec,
)
from crowd_bellman.runner import _build_group_models, _build_initial_group_density, simulate_case


def _load_scene_case(config_path: Path):
    run_spec = load_run_config(config_path)
    scene_spec = load_scene_spec(run_spec.scene_path)
    population_spec = load_population_spec(run_spec.population_path)
    route_spec = load_route_spec(run_spec.routes_path)
    bundle = compile_scene(scene_spec=scene_spec, cfg=run_spec.simulation)
    scene, case = compile_case(bundle=bundle, population_spec=population_spec, route_spec=route_spec)
    return run_spec, scene, case


def _prepare_kernel_state(config_path: Path):
    run_spec, scene, case = _load_scene_case(config_path)
    groups = _build_group_models(case)
    rho_by_group = _build_initial_group_density(scene, case, groups)
    step_factors = {
        key: precompute_step_factors(
            walkable=case.walkable,
            dx=run_spec.simulation.dx,
            m11=group.m11,
            m12=group.m12,
            m22=group.m22,
        )
        for key, group in groups.items()
    }
    rho_tot = compute_total_density(rho_by_group)
    speed = greenshields_speed(rho_tot, run_spec.simulation.vmax, run_spec.simulation.rho_max)
    return run_spec, case, groups, speed, step_factors


def _solve_one_group(case, group, speed, step_factor, f_eps: float, *, bellman_backend: str, direction_backend: str):
    phi = solve_bellman(
        walkable=case.walkable,
        exit_mask=group.goal_mask,
        allowed_mask=group.allowed_mask,
        speed=speed,
        step_factor=step_factor,
        f_eps=f_eps,
        backend=bellman_backend,
    )
    recover_optimal_direction(
        walkable=case.walkable,
        exit_mask=group.goal_mask,
        allowed_mask=group.allowed_mask,
        speed=speed,
        step_factor=step_factor,
        phi=phi,
        f_eps=f_eps,
        backend=direction_backend,
    )


def _benchmark_serial_kernel(case, groups, speed, step_factors, f_eps: float, *, bellman_backend: str, direction_backend: str) -> float:
    start = time.perf_counter()
    for key, group in groups.items():
        _solve_one_group(
            case,
            group,
            speed,
            step_factors[key],
            f_eps,
            bellman_backend=bellman_backend,
            direction_backend=direction_backend,
        )
    return time.perf_counter() - start


def _benchmark_threaded_kernel(case, groups, speed, step_factors, f_eps: float, *, bellman_backend: str, direction_backend: str) -> float:
    workers = min(len(groups), os.cpu_count() or 1)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _solve_one_group,
                case,
                group,
                speed,
                step_factors[key],
                f_eps,
                bellman_backend=bellman_backend,
                direction_backend=direction_backend,
            )
            for key, group in groups.items()
        ]
        for future in futures:
            future.result()
    return time.perf_counter() - start


def _time_repeated(fn, repeats: int) -> dict[str, float]:
    samples = []
    fn()
    for _ in range(repeats):
        samples.append(fn())
    return {
        "mean_seconds": mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def _benchmark_full_simulation(config_path: Path, *, steps: int | None, time_horizon: float | None, save_every: int, bellman_backend: str, direction_backend: str, repeats: int):
    run_spec, scene, case = _load_scene_case(config_path)
    cfg = run_spec.simulation
    if steps is not None:
        cfg = replace(cfg, steps=steps)
    if time_horizon is not None:
        cfg = replace(cfg, time_horizon=time_horizon)
    cfg = replace(
        cfg,
        save_every=save_every,
        bellman_backend=bellman_backend,
        direction_recovery_backend=direction_backend,
    )

    def timed_once() -> float:
        with tempfile.TemporaryDirectory() as temp_dir:
            start = time.perf_counter()
            simulate_case(
                cfg=cfg,
                scene=scene,
                case=case,
                output_dir=Path(temp_dir),
                objective_cfg=run_spec.objective,
            )
            return time.perf_counter() - start

    return _time_repeated(timed_once, repeats=repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Bellman/recovery backends for the macroscopic crowd model.")
    parser.add_argument("--config", required=True, help="Path to the run TOML file")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark repetitions for each method")
    parser.add_argument("--steps", type=int, default=None, help="Optional step override for end-to-end simulation")
    parser.add_argument("--time-horizon", type=float, default=None, help="Optional time horizon override for end-to-end simulation")
    parser.add_argument("--save-every", type=int, default=9999, help="Snapshot interval for end-to-end benchmark")
    parser.add_argument("--output", default=None, help="Optional JSON path for benchmark results")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    run_spec, case, groups, speed, step_factors = _prepare_kernel_state(config_path)
    f_eps = run_spec.simulation.bellman_f_eps

    kernel_results = {
        "serial_python": _time_repeated(
            lambda: _benchmark_serial_kernel(
                case,
                groups,
                speed,
                step_factors,
                f_eps,
                bellman_backend="python",
                direction_backend="python",
            ),
            repeats=args.repeats,
        ),
        "serial_optimized": _time_repeated(
            lambda: _benchmark_serial_kernel(
                case,
                groups,
                speed,
                step_factors,
                f_eps,
                bellman_backend="optimized",
                direction_backend="vectorized",
            ),
            repeats=args.repeats,
        ),
    }

    if len(groups) > 1:
        kernel_results["threaded_optimized"] = _time_repeated(
            lambda: _benchmark_threaded_kernel(
                case,
                groups,
                speed,
                step_factors,
                f_eps,
                bellman_backend="optimized",
                direction_backend="vectorized",
            ),
            repeats=args.repeats,
        )

    full_results = {
        "python": _benchmark_full_simulation(
            config_path,
            steps=args.steps,
            time_horizon=args.time_horizon,
            save_every=args.save_every,
            bellman_backend="python",
            direction_backend="python",
            repeats=args.repeats,
        ),
        "optimized": _benchmark_full_simulation(
            config_path,
            steps=args.steps,
            time_horizon=args.time_horizon,
            save_every=args.save_every,
            bellman_backend="optimized",
            direction_backend="vectorized",
            repeats=args.repeats,
        ),
    }

    payload = {
        "config_path": str(config_path),
        "simulation": asdict(run_spec.simulation),
        "group_count": len(groups),
        "kernel_benchmark": kernel_results,
        "full_simulation_benchmark": full_results,
    }

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
