from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from crowd_bellman.compilers.config_compiler import compile_case, compile_scene
from crowd_bellman.core import greenshields_speed, precompute_step_factors, recover_optimal_direction, solve_bellman
from crowd_bellman.loaders.config_loader import (
    load_population_spec,
    load_route_spec,
    load_run_config,
    load_scene_spec,
)
from crowd_bellman.runner import simulate_case


ROOT = Path(__file__).resolve().parents[1]


def _load_scene_case(run_relative_path: str):
    run_path = ROOT / run_relative_path
    run_spec = load_run_config(run_path)
    scene_spec = load_scene_spec(run_spec.scene_path)
    population_spec = load_population_spec(run_spec.population_path)
    route_spec = load_route_spec(run_spec.routes_path)
    bundle = compile_scene(scene_spec=scene_spec, cfg=run_spec.simulation)
    scene, case = compile_case(bundle=bundle, population_spec=population_spec, route_spec=route_spec)
    return run_spec, scene, case


class CoreBackendTests(unittest.TestCase):
    def test_solve_bellman_backends_match(self) -> None:
        run_spec, scene, case = _load_scene_case("scenes/examples/multi_stage/run.toml")
        first_key = next(iter(case.groups))
        group = case.groups[first_key]
        step_factor = precompute_step_factors(
            walkable=case.walkable,
            dx=run_spec.simulation.dx,
            m11=group.m11,
            m12=group.m12,
            m22=group.m22,
        )
        speed = greenshields_speed(scene.initial_rho, run_spec.simulation.vmax, run_spec.simulation.rho_max)

        phi_python = solve_bellman(
            walkable=case.walkable,
            exit_mask=group.goal_mask,
            allowed_mask=group.allowed_mask,
            speed=speed,
            step_factor=step_factor,
            f_eps=run_spec.simulation.bellman_f_eps,
            backend="python",
        )
        phi_optimized = solve_bellman(
            walkable=case.walkable,
            exit_mask=group.goal_mask,
            allowed_mask=group.allowed_mask,
            speed=speed,
            step_factor=step_factor,
            f_eps=run_spec.simulation.bellman_f_eps,
            backend="optimized",
        )

        np.testing.assert_allclose(phi_python, phi_optimized, atol=1.0e-12, rtol=0.0)

    def test_direction_recovery_backends_match(self) -> None:
        run_spec, scene, case = _load_scene_case("scenes/examples/multi_stage/run.toml")
        first_key = next(iter(case.groups))
        group = case.groups[first_key]
        step_factor = precompute_step_factors(
            walkable=case.walkable,
            dx=run_spec.simulation.dx,
            m11=group.m11,
            m12=group.m12,
            m22=group.m22,
        )
        speed = greenshields_speed(scene.initial_rho, run_spec.simulation.vmax, run_spec.simulation.rho_max)
        phi = solve_bellman(
            walkable=case.walkable,
            exit_mask=group.goal_mask,
            allowed_mask=group.allowed_mask,
            speed=speed,
            step_factor=step_factor,
            f_eps=run_spec.simulation.bellman_f_eps,
            backend="optimized",
        )

        ux_python, uy_python = recover_optimal_direction(
            walkable=case.walkable,
            exit_mask=group.goal_mask,
            allowed_mask=group.allowed_mask,
            speed=speed,
            step_factor=step_factor,
            phi=phi,
            f_eps=run_spec.simulation.bellman_f_eps,
            backend="python",
        )
        ux_vectorized, uy_vectorized = recover_optimal_direction(
            walkable=case.walkable,
            exit_mask=group.goal_mask,
            allowed_mask=group.allowed_mask,
            speed=speed,
            step_factor=step_factor,
            phi=phi,
            f_eps=run_spec.simulation.bellman_f_eps,
            backend="vectorized",
        )

        np.testing.assert_allclose(ux_python, ux_vectorized, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(uy_python, uy_vectorized, atol=0.0, rtol=0.0)

    def test_simulate_case_summary_matches_between_backends(self) -> None:
        run_spec, scene, case = _load_scene_case("scenes/examples/multi_stage/run.toml")
        cfg_baseline = replace(
            run_spec.simulation,
            steps=12,
            time_horizon=3.0,
            save_every=9999,
            bellman_backend="python",
            direction_recovery_backend="python",
        )
        cfg_optimized = replace(
            cfg_baseline,
            bellman_backend="optimized",
            direction_recovery_backend="vectorized",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            summary_baseline = simulate_case(
                cfg=cfg_baseline,
                scene=scene,
                case=case,
                output_dir=temp_root / "baseline",
                objective_cfg=run_spec.objective,
            )
            summary_optimized = simulate_case(
                cfg=cfg_optimized,
                scene=scene,
                case=case,
                output_dir=temp_root / "optimized",
                objective_cfg=run_spec.objective,
            )

        scalar_keys = [
            "final_time",
            "final_sink_cumulative",
            "mean_density_avg",
            "peak_density_max",
            "velocity_discontinuity_avg",
            "density_gradient_avg",
            "j1_total_travel_time",
            "j2_high_density_exposure",
            "j5_channel_flux_variance",
            "objective_value",
        ]
        for key in scalar_keys:
            self.assertAlmostEqual(summary_baseline[key], summary_optimized[key], places=12)

        for key in ("group_count", "transition_count"):
            self.assertEqual(summary_baseline[key], summary_optimized[key])

        for mapping_key in ("channel_flux_cumulative", "channel_flux_share", "channel_time_mean_density"):
            for name, value in summary_baseline[mapping_key].items():
                self.assertAlmostEqual(value, summary_optimized[mapping_key][name], places=12)


if __name__ == "__main__":
    unittest.main()
