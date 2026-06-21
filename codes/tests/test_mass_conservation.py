from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from crowd_bellman.core import (
    compute_face_fluxes,
    limit_fluxes_to_density_capacity,
    update_density,
    update_density_from_fluxes,
)
from crowd_bellman.metrics import init_case_stats, record_step, save_case_timeseries


class MassConservationTests(unittest.TestCase):
    def test_update_density_preserves_mass_at_internal_wall(self) -> None:
        walkable = np.ones((3, 4), dtype=bool)
        walkable[1, 2] = False
        rho = np.zeros((3, 4), dtype=float)
        rho[1, 1] = 1.0
        vx = np.ones((3, 4), dtype=float)
        vy = np.zeros((3, 4), dtype=float)
        exit_mask = np.zeros_like(walkable, dtype=bool)

        updated, fx, _fy, sink_mass = update_density(
            rho=rho,
            walkable=walkable,
            exit_mask=exit_mask,
            vx=vx,
            vy=vy,
            dx=1.0,
            dt=0.1,
        )

        self.assertAlmostEqual(float(np.sum(rho[walkable])), 1.0)
        self.assertAlmostEqual(float(np.sum(updated[walkable])), 1.0)
        self.assertAlmostEqual(sink_mass, 0.0)
        self.assertAlmostEqual(fx[1, 1], 0.0)

    def test_limited_flux_path_preserves_mass_at_internal_wall(self) -> None:
        walkable = np.array([[True, True, False]], dtype=bool)
        rho = np.array([[0.0, 1.0, 0.0]], dtype=float)
        vx = np.ones_like(rho)
        vy = np.zeros_like(rho)
        exit_mask = np.zeros_like(walkable, dtype=bool)

        fx, fy = compute_face_fluxes(rho, vx, vy, walkable=walkable)
        updated, sink_mass = update_density_from_fluxes(
            rho=rho,
            walkable=walkable,
            exit_mask=exit_mask,
            fx=fx,
            fy=fy,
            dx=1.0,
            dt=0.1,
        )

        self.assertAlmostEqual(float(np.sum(updated[walkable])), 1.0)
        self.assertAlmostEqual(sink_mass, 0.0)
        self.assertAlmostEqual(fx[0, 1], 0.0)

    def test_closed_domain_mass_balance_residual_stays_near_zero(self) -> None:
        walkable = np.ones((5, 5), dtype=bool)
        walkable[[0, -1], :] = False
        walkable[:, [0, -1]] = False
        walkable[2, 3] = False
        rho = np.zeros((5, 5), dtype=float)
        rho[2, 2] = 1.0
        vx = np.ones((5, 5), dtype=float)
        vy = np.zeros((5, 5), dtype=float)
        exit_mask = np.zeros_like(walkable, dtype=bool)

        for _step in range(5):
            rho, _fx, _fy, sink_mass = update_density(
                rho=rho,
                walkable=walkable,
                exit_mask=exit_mask,
                vx=vx,
                vy=vy,
                dx=1.0,
                dt=0.1,
            )
            self.assertAlmostEqual(sink_mass, 0.0)

        self.assertAlmostEqual(float(np.sum(rho[walkable])), 1.0)

    def test_record_step_writes_mass_balance_diagnostics(self) -> None:
        walkable = np.ones((2, 2), dtype=bool)
        rho = np.full((2, 2), 0.25, dtype=float)
        stats = init_case_stats([], initial_total_mass=1.0, walkable_area=4.0)

        record_step(
            stats=stats,
            time_value=0.1,
            rho=rho,
            walkable=walkable,
            vx=np.zeros_like(rho),
            vy=np.zeros_like(rho),
            fx=np.zeros((2, 1), dtype=float),
            fy=np.zeros((1, 2), dtype=float),
            sink_total=0.0,
            dt=0.1,
            dx=1.0,
            rho_safe=3.5,
            channel_masks={},
            probe_x={},
            inflow_total=0.0,
            cap_removed_total=0.0,
        )

        self.assertEqual(stats.current_mass, [1.0])
        self.assertEqual(stats.mass_balance_residual, [0.0])

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "timeseries.csv"
            save_case_timeseries(csv_path, stats)
            text = csv_path.read_text(encoding="utf-8")

        self.assertIn("current_mass", text.splitlines()[0])
        self.assertIn("mass_balance_residual", text.splitlines()[0])

    def test_record_step_counts_axis_aligned_flux_for_tilted_channel(self) -> None:
        walkable = np.ones((3, 3), dtype=bool)
        rho = np.zeros((3, 3), dtype=float)
        stats = init_case_stats(["diagonal"], initial_total_mass=0.0, walkable_area=9.0)
        channel_mask = np.zeros((3, 3), dtype=bool)
        channel_mask[1, 1] = True
        fx = np.zeros((3, 2), dtype=float)
        fy = np.zeros((2, 3), dtype=float)
        fy[1, 1] = 2.0

        record_step(
            stats=stats,
            time_value=0.5,
            rho=rho,
            walkable=walkable,
            vx=np.zeros_like(rho),
            vy=np.zeros_like(rho),
            fx=fx,
            fy=fy,
            sink_total=0.0,
            dt=0.5,
            dx=1.0,
            rho_safe=3.5,
            channel_masks={"diagonal": channel_mask},
            probe_x={"diagonal": 1},
            channel_axes={"diagonal": (0.0, 1.0)},
            channel_flux_directions={"diagonal": "E"},
            inflow_total=0.0,
            cap_removed_total=0.0,
        )

        self.assertAlmostEqual(stats.channel_flux_cumulative["diagonal"], 1.0)

    def test_capacity_limiter_keeps_overflowing_inflow_upstream(self) -> None:
        walkable = np.ones((3, 4), dtype=bool)
        rho = np.zeros((3, 4), dtype=float)
        rho[1, 1] = 5.0
        rho[1, 2] = 4.9
        fx = np.zeros((3, 3), dtype=float)
        fy = np.zeros((2, 4), dtype=float)
        fx[1, 1] = 1.0

        limited_fx_by_group, limited_fy_by_group, diagnostics = limit_fluxes_to_density_capacity(
            rho_by_group={(1, 1): rho},
            fx_by_group={(1, 1): fx},
            fy_by_group={(1, 1): fy},
            walkable=walkable,
            rho_max=5.0,
            dx=1.0,
            dt=0.2,
        )
        updated, sink_mass = update_density_from_fluxes(
            rho=rho,
            walkable=walkable,
            exit_mask=np.zeros_like(walkable, dtype=bool),
            fx=limited_fx_by_group[(1, 1)],
            fy=limited_fy_by_group[(1, 1)],
            dx=1.0,
            dt=0.2,
        )

        self.assertAlmostEqual(limited_fx_by_group[(1, 1)][1, 1], 0.5)
        self.assertAlmostEqual(float(np.max(updated[walkable])), 5.0)
        self.assertAlmostEqual(float(np.sum(updated[walkable])), float(np.sum(rho[walkable])))
        self.assertAlmostEqual(sink_mass, 0.0)
        self.assertAlmostEqual(float(diagnostics["limited_mass"]), 0.1)
        self.assertTrue(bool(diagnostics["binding"]))

    def test_capacity_limiter_scales_multiple_groups_conservatively(self) -> None:
        walkable = np.ones((3, 4), dtype=bool)
        rho_a = np.zeros((3, 4), dtype=float)
        rho_b = np.zeros((3, 4), dtype=float)
        rho_a[1, 1] = 2.0
        rho_b[1, 1] = 3.0
        rho_a[1, 2] = 2.0
        rho_b[1, 2] = 2.9
        fx_a = np.zeros((3, 3), dtype=float)
        fx_b = np.zeros((3, 3), dtype=float)
        fy = np.zeros((2, 4), dtype=float)
        fx_a[1, 1] = 0.4
        fx_b[1, 1] = 0.6

        limited_fx_by_group, _limited_fy_by_group, diagnostics = limit_fluxes_to_density_capacity(
            rho_by_group={(1, 1): rho_a, (1, 2): rho_b},
            fx_by_group={(1, 1): fx_a, (1, 2): fx_b},
            fy_by_group={(1, 1): fy, (1, 2): fy},
            walkable=walkable,
            rho_max=5.0,
            dx=1.0,
            dt=0.2,
        )

        self.assertAlmostEqual(limited_fx_by_group[(1, 1)][1, 1], 0.2)
        self.assertAlmostEqual(limited_fx_by_group[(1, 2)][1, 1], 0.3)
        self.assertAlmostEqual(float(diagnostics["limited_mass"]), 0.1)


if __name__ == "__main__":
    unittest.main()
