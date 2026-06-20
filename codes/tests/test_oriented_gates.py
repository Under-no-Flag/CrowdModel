from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_bellman.compilers.config_compiler import compile_case, compile_scene
from crowd_bellman.loaders.config_loader import load_scene_spec
from crowd_bellman.runner import _apply_internal_gate_limits
from crowd_bellman.scenes import ChannelGateModel, GateCapacitySchedule, SimulationConfig
from crowd_bellman.spec.population_spec import PopulationSpec
from crowd_bellman.spec.route_spec import CapacityControlSpec, CaseRouteSpec, StageSpec


class OrientedGateTests(unittest.TestCase):
    def test_diagonal_channel_capacity_gate_compiles_oriented_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "diagonal_channel"
                    shape = "polygon"
                    points = [[2, 1], [7, 6], [6, 7], [1, 2]]
                    axis = [1, 1]

                    [[regions]]
                    name = "goal"
                    x0 = 8
                    x1 = 10
                    y0 = 8
                    y1 = 10

                    [[channels]]
                    name = "diagonal_channel"
                    region = "diagonal_channel"
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=12, ny=12))
            _scene, case = compile_case(
                bundle,
                PopulationSpec(),
                CaseRouteSpec(
                    case_id="diagonal_gate",
                    title="diagonal gate",
                    stages=(
                        StageSpec(
                            stage_id="main",
                            group_key=(1, 1),
                            goal_regions=("goal",),
                        ),
                    ),
                    capacity_controls=(
                        CapacityControlSpec(
                            channel="diagonal_channel",
                            side="plus",
                            rate=1.0,
                        ),
                    ),
                ),
            )

        gate = case.gates["diagonal_channel:plus"]
        self.assertEqual(gate.face_axis, "oriented")
        self.assertAlmostEqual(gate.normal[0], 2.0**-0.5)
        self.assertAlmostEqual(gate.normal[1], 2.0**-0.5)
        self.assertGreater(len(gate.face_x_rows), 0)
        self.assertGreater(len(gate.face_y_rows), 0)

    def test_oriented_gate_limits_x_and_y_normal_fluxes_together(self) -> None:
        gate = ChannelGateModel(
            gate_id="diagonal:plus",
            channel="diagonal",
            side="plus",
            face_axis="oriented",
            face_index=0,
            face_rows=np.array([], dtype=int),
            waiting_mask=np.zeros((4, 4), dtype=bool),
            face_x_rows=np.array([1], dtype=int),
            face_x_cols=np.array([1], dtype=int),
            face_x_signs=np.array([1.0]),
            face_y_rows=np.array([1], dtype=int),
            face_y_cols=np.array([2], dtype=int),
            face_y_signs=np.array([1.0]),
            normal=(2.0**-0.5, 2.0**-0.5),
        )
        fx = np.zeros((4, 3), dtype=float)
        fy = np.zeros((3, 4), dtype=float)
        fx[1, 1] = 3.0
        fy[1, 2] = 1.0

        limited_fx, limited_fy, diagnostics = _apply_internal_gate_limits(
            fx_by_group={(1, 1): fx},
            fy_by_group={(1, 1): fy},
            gates={"diagonal:plus": gate},
            schedules=(
                GateCapacitySchedule(
                    gate_id="diagonal:plus",
                    channel="diagonal",
                    side="plus",
                    rate=2.0,
                ),
            ),
            time_value=0.0,
            dx=1.0,
        )

        self.assertAlmostEqual(diagnostics["diagonal:plus"]["attempted_rate"], 4.0)
        self.assertAlmostEqual(diagnostics["diagonal:plus"]["actual_rate"], 2.0)
        self.assertAlmostEqual(diagnostics["diagonal:plus"]["lambda"], 0.5)
        self.assertAlmostEqual(limited_fx[(1, 1)][1, 1], 1.5)
        self.assertAlmostEqual(limited_fy[(1, 1)][1, 2], 0.5)


if __name__ == "__main__":
    unittest.main()
