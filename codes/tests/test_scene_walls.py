from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_bellman.compilers.config_compiler import compile_case, compile_scene
from crowd_bellman.core import DIRECTIONS
from crowd_bellman.loaders.config_loader import load_scene_spec
from crowd_bellman.scenes import SimulationConfig
from crowd_bellman.spec.population_spec import PopulationSpec
from crowd_bellman.spec.route_spec import CaseRouteSpec, ControlSpec, StageSpec


class SceneWallTests(unittest.TestCase):
    def test_legacy_rect_regions_default_to_rect_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "legacy_rect"
                    x0 = 1
                    x1 = 4
                    y0 = 2
                    y1 = 6
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))

        self.assertEqual(scene_spec.regions[0].shape, "rect")
        self.assertEqual((scene_spec.regions[0].x0, scene_spec.regions[0].x1), (1, 4))
        self.assertTrue(bundle.region_masks["legacy_rect"][2, 1])
        self.assertTrue(bundle.region_masks["legacy_rect"][5, 3])
        self.assertFalse(bundle.region_masks["legacy_rect"][2, 4])

    def test_loads_polygon_regions_from_scene_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "rotated_channel"
                    shape = "polygon"
                    points = [[2, 1], [6, 3], [5, 5], [1, 3]]
                    axis = [4, 2]
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)

        self.assertEqual(len(scene_spec.regions), 1)
        self.assertEqual(scene_spec.regions[0].name, "rotated_channel")
        self.assertEqual(scene_spec.regions[0].shape, "polygon")
        self.assertEqual(scene_spec.regions[0].points, ((2.0, 1.0), (6.0, 3.0), (5.0, 5.0), (1.0, 3.0)))
        self.assertEqual(scene_spec.regions[0].axis, (4.0, 2.0))

    def test_polygon_region_uses_cell_centers_and_matches_rect_edges(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "rect_region"
                    x0 = 2
                    x1 = 5
                    y0 = 2
                    y1 = 5

                    [[regions]]
                    name = "polygon_region"
                    shape = "polygon"
                    points = [[2, 2], [5, 2], [5, 5], [2, 5]]
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))

        self.assertTrue((bundle.region_masks["rect_region"] == bundle.region_masks["polygon_region"]).all())
        self.assertTrue(bundle.region_masks["polygon_region"][2, 2])
        self.assertTrue(bundle.region_masks["polygon_region"][4, 4])
        self.assertFalse(bundle.region_masks["polygon_region"][2, 1])
        self.assertFalse(bundle.region_masks["polygon_region"][4, 5])

    def test_polygon_obstacles_block_compiled_walkable_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false
                    obstacles = ["solid_triangle"]

                    [[regions]]
                    name = "solid_triangle"
                    shape = "polygon"
                    points = [[2, 2], [6, 2], [4, 6]]
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))

        self.assertFalse(bundle.scene.walkable[2, 3])
        self.assertFalse(bundle.scene.walkable[4, 4])
        self.assertTrue(bundle.scene.walkable[1, 4])
        self.assertTrue(bundle.scene.walkable[6, 4])

    def test_polygon_region_axis_defaults_to_longest_edge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "diagonal_channel"
                    shape = "polygon"
                    points = [[1, 1], [5, 4], [4, 5], [0, 2]]
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))

        axis_x, axis_y = bundle.region_axes["diagonal_channel"]
        self.assertAlmostEqual(axis_x, 0.8)
        self.assertAlmostEqual(axis_y, 0.6)

    def test_region_axis_control_applies_tensor_and_nearest_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "axis_channel"
                    x0 = 1
                    x1 = 4
                    y0 = 1
                    y1 = 4
                    axis = [1, 0]

                    [[regions]]
                    name = "goal"
                    x0 = 5
                    x1 = 7
                    y0 = 1
                    y1 = 4
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))
            _scene, case = compile_case(
                bundle,
                PopulationSpec(),
                CaseRouteSpec(
                    case_id="axis_case",
                    title="axis case",
                    stages=(
                        StageSpec(
                            stage_id="main",
                            group_key=(1, 1),
                            goal_regions=("goal",),
                            controls=(
                                ControlSpec(
                                    mode="region_axis",
                                    region="axis_channel",
                                    alpha=3.0,
                                    beta=1.0,
                                    direction="minus",
                                ),
                            ),
                        ),
                    ),
                ),
            )

        group = case.groups[(1, 1)]
        self.assertEqual(group.allowed_mask[2, 2], DIRECTIONS.bits[DIRECTIONS.names.index("W")])
        self.assertEqual(group.allowed_mask[2, 5], DIRECTIONS.all_mask)
        self.assertAlmostEqual(group.m11[2, 2], 3.0)
        self.assertAlmostEqual(group.m12[2, 2], 0.0)
        self.assertAlmostEqual(group.m22[2, 2], 1.0)

    def test_loads_polyline_walls_from_scene_toml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[walls]]
                    name = "diagonal_wall"
                    points = [[2, 2], [5, 5]]
                    width = 1.0
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)

        self.assertEqual(len(scene_spec.walls), 1)
        self.assertEqual(scene_spec.walls[0].name, "diagonal_wall")
        self.assertEqual(scene_spec.walls[0].points, ((2.0, 2.0), (5.0, 5.0)))
        self.assertEqual(scene_spec.walls[0].width, 1.0)

    def test_polyline_walls_block_compiled_walkable_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_path = Path(temp_dir) / "scene.toml"
            scene_path.write_text(
                textwrap.dedent(
                    """
                    block_boundaries = false

                    [[regions]]
                    name = "whole_area"
                    x0 = 0
                    x1 = 8
                    y0 = 0
                    y1 = 8

                    [[walls]]
                    name = "vertical_wall"
                    points = [[3, 1], [3, 6]]
                    width = 1.0
                    """
                ),
                encoding="utf-8",
            )

            scene_spec = load_scene_spec(scene_path)
            bundle = compile_scene(scene_spec, SimulationConfig(nx=8, ny=8))

        self.assertFalse(bundle.scene.walkable[1, 3])
        self.assertFalse(bundle.scene.walkable[3, 3])
        self.assertFalse(bundle.scene.walkable[6, 3])
        self.assertTrue(bundle.scene.walkable[3, 2])
        self.assertTrue(bundle.scene.walkable[3, 4])
        self.assertIn("vertical_wall", bundle.region_masks)


if __name__ == "__main__":
    unittest.main()
