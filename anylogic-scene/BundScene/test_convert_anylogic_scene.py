from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from convert_anylogic_scene import (
    DEFAULT_DX,
    DEFAULT_INFLOW_RATE,
    DEFAULT_INFLOW_RHO_CAP,
    DEFAULT_NX,
    DEFAULT_NY,
    extract_regions,
    extract_wall_paths,
    write_draft_toml,
)


ALP_PATH = ROOT / "BundScene.alp"


class AnyLogicConversionTests(unittest.TestCase):
    def test_default_grid_resolution_uses_higher_density(self) -> None:
        self.assertEqual(DEFAULT_NX, 512)
        self.assertEqual(DEFAULT_NY, 224)
        self.assertEqual(DEFAULT_DX, 0.125)

    def test_extracts_anylogic_paths_as_wall_records(self) -> None:
        frame, walls = extract_wall_paths(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        self.assertEqual(frame.width, 2053)
        self.assertEqual({wall.name for wall in walls}, {"path2", "path3", "path5", "path6", "path7"})
        self.assertTrue(all(wall.kind == "wall" for wall in walls))
        self.assertTrue(all(len(wall.grid_points) >= 2 for wall in walls))

        path6 = next(wall for wall in walls if wall.name == "path6")
        self.assertEqual(path6.raw_point_count, 93)
        self.assertLess(len(path6.grid_points), path6.raw_point_count)
        self.assertEqual(path6.point_mode, "anchors")

    def test_rotated_rectangle_uses_anylogic_xy_as_rotation_anchor(self) -> None:
        frame, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)
        channel_2 = next(region for region in regions if region.name == "channel_2")

        expected_x = (channel_2.x - frame.x) / frame.width * DEFAULT_NX
        expected_y = (channel_2.y - frame.y) / frame.height * DEFAULT_NY

        self.assertAlmostEqual(channel_2.grid_corners[0][0], expected_x)
        self.assertAlmostEqual(channel_2.grid_corners[0][1], expected_y)

    def test_wall_paths_are_written_as_polyline_walls(self) -> None:
        _, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)
        _, walls = extract_wall_paths(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_draft_toml(output_dir, regions, wall_paths=walls, nx=DEFAULT_NX, ny=DEFAULT_NY, dx=0.5)
            scene_text = (output_dir / "scene.toml").read_text(encoding="utf-8")

        self.assertIn("[[walls]]", scene_text)
        self.assertIn('name = "path6"', scene_text)
        self.assertIn('shape = "polyline"', scene_text)
        self.assertIn("points = [[", scene_text)
        self.assertIn("width = ", scene_text)
        self.assertNotIn('name = "wall_path', scene_text)

    def test_rectangle_regions_are_written_as_polygon_regions(self) -> None:
        _, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_draft_toml(output_dir, regions, nx=DEFAULT_NX, ny=DEFAULT_NY, dx=0.5)
            scene_text = (output_dir / "scene.toml").read_text(encoding="utf-8")

        self.assertIn("[[regions]]", scene_text)
        self.assertIn('name = "channel_1"', scene_text)
        self.assertIn('shape = "polygon"', scene_text)
        self.assertIn("points = [[", scene_text)
        self.assertIn("axis = [", scene_text)
        self.assertNotIn("x0 = ", scene_text)

    def test_auxiliary_route_regions_are_preserved(self) -> None:
        _, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        route_regions = {region.name: region.kind for region in regions if region.kind == "route"}

        self.assertEqual(route_regions["decision_channel_split"], "route")
        self.assertEqual(route_regions["merge_after_channels"], "route")
        for index in range(1, 5):
            self.assertEqual(route_regions[f"post_channel_{index}"], "route")

    def test_default_routes_use_channel_split_when_auxiliary_regions_exist(self) -> None:
        _, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_draft_toml(output_dir, regions, nx=DEFAULT_NX, ny=DEFAULT_NY, dx=0.5)
            population_text = (output_dir / "population.toml").read_text(encoding="utf-8")
            routes_text = (output_dir / "routes.toml").read_text(encoding="utf-8")

        self.assertIn('case_id = "bund_anylogic_simple_route"', routes_text)
        self.assertIn('stage_id = "to_decision_channel_split"', population_text)
        self.assertIn('goal_region = "decision_channel_split"', routes_text)
        self.assertIn('decision_region = "decision_channel_split"', routes_text)
        self.assertIn('stage_id = "to_pre_channel_1"', routes_text)
        self.assertIn('goal_region = "goal_region11"', routes_text)
        self.assertIn('next_stage = "to_channel_1"', routes_text)
        self.assertIn('stage_id = "to_pre_channel_2"', routes_text)
        self.assertIn('goal_region = "goal_region"', routes_text)
        self.assertIn('stage_id = "to_pre_channel_3"', routes_text)
        self.assertIn('goal_region = "goal_region2"', routes_text)
        self.assertIn('stage_id = "to_pre_channel_4"', routes_text)
        self.assertIn('goal_region = "goal_region5"', routes_text)
        self.assertIn('stage_id = "to_channel_1"', routes_text)
        self.assertIn('goal_region = "channel_1"', routes_text)
        self.assertIn('next_stage = "post_channel_1"', routes_text)
        self.assertIn('stage_id = "post_channel_1"', routes_text)
        self.assertIn('goal_region = "post_channel_1"', routes_text)
        self.assertIn('next_stage = "to_merge_after_channels"', routes_text)
        self.assertIn('stage_id = "to_channel_4"', routes_text)
        self.assertIn('stage_id = "post_channel_4"', routes_text)
        self.assertIn('goal_region = "post_channel_4"', routes_text)
        self.assertIn('stage_id = "to_merge_after_channels"', routes_text)
        self.assertIn('goal_region = "merge_after_channels"', routes_text)
        self.assertIn('next_stage = "to_exits"', routes_text)
        self.assertIn('goal_region = "exits"', routes_text)
        self.assertIn('sink_region = "exits"', routes_text)

    def test_population_adds_continuous_inflow_at_initial_region(self) -> None:
        _, regions = extract_regions(ALP_PATH, nx=DEFAULT_NX, ny=DEFAULT_NY)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_draft_toml(output_dir, regions, nx=DEFAULT_NX, ny=DEFAULT_NY, dx=0.5)
            population_text = (output_dir / "population.toml").read_text(encoding="utf-8")

        self.assertIn("[[initial_groups]]", population_text)
        self.assertIn("[[inflow_groups]]", population_text)
        self.assertIn('group_id = "bund_anylogic_inflow"', population_text)
        self.assertIn('stage_id = "to_decision_channel_split"', population_text)
        self.assertIn('region = "initial_groups"', population_text)
        self.assertIn(f"rate = {DEFAULT_INFLOW_RATE}", population_text)
        self.assertIn(f"rho_cap = {DEFAULT_INFLOW_RHO_CAP}", population_text)
        self.assertNotIn("time_end", population_text)


if __name__ == "__main__":
    unittest.main()
