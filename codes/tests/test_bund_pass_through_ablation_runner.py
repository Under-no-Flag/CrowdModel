from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_pass_through_ablation_runner import (
    CENTER_PREFIX,
    VARIANT_SPECS,
    add_center_goal_regions,
    build_region_edge_metrics,
    make_route_variant,
    shrink_polygon,
)


class BundPassThroughAblationRunnerTests(unittest.TestCase):
    def test_shrink_polygon_keeps_centroid_and_moves_vertices_inward(self) -> None:
        points = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]

        shrunk = shrink_polygon(points, factor=0.25)

        self.assertEqual(len(shrunk), 4)
        self.assertAlmostEqual(sum(x for x, _ in shrunk) / 4.0, 2.0)
        self.assertAlmostEqual(sum(y for _, y in shrunk) / 4.0, 1.0)
        self.assertGreater(shrunk[0][0], points[0][0])
        self.assertGreater(shrunk[0][1], points[0][1])

    def test_add_center_goal_regions_appends_center_polygons_once(self) -> None:
        scene = {
            "regions": [
                {
                    "name": "goal_region2",
                    "shape": "polygon",
                    "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]],
                    "axis": [1.0, 0.0],
                }
            ]
        }

        updated = add_center_goal_regions(scene, ["goal_region2"])
        updated_again = add_center_goal_regions(updated, ["goal_region2"])

        names = [region["name"] for region in updated_again["regions"]]
        self.assertEqual(names.count(f"{CENTER_PREFIX}goal_region2"), 1)
        center_region = next(region for region in updated_again["regions"] if region["name"] == f"{CENTER_PREFIX}goal_region2")
        self.assertEqual(center_region["shape"], "polygon")
        self.assertEqual(center_region["axis"], [1.0, 0.0])

    def test_route_variants_change_goal_regions_but_keep_decision_regions(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "base"},
            "stages": [
                {
                    "stage_id": "to_pre_channel_3",
                    "group_key": [1, 4],
                    "goal_region": "channel_3",
                    "decision_regions": ["goal_region2", "channel_3"],
                    "next_stage": "to_channel_3",
                    "transition_direction": "inherit_target",
                    "kappa": 2.0,
                },
                {
                    "stage_id": "to_merge_after_channels",
                    "group_key": [1, 10],
                    "goal_region": "exits",
                    "decision_regions": ["merge_after_channels", "exits"],
                    "next_stage": "to_exits",
                    "transition_direction": "inherit_target",
                    "kappa": 2.0,
                },
            ],
        }

        full_goal = make_route_variant(base_routes, VARIANT_SPECS["a_full_goal"], case_id="a", kappa=8.0)
        center_goal = make_route_variant(base_routes, VARIANT_SPECS["b_center_goal"], case_id="b", kappa=8.0)
        pass_through = make_route_variant(base_routes, VARIANT_SPECS["c_pass_through"], case_id="c", kappa=8.0)

        self.assertEqual(full_goal["stages"][0]["goal_region"], "goal_region2")
        self.assertEqual(center_goal["stages"][0]["goal_region"], f"{CENTER_PREFIX}goal_region2")
        self.assertEqual(pass_through["stages"][0]["goal_region"], "channel_3")
        self.assertEqual(pass_through["stages"][0]["decision_regions"], ["goal_region2", "channel_3"])
        self.assertEqual(pass_through["stages"][0]["kappa"], 8.0)
        self.assertEqual(full_goal["stages"][1]["goal_region"], "merge_after_channels")
        self.assertEqual(pass_through["stages"][1]["goal_region"], "exits")

    def test_build_region_edge_metrics_reports_boundary_and_interior_density(self) -> None:
        density = np.zeros((5, 5), dtype=float)
        density[1:4, 1:4] = 1.0
        density[1, 1:4] = 3.0
        region_masks = {"box": np.zeros((5, 5), dtype=bool)}
        region_masks["box"][1:4, 1:4] = True
        walkable = np.ones((5, 5), dtype=bool)

        rows = build_region_edge_metrics(
            density,
            region_masks=region_masks,
            walkable=walkable,
            regions=["box"],
            case_id="case_a",
            step=10,
            time_value=1.0,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["region"], "box")
        self.assertGreater(row["boundary_mean"], row["interior_mean"])
        self.assertGreater(row["boundary_to_interior"], 1.0)


if __name__ == "__main__":
    unittest.main()
