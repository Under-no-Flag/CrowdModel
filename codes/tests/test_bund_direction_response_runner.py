from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_direction_response_runner import (
    DirectionCase,
    DirectionSetting,
    _direction_response_conclusion,
    _dump_routes_toml,
    _reverse_settings,
    _routes_for_case,
    parse_direction_setting,
)


class BundDirectionResponseRunnerTests(unittest.TestCase):
    def test_parse_direction_setting_requires_supported_axis_direction(self) -> None:
        setting = parse_direction_setting("channel_3:minus")

        self.assertEqual(setting.channel, "channel_3")
        self.assertEqual(setting.direction, "minus")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_direction_setting("channel_3")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_direction_setting("channel_3:east")

    def test_reverse_settings_flip_plus_minus_and_preserve_closed(self) -> None:
        reversed_settings = _reverse_settings(
            (
                DirectionSetting("channel_3", "minus"),
                DirectionSetting("channel_4", "plus"),
                DirectionSetting("channel_5", "closed"),
            )
        )

        self.assertEqual(
            [(setting.channel, setting.direction) for setting in reversed_settings],
            [("channel_3", "plus"), ("channel_4", "minus"), ("channel_5", "closed")],
        )

    def test_routes_for_case_adds_region_axis_controls_to_every_stage(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "to_channel_3",
                    "group_key": [1, 2],
                    "goal_region": "channel_3",
                },
                {
                    "stage_id": "to_goal_region3",
                    "group_key": [1, 3],
                    "goal_region": "goal_region3",
                },
            ],
        }
        case = DirectionCase(
            case_id="bund_s_forward",
            title="Bund forward",
            family="direction_scan",
            settings=(DirectionSetting("channel_3", "minus"),),
            description="test",
        )

        routes = _routes_for_case(base_routes, case, alpha=2.8, beta=0.35)
        rendered = _dump_routes_toml(routes)

        self.assertIn('case_id = "bund_s_forward"', rendered)
        self.assertEqual(rendered.count("[[stages.controls]]"), 2)
        self.assertIn('mode = "region_axis"', rendered)
        self.assertIn('region = "channel_3"', rendered)
        self.assertIn('axis_region = "channel_3"', rendered)
        self.assertIn('direction = "minus"', rendered)
        self.assertIn("alpha = 2.8", rendered)
        self.assertIn("beta = 0.35", rendered)
        self.assertEqual(base_routes["case"]["case_id"], "base")

    def test_direction_response_conclusion_uses_objective_span(self) -> None:
        rows = [
            {
                "case_id": "bund_s_no_control",
                "family": "reference",
                "objective_value": 1.0,
                "objective_without_j5": 0.5,
            },
            {
                "case_id": "bund_s_reverse",
                "family": "direction_scan",
                "objective_value": 1.25,
                "objective_without_j5": 0.55,
            },
        ]

        conclusion = _direction_response_conclusion(rows, objective_tolerance=1.0e-4)

        self.assertEqual(conclusion["verdict"], "supports_direction_control_changes_objective")
        self.assertAlmostEqual(conclusion["objective_span"], 0.25)
        self.assertAlmostEqual(conclusion["objective_without_j5_span"], 0.05)


if __name__ == "__main__":
    unittest.main()
