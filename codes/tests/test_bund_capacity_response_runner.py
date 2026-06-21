from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_capacity_response_runner import (
    CapacityCase,
    CapacityControl,
    GateRef,
    _capacity_response_conclusion,
    _dump_routes_toml,
    _level_cases,
    _ref_rates_from_probe,
    _routes_for_case,
    parse_gate_ref,
)


class BundCapacityResponseRunnerTests(unittest.TestCase):
    def test_parse_gate_ref_requires_side(self) -> None:
        gate = parse_gate_ref("channel_3:minus")

        self.assertEqual(gate.channel, "channel_3")
        self.assertEqual(gate.side, "minus")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_gate_ref("channel_3")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_gate_ref("channel_3:left")

    def test_routes_for_case_keeps_stages_and_adds_capacity_controls(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "to_channel_3",
                    "group_key": [1, 2],
                    "goal_region": "channel_3",
                }
            ],
        }
        case = CapacityCase(
            case_id="bund_q_low",
            title="Bund low q",
            family="level_scan",
            controls=(
                CapacityControl(
                    gate=GateRef("channel_3", "minus"),
                    rate=0.25,
                    waiting_width=9,
                ),
            ),
            description="test",
        )

        routes = _routes_for_case(base_routes, case)
        rendered = _dump_routes_toml(routes)

        self.assertIn('case_id = "bund_q_low"', rendered)
        self.assertIn('stage_id = "to_channel_3"', rendered)
        self.assertIn("[[capacity_controls]]", rendered)
        self.assertIn('channel = "channel_3"', rendered)
        self.assertIn('side = "minus"', rendered)
        self.assertIn("rate = 0.25", rendered)
        self.assertIn("waiting_width = 9", rendered)
        self.assertEqual(base_routes["case"]["case_id"], "base")

    def test_ref_rates_from_probe_uses_average_attempted_rate_with_floor(self) -> None:
        summary = {
            "final_time": 20.0,
            "gate_attempted_cumulative": {
                "channel_3:minus": 5.0,
                "channel_4:plus": 0.0,
            },
        }

        rates = _ref_rates_from_probe(
            summary,
            (GateRef("channel_3", "minus"), GateRef("channel_4", "plus")),
            rate_floor=0.05,
        )

        self.assertAlmostEqual(rates["channel_3:minus"], 0.25)
        self.assertAlmostEqual(rates["channel_4:plus"], 0.05)

    def test_level_case_names_follow_multiplier_value(self) -> None:
        cases = _level_cases(
            {"channel_3:minus": 1.0},
            multipliers=(0.3,),
            waiting_width=8,
        )

        self.assertEqual(cases[0].case_id, "bund_q_low")
        self.assertEqual(cases[0].controls[0].rate, 0.3)

    def test_conclusion_requires_objective_change_and_binding(self) -> None:
        rows = [
            {
                "case_id": "bund_q_no_limit",
                "family": "reference",
                "objective_value": 1.0,
                "gate_attempted": 0.0,
                "gate_rejected": 0.0,
                "binding_time_ratio_max": 0.0,
            },
            {
                "case_id": "bund_q_low",
                "family": "level_scan",
                "objective_value": 1.2,
                "gate_attempted": 3.0,
                "gate_rejected": 1.0,
                "binding_time_ratio_max": 0.5,
            },
        ]

        conclusion = _capacity_response_conclusion(rows, objective_tolerance=1.0e-4)

        self.assertEqual(conclusion["verdict"], "supports_capacity_control_changes_objective")
        self.assertAlmostEqual(conclusion["objective_span"], 0.2)
        self.assertAlmostEqual(conclusion["max_gate_rejected"], 1.0)


if __name__ == "__main__":
    unittest.main()
