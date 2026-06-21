from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_capacity_response_runner import _dump_routes_toml
from bund_hcmbo_transfer_runner import (
    DEFAULT_CHANNEL_MAP,
    _capacity_controls_from_q,
    _dump_population_toml,
    _direction_controls_from_g6,
    _field_data_config_from_args,
    _control_enablement_from_args,
    _population_with_scaled_inflows,
    _routes_from_hcmbo_control,
    _simulation_overrides_from_args,
    _translate_direction_state,
    build_arg_parser,
)


class BundHCMBOTransferRunnerTests(unittest.TestCase):
    def test_translate_direction_state_maps_g6_horizontal_states_to_channel_axis(self) -> None:
        self.assertEqual(_translate_direction_state("E"), "plus")
        self.assertEqual(_translate_direction_state("W"), "minus")
        self.assertEqual(_translate_direction_state("FREE"), "both")
        self.assertEqual(_translate_direction_state("CLOSED"), "closed")

    def test_direction_controls_use_region_axis_for_bund_channels(self) -> None:
        controls = _direction_controls_from_g6(
            {"top": "W", "middle": "E", "lower_middle": "E", "bottom": "W"},
            channel_map=DEFAULT_CHANNEL_MAP,
            alpha=2.8,
            beta=0.35,
        )

        self.assertEqual(len(controls), 4)
        self.assertEqual(controls[0]["mode"], "region_axis")
        self.assertEqual(controls[0]["region"], "channel_1")
        self.assertEqual(controls[0]["axis_region"], "channel_1")
        self.assertEqual(controls[0]["direction"], "minus")
        self.assertEqual(controls[1]["region"], "channel_2")
        self.assertEqual(controls[1]["direction"], "plus")

    def test_capacity_controls_map_g6_gates_to_bund_channels_and_time_segments(self) -> None:
        controls = _capacity_controls_from_q(
            {
                "top:minus": [0.2, 0.4],
                "middle:plus": [1.0, 3.0],
            },
            channel_map=DEFAULT_CHANNEL_MAP,
            duration=20.0,
            waiting_width=7,
            capacity_scale=2.0,
        )

        self.assertEqual(
            controls,
            [
                {
                    "channel": "channel_1",
                    "side": "minus",
                    "rate": 0.4,
                    "time_start": 0.0,
                    "time_end": 10.0,
                    "waiting_width": 7,
                },
                {
                    "channel": "channel_1",
                    "side": "minus",
                    "rate": 0.8,
                    "time_start": 10.0,
                    "time_end": 20.0,
                    "waiting_width": 7,
                },
                {
                    "channel": "channel_2",
                    "side": "plus",
                    "rate": 2.0,
                    "time_start": 0.0,
                    "time_end": 10.0,
                    "waiting_width": 7,
                },
                {
                    "channel": "channel_2",
                    "side": "plus",
                    "rate": 6.0,
                    "time_start": 10.0,
                    "time_end": 20.0,
                    "waiting_width": 7,
                },
            ],
        )

    def test_routes_from_hcmbo_control_adds_direction_and_capacity_controls(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "to_decision_channel_split",
                    "group_key": [1, 1],
                    "goal_region": "decision_channel_split",
                },
                {
                    "stage_id": "via_channel_1",
                    "group_key": [1, 2],
                    "goal_region": "post_channel_1",
                },
            ],
        }
        control = {
            "directions": {"top": "W", "middle": "E", "lower_middle": "E", "bottom": "W"},
            "q_by_gate": {"top:minus": [0.2], "middle:plus": [1.0]},
        }

        routes = _routes_from_hcmbo_control(
            base_routes,
            control,
            channel_map=DEFAULT_CHANNEL_MAP,
            case_id="bund_hcmbo_transfer",
            duration=40.0,
            alpha=2.8,
            beta=0.35,
            waiting_width=8,
            capacity_scale=1.0,
        )
        rendered = _dump_routes_toml(routes)

        self.assertIn('case_id = "bund_hcmbo_transfer"', rendered)
        self.assertEqual(rendered.count("[[stages.controls]]"), 8)
        self.assertIn('mode = "region_axis"', rendered)
        self.assertIn('region = "channel_1"', rendered)
        self.assertIn('direction = "minus"', rendered)
        self.assertIn("[[capacity_controls]]", rendered)
        self.assertIn('channel = "channel_1"', rendered)
        self.assertIn('side = "minus"', rendered)
        self.assertIn("rate = 0.2", rendered)
        self.assertEqual(base_routes["case"]["case_id"], "base")

    def test_routes_from_hcmbo_control_can_override_transition_kappa(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "split",
                    "group_key": [1, 1],
                    "goal_region": "decision",
                    "targets": [{"stage_id": "via_channel_1", "probability": 1.0}],
                },
                {
                    "stage_id": "via_channel_1",
                    "group_key": [1, 2],
                    "goal_region": "channel_1",
                    "next_stage": "exit",
                },
                {
                    "stage_id": "exit",
                    "group_key": [1, 3],
                    "goal_region": "exits",
                    "sink_region": "exits",
                },
            ],
        }
        control = {
            "directions": {"top": "W", "middle": "E", "lower_middle": "E", "bottom": "W"},
            "q_by_gate": {"top:minus": [0.2]},
        }

        routes = _routes_from_hcmbo_control(
            base_routes,
            control,
            channel_map=DEFAULT_CHANNEL_MAP,
            case_id="bund_hcmbo_transfer",
            duration=40.0,
            alpha=2.8,
            beta=0.35,
            waiting_width=8,
            capacity_scale=1.0,
            transition_kappa=8.0,
        )

        self.assertEqual(routes["stages"][0]["kappa"], 8.0)
        self.assertEqual(routes["stages"][1]["kappa"], 8.0)
        self.assertNotIn("kappa", routes["stages"][2])
        self.assertNotIn("kappa", base_routes["stages"][0])

    def test_routes_from_hcmbo_control_can_disable_capacity_controls(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "via_channel_1",
                    "group_key": [1, 1],
                    "goal_region": "channel_1",
                },
            ],
        }
        control = {
            "directions": {"top": "W", "middle": "E", "lower_middle": "E", "bottom": "W"},
            "q_by_gate": {"top:minus": [0.2], "middle:plus": [1.0]},
        }

        routes = _routes_from_hcmbo_control(
            base_routes,
            control,
            channel_map=DEFAULT_CHANNEL_MAP,
            case_id="bund_hcmbo_transfer",
            duration=40.0,
            alpha=2.8,
            beta=0.35,
            waiting_width=8,
            capacity_scale=1.0,
            include_capacity_controls=False,
        )
        rendered = _dump_routes_toml(routes)

        self.assertEqual(routes["capacity_controls"], [])
        self.assertNotIn("[[capacity_controls]]", rendered)
        self.assertIn("[[stages.controls]]", rendered)
        self.assertIn('mode = "region_axis"', rendered)

    def test_routes_from_hcmbo_control_can_disable_direction_controls(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "via_channel_1",
                    "group_key": [1, 1],
                    "goal_region": "channel_1",
                },
            ],
        }
        control = {
            "directions": {"top": "W", "middle": "E", "lower_middle": "E", "bottom": "W"},
            "q_by_gate": {"top:minus": [0.2], "middle:plus": [1.0]},
        }

        routes = _routes_from_hcmbo_control(
            base_routes,
            control,
            channel_map=DEFAULT_CHANNEL_MAP,
            case_id="bund_hcmbo_transfer",
            duration=40.0,
            alpha=2.8,
            beta=0.35,
            waiting_width=8,
            capacity_scale=1.0,
            include_direction_controls=False,
        )
        rendered = _dump_routes_toml(routes)

        self.assertNotIn("[[stages.controls]]", rendered)
        self.assertNotIn('mode = "region_axis"', rendered)
        self.assertIn("[[capacity_controls]]", rendered)

    def test_field_data_cli_config_is_optional_and_uses_explicit_interval(self) -> None:
        parser = build_arg_parser()

        disabled = parser.parse_args([])
        self.assertEqual(_field_data_config_from_args(disabled, {}), {"enabled": False})

        enabled = parser.parse_args(
            [
                "--save-field-data",
                "--field-save-every",
                "7",
                "--field-dtype",
                "float64",
                "--field-output-dir",
                "field_arrays",
            ]
        )
        self.assertEqual(
            _field_data_config_from_args(enabled, {"save_every": 40}),
            {
                "enabled": True,
                "output_dir_name": "field_arrays",
                "dtype": "float64",
                "save_every": 7,
            },
        )

    def test_control_switch_can_enable_or_disable_all_transferred_controls(self) -> None:
        parser = build_arg_parser()

        self.assertEqual(_control_enablement_from_args(parser.parse_args([])), (True, True))
        self.assertEqual(_control_enablement_from_args(parser.parse_args(["--apply-controls"])), (True, True))
        self.assertEqual(_control_enablement_from_args(parser.parse_args(["--no-controls"])), (False, False))
        self.assertEqual(
            _control_enablement_from_args(parser.parse_args(["--apply-controls", "--disable-capacity-controls"])),
            (True, False),
        )
        self.assertEqual(
            _control_enablement_from_args(parser.parse_args(["--disable-direction-controls"])),
            (False, True),
        )

    def test_simulation_overrides_can_set_rho_max(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["--rho-max", "10.0"])

        self.assertEqual(_simulation_overrides_from_args(args)["rho_max"], 10.0)

    def test_population_with_scaled_inflows_doubles_only_continuous_rates(self) -> None:
        population = {
            "initial_groups": [
                {
                    "group_id": "initial",
                    "stage_id": "to_decision",
                    "region": "initial_groups",
                    "density": 2.0,
                }
            ],
            "inflow_groups": [
                {
                    "group_id": "inflow",
                    "stage_id": "to_decision",
                    "region": "initial_groups",
                    "rate": 0.08,
                    "time_start": 0.0,
                    "rho_cap": 4.8,
                }
            ],
        }

        scaled = _population_with_scaled_inflows(population, 2.0)
        rendered = _dump_population_toml(scaled)

        self.assertEqual(population["inflow_groups"][0]["rate"], 0.08)
        self.assertEqual(scaled["initial_groups"][0]["density"], 2.0)
        self.assertEqual(scaled["inflow_groups"][0]["rate"], 0.16)
        self.assertIn("[[initial_groups]]", rendered)
        self.assertIn("[[inflow_groups]]", rendered)
        self.assertIn("rate = 0.16", rendered)

    def test_wall_avoidance_cli_values_are_simulation_overrides(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--wall-avoidance-weight",
                "2.5",
                "--wall-avoidance-sigma-cells",
                "1.5",
                "--wall-avoidance-radius-cells",
                "4.0",
                "--wall-clearance-cells",
                "1",
                "--allow-diagonal-corner-cutting",
            ]
        )

        self.assertEqual(
            _simulation_overrides_from_args(args),
            {
                "block_diagonal_corner_cutting": False,
                "wall_avoidance_weight": 2.5,
                "wall_avoidance_sigma_cells": 1.5,
                "wall_avoidance_radius_cells": 4.0,
                "wall_clearance_cells": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
