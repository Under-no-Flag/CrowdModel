from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_hcmbo_optimization_runner import (
    _load_records_from_existing_summaries,
    build_arg_parser,
    build_bund_controlled_routes,
    build_small_budget_config,
    prepare_bund_hcmbo_base_dir,
    qbar_from_bund_reference,
    resolve_direction_candidates,
)
from crowd_bellman.g5_hcmbo import ALL_GATE_IDS, DEFAULT_PRIOR_DIRECTIONS, V2ControlVector


class BundHCMBOOptimizationRunnerTests(unittest.TestCase):
    def test_build_bund_controlled_routes_maps_hcmbo_channels_to_bund_channels(self) -> None:
        base_routes = {
            "case": {"case_id": "base", "title": "Base"},
            "stages": [
                {
                    "stage_id": "to_channel_1",
                    "group_key": [1, 1],
                    "goal_region": "channel_1",
                }
            ],
        }
        q_profiles = []
        for gate_id in ALL_GATE_IDS:
            q_profiles.append((0.25,) if gate_id == "top:minus" else (0.0,))
        control = V2ControlVector(
            directions=("W", "E", "E", "W"),
            q_by_gate=tuple(q_profiles),
        ).normalized()

        routes = build_bund_controlled_routes(
            base_routes,
            control,
            case_id="case",
            duration=10.0,
            alpha=2.8,
            beta=0.35,
            waiting_width=7,
        )

        controls = routes["stages"][0]["controls"]
        self.assertTrue(any(item["region"] == "channel_1" and item["direction"] == "minus" for item in controls))
        self.assertTrue(any(item["region"] == "channel_2" and item["direction"] == "plus" for item in controls))
        capacity = [item for item in routes["capacity_controls"] if item["channel"] == "channel_1" and item["side"] == "minus"]
        self.assertEqual(len(capacity), 1)
        self.assertEqual(capacity[0]["rate"], 0.25)
        self.assertEqual(capacity[0]["waiting_width"], 7)

    def test_qbar_from_bund_reference_maps_gate_ids_back_to_hcmbo_space(self) -> None:
        summary = {
            "final_time": 10.0,
            "gate_attempted_cumulative": {
                "channel_1:minus": 3.0,
                "channel_2:plus": 5.0,
            },
        }
        config = build_small_budget_config(build_arg_parser().parse_args([]))

        qbar = qbar_from_bund_reference(summary, config=config)

        self.assertAlmostEqual(qbar["top:minus"], 0.36)
        self.assertAlmostEqual(qbar["middle:plus"], 0.6)
        self.assertEqual(qbar["bottom:plus"], config.min_qbar)

    def test_prepare_bund_hcmbo_base_dir_scales_inflow_and_adds_center_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            base_dir = prepare_bund_hcmbo_base_dir(
                source_base_dir=Path(__file__).resolve().parents[2] / "anylogic-scene" / "BundScene" / "converted",
                output_root=output_root,
                route_variant="b_center_goal",
                transition_kappa=32.0,
                inflow_rate_scale=3.0,
            )
            scene_text = (base_dir / "scene.toml").read_text(encoding="utf-8")
            routes_text = (base_dir / "routes.toml").read_text(encoding="utf-8")
            population_text = (base_dir / "population.toml").read_text(encoding="utf-8")

        self.assertIn('name = "center_goal_region11"', scene_text)
        self.assertIn('goal_region = "center_goal_region11"', routes_text)
        self.assertIn("rate = 0.24", population_text)

    def test_default_hcmbo_budget_is_small_and_one_segment(self) -> None:
        args = build_arg_parser().parse_args([])

        config = build_small_budget_config(args)

        self.assertEqual(config.time_segments, 1)
        self.assertEqual(config.initial_samples, 2)
        self.assertEqual(config.bo_iterations, 1)
        self.assertEqual(config.random_search_evaluations, 0)
        self.assertEqual(config.dfo_evaluations, 0)
        self.assertEqual(args.screen_steps, 80)
        self.assertEqual(args.high_fidelity_steps, 240)
        self.assertEqual(args.rho_max, 4.0)
        self.assertEqual(args.direction_candidate_limit, 12)
        self.assertFalse(args.resume_existing)

    def test_resume_record_loader_rebuilds_control_from_summary(self) -> None:
        args = build_arg_parser().parse_args(["--time-segments", "2"])
        config = build_small_budget_config(args)
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            case_dir.mkdir()
            q_by_gate = {gate_id: [0.1, 0.2] for gate_id in ALL_GATE_IDS}
            summary = {
                "case_id": "case",
                "final_time": 1.0,
                "final_cap_removed_cumulative": 0.0,
                "normalization_context": {"total_mass_reference": 1.0, "evaluation_time": 1.0},
                "objective": {"j1_eval": 1.0, "j2_eval": 2.0, "j5_eval": 3.0},
                "bund_hcmbo_optimization": {
                    "source": "resume_test",
                    "phase": "high_fidelity",
                    "fidelity": "hf",
                    "eval_id": 7,
                    "control": {
                        "directions": {
                            "top": "FREE",
                            "middle": "E",
                            "lower_middle": "W",
                            "bottom": "FREE",
                        },
                        "q_by_gate": q_by_gate,
                    },
                    "config_path": "run.toml",
                },
            }
            (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

            records = _load_records_from_existing_summaries(
                Path(tmp),
                config=config,
                qbar_by_gate={gate_id: 1.0 for gate_id in ALL_GATE_IDS},
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].eval_id, 7)
        self.assertEqual(records[0].phase, "high_fidelity")
        self.assertEqual(records[0].control.directions, ("FREE", "E", "W", "FREE"))
        self.assertEqual(records[0].control.q_by_gate[0], (0.1, 0.2))

    def test_default_direction_candidates_generate_twelve_hcmbo_candidates(self) -> None:
        args = build_arg_parser().parse_args([])
        config = build_small_budget_config(args)

        candidates = resolve_direction_candidates(None, config=config, rng=np.random.default_rng(args.random_seed))

        self.assertEqual(len(candidates), 12)
        self.assertEqual(len(set(candidates)), 12)
        self.assertIn(tuple("FREE" for _ in range(4)), candidates)
        self.assertIn(DEFAULT_PRIOR_DIRECTIONS, candidates)

    def test_explicit_direction_candidates_override_generated_defaults(self) -> None:
        args = build_arg_parser().parse_args(["--direction-candidate", "FREE,E,W,FREE"])
        config = build_small_budget_config(args)

        candidates = resolve_direction_candidates(args.direction_candidate, config=config, rng=np.random.default_rng(args.random_seed))

        self.assertEqual(candidates, [("FREE", "E", "W", "FREE")])


if __name__ == "__main__":
    unittest.main()
