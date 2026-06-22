from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bund_control_comparison_runner import (
    DEFAULT_CASE_SPECS,
    build_case_commands,
    generate_objective_timeseries_outputs,
    load_objective_timeseries,
    load_field_timeseries,
    prepare_route_variant_base_dir,
    summarize_case_outputs,
)


class BundControlComparisonRunnerTests(unittest.TestCase):
    def test_build_case_commands_keep_control_and_no_control_settings_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = build_case_commands(
                comparison_root=root,
                base_dir=Path("anylogic-scene/BundScene/converted"),
                control_json=Path("codes/results/custom_control.json"),
                steps=1600,
                time_horizon=50.0,
                save_every=10,
                rho_max=4.0,
                inflow_rate_scale=6.0,
                transition_kappa=32.0,
                capacity_scale=2.0,
                field_dtype="float32",
                python_executable="python",
            )

        self.assertEqual(set(commands), {"controlled", "uncontrolled"})
        controlled = commands["controlled"]
        uncontrolled = commands["uncontrolled"]
        self.assertIn("--apply-controls", controlled.argv)
        self.assertIn("--no-controls", uncontrolled.argv)
        for case in (controlled, uncontrolled):
            self.assertIn("--save-field-data", case.argv)
            self.assertEqual(case.argv[case.argv.index("--control-json") + 1], str((Path(__file__).resolve().parents[2] / "codes/results/custom_control.json").resolve()))
            self.assertEqual(case.argv[case.argv.index("--field-save-every") + 1], "10")
            self.assertEqual(case.argv[case.argv.index("--save-every") + 1], "10")
            self.assertEqual(case.argv[case.argv.index("--steps") + 1], "1600")
            self.assertEqual(case.argv[case.argv.index("--inflow-rate-scale") + 1], "6")
            self.assertTrue(str(case.output_root).endswith(case.spec.case_id))

    def test_prepare_route_variant_base_dir_writes_b_center_goal_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = prepare_route_variant_base_dir(
                source_base_dir=Path(__file__).resolve().parents[2] / "anylogic-scene" / "BundScene" / "converted",
                comparison_root=root,
                route_variant="b_center_goal",
                transition_kappa=32.0,
            )

            scene_text = (base_dir / "scene.toml").read_text(encoding="utf-8")
            routes_text = (base_dir / "routes.toml").read_text(encoding="utf-8")

        self.assertIn('name = "center_goal_region11"', scene_text)
        self.assertIn('goal_region = "center_goal_region11"', routes_text)
        self.assertIn('decision_regions = ["goal_region11", "channel_1"]', routes_text)

    def test_load_field_timeseries_reads_saved_density_and_speed_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fields_dir = Path(tmp) / "fields"
            fields_dir.mkdir()
            files = []
            for step, value in ((0, 1.0), (10, 3.0)):
                filename = f"field_step_{step:04d}.npz"
                np.savez_compressed(
                    fields_dir / filename,
                    step=np.array([step], dtype=np.int64),
                    time=np.array([0.1 * step], dtype=np.float64),
                    dt=np.array([0.1], dtype=np.float64),
                    rho=np.array([[0.0, value], [value + 1.0, 0.0]], dtype=np.float32),
                    speed=np.full((2, 2), value, dtype=np.float32),
                    vx=np.zeros((2, 2), dtype=np.float32),
                    vy=np.zeros((2, 2), dtype=np.float32),
                )
                files.append({"file": filename, "step": step, "time": 0.1 * step})
            (fields_dir / "fields_manifest.json").write_text(
                json.dumps({"files": files, "fields": ["rho", "speed", "vx", "vy"], "shape": [2, 2]}),
                encoding="utf-8",
            )

            rows = load_field_timeseries(fields_dir, label="controlled")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["case"], "controlled")
        self.assertEqual(rows[1]["step"], 10)
        self.assertAlmostEqual(rows[0]["density_sum"], 3.0)
        self.assertAlmostEqual(rows[1]["density_max"], 4.0)
        self.assertAlmostEqual(rows[1]["speed_mean"], 3.0)

    def test_summarize_case_outputs_flattens_comparison_metrics(self) -> None:
        case_dir = Path("case")
        summary = {
            "case_id": "case",
            "objective_value": 1.2,
            "final_sink_cumulative": 2.0,
            "final_inflow_cumulative": 6.0,
            "final_mass": 4.0,
            "peak_density_max": 3.5,
            "objective_terms_normalized": {"j1_total_travel_time": 0.4, "j2_high_density_exposure": 0.2},
            "channel_flux_share": {"channel_1": 0.25},
            "gate_rejected_cumulative": {"channel_1:plus": 0.5},
        }

        row = summarize_case_outputs(DEFAULT_CASE_SPECS[0], case_dir, summary)

        self.assertEqual(row["label"], DEFAULT_CASE_SPECS[0].label)
        self.assertEqual(row["case_id"], "case")
        self.assertEqual(row["case_dir"], str(case_dir))
        self.assertEqual(row["objective_value"], 1.2)
        self.assertEqual(row["j1_normalized"], 0.4)
        self.assertEqual(row["j2_normalized"], 0.2)
        self.assertEqual(row["channel_flux_share.channel_1"], 0.25)
        self.assertEqual(row["gate_rejected_total"], 0.5)

    def test_load_objective_timeseries_matches_summary_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "controlled" / "controlled"
            case_dir.mkdir(parents=True)
            summary = {
                "objective_value": 1.5,
                "normalization_context": {
                    "j1_denominator": 10.0,
                    "j2_denominator": 20.0,
                },
                "objective": {
                    "lambda_j1": 1.0,
                    "lambda_j2": 1.0,
                    "lambda_j5": 1.0,
                    "j1_scale": 1.0,
                    "j2_scale": 0.5,
                    "j5_scale": 1.0,
                    "j1_eval": 1.0,
                    "j2_eval": 0.4,
                    "j5_eval": 0.1,
                    "objective_value": 1.5,
                },
            }
            (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (case_dir / "timeseries.csv").write_text(
                "\n".join(
                    [
                        "time,dt,travel_time_cumulative,high_density_exposure_cumulative,channel_density_channel_1,channel_density_channel_2",
                        "1.0,1.0,5.0,2.0,1.0,0.0",
                        "2.0,1.0,10.0,4.0,2.0,0.0",
                    ]
                ),
                encoding="utf-8",
            )

            rows, metadata = load_objective_timeseries(case_dir, label="controlled")

        self.assertEqual(len(rows), 2)
        self.assertEqual(metadata["j5_source"], "channel_density_exposure_variance_proxy_scaled_to_summary")
        self.assertAlmostEqual(float(rows[-1]["j1_eval"]), 1.0)
        self.assertAlmostEqual(float(rows[-1]["j2_eval"]), 0.4)
        self.assertAlmostEqual(float(rows[-1]["j5_eval"]), 0.1)
        self.assertAlmostEqual(float(rows[-1]["objective_value"]), 1.5)

    def test_generate_objective_timeseries_outputs_writes_csv_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for spec in DEFAULT_CASE_SPECS:
                case_dir = root / spec.case_id / spec.case_id
                case_dir.mkdir(parents=True)
                summary = {
                    "objective_value": 1.5,
                    "normalization_context": {
                        "j1_denominator": 10.0,
                        "j2_denominator": 20.0,
                    },
                    "objective": {
                        "lambda_j1": 1.0,
                        "lambda_j2": 1.0,
                        "lambda_j5": 1.0,
                        "j1_scale": 1.0,
                        "j2_scale": 0.5,
                        "j5_scale": 1.0,
                        "j5_eval": 0.1,
                    },
                }
                (case_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                (case_dir / "timeseries.csv").write_text(
                    "\n".join(
                        [
                            "time,dt,travel_time_cumulative,high_density_exposure_cumulative,channel_density_channel_1,channel_density_channel_2",
                            "1.0,1.0,5.0,2.0,1.0,0.0",
                            "2.0,1.0,10.0,4.0,2.0,0.0",
                        ]
                    ),
                    encoding="utf-8",
                )

            payload = generate_objective_timeseries_outputs(root)

            self.assertTrue((root / "objective_timeseries.csv").exists())
            self.assertTrue((root / "objective_timeseries_metadata.json").exists())
            self.assertTrue((root / "figures" / "all_j_timeseries_stacked.png").exists())
            self.assertTrue((root / "figures" / "all_j_timeseries_lines.png").exists())
            self.assertIn("objective_timeseries_csv", payload)


if __name__ == "__main__":
    unittest.main()
