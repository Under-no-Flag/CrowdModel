from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_bellman.field_recorder import make_field_data_observer_factory
from crowd_bellman.scenes import SimulationConfig


class FieldDataRecorderTests(unittest.TestCase):
    def test_saves_density_velocity_fields_on_interval_and_final_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_output_dir = Path(tmp)
            factory = make_field_data_observer_factory(
                save_every=2,
                dtype="float32",
                output_dir_name="fields",
            )
            observer = factory(
                case_output_dir=case_output_dir,
                simulation=SimulationConfig(save_every=5),
            )
            self.assertIsNotNone(observer)
            assert observer is not None

            rho = np.arange(6, dtype=float).reshape(2, 3)
            for step, is_final in ((0, False), (1, False), (3, True)):
                observer(
                    {
                        "step": step,
                        "time": 0.25 * step,
                        "dt": 0.25,
                        "rho": rho + step,
                        "speed": rho + step + 10.0,
                        "vx": rho + step + 20.0,
                        "vy": rho + step + 30.0,
                        "is_final": is_final,
                    }
                )

            fields_dir = case_output_dir / "fields"
            step0_path = fields_dir / "field_step_0000.npz"
            step1_path = fields_dir / "field_step_0001.npz"
            step3_path = fields_dir / "field_step_0003.npz"
            self.assertTrue(step0_path.exists())
            self.assertFalse(step1_path.exists())
            self.assertTrue(step3_path.exists())

            with np.load(step3_path) as payload:
                self.assertEqual(payload["rho"].shape, (2, 3))
                self.assertEqual(payload["rho"].dtype, np.dtype("float32"))
                self.assertEqual(payload["vx"].dtype, np.dtype("float32"))
                self.assertEqual(payload["vy"].dtype, np.dtype("float32"))
                self.assertEqual(payload["speed"].dtype, np.dtype("float32"))
                self.assertAlmostEqual(float(payload["time"][0]), 0.75)
                self.assertEqual(int(payload["step"][0]), 3)

            manifest = json.loads((fields_dir / "fields_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["save_every"], 2)
            self.assertEqual(manifest["shape"], [2, 3])
            self.assertEqual(manifest["dtype"], "float32")
            self.assertEqual(manifest["fields"], ["rho", "speed", "vx", "vy"])
            self.assertEqual(
                [item["file"] for item in manifest["files"]],
                ["field_step_0000.npz", "field_step_0003.npz"],
            )


if __name__ == "__main__":
    unittest.main()
