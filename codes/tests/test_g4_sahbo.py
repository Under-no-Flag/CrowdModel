from __future__ import annotations

import unittest
from pathlib import Path

from g4_runner import _load_g4_config
from crowd_bellman.g4_sahbo import (
    CHANNEL_STATES,
    ControlVector,
    _control_digest,
    _short_source_label,
    generate_direction_neighbors,
    proxy_score,
)


class G4SAHBOTests(unittest.TestCase):
    def test_control_vector_normalizes_aliases_and_eta(self) -> None:
        control = ControlVector(("both", "c", "W", "E"), (0.2, 4.0, 8.0, 12.0)).normalized()

        self.assertEqual(control.directions, ("FREE", "CLOSED", "W", "E"))
        self.assertEqual(control.eta, (1.0, 4.0, 8.0, 12.0))

    def test_direction_neighbors_radius_one_changes_at_most_one_channel(self) -> None:
        base = ("FREE", "FREE", "FREE", "FREE")
        neighbors = generate_direction_neighbors(base, radius=1)

        self.assertEqual(len(neighbors), 1 + 4 * (len(CHANNEL_STATES) - 1))
        self.assertIn(base, neighbors)
        for neighbor in neighbors:
            changes = sum(left != right for left, right in zip(base, neighbor))
            self.assertLessEqual(changes, 1)

    def test_proxy_penalizes_closing_loaded_channel(self) -> None:
        incumbent = {
            "channel_flux_share": {
                "top": 0.8,
                "middle": 0.1,
                "lower_middle": 0.05,
                "bottom": 0.1,
            }
        }
        close_loaded = proxy_score(
            directions=("CLOSED", "E", "W", "W"),
            eta=(8.0, 8.0, 8.0, 8.0),
            incumbent_summary=incumbent,
        )
        close_light = proxy_score(
            directions=("E", "W", "W", "CLOSED"),
            eta=(8.0, 8.0, 8.0, 8.0),
            incumbent_summary=incumbent,
        )

        self.assertGreater(close_loaded, close_light)

    def test_g4_toml_config_loads_and_resolves_paths(self) -> None:
        config_path = Path("codes/scenes/examples/g4_sahbo_vs_grid/g4.toml")
        config = _load_g4_config(config_path)

        self.assertEqual(config["mode"], "both")
        self.assertTrue(Path(str(config["baseline_config"])).is_absolute())
        self.assertTrue(str(config["baseline_config"]).endswith("g2_multistage_directional\\run_baseline.toml") or str(config["baseline_config"]).endswith("g2_multistage_directional/run_baseline.toml"))
        self.assertTrue(Path(str(config["output_root"])).is_absolute())
        self.assertIsInstance(config["simulation_overrides"]["steps"], int)
        self.assertGreater(config["simulation_overrides"]["steps"], 0)
        self.assertEqual(config["sahbo"]["initial_directions"], ("FREE", "FREE", "FREE", "FREE"))
        self.assertEqual(config["sahbo"]["initial_eta"], (8.0, 8.0, 8.0, 8.0))
        self.assertEqual(config["grid"]["eta_values"], (1.0, 4.0, 8.0, 12.0))
        self.assertGreaterEqual(len(config["grid"]["direction_sets"]), 3)

    def test_generated_g4_labels_stay_short_for_windows_paths(self) -> None:
        control = ControlVector(("FREE", "E", "FREE", "W"), (7.997191234, 7.997191234, 7.997191234, 7.997191234)).normalized()
        case_id = f"g4_{7:04d}_{_short_source_label('sahbo_iter0_eta_candidate_bt0')}_{_control_digest(control)}"

        self.assertLessEqual(len(case_id), 40)
        self.assertNotIn("7p997191234", case_id)


if __name__ == "__main__":
    unittest.main()
