from __future__ import annotations

import unittest

from crowd_bellman.g4_sahbo import (
    CHANNEL_STATES,
    ControlVector,
    generate_direction_neighbors,
    proxy_score,
)


class G4SAHBOTests(unittest.TestCase):
    def test_control_vector_normalizes_aliases_and_eta(self) -> None:
        control = ControlVector(("both", "c", "W"), (0.2, 4.0, 8.0)).normalized()

        self.assertEqual(control.directions, ("FREE", "CLOSED", "W"))
        self.assertEqual(control.eta, (1.0, 4.0, 8.0))

    def test_direction_neighbors_radius_one_changes_at_most_one_channel(self) -> None:
        base = ("FREE", "FREE", "FREE")
        neighbors = generate_direction_neighbors(base, radius=1)

        self.assertEqual(len(neighbors), 1 + 3 * (len(CHANNEL_STATES) - 1))
        self.assertIn(base, neighbors)
        for neighbor in neighbors:
            changes = sum(left != right for left, right in zip(base, neighbor))
            self.assertLessEqual(changes, 1)

    def test_proxy_penalizes_closing_loaded_channel(self) -> None:
        incumbent = {
            "channel_flux_share": {
                "top": 0.8,
                "middle": 0.1,
                "bottom": 0.1,
            }
        }
        close_loaded = proxy_score(
            directions=("CLOSED", "E", "W"),
            eta=(8.0, 8.0, 8.0),
            incumbent_summary=incumbent,
        )
        close_light = proxy_score(
            directions=("E", "W", "CLOSED"),
            eta=(8.0, 8.0, 8.0),
            incumbent_summary=incumbent,
        )

        self.assertGreater(close_loaded, close_light)


if __name__ == "__main__":
    unittest.main()
