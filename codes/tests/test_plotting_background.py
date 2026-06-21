from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowd_bellman.plotting import _background_content_limits, save_case_snapshot


class PlottingBackgroundTests(unittest.TestCase):
    def test_background_content_limits_crop_to_non_white_scene_content(self) -> None:
        background = np.ones((100, 200, 3), dtype=float)
        background[40:60, 50:150, :] = 0.0

        xlim, ylim = _background_content_limits(background, data_width=200, data_height=100)

        self.assertGreater(xlim[0], 0.0)
        self.assertLess(xlim[1], 200.0)
        self.assertLess(ylim[0], 100.0)
        self.assertGreater(ylim[1], 0.0)
        self.assertLess(xlim[0], 50.0)
        self.assertGreater(xlim[1], 150.0)

    def test_save_case_snapshot_accepts_scene_background_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            background_path = temp_path / "grid_overlay.png"
            output_path = temp_path / "snapshot.png"

            plt.imsave(background_path, np.ones((4, 6, 3), dtype=float))

            rho = np.zeros((4, 6), dtype=float)
            rho[1:3, 2:4] = 1.0
            walkable = np.ones((4, 6), dtype=bool)
            save_case_snapshot(
                path=output_path,
                title="background snapshot",
                rho=rho,
                phi=np.zeros_like(rho),
                ux=np.zeros_like(rho),
                uy=np.zeros_like(rho),
                walkable=walkable,
                rho_max=5.0,
                scene_background_path=background_path,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
