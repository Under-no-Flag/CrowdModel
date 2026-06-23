from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_bund_comparison_panel import (
    crop_bounds_from_walls,
    parse_steps,
    render_comparison_panel,
    select_entry_for_step,
)
from render_refined_density_heatmap import load_wall_polylines


class BundComparisonPanelTests(unittest.TestCase):
    def test_parse_steps_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(parse_steps(["40,400", "800", "1590"]), [40, 400, 800, 1590])

    def test_select_entry_for_step_can_use_nearest(self) -> None:
        entries = [{"file": "field_step_0000.npz", "step": 0}, {"file": "field_step_0020.npz", "step": 20}]

        self.assertEqual(select_entry_for_step(entries, 18, nearest=True)["step"], 20)
        with self.assertRaisesRegex(ValueError, "Step 18 not found"):
            select_entry_for_step(entries, 18, nearest=False)

    def test_crop_bounds_from_walls_uses_polyline_extent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = Path(tmp) / "scene.toml"
            scene_path.write_text(
                """
                [[walls]]
                name = "wall"
                shape = "polyline"
                points = [[4, 3], [10, 8]]
                width = 1.0
                """,
                encoding="utf-8",
            )
            walls = load_wall_polylines(scene_path)

            self.assertEqual(crop_bounds_from_walls(walls, shape=(20, 30), padding=2), (1, 11, 2, 13))

    def test_render_comparison_panel_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            for case_key in ("controlled", "uncontrolled"):
                case_dir = comparison_dir / case_key / case_key
                fields_dir = case_dir / "fields"
                fields_dir.mkdir(parents=True)
                files = []
                for step in (0, 10, 20, 30):
                    rho = np.zeros((8, 12), dtype=np.float32)
                    rho[3:5, 6:9] = 0.5 + step / 10.0
                    np.savez_compressed(
                        fields_dir / f"field_step_{step:04d}.npz",
                        step=np.array([step], dtype=np.int64),
                        time=np.array([float(step)], dtype=np.float64),
                        dt=np.array([1.0], dtype=np.float64),
                        rho=rho,
                    )
                    files.append({"file": f"field_step_{step:04d}.npz", "step": step, "time": float(step)})
                np.savez_compressed(fields_dir / "static_masks.npz", walkable=np.ones((8, 12), dtype=bool))
                (fields_dir / "fields_manifest.json").write_text(
                    json.dumps({"files": files, "static_masks": "static_masks.npz"}),
                    encoding="utf-8",
                )
                scene_dir = case_dir / "config_snapshot"
                scene_dir.mkdir()
                scene_path = scene_dir / "scene.toml"
                scene_path.write_text(
                    """
                    [[walls]]
                    name = "bund_wall"
                    shape = "polyline"
                    points = [[2, 2], [9, 5]]
                    width = 1.0
                    """,
                    encoding="utf-8",
                )
                (case_dir / "summary.json").write_text(
                    json.dumps({"config_snapshot": {"files": {"scene": str(scene_path)}}}),
                    encoding="utf-8",
                )
            output = root / "panel.png"

            render_comparison_panel(
                comparison_dir,
                steps=[0, 10, 20, 30],
                output=output,
                scale=2,
                smooth_sigma=0.0,
                vmax=6.0,
                dpi=80,
                crop_padding=1,
                nearest=False,
                gamma=0.42,
                wall_color="#111111",
                wall_alpha=0.95,
                wall_linewidth=None,
                title_fontsize=8.0,
                row_label_fontsize=8.0,
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
