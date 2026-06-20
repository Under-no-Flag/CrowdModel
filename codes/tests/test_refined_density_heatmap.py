from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib.colors import PowerNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from render_refined_density_heatmap import (
    density_percentile_limits,
    crop_density,
    load_density_frame,
    make_density_norm,
    parse_percentile_pair,
    render_density_heatmap,
    resize_image_to_shape,
    resolve_background_path,
    resolve_fields_dir,
    select_manifest_entries,
    smooth_density,
    upsample_density,
)


class RefinedDensityHeatmapTests(unittest.TestCase):
    def _write_fields_fixture(self, root: Path) -> Path:
        fields_dir = root / "case_a" / "fields"
        fields_dir.mkdir(parents=True)
        files = []
        for step in (0, 4):
            filename = f"field_step_{step:04d}.npz"
            np.savez_compressed(
                fields_dir / filename,
                step=np.array([step], dtype=np.int64),
                time=np.array([0.1 * step], dtype=np.float64),
                dt=np.array([0.1], dtype=np.float64),
                rho=np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32) + step,
            )
            files.append({"file": filename, "step": step, "time": 0.1 * step})
        (fields_dir / "fields_manifest.json").write_text(
            json.dumps({"files": files, "fields": ["rho"], "shape": [2, 2]}),
            encoding="utf-8",
        )
        return fields_dir

    def _write_background_fixture(self, root: Path) -> Path:
        scene_dir = root / "scene"
        scene_dir.mkdir(parents=True)
        scene_path = scene_dir / "scene.toml"
        scene_path.write_text("[scene]\n", encoding="utf-8")
        background_path = scene_dir / "grid_overlay.png"
        background = np.ones((4, 4, 3), dtype=np.float32)
        background[0, 0, :] = 0.0
        import matplotlib.pyplot as plt

        plt.imsave(background_path, background)
        summary_path = root / "case_a" / "summary.json"
        summary_path.write_text(json.dumps({"scene_path": str(scene_path)}), encoding="utf-8")
        return background_path

    def test_resolves_case_or_output_root_to_fields_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields_dir = self._write_fields_fixture(root)

            self.assertEqual(resolve_fields_dir(fields_dir), fields_dir)
            self.assertEqual(resolve_fields_dir(root / "case_a"), fields_dir)
            self.assertEqual(resolve_fields_dir(root), fields_dir)

    def test_resolves_background_from_case_summary_scene_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields_dir = self._write_fields_fixture(root)
            background_path = self._write_background_fixture(root)

            self.assertEqual(resolve_background_path(None, fields_dir), background_path)

    def test_select_manifest_entries_defaults_to_latest_or_exact_step(self) -> None:
        entries = [
            {"file": "field_step_0000.npz", "step": 0},
            {"file": "field_step_0004.npz", "step": 4},
        ]

        self.assertEqual(select_manifest_entries(entries, step=None, render_all=False), [entries[-1]])
        self.assertEqual(select_manifest_entries(entries, step=0, render_all=False), [entries[0]])
        self.assertEqual(select_manifest_entries(entries, step=None, render_all=True), entries)

    def test_upsample_and_smooth_density_produce_larger_smooth_grid(self) -> None:
        density = np.array([[0.0, 4.0], [2.0, 6.0]], dtype=float)

        upsampled = upsample_density(density, scale=3)
        smoothed = smooth_density(upsampled, sigma=0.8)

        self.assertEqual(upsampled.shape, (6, 6))
        self.assertEqual(smoothed.shape, (6, 6))
        self.assertGreater(float(smoothed[2, 2]), 0.0)
        self.assertLess(float(smoothed[2, 2]), 6.0)

    def test_crop_density_keeps_active_region_with_padding(self) -> None:
        density = np.zeros((8, 10), dtype=float)
        density[3:5, 4:7] = 2.0

        cropped = crop_density(density, threshold=0.1, padding=1)
        uncropped = crop_density(np.zeros((4, 5), dtype=float), threshold=0.1, padding=1)

        self.assertEqual(cropped.shape, (4, 5))
        self.assertEqual(uncropped.shape, (4, 5))

    def test_resize_image_to_shape_matches_density_display_shape(self) -> None:
        image = np.zeros((4, 6, 3), dtype=float)
        image[:, :, 0] = 1.0

        resized = resize_image_to_shape(image, target_shape=(8, 10))

        self.assertEqual(resized.shape, (8, 10, 3))
        self.assertGreater(float(np.mean(resized[:, :, 0])), 0.9)

    def test_density_percentile_limits_ignore_zero_background(self) -> None:
        limits = density_percentile_limits(
            [
                np.array([[0.0, 1.0, 2.0, 100.0]], dtype=float),
                np.array([[0.0, 3.0, 4.0, 5.0]], dtype=float),
            ],
            percentiles=(0.0, 50.0),
            threshold=0.0,
        )

        self.assertEqual(limits, (1.0, 3.5))

    def test_parse_percentile_pair_and_power_norm(self) -> None:
        self.assertEqual(parse_percentile_pair("2,98"), (2.0, 98.0))
        norm = make_density_norm(vmin=0.5, vmax=4.5, norm_mode="power", gamma=0.55)

        self.assertIsInstance(norm, PowerNorm)

    def test_render_density_heatmap_writes_single_density_png_with_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields_dir = self._write_fields_fixture(root)
            background_path = self._write_background_fixture(root)
            entry = select_manifest_entries(
                json.loads((fields_dir / "fields_manifest.json").read_text(encoding="utf-8"))["files"],
                step=4,
                render_all=False,
            )[0]
            frame = load_density_frame(fields_dir, entry)
            output_path = root / "density_step_0004.png"

            render_density_heatmap(
                frame,
                output_path,
                scale=2,
                smooth_sigma=0.5,
                vmax=8.0,
                dpi=80,
                background_path=background_path,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
