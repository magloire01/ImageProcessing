"""Numerical checks for the WaterVision pipeline.

These tests do not open the window. They lock the definitions that the
interface explains: luma weights, HSV angles, Fourier inversion, the fact
that a smooth tan region is not accepted by the hybrid score, and the rule
that square meters stay undefined until a pixel size is provided.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from watervision.processing.color import rgb_to_gray, rgb_to_hsv
from watervision.processing.demo import demo_regions, make_demo_scene
from watervision.processing.filters import preprocess
from watervision.processing.fourier import (
    butterworth_highpass,
    butterworth_lowpass,
    fft2_shifted,
    ideal_lowpass,
    reconstruct,
)
from watervision.processing.metrics import calibration_area, estimate_mask, evaluate_mask
from watervision.processing.morphology import fill_holes, opening, remove_small_regions
from watervision.processing.report import write_csv, write_pdf
from watervision.processing.segmentation import hue_in_range, segment
from watervision.state import default_params


class ColorTests(unittest.TestCase):
    def test_luma_uses_bt601_weights(self):
        red = np.zeros((2, 2, 3), dtype=np.uint8)
        red[..., 0] = 255
        gray = rgb_to_gray(red)
        self.assertTrue(np.allclose(gray, 0.299, atol=1e-5))

    def test_primary_hues(self):
        colors = {
            (255, 0, 0): 0.0,
            (0, 255, 0): 120.0,
            (0, 0, 255): 240.0,
        }
        for rgb, hue in colors.items():
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            image[..., 0] = rgb[0]
            image[..., 1] = rgb[1]
            image[..., 2] = rgb[2]
            hsv = rgb_to_hsv(image)
            self.assertTrue(np.allclose(hsv[..., 0], hue, atol=1e-3), msg=f"hue of {rgb}")
            self.assertTrue(np.allclose(hsv[..., 1], 1.0, atol=1e-4))
            self.assertTrue(np.allclose(hsv[..., 2], 1.0, atol=1e-4))

    def test_hue_interval_can_cross_zero(self):
        hue = np.array([0.0, 355.0, 180.0, 10.0], dtype=np.float32)
        inside = hue_in_range(hue, 350.0, 15.0)
        self.assertEqual(inside.tolist(), [True, True, False, True])


class FourierTests(unittest.TestCase):
    def test_transform_roundtrip(self):
        rng = np.random.default_rng(0)
        image = rng.random((48, 50), dtype=np.float32)
        recovered = reconstruct(fft2_shifted(image))
        self.assertTrue(np.allclose(image, recovered, atol=1e-4))

    def test_lowpass_keeps_dc_and_highpass_removes_it(self):
        shape = (32, 40)
        low = butterworth_lowpass(shape, 0.2, 2)
        high = butterworth_highpass(shape, 0.2, 2)
        center = (shape[0] // 2, shape[1] // 2)
        self.assertAlmostEqual(float(low[center]), 1.0, places=5)
        self.assertAlmostEqual(float(high[center]), 0.0, places=5)
        # Half-power: H(D0) = 1/2. The cutoff fraction is relative to the corner.
        ideal = ideal_lowpass(shape, 0.25)
        self.assertGreater(float(ideal.sum()), 1.0)
        self.assertLess(float(ideal.mean()), 0.5)

    def test_lowpass_reduces_a_checkerboard(self):
        checker = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.float32)
        spectrum = fft2_shifted(checker)
        from watervision.processing.fourier import apply_transfer

        filtered = reconstruct(apply_transfer(spectrum, butterworth_lowpass(checker.shape, 0.08, 4)))
        self.assertLess(float(filtered.var()), float(checker.var()) * 0.25)


class MorphologyTests(unittest.TestCase):
    def test_opening_removes_a_speck_and_keeps_a_block(self):
        mask = np.zeros((24, 24), dtype=bool)
        mask[2, 2] = True
        mask[8:20, 8:20] = True
        opened = opening(mask, 1)
        self.assertFalse(opened[2, 2])
        self.assertTrue(opened[12, 12])

    def test_fill_holes_keeps_a_bay_open(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:16, 4:16] = True
        mask[6:14, 6:14] = False
        self.assertTrue(fill_holes(mask)[10, 10])

    def test_small_components_are_removed(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[1:3, 1:4] = True  # 6 pixels
        mask[10:20, 10:20] = True  # 100 pixels
        cleaned = remove_small_regions(mask, 10)
        self.assertEqual(int(cleaned.sum()), 100)


class MetricTests(unittest.TestCase):
    def test_percentage_and_missing_calibration(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[:25, :] = True
        params = default_params()
        stats = estimate_mask(mask, params)
        self.assertAlmostEqual(stats["percentage"], 25.0)
        self.assertIsNone(stats["physical_area_m2"])
        self.assertIn("not a ground area", stats["note"])

    def test_physical_area_uses_both_pixel_sides(self):
        params = default_params()
        params["use_calibration"] = True
        params["pixel_width_m"] = 2.0
        params["pixel_height_m"] = 0.5
        self.assertAlmostEqual(calibration_area(10, params), 10.0)

    def test_calibration_rejects_non_positive_size(self):
        params = default_params()
        params["use_calibration"] = True
        params["pixel_width_m"] = 0.0
        params["pixel_height_m"] = 1.0
        with self.assertRaises(ValueError):
            calibration_area(5, params)

    def test_perfect_and_empty_scores(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:8, 2:8] = True
        perfect = evaluate_mask(mask, mask)
        self.assertAlmostEqual(perfect["iou"], 1.0)
        self.assertAlmostEqual(perfect["f1"], 1.0)
        empty = evaluate_mask(np.zeros_like(mask), np.zeros_like(mask))
        self.assertAlmostEqual(empty["iou"], 1.0)

    def test_iou_hand_count(self):
        prediction = np.zeros((10, 10), dtype=bool)
        reference = np.zeros((10, 10), dtype=bool)
        prediction[:5, :5] = True  # 25
        reference[:5, :6] = True  # 30, intersection 25
        scores = evaluate_mask(prediction, reference)
        self.assertAlmostEqual(scores["iou"], 25 / 30)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["recall"], 25 / 30)


class DemoTests(unittest.TestCase):
    def test_methods_separate_smooth_sand_from_water(self):
        rgb, truth = make_demo_scene()
        regions = demo_regions()
        result = segment(rgb, default_params())
        hybrid = result.clean["hybrid"]
        fourier = result.clean["fourier"]
        hsv = result.clean["hsv"]
        lake = float(hybrid[regions["lake"]].mean())
        sand_hybrid = float(hybrid[regions["sand"]].mean())
        sand_fourier = float(fourier[regions["sand"]].mean())
        sand_hsv = float(hsv[regions["sand"]].mean())
        self.assertGreater(lake, 0.75, msg=f"lake coverage {lake:.3f}")
        self.assertLess(sand_hybrid, 0.25, msg=f"hybrid sand {sand_hybrid:.3f}")
        self.assertGreater(sand_fourier, 0.55, msg=f"fourier sand {sand_fourier:.3f}")
        self.assertLess(sand_hsv, 0.2, msg=f"hsv sand {sand_hsv:.3f}")
        self.assertGreater(sand_fourier, sand_hybrid)
        scores = evaluate_mask(hybrid, truth)
        self.assertGreater(scores["f1"], 0.7)
        # Frequency alone is a worse description of this scene than the hybrid.
        fourier_scores = evaluate_mask(fourier, truth)
        self.assertGreater(scores["iou"], fourier_scores["iou"])

    def test_enclosed_green_land_is_not_filled_as_water(self):
        # A blue field with a green block in the middle is the globe case:
        # hole filling would paint the block, but that block is not blue.
        rgb = np.zeros((80, 80, 3), dtype=np.uint8)
        rgb[..., 2] = 180
        rgb[..., 0] = 20
        rgb[..., 1] = 40
        rgb[25:55, 25:55] = (40, 160, 40)
        result = segment(rgb, default_params())
        land = result.clean["hybrid"][25:55, 25:55]
        ocean = result.clean["hybrid"][5:20, 5:20]
        self.assertLess(float(land.mean()), 0.05)
        self.assertGreater(float(ocean.mean()), 0.9)


class ReportTests(unittest.TestCase):
    def test_csv_and_pdf_roundtrip_files(self):
        rgb, _truth = make_demo_scene(shape=(80, 100), seed=1)
        view = {
            "working": rgb,
            "filename": "demo.png",
            "params": default_params(),
            "metrics": {},
            "evaluation": {},
            "masks": {},
            "spectrum_rgb": None,
            "filtered_rgb": None,
            "filter_label": "",
            "ground_truth_name": "",
            "ground_truth_resized": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "out.csv"
            pdf_path = Path(folder) / "out.pdf"
            write_csv(str(csv_path), view)
            write_pdf(str(pdf_path), view)
            text = csv_path.read_text(encoding="utf-8-sig")
            self.assertIn("water_percentage", text)
            self.assertGreater(pdf_path.stat().st_size, 1000)

    def test_preprocess_identity_matches_the_input(self):
        rgb, _truth = make_demo_scene(shape=(40, 50), seed=2)
        params = default_params()
        out = preprocess(rgb, params)
        self.assertTrue(np.array_equal(out, rgb))


if __name__ == "__main__":
    unittest.main()
