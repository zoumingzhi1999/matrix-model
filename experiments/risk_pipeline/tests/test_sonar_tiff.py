import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))

from risk_pipeline.data_io import (  # noqa: E402
    load_sonar_tiff_image,
    resolve_sonar_source_paths,
)


class SonarTiffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_directory_discovers_only_tiff_in_filename_order(self):
        for name in ("z.xtf", "a.tif", "B.TIFF", "ignored.png", "notes.txt"):
            (self.root / name).write_bytes(b"input placeholder")
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "hidden.tif").write_bytes(b"nested placeholder")

        self.assertEqual(
            resolve_sonar_source_paths(self.root),
            [self.root / "B.TIFF", self.root / "a.tif"],
        )
        self.assertEqual(
            resolve_sonar_source_paths(self.root, allow_xtf=True),
            [self.root / "B.TIFF", self.root / "a.tif", self.root / "z.xtf"],
        )

    def test_explicit_non_tiff_requires_legacy_xtf_input(self):
        for suffix in (".png", ".jpg", ".xtf"):
            source = self.root / f"sonar{suffix}"
            source.write_bytes(b"input placeholder")
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "TIFF"):
                resolve_sonar_source_paths(source)
        self.assertEqual(
            resolve_sonar_source_paths(self.root / "sonar.xtf", allow_xtf=True),
            [self.root / "sonar.xtf"],
        )

    def test_missing_and_tiff_free_sources_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            resolve_sonar_source_paths(self.root / "missing.tif")
        (self.root / "sonar.xtf").write_bytes(b"legacy input")
        with self.assertRaisesRegex(ValueError, "no supported TIFF"):
            resolve_sonar_source_paths(self.root)

    def test_uint16_and_float32_preserve_intensities_and_dimensions(self):
        cases = (
            (np.asarray([[0, 1000, 32768], [65535, 2000, 4000]], dtype="uint16"), 65535.0),
            (np.asarray([[0.0, 0.1, 0.25], [0.5, 0.75, 1.0]], dtype="float32"), 1.0),
        )
        for pixels, divisor in cases:
            for suffix in (".tif", ".TiFF"):
                with self.subTest(dtype=pixels.dtype, suffix=suffix):
                    path = self.root / f"sonar{suffix}"
                    Image.fromarray(pixels).save(path)
                    result = load_sonar_tiff_image(path)
                    self.assertEqual(result.shape, pixels.shape)
                    self.assertEqual(result.dtype, np.dtype("float32"))
                    np.testing.assert_allclose(result, pixels.astype("float32") / divisor)

    def test_eight_bit_and_rgb_tiff_use_existing_grayscale_conversion(self):
        for mode in ("L", "RGB"):
            with self.subTest(mode=mode):
                path = self.root / f"sonar_{mode}.tif"
                original = Image.new(mode, (17, 11), 96 if mode == "L" else (30, 90, 150))
                original.save(path)
                expected = np.asarray(original.convert("L"), dtype="float32") / 255.0
                np.testing.assert_array_equal(load_sonar_tiff_image(path), expected)

    def test_float_non_finite_values_are_rejected(self):
        path = self.root / "sonar.tif"
        Image.fromarray(np.asarray([[0.0, np.nan]], dtype="float32")).save(path)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_sonar_tiff_image(path)

    def test_png_renamed_to_tiff_is_rejected(self):
        path = self.root / "renamed.tif"
        Image.new("L", (16, 16)).save(path, format="PNG")
        with self.assertRaisesRegex(ValueError, "contain TIFF data"):
            load_sonar_tiff_image(path)

    def test_multi_page_tiff_is_rejected_without_losing_pages(self):
        path = self.root / "multipage.tiff"
        Image.new("L", (16, 16), 20).save(
            path, save_all=True, append_images=[Image.new("L", (16, 16), 220)],
        )
        with self.assertRaisesRegex(ValueError, "Multi-page"):
            load_sonar_tiff_image(path)


if __name__ == "__main__":
    unittest.main()
