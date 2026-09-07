import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments"))

from risk_pipeline.data_io import resolve_image_source_paths  # noqa: E402


class ImageSourcePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_explicit_source_preserves_loader_validation(self):
        source = self.root / "source.custom"
        source.write_bytes(b"validated later by the source loader")

        with patch.object(Image, "open", side_effect=AssertionError("unexpected decode")):
            self.assertEqual(resolve_image_source_paths(str(source)), [source.resolve()])

    def test_directory_filters_formats_and_orders_filenames(self):
        names = ["z.XTF", "c.TiF", "B.PNG", "a.JpEg", "ignore.csv", "notes.txt"]
        for name in names:
            (self.root / name).write_bytes(b"source placeholder")
        nested = self.root / "nested.png"
        nested.mkdir()
        (nested / "hidden.png").write_bytes(b"nested source")

        optical = resolve_image_source_paths(self.root)
        sonar = resolve_image_source_paths(self.root, allow_xtf=True)

        self.assertEqual([path.name for path in optical], ["B.PNG", "a.JpEg", "c.TiF"])
        self.assertEqual([path.name for path in sonar], ["B.PNG", "a.JpEg", "c.TiF", "z.XTF"])
        self.assertTrue(all(path.is_absolute() for path in sonar))

    def test_directory_uses_registered_decoders(self):
        for name in ("input.custom", "output.saveonly"):
            (self.root / name).write_bytes(b"source placeholder")
        with patch.object(Image, "registered_extensions", return_value={
            ".custom": "CUSTOM_READER", ".saveonly": "CUSTOM_WRITER",
        }), patch.dict(Image.OPEN, {"CUSTOM_READER": object()}):
            paths = resolve_image_source_paths(self.root)
        self.assertEqual(paths, [(self.root / "input.custom").resolve()])

    def test_empty_or_metadata_only_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supported files"):
            resolve_image_source_paths(self.root)
        (self.root / "metadata.json").write_text("{}", encoding="ascii")
        (self.root / "records.csv").write_text("timestamp,value", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "no supported files"):
            resolve_image_source_paths(self.root)

    def test_xtf_only_directory_requires_sonar_mode(self):
        source = self.root / "sonar.XTF"
        source.write_bytes(b"XTF placeholder")
        with self.assertRaisesRegex(ValueError, "no supported files"):
            resolve_image_source_paths(self.root)
        self.assertEqual(resolve_image_source_paths(self.root, allow_xtf=True), [source.resolve()])

    def test_missing_source_is_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            resolve_image_source_paths(self.root / "missing.png")


if __name__ == "__main__":
    unittest.main()
