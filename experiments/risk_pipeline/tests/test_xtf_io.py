import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from risk_pipeline.xtf_io import load_xtf_sonar_image  # noqa: E402


class XtfIoTests(unittest.TestCase):
    def test_real_retained_xtf_decodes_all_pings_without_resize(self):
        matches = sorted(
            (ROOT / "data" / "multimodal_training" / "runs").glob(
                "*/raw/sonar/**/test_sample.xtf"
            )
        )
        if not matches:
            self.skipTest("Retained real XTF training sample is not available.")

        image, metadata = load_xtf_sonar_image(matches[-1])

        self.assertEqual(image.shape, (931, 4096))
        self.assertEqual(metadata["sonar_ping_count"], 931)
        self.assertEqual(metadata["channel_count"], 2)
        self.assertFalse(metadata["resize_applied"])
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)

    def test_non_xtf_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sonar.png"
            path.write_bytes(b"not-an-xtf")
            with self.assertRaisesRegex(ValueError, "raw .xtf"):
                load_xtf_sonar_image(path)

    def test_real_three_channel_echosounder_xtf_uses_structured_fallback(self):
        matches = sorted(
            (ROOT / "data" / "multimodal_training" / "runs").glob(
                "*/raw/sonar/zenodo_5558794/2018_10_02_11_25_53.xtf"
            )
        )
        if not matches:
            self.skipTest("Retained three-channel XTF dataset is not available.")

        image, metadata = load_xtf_sonar_image(matches[-1])

        self.assertEqual(image.shape, (7980, 4800))
        self.assertEqual(metadata["sonar_ping_count"], 7980)
        self.assertEqual(metadata["channel_count"], 3)
        self.assertEqual(
            metadata["decoder_mode"],
            "pyxtf_structured_multichannel_fallback",
        )
        self.assertEqual(
            {channel["channel_type"] for channel in metadata["channels"]},
            {"subbottom"},
        )
        self.assertFalse(metadata["resize_applied"])


if __name__ == "__main__":
    unittest.main()
