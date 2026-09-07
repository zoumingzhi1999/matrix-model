import hashlib
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


EXPERIMENTS = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from risk_pipeline.artifacts import (  # noqa: E402
    copy_file_once,
    write_csv_once,
    write_text_once,
)


class ArtifactImmutabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="artifact_immutability_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def assert_unchanged(self, path, original_bytes):
        current_bytes = path.read_bytes()
        self.assertEqual(current_bytes, original_bytes)
        self.assertEqual(
            hashlib.sha256(current_bytes).hexdigest(),
            hashlib.sha256(original_bytes).hexdigest(),
        )

    def test_csv_duplicate_write_preserves_original_file(self):
        path = self.root / "features" / "first_stage.csv"
        original = pd.DataFrame({"feature": [0.25, 1.5], "sensor": ["a", "b"]})
        replacement = pd.DataFrame({"feature": [-90.0], "sensor": ["replacement"]})
        write_csv_once(original, path)
        original_bytes = path.read_bytes()
        pd.testing.assert_frame_equal(pd.read_csv(path), original)

        with self.assertRaises(FileExistsError):
            write_csv_once(replacement, path)

        self.assert_unchanged(path, original_bytes)

    def test_text_duplicate_write_preserves_original_file(self):
        path = self.root / "risk" / "matrix.json"
        original = '{"total_risk": 0.25}\n'
        write_text_once(path, original)
        original_bytes = path.read_bytes()
        self.assertEqual(original_bytes, original.encode("utf-8"))

        with self.assertRaises(FileExistsError):
            write_text_once(path, '{"total_risk": 0.99}\n')

        self.assert_unchanged(path, original_bytes)

    def test_copy_duplicate_write_preserves_original_file(self):
        source = self.root / "source.bin"
        replacement = self.root / "replacement.bin"
        destination = self.root / "retained" / "source.bin"
        original_bytes = bytes(range(256)) * 16
        source.write_bytes(original_bytes)
        replacement.write_bytes(b"replacement content")
        copy_file_once(source, destination)
        self.assert_unchanged(destination, original_bytes)

        with self.assertRaises(FileExistsError):
            copy_file_once(replacement, destination)

        self.assert_unchanged(destination, original_bytes)

    def test_concurrent_csv_writes_have_one_complete_winner(self):
        path = self.root / "features" / "shared.csv"
        frames = [
            pd.DataFrame({"writer": [writer] * 2048, "value": range(2048)})
            for writer in ("first", "second")
        ]
        barrier = threading.Barrier(2, timeout=10)

        def attempt_write(writer):
            barrier.wait()
            try:
                write_csv_once(frames[writer], path)
            except FileExistsError:
                return None
            return writer

        with ThreadPoolExecutor(max_workers=2) as executor:
            attempts = [executor.submit(attempt_write, writer) for writer in range(2)]
            results = [attempt.result(timeout=20) for attempt in attempts]

        winners = [writer for writer in results if writer is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(results.count(None), 1)
        winner = frames[winners[0]]
        expected_bytes = winner.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
        self.assert_unchanged(path, expected_bytes)
        pd.testing.assert_frame_equal(pd.read_csv(path), winner)


if __name__ == "__main__":
    unittest.main()
