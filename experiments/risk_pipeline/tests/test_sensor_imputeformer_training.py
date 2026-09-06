import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from train_sensor_imputeformer import (
    MASK_PATTERNS,
    METRIC_NAMES,
    SENSOR_COLUMNS,
    _metrics_from_sums,
    apply_output_calibration,
    frequency_imputation_loss,
    load_saved_selection,
    load_ocean_sensor_splits,
    make_artificial_missing_mask,
    sliding_windows,
    window_starts,
)


class SensorImputeFormerTrainingTests(unittest.TestCase):
    def _dataset_run(self, root: Path, include_site: bool = False) -> Path:
        run_dir = root / "runs" / "20260905_000000_000000"
        split_dir = run_dir / "splits"
        split_dir.mkdir(parents=True)
        index = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
        values = np.arange(96 * len(SENSOR_COLUMNS), dtype="float64").reshape(
            96, len(SENSOR_COLUMNS)
        )
        frame = pd.DataFrame(values, columns=SENSOR_COLUMNS)
        frame.insert(0, "timestamp", index)
        if include_site:
            frame["site_id"] = "forbidden"
        boundaries = {"train": (0, 64), "validation": (64, 80), "test": (80, 96)}
        for name, (start, end) in boundaries.items():
            frame.iloc[start:end].to_csv(split_dir / f"{name}.csv", index=False)
        manifest = {
            "entity_type": "sensor",
            "uses_site_entity": False,
            "uses_coordinates": False,
            "model_columns": ["timestamp", *SENSOR_COLUMNS],
        }
        (run_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return run_dir

    def test_sliding_windows_preserve_every_available_window(self):
        values = np.arange(10 * 3, dtype="float32").reshape(10, 3)
        windows = sliding_windows(values, window=4)
        self.assertEqual(windows.shape, (7, 4, 3))
        np.testing.assert_array_equal(windows[0], values[:4])
        np.testing.assert_array_equal(windows[-1], values[-4:])

    def test_tiled_windows_cover_every_source_row(self):
        starts = window_starts(length=10, window=4, stride=4)
        covered = np.zeros(10, dtype=bool)
        for start in starts:
            covered[start : start + 4] = True
        self.assertTrue(covered.all())
        self.assertEqual(starts, [0, 4, 6])

    def test_single_fft_loss_is_equivalent_to_original_expression(self):
        generator = torch.Generator().manual_seed(7)
        prediction = torch.randn((2, 32, 5, 1), generator=generator)
        target = torch.randn((2, 32, 5, 1), generator=generator)
        mask = torch.rand((2, 32, 5, 1), generator=generator) > 0.7
        alpha = 0.05
        missing_rate = mask.float().mean()
        original = (
            alpha * missing_rate / (1.0 - missing_rate)
            * torch.abs(
                torch.fft.rfft(torch.where(mask, prediction, target), dim=1)
                - torch.fft.rfft(target, dim=1)
            ).mean()
        )
        optimized = frequency_imputation_loss(
            prediction, target, mask, alpha
        )
        torch.testing.assert_close(optimized, original)

    def test_all_mask_patterns_only_hide_observed_values(self):
        observed = torch.ones((8, 24, 5, 1), dtype=torch.bool)
        observed[:, 3:7, 2, :] = False
        for index, pattern in enumerate(MASK_PATTERNS):
            target = make_artificial_missing_mask(
                observed, np.random.default_rng(index), pattern
            )
            self.assertTrue(target.any())
            self.assertFalse((target & ~observed).any())

    def test_sensor_outage_selects_only_a_sensor_with_ground_truth(self):
        observed = torch.zeros((3, 24, 5, 1), dtype=torch.bool)
        observed[:, ::4, 3, :] = True
        target = make_artificial_missing_mask(
            observed,
            np.random.default_rng(11),
            "single_sensor_outage",
        )
        self.assertTrue(torch.equal(target, observed))

    def test_loads_exact_five_sensor_schema_without_site_or_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._dataset_run(root)
            splits, _, selected_run = load_ocean_sensor_splits(root)
        self.assertEqual(selected_run, run_dir.resolve())
        for frame in splits.values():
            self.assertEqual(tuple(frame.columns), SENSOR_COLUMNS)
        self.assertLess(splits["train"].index[-1], splits["validation"].index[0])
        self.assertLess(splits["validation"].index[-1], splits["test"].index[0])

    def test_rejects_any_site_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._dataset_run(root, include_site=True)
            with self.assertRaisesRegex(ValueError, "Site, station"):
                load_ocean_sensor_splits(root)

    def test_metrics_include_mae_mse_rmse_and_mre(self):
        metrics = _metrics_from_sums(
            absolute_error_sum=6.0,
            squared_error_sum=14.0,
            absolute_target_sum=12.0,
            count=3,
        )
        self.assertEqual(tuple(metrics), METRIC_NAMES)
        self.assertAlmostEqual(metrics["MAE"], 2.0)
        self.assertAlmostEqual(metrics["MSE"], 14.0 / 3.0)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(14.0 / 3.0))
        self.assertAlmostEqual(metrics["MRE"], 0.5)

    def test_resume_loads_saved_best_epoch_and_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history_path = root / "model_selection_history.csv"
            state_path = root / "best_selection_state.pt"
            pd.DataFrame([
                {"epoch": 1, "validation_normalized_MAE": 0.9},
                {"epoch": 2, "validation_normalized_MAE": 0.7},
                {"epoch": 3, "validation_normalized_MAE": 0.8},
            ]).to_csv(history_path, index=False)
            expected = torch.nn.Linear(2, 1)
            torch.save(expected.state_dict(), state_path)
            actual = torch.nn.Linear(2, 1)
            actual, history, best_epoch = load_saved_selection(
                actual, history_path, state_path
            )

        self.assertEqual(best_epoch, 2)
        self.assertEqual(len(history), 3)
        for name, value in expected.state_dict().items():
            torch.testing.assert_close(actual.state_dict()[name], value)

    def test_output_calibration_is_applied_per_sensor(self):
        prediction = np.ones((1, 2, len(SENSOR_COLUMNS), 1), dtype="float32")
        calibration = {
            sensor_id: {"scale": index + 1.0, "bias": index / 10.0}
            for index, sensor_id in enumerate(SENSOR_COLUMNS)
        }
        actual = apply_output_calibration(prediction, calibration)
        for index, sensor_id in enumerate(SENSOR_COLUMNS):
            np.testing.assert_allclose(
                actual[:, :, index, :], index + 1.0 + index / 10.0
            )


if __name__ == "__main__":
    unittest.main()
