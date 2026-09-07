import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imputeformer.models import ImputeFormerModel  # noqa: E402
from risk_pipeline.sensor_imputation import align_with_pretrained_imputeformer  # noqa: E402
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS  # noqa: E402
from risk_pipeline.temporal_context import relative_time_seconds  # noqa: E402


class _InterpolationWithResidual(torch.nn.Module):
    def __init__(self, residual=0.5, time_aware=True):
        super().__init__()
        self.residual = torch.nn.Parameter(torch.tensor(residual))
        self.time_aware = time_aware

    def forward(self, x, temporal, mask):
        baseline = (
            ImputeFormerModel.time_aware_components(x, temporal, mask)[0]
            if self.time_aware else ImputeFormerModel.visible_interpolation(x, mask)
        )
        return baseline + self.residual


class TimeAwareGlobalContextTests(unittest.TestCase):
    def _frame(self, length=19, irregular=False):
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        if irregular:
            gaps = np.resize(np.array([200_000, 1_000_000_000, 3_600_000_000_000]), length - 1)
            index = base + pd.to_timedelta(np.concatenate([[0], gaps.cumsum()]), unit="ns")
        else:
            index = pd.date_range(base, periods=length, freq="s")
        return pd.DataFrame(np.zeros((length, 5)), index=index, columns=OCEAN_SENSOR_IDS)

    def _infer(self, frame, time_aware=True, residual=0.5, mean=None, std=None, holdout_ratio=0.0):
        checkpoint = {
            "model_config": {
                "window": 6, "inference_stride": 3, "variable_length": True,
                "time_aware": time_aware,
            },
            "normalization": {
                "mean": np.zeros(5) if mean is None else mean,
                "std": np.ones(5) if std is None else std,
            },
            "temporal_encoding": "bounded_log_interval_v1",
            "supported_sampling_interval_seconds": {"minimum": 0.0, "maximum": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(checkpoint, path)
            return align_with_pretrained_imputeformer(
                frame, path, preloaded_model=_InterpolationWithResidual(residual, time_aware),
                preloaded_checkpoint=checkpoint, evaluation_holdout_ratio=holdout_ratio,
            )

    def test_long_outage_uses_distant_endpoints_and_actual_irregular_time(self):
        frame = self._frame(irregular=True)
        frame.iloc[:, 0] = np.nan
        frame.iloc[0, 0], frame.iloc[-1, 0] = 2.0, 20.0

        result, metadata = self._infer(frame)

        elapsed = relative_time_seconds(frame.index)
        expected = np.interp(elapsed, elapsed[[0, -1]], [2.0, 20.0])
        np.testing.assert_allclose(result.iloc[:, 0], expected, rtol=1e-12, atol=1e-12)
        pd.testing.assert_index_equal(result.index, frame.index)
        self.assertEqual(metadata["window_fusion"], "global_visible_time_baseline_with_context_matched_residual_mean")

    def test_local_residual_is_retained_when_window_has_both_endpoints(self):
        frame = self._frame()
        frame.iloc[:, 0] = np.arange(len(frame), dtype="float64")
        frame.iloc[8:10, 0] = np.nan

        result, _ = self._infer(frame)

        np.testing.assert_allclose(result.iloc[8:10, 0], [8.5, 9.5], atol=1e-6)
        visible = frame.notna().to_numpy()
        np.testing.assert_array_equal(result.to_numpy()[visible], frame.to_numpy()[visible])

    def test_single_distant_observation_is_carried_into_windows_without_context(self):
        for observed_position in (0, 18):
            with self.subTest(observed_position=observed_position):
                frame = self._frame()
                frame.iloc[:, 0] = np.nan
                frame.iloc[observed_position, 0] = 17.0
                result, _ = self._infer(frame)
                self.assertEqual(result.iloc[9, 0], 17.0)
                self.assertEqual(result.iloc[observed_position, 0], 17.0)

    def test_globally_unobserved_sensor_keeps_learned_residual_over_checkpoint_mean(self):
        frame = self._frame()
        frame.iloc[:, 0] = np.nan
        result, metadata = self._infer(frame, mean=np.array([10, 0, 0, 0, 0]), std=np.array([2, 1, 1, 1, 1]))
        np.testing.assert_allclose(result.iloc[:, 0], 11.0)
        self.assertIn(OCEAN_SENSOR_IDS[0], metadata["no_observed_sensor_ids"])

    def test_evaluation_baseline_cannot_see_hidden_truth(self):
        frame = self._frame()
        frame.iloc[:, 0] = np.nan
        frame.iloc[[0, 9, 18], 0] = [0.0, 100.0, 0.0]
        holdout = np.zeros(frame.shape, dtype=bool)
        holdout[9, 0] = True
        with patch("risk_pipeline.sensor_imputation.deterministic_observed_holdout", return_value=holdout):
            result, metadata = self._infer(frame, holdout_ratio=0.1)
        self.assertEqual(result.iloc[9, 0], 100.0)
        sensor_metric = next(row for row in metadata["evaluation_metrics"] if row["sensor_id"] == OCEAN_SENSOR_IDS[0])
        self.assertEqual(sensor_metric["MAE"], 100.0)

    def test_legacy_window_behavior_remains_unchanged(self):
        frame = self._frame()
        frame.iloc[:, 0] = np.nan
        frame.iloc[0, 0], frame.iloc[-1, 0] = 2.0, 20.0
        result, metadata = self._infer(frame, time_aware=False, residual=0.0)
        self.assertEqual(result.iloc[9, 0], 0.0)
        self.assertEqual(metadata["window_fusion"], "observed_context_priority_mean")


if __name__ == "__main__":
    unittest.main()
