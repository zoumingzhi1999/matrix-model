import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from imputeformer.models import ImputeFormerModel  # noqa: E402
from risk_pipeline.sensor_imputation import (  # noqa: E402
    _build_model,
    align_with_pretrained_imputeformer,
    load_sensor_imputeformer_checkpoint,
)
from risk_pipeline.sensor_schema import OCEAN_SENSOR_IDS  # noqa: E402
from risk_pipeline.temporal_context import (  # noqa: E402
    RAW_TEMPORAL_FEATURE_NAMES,
    raw_temporal_features,
    relative_time_seconds,
)


class VisibleMedianCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _model(self, calibrated=True, zero_readout=False, no_context_prior=None):
        config = {
            "window": 8, "inference_stride": 4, "variable_length": True,
            "time_aware": True, "static_sensor_embedding": True,
            "use_interpolation_residual": True,
            "num_nodes": 5, "input_dim": 13, "output_dim": 1,
            "input_embedding_dim": 4, "learnable_embedding_dim": 4,
            "feed_forward_dim": 8, "num_temporal_heads": 2,
            "num_layers": 1, "dim_proj": 2, "dropout": 0.0,
        }
        if calibrated:
            config["visible_median_calibration"] = True
        if no_context_prior is not None:
            config["no_context_prior"] = no_context_prior
        torch.manual_seed(319)
        model = _build_model(config).cpu().eval()
        if calibrated:
            model.context_median_gain.copy_(torch.arange(15).reshape(5, 3) / 20.0)
        if zero_readout:
            torch.nn.init.zeros_(model.readout.net[-1].weight)
            torch.nn.init.zeros_(model.readout.net[-1].bias)
        return model, config

    def _checkpoint(self, model, config):
        return {
            "format_version": 5,
            "model_type": "ImputeFormerModel", "entity_type": "sensor",
            "uses_site_entity": False, "uses_coordinates": False,
            "sensor_ids": list(OCEAN_SENSOR_IDS),
            "model_config": config, "model_state_dict": model.state_dict(),
            "normalization": {"mean": np.zeros(5), "std": np.ones(5)},
            "temporal_feature_names": RAW_TEMPORAL_FEATURE_NAMES,
            "supported_sampling_interval_seconds": {"minimum": 0.0, "maximum": None},
        }

    def _input(self):
        values = torch.arange(2 * 9 * 5, dtype=torch.float32).reshape(2, 9, 5, 1) / 4.0
        mask = torch.zeros_like(values, dtype=torch.bool)
        for sensor, positions in enumerate(((1, 3, 5, 7), (2, 4, 6), (1, 7), (4,), ())):
            mask[:, positions, sensor] = True
        index = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(
            [0, 0.0002, 1, 5, 7, 30, 60, 3600, 4000], unit="s",
        )
        temporal = torch.from_numpy(raw_temporal_features(index)).unsqueeze(0).expand(2, -1, -1)
        return values, temporal, mask

    def test_visible_median_matches_numpy_for_even_odd_single_and_empty_counts(self):
        values, _, mask = self._input()
        expected = np.zeros((2, 1, 5, 1), dtype="float64")
        for batch in range(2):
            for sensor in range(5):
                visible = values[batch, :, sensor, 0][mask[batch, :, sensor, 0]].numpy()
                if len(visible):
                    expected[batch, 0, sensor, 0] = np.median(visible)
        actual = ImputeFormerModel.visible_median(values.double(), mask)
        self.assertEqual(actual.shape, (2, 1, 5, 1))
        self.assertEqual(actual.dtype, torch.float64)
        np.testing.assert_array_equal(actual.numpy(), expected)
        even_pair = torch.tensor([1.0, 9.0]).reshape(1, 2, 1, 1)
        self.assertEqual(ImputeFormerModel.visible_median(even_pair, torch.ones_like(even_pair)).item(), 5.0)

    def test_hidden_nonfinite_values_cannot_change_median_or_calibrated_predictions(self):
        model, _ = self._model()
        values, temporal, mask = self._input()
        with torch.inference_mode():
            expected_median = model.visible_median(values, mask)
            expected_prediction = model(values, temporal, mask.float())
            for hidden in (float("nan"), float("inf"), -float("inf"), -1e12, 1e12):
                with self.subTest(hidden=hidden):
                    changed = values.masked_fill(~mask, hidden)
                    median = model.visible_median(changed, mask)
                    prediction = model(changed, temporal, mask.float())
                    torch.testing.assert_close(median, expected_median, rtol=0, atol=0)
                    torch.testing.assert_close(prediction, expected_prediction, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_all_missing_median_and_zero_readout_prediction_are_finite_zero(self):
        model, _ = self._model(zero_readout=True)
        for length in (1, 9):
            with self.subTest(length=length), torch.inference_mode():
                values = torch.full((2, length, 5, 1), float("nan"))
                mask = torch.zeros_like(values)
                index = pd.date_range("2026-01-01", periods=length, freq="h", tz="UTC")
                temporal = torch.from_numpy(raw_temporal_features(index)).unsqueeze(0).expand(2, -1, -1)
                torch.testing.assert_close(model.visible_median(values, mask), torch.zeros((2, 1, 5, 1)))
                torch.testing.assert_close(model(values, temporal, mask), torch.zeros_like(values))

    def test_fewer_than_three_visible_values_leave_uncalibrated_output_unchanged(self):
        calibrated, _ = self._model()
        uncalibrated, _ = self._model(calibrated=False)
        calibrated.context_median_gain.fill_(0.8)
        values, temporal, mask = self._input()
        mask.zero_()
        for sensor, positions in enumerate(((), (4,), (1, 7), (2,), (2, 6))):
            mask[:, positions, sensor] = True
        with torch.inference_mode():
            original = uncalibrated(values, temporal, mask.float())
            corrected = calibrated(values, temporal, mask.float())
        torch.testing.assert_close(corrected, original, rtol=0, atol=0)

    def test_calibrated_forward_adds_context_gain_times_numpy_median_minus_baseline(self):
        calibrated, _ = self._model()
        uncalibrated, _ = self._model(calibrated=False)
        values, temporal, mask = self._input()
        correction = np.zeros(values.shape, dtype="float64")
        gain = calibrated.context_median_gain.numpy()
        elapsed = temporal[0, :, 0].numpy()
        for batch in range(2):
            for sensor in range(5):
                selected = np.flatnonzero(mask[batch, :, sensor, 0].numpy())
                if len(selected) < 3:
                    continue
                observed = values[batch, selected, sensor, 0].numpy()
                baseline = np.interp(elapsed, elapsed[selected], observed)
                for position in range(len(elapsed)):
                    context = int((selected <= position).any()) + int((selected >= position).any())
                    correction[batch, position, sensor, 0] = gain[sensor, context] * (np.median(observed) - baseline[position])
        with torch.inference_mode():
            original = uncalibrated(values, temporal, mask.float()).numpy()
            corrected = calibrated(values, temporal, mask.float()).numpy()
        self.assertGreater(np.abs(correction).max(), 0.0)
        np.testing.assert_allclose(corrected, original + correction, rtol=1e-6, atol=1e-6)

    def test_production_long_gap_keeps_global_baseline_when_local_median_would_shrink(self):
        model, config = self._model(zero_readout=True)
        model.context_median_gain.fill_(0.75)
        offsets_ns = np.concatenate([[0], np.cumsum(np.resize(
            np.array([200_000, 1_000_000_000, 3_600_000_000_000], dtype="int64"), 30,
        ))])
        index = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(offsets_ns, unit="ns")
        frame = pd.DataFrame(np.zeros((31, 5)), index=index, columns=OCEAN_SENSOR_IDS)
        frame.iloc[:, 0] = np.nan
        positions = [0, 1, 2, 28, 29, 30]
        frame.iloc[positions, 0] = [1.0, 3.0, 5.0, 40.0, 45.0, 50.0]
        local = frame.iloc[:8]
        with torch.inference_mode():
            local_prediction = model(
                torch.from_numpy(local.fillna(0).to_numpy(dtype="float32", copy=True)[None, :, :, None]),
                torch.from_numpy(raw_temporal_features(local.index)[None]),
                torch.from_numpy(local.notna().to_numpy(dtype="float32", copy=True)[None, :, :, None]),
            )
        self.assertAlmostEqual(local_prediction[0, 3, 0, 0].item(), 3.5)
        checkpoint = self._checkpoint(model, config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "median.pt"
            torch.save(checkpoint, path)
            result, _ = align_with_pretrained_imputeformer(
                frame, path, device="cpu", preloaded_model=model,
                preloaded_checkpoint=checkpoint,
            )
        elapsed = relative_time_seconds(index)
        expected = np.interp(elapsed, elapsed[positions], frame.iloc[positions, 0])
        np.testing.assert_allclose(result.iloc[:, 0], expected, rtol=1e-12, atol=1e-12)
        visible = frame.notna().to_numpy()
        np.testing.assert_array_equal(result.to_numpy()[visible], frame.to_numpy()[visible])

    def test_checkpoint_roundtrip_preserves_gain_and_accepts_legacy_absent_flag(self):
        values, temporal, mask = self._input()
        with tempfile.TemporaryDirectory() as temporary:
            for calibrated in (False, True):
                with self.subTest(calibrated=calibrated):
                    model, config = self._model(calibrated=calibrated)
                    checkpoint = self._checkpoint(model, config)
                    path = Path(temporary) / f"median_{calibrated}.pt"
                    torch.save(checkpoint, path)
                    loaded, saved = load_sensor_imputeformer_checkpoint(path, device="cpu")
                    self.assertEqual(loaded.visible_median_calibration, calibrated)
                    self.assertEqual(hasattr(loaded, "context_median_gain"), calibrated)
                    if calibrated:
                        self.assertEqual(tuple(loaded.context_median_gain.shape), (5, 3))
                        torch.testing.assert_close(loaded.context_median_gain, model.context_median_gain)
                        self.assertIn("context_median_gain", saved["model_state_dict"])
                        self.assertNotIn("context_median_gain", dict(loaded.named_parameters()))
                    else:
                        self.assertNotIn("visible_median_calibration", saved["model_config"])
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            loaded(values, temporal, mask.float()),
                            model(values, temporal, mask.float()), rtol=0, atol=0,
                        )

    def test_no_context_prior_applies_only_to_sensors_without_any_visible_value(self):
        prior = [-0.5, 1.25, 4.0, -0.75, 2.5]
        model, _ = self._model(zero_readout=True, no_context_prior=prior)
        values, temporal, mask = self._input()
        values.fill_(float("nan"))
        mask.zero_()
        expected = torch.tensor(prior).view(1, 1, 5, 1).expand_as(values).clone()
        with torch.inference_mode():
            all_missing = model(values, temporal, mask.float())
            torch.testing.assert_close(all_missing, expected, rtol=0, atol=0)
            mask[:, 4, 2] = True
            values[:, 4, 2] = 4.25
            expected[:, :, 2] = 4.25
            one_visible_sensor = model(values, temporal, mask.float())
        torch.testing.assert_close(one_visible_sensor, expected, rtol=0, atol=0)

    def test_production_removes_local_prior_when_full_input_has_distant_observations(self):
        model, config = self._model(zero_readout=True, no_context_prior=[99.0, 2.0, 3.0, 4.0, 5.0])
        index = pd.date_range("2026-01-01", periods=31, freq="200us", tz="UTC")
        frame = pd.DataFrame(np.zeros((31, 5)), index=index, columns=OCEAN_SENSOR_IDS)
        frame.iloc[:, :2] = np.nan
        frame.iloc[0, 0], frame.iloc[-1, 0] = 10.0, 20.0
        checkpoint = self._checkpoint(model, config)
        checkpoint["normalization"] = {
            "mean": np.array([0.0, 100.0, 0.0, 0.0, 0.0]),
            "std": np.array([1.0, 3.0, 1.0, 1.0, 1.0]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prior.pt"
            torch.save(checkpoint, path)
            result, _ = align_with_pretrained_imputeformer(
                frame, path, device="cpu", preloaded_model=model,
                preloaded_checkpoint=checkpoint,
            )
        elapsed = relative_time_seconds(index)
        expected = np.interp(elapsed, elapsed[[0, -1]], [10.0, 20.0])
        np.testing.assert_allclose(result.iloc[:, 0], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result.iloc[:, 1], np.full(31, 106.0))
        self.assertEqual(result.iloc[0, 0], 10.0)
        self.assertEqual(result.iloc[-1, 0], 20.0)

    def test_no_context_prior_checkpoint_roundtrip_and_legacy_none_load_strictly(self):
        values, temporal, mask = self._input()
        values.fill_(float("nan"))
        mask.zero_()
        with tempfile.TemporaryDirectory() as temporary:
            for prior in (None, [-1.0, -0.5, 0.0, 0.5, 1.0]):
                with self.subTest(prior=prior):
                    model, config = self._model(zero_readout=True, no_context_prior=prior)
                    config["no_context_prior"] = prior
                    checkpoint = self._checkpoint(model, config)
                    path = Path(temporary) / f"prior_{prior is not None}.pt"
                    torch.save(checkpoint, path)
                    loaded, saved = load_sensor_imputeformer_checkpoint(path, device="cpu")
                    if prior is None:
                        self.assertIsNone(loaded.no_context_prior)
                        self.assertNotIn("no_context_prior", saved["model_state_dict"])
                    else:
                        torch.testing.assert_close(loaded.no_context_prior.flatten(), torch.tensor(prior))
                        self.assertNotIn("no_context_prior", dict(loaded.named_parameters()))
                        self.assertIn("no_context_prior", saved["model_state_dict"])
                    with torch.inference_mode():
                        torch.testing.assert_close(
                            loaded(values, temporal, mask.float()),
                            model(values, temporal, mask.float()), rtol=0, atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
