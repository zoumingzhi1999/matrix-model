import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from imputeformer.models.ImputeFormer import ImputeFormerModel
from risk_pipeline.temporal_context import raw_temporal_features


class TimeAwareImputeFormerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_model(self, static_sensor_embedding=False):
        torch.manual_seed(73)
        return ImputeFormerModel(
            num_nodes=5,
            input_dim=13,
            input_embedding_dim=8,
            learnable_embedding_dim=8,
            feed_forward_dim=16,
            num_temporal_heads=2,
            num_layers=1,
            dropout=0.0,
            windows=32,
            dim_proj=4,
            variable_length=True,
            use_interpolation_residual=True,
            time_aware=True,
            static_sensor_embedding=static_sensor_embedding,
        ).cpu().eval()

    def make_temporal(self, offsets, batch_size=1):
        index = pd.Timestamp("2026-09-06T12:00:00+08:00") + pd.to_timedelta(
            np.asarray(offsets, dtype="float64"), unit="s"
        )
        temporal = torch.from_numpy(raw_temporal_features(index))
        self.assertEqual(temporal.dtype, torch.float64)
        self.assertEqual(temporal.shape, (len(offsets), 6))
        return temporal.unsqueeze(0).expand(batch_size, -1, -1)

    def make_sparse_input(self):
        values = torch.full((2, 7, 5, 1), 99999.0)
        visible = torch.zeros_like(values, dtype=torch.bool)
        visible[:, 1, :4] = True
        visible[:, 4, :4] = True
        values[:, 1, :4] = 2.0
        values[:, 4, :4] = 8.0
        temporal = self.make_temporal([0, 1, 2, 6, 10, 20, 100], batch_size=2)
        return values, temporal, visible

    def test_irregular_interpolation_uses_elapsed_seconds(self):
        values = torch.full((1, 3, 5, 1), float("nan"))
        visible = torch.zeros_like(values, dtype=torch.bool)
        values[:, 0] = 0.0
        values[:, 2] = 10.0
        visible[:, [0, 2]] = True
        temporal = self.make_temporal([0, 1, 10])

        baseline, _, _ = ImputeFormerModel.time_aware_components(
            values, temporal, visible
        )
        expected = torch.tensor([0.0, 1.0, 10.0])
        torch.testing.assert_close(baseline[0, :, 0, 0], expected, rtol=0, atol=0)

    def test_zero_readout_returns_true_time_interpolation_at_different_rates(self):
        model = self.make_model()
        torch.nn.init.zeros_(model.readout.net[-1].weight)
        torch.nn.init.zeros_(model.readout.net[-1].bias)
        values = torch.full((1, 5, 5, 1), float("inf"))
        visible = torch.zeros_like(values, dtype=torch.bool)
        values[:, 1, :4] = 2.0
        values[:, 3, :4] = 8.0
        visible[:, [1, 3], :4] = True
        expected = torch.tensor([2.0, 2.0, 3.0, 8.0, 8.0])

        with torch.inference_mode():
            for interval in (0.0001, 1, 2, 5, 10, 30, 60, 3600):
                with self.subTest(interval=interval):
                    temporal = self.make_temporal(
                        np.array([0, 1, 2, 7, 20], dtype="float64") * interval
                    )
                    prediction = model(values, temporal, visible.float())
                    torch.testing.assert_close(prediction[0, :, 0, 0], expected)
                    torch.testing.assert_close(
                        prediction[:, :, 4], torch.zeros((1, 5, 1)), rtol=0, atol=0
                    )

    def test_time_gaps_change_context_even_when_values_do_not_change(self):
        values = torch.full((1, 3, 5, 1), 4.0)
        visible = torch.ones_like(values, dtype=torch.bool)
        visible[:, 1] = False
        early = ImputeFormerModel.time_aware_components(
            values, self.make_temporal([0, 1, 10]), visible
        )
        later = ImputeFormerModel.time_aware_components(
            values, self.make_temporal([0, 5, 10]), visible
        )

        torch.testing.assert_close(early[0], later[0], rtol=0, atol=0)
        torch.testing.assert_close(early[2], later[2], rtol=0, atol=0)
        for feature_index in (2, 3, 4, 8):
            with self.subTest(feature_index=feature_index):
                self.assertFalse(torch.equal(
                    early[1][:, 1, :, feature_index],
                    later[1][:, 1, :, feature_index],
                ))

    def test_context_and_local_normalization_ignore_hidden_values(self):
        values, temporal, visible = self.make_sparse_input()
        baseline, features, scale = ImputeFormerModel.time_aware_components(
            values, temporal, visible
        )
        torch.testing.assert_close(scale[:, :, :4], torch.full((2, 1, 4, 1), 3.0))
        torch.testing.assert_close(scale[:, :, 4], torch.ones((2, 1, 1)))
        torch.testing.assert_close(baseline[:, :, 4], torch.zeros((2, 7, 1)))
        expected_count = np.log1p(2.0) / (1.0 + np.log1p(2.0))
        torch.testing.assert_close(
            features[:, :, :4, 12], torch.full((2, 7, 4), expected_count)
        )
        torch.testing.assert_close(features[:, :, 4, 12], torch.zeros((2, 7)))

        for hidden_value in (-1e9, float("nan"), float("inf"), -float("inf")):
            with self.subTest(hidden_value=hidden_value):
                changed = values.masked_fill(~visible, hidden_value)
                actual = ImputeFormerModel.time_aware_components(changed, temporal, visible)
                for result, expected in zip(actual, (baseline, features, scale)):
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(result).all())

    def test_model_predictions_do_not_leak_hidden_targets(self):
        model = self.make_model()
        values, temporal, visible = self.make_sparse_input()
        with torch.inference_mode():
            reference = model(values, temporal, visible.float())
            for hidden_value in (-1e9, float("nan"), float("inf"), -float("inf")):
                with self.subTest(hidden_value=hidden_value):
                    changed = values.masked_fill(~visible, hidden_value)
                    actual = model(changed, temporal, visible.float())
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_model_preserves_requested_lengths_on_irregular_timestamps(self):
        model = self.make_model()
        with torch.inference_mode():
            for length in (1, 100, 256, 500, 1000, 2000, 3600, 3601):
                with self.subTest(length=length):
                    values = torch.randn((2, length, 5, 1))
                    visible = torch.rand(values.shape) > 0.6
                    offsets = np.cumsum(np.resize([0.0001, 1.0, 60.0], length))
                    temporal = self.make_temporal(offsets, batch_size=2)
                    prediction = model(values, temporal, visible.float())
                    self.assertEqual(prediction.shape, values.shape)
                    self.assertEqual(prediction.dtype, torch.float32)
                    self.assertEqual(prediction.device.type, "cpu")
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_all_missing_values_produce_finite_context_and_predictions(self):
        model = self.make_model()
        with torch.inference_mode():
            for length in (1, 100, 1000):
                with self.subTest(length=length):
                    values = torch.full((1, length, 5, 1), float("nan"))
                    visible = torch.zeros_like(values, dtype=torch.bool)
                    temporal = self.make_temporal(np.arange(length) * 3600.0)
                    baseline, features, scale = model.time_aware_components(
                        values, temporal, visible
                    )
                    torch.testing.assert_close(baseline, torch.zeros_like(values))
                    torch.testing.assert_close(scale, torch.ones((1, 1, 5, 1)))
                    self.assertTrue(torch.isfinite(features).all())
                    prediction = model(values, temporal, visible.float())
                    self.assertTrue(torch.isfinite(prediction).all())

    def test_static_sensor_embedding_supports_different_lengths_without_new_parameters(self):
        model = self.make_model(static_sensor_embedding=True)
        embedding = model.learnable_embedding
        self.assertEqual(embedding.shape, (1, 5, 8))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        with torch.inference_mode():
            for length in (1, 137, 3601):
                with self.subTest(length=length):
                    values = torch.randn((1, length, 5, 1))
                    visible = torch.rand(values.shape) > 0.6
                    offsets = np.cumsum(np.resize([0.0001, 2.0, 60.0], length))
                    prediction = model(values, self.make_temporal(offsets), visible.float())
                    self.assertEqual(prediction.shape, values.shape)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertIs(model.learnable_embedding, embedding)
                    self.assertEqual(embedding.shape, (1, 5, 8))
                    self.assertEqual(
                        sum(parameter.numel() for parameter in model.parameters()), parameter_count
                    )

    def test_static_sensor_embedding_accumulates_gradients_from_distant_times(self):
        model = self.make_model(static_sensor_embedding=True)
        values = torch.randn((1, 137, 5, 1))
        visible = torch.zeros_like(values, dtype=torch.bool)
        visible[:, ::3] = True
        temporal = self.make_temporal(np.cumsum(np.resize([0.1, 1.0, 10.0], 137)))
        prediction = model(values, temporal, visible.float())
        early_loss = prediction[:, 5].sum()
        late_loss = prediction[:, 131].sum()
        embedding = model.learnable_embedding

        early_gradient = torch.autograd.grad(early_loss, embedding, retain_graph=True)[0]
        late_gradient = torch.autograd.grad(late_loss, embedding, retain_graph=True)[0]
        combined_gradient = torch.autograd.grad(early_loss + late_loss, embedding)[0]
        for gradient in (early_gradient, late_gradient, combined_gradient):
            self.assertEqual(gradient.shape, (1, 5, 8))
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        torch.testing.assert_close(combined_gradient, early_gradient + late_gradient)

    def test_static_sensor_embedding_predictions_ignore_hidden_values(self):
        model = self.make_model(static_sensor_embedding=True)
        values, temporal, visible = self.make_sparse_input()
        with torch.inference_mode():
            reference = model(values, temporal, visible.float())
            for hidden_value in (-1e9, float("nan"), float("inf"), -float("inf")):
                with self.subTest(hidden_value=hidden_value):
                    changed = values.masked_fill(~visible, hidden_value)
                    actual = model(changed, temporal, visible.float())
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
