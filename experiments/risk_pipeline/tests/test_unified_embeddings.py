import io
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))

from multisource_unified import (  # noqa: E402
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    FeatureFusionTransformer,
    feature_input_dims,
)
from risk_pipeline.run_encapsulated_pipeline import (  # noqa: E402
    encode_second_stage_features,
    reconstruct_second_stage_features,
    run_transformer_stage,
)
from risk_pipeline.single_feature_risk import (  # noqa: E402
    build_second_stage_feature_risk_matrices,
)


class UnifiedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)
        self.model = self._model()
        rng = np.random.default_rng(285)
        self.frames = {}
        self.normalization = {}
        self.z_columns = [f"z_{index:03d}" for index in range(128)]
        for index, (modality, row_count) in enumerate(
            zip(FUSION_MODALITIES, (3, 5, 2))
        ):
            columns = list(FUSION_FEATURE_COLUMNS[modality])
            values = rng.normal(size=(row_count, len(columns))).astype("float32")
            values *= np.arange(1, len(columns) + 1, dtype="float32")
            values += index + 2
            frame = pd.DataFrame(values, columns=columns)
            frame.insert(0, "source", [f"{modality}_{row}" for row in range(row_count)])
            frame.insert(1, "source_row", list(reversed(range(row_count))))
            if modality == "time_series":
                frame.insert(0, "sensor_id", ["depth", "temperature", "current"])
            else:
                frame.insert(2, "tile_index", np.arange(row_count, dtype="int64") * 3)
            frame.index = np.arange(row_count, dtype="int64") * 2 + 7
            self.frames[modality] = frame
            self.normalization[modality] = {
                "columns": columns,
                "mean": [0.5 * (index + 1) * (column + 1) for column in range(len(columns))],
                "scale": [0.7 + 0.3 * column for column in range(len(columns))],
            }

    @staticmethod
    def _model(d_model=128):
        return FeatureFusionTransformer(
            feature_input_dims(),
            d_model=d_model,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
        ).eval()

    def _oracle(self):
        normalized = {}
        for modality, frame in self.frames.items():
            spec = self.normalization[modality]
            values = frame.loc[:, list(FUSION_FEATURE_COLUMNS[modality])].to_numpy(
                dtype="float32"
            )
            normalized[modality] = torch.tensor(
                (values - np.asarray(spec["mean"], dtype="float32"))
                / np.asarray(spec["scale"], dtype="float32")
            )

        expected = {}
        for modality_index, target in enumerate(FUSION_MODALITIES):
            vectors, tokens, decoded = [], [], []
            for row in range(len(self.frames[target])):
                inputs = {
                    modality: (
                        values[row:row + 1]
                        if modality == target
                        else values.mean(dim=0, keepdim=True)
                    )
                    for modality, values in normalized.items()
                }
                with torch.inference_mode():
                    output = self.model(inputs)
                vectors.append(output["unified"][0].numpy())
                tokens.append(output["tokens"][0, modality_index].numpy())
                decoded.append(output["unified_reconstruction"][target][0].numpy())
            spec = self.normalization[target]
            original_units = (
                np.stack(decoded) * np.asarray(spec["scale"], dtype="float32")
                + np.asarray(spec["mean"], dtype="float32")
            )
            expected[target] = (np.stack(vectors), np.stack(tokens), original_units)
        return expected

    @staticmethod
    def _risk_config():
        return {
            "calibration_feature_stage": "stage3_transformer_unified_reconstruction",
            "modalities": {
                modality: {
                    "selected_feature": FUSION_FEATURE_COLUMNS[modality][1],
                    "fixed_weight": weight,
                    "scaler_mean": 0.25,
                    "scaler_scale": 1.7,
                    "logistic_coefficient": 0.8,
                    "logistic_intercept": -0.3,
                }
                for modality, weight in zip(FUSION_MODALITIES, (0.2, 0.4, 0.4))
            },
        }

    def test_each_source_row_exports_cls_with_identifiers_and_original_order(self):
        expected = self._oracle()
        captured = []

        def capture_forward(model, inputs, output):
            captured.append(output["unified"].detach().cpu().numpy().copy())

        hook = self.model.register_forward_hook(capture_forward)
        try:
            embeddings, reconstructed = encode_second_stage_features(
                self.model, self.normalization, self.frames, batch_size=2
            )
        finally:
            hook.remove()

        self.assertEqual(set(embeddings), set(FUSION_MODALITIES))
        self.assertEqual(len(captured), sum(math.ceil(len(frame) / 2) for frame in self.frames.values()))
        captured_offset = 0
        for modality, source in self.frames.items():
            with self.subTest(modality=modality):
                metadata = source.drop(columns=list(FUSION_FEATURE_COLUMNS[modality]))
                self.assertEqual(
                    embeddings[modality].columns.tolist(), metadata.columns.tolist() + self.z_columns
                )
                pd.testing.assert_frame_equal(
                    embeddings[modality].loc[:, metadata.columns],
                    metadata.reset_index(drop=True),
                )
                pd.testing.assert_frame_equal(
                    reconstructed[modality].loc[:, metadata.columns],
                    metadata.reset_index(drop=True),
                )
                exported = pd.read_csv(io.StringIO(embeddings[modality].to_csv(index=False)))
                actual = exported.loc[:, self.z_columns].to_numpy(dtype="float32")
                np.testing.assert_allclose(actual, expected[modality][0], rtol=1e-5, atol=1e-6)
                self.assertFalse(np.allclose(actual, expected[modality][1], rtol=1e-5, atol=1e-6))
                batch_count = math.ceil(len(source) / 2)
                from_same_forward = np.concatenate(
                    captured[captured_offset:captured_offset + batch_count]
                )
                np.testing.assert_array_equal(actual, from_same_forward)
                captured_offset += batch_count
                np.testing.assert_allclose(
                    reconstructed[modality].loc[:, list(FUSION_FEATURE_COLUMNS[modality])],
                    expected[modality][2],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_export_is_invariant_to_batch_size_for_unequal_source_lengths(self):
        embeddings_one, decoded_one = encode_second_stage_features(
            self.model, self.normalization, self.frames, batch_size=1
        )
        embeddings_two, decoded_two = encode_second_stage_features(
            self.model, self.normalization, self.frames, batch_size=2
        )
        for modality in FUSION_MODALITIES:
            with self.subTest(modality=modality):
                pd.testing.assert_frame_equal(
                    embeddings_one[modality], embeddings_two[modality],
                    check_exact=False, rtol=1e-5, atol=1e-6,
                )
                pd.testing.assert_frame_equal(
                    decoded_one[modality], decoded_two[modality],
                    check_exact=False, rtol=1e-5, atol=1e-6,
                )

    def test_exported_vectors_decode_to_the_existing_calibrated_risk(self):
        embeddings, reconstructed = encode_second_stage_features(
            self.model, self.normalization, self.frames, batch_size=2
        )
        expected = self._oracle()
        independently_decoded = {}
        oracle_reconstructed = {}
        for modality in FUSION_MODALITIES:
            exported = pd.read_csv(io.StringIO(embeddings[modality].to_csv(index=False)))
            vectors = torch.tensor(exported.loc[:, self.z_columns].to_numpy(dtype="float32"))
            with torch.inference_mode():
                decoded = self.model.unified_decoders[modality](vectors).numpy()
            spec = self.normalization[modality]
            decoded = (
                decoded * np.asarray(spec["scale"], dtype="float32")
                + np.asarray(spec["mean"], dtype="float32")
            )
            columns = list(FUSION_FEATURE_COLUMNS[modality])
            np.testing.assert_allclose(
                decoded, reconstructed[modality].loc[:, columns], rtol=1e-5, atol=1e-6
            )
            metadata = self.frames[modality].drop(columns=columns).reset_index(drop=True)
            independently_decoded[modality] = pd.concat(
                [metadata, pd.DataFrame(decoded, columns=columns)], axis=1
            )
            oracle_reconstructed[modality] = pd.concat(
                [metadata, pd.DataFrame(expected[modality][2], columns=columns)], axis=1
            )

        legacy = reconstruct_second_stage_features(
            self.model, self.normalization, self.frames, batch_size=2
        )
        omitted, without_embeddings = encode_second_stage_features(
            self.model, self.normalization, self.frames, batch_size=2,
            include_embeddings=False,
        )
        self.assertEqual(omitted, {})
        expected_risk = build_second_stage_feature_risk_matrices(
            oracle_reconstructed, self._risk_config()
        )
        for label, frames in (
            ("exported", independently_decoded),
            ("reconstructed", reconstructed),
            ("legacy", legacy),
            ("without_embeddings", without_embeddings),
        ):
            with self.subTest(result=label):
                actual_risk = build_second_stage_feature_risk_matrices(frames, self._risk_config())
                for actual, expected_frame in zip(actual_risk, expected_risk):
                    pd.testing.assert_frame_equal(
                        actual, expected_frame, check_exact=False, rtol=1e-5, atol=1e-6
                    )

    def test_transformer_stage_keeps_legacy_tuple_and_opt_in_row_embeddings(self):
        bundle = SimpleNamespace(
            fusion_model=self.model,
            fusion_checkpoint={"normalization": self.normalization},
        )
        arguments = [self.frames[modality] for modality in FUSION_MODALITIES]
        legacy = run_transformer_stage(*arguments, SimpleNamespace(), model_bundle=bundle)
        exported = run_transformer_stage(
            *arguments, SimpleNamespace(), model_bundle=bundle, include_row_embeddings=True
        )
        self.assertEqual(len(legacy), 5)
        self.assertEqual(len(exported), 6)
        for index in (0, 2, 3, 4):
            pd.testing.assert_frame_equal(legacy[index], exported[index])
        for modality in FUSION_MODALITIES:
            pd.testing.assert_frame_equal(legacy[1][modality], exported[1][modality])
            self.assertEqual(len(exported[5][modality]), len(self.frames[modality]))
            self.assertEqual(exported[5][modality].filter(regex=r"^z_\d+$").columns.tolist(), self.z_columns)
        np.testing.assert_allclose(
            exported[0].loc[:, self.z_columns],
            exported[5]["time_series"].loc[:, self.z_columns],
            rtol=1e-5, atol=1e-6,
        )

    def test_production_row_embedding_export_requires_128_dimensions(self):
        bundle = SimpleNamespace(
            fusion_model=self._model(d_model=32),
            fusion_checkpoint={"normalization": self.normalization},
        )
        arguments = [self.frames[modality] for modality in FUSION_MODALITIES]
        legacy = run_transformer_stage(*arguments, SimpleNamespace(), model_bundle=bundle)
        self.assertEqual(len(legacy), 5)
        with self.assertRaisesRegex(ValueError, "128"):
            run_transformer_stage(
                *arguments, SimpleNamespace(), model_bundle=bundle, include_row_embeddings=True
            )


if __name__ == "__main__":
    unittest.main()
