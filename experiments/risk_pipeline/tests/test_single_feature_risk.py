import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from risk_pipeline.single_feature_risk import (  # noqa: E402
    build_second_stage_feature_risk_matrices,
    build_single_feature_risk_matrices,
    load_fixed_risk_config,
    risk_level,
    validate_fixed_risk_config,
)
from risk_pipeline.modality_features import (  # noqa: E402
    IMAGE_FEATURE_COLUMNS,
    extract_image_tile_first_stage_features,
)


class SingleFeatureRiskTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "calibration_feature_stage": (
                "stage3_transformer_unified_reconstruction"
            ),
            "modalities": {
                name: {
                    "selected_feature": feature,
                    "fixed_weight": weight,
                    "scaler_mean": 0.0,
                    "scaler_scale": 1.0,
                    "logistic_coefficient": 1.0,
                    "logistic_intercept": 0.0,
                    "validation_auc": 0.8,
                    "test_auc": 0.79,
                }
                for name, feature, weight in (
                    ("time_series", "std", 0.20),
                    ("optical_image", "contrast", 0.40),
                    ("sonar_image", "entropy", 0.40),
                )
            }
        }
        self.time = pd.DataFrame({
            "sensor_id": ["depth_001", "depth_002", "temperature_001", "temperature_002"],
            "std": [0.1, 0.2, 0.3, 0.4],
            "energy": [1.0, 2.0, 3.0, 4.0],
        })
        self.optical = pd.DataFrame([{"source": "o", "contrast": 0.4}])
        self.sonar = pd.DataFrame([{"source": "s", "entropy": 0.6}])

    def test_only_configured_existing_columns_are_selected(self):
        vectors, matrix, overall = build_single_feature_risk_matrices(
            self.time,
            self.optical,
            self.sonar,
            self.config,
        )
        time_rows = vectors[vectors["modality"] == "time_series"]
        self.assertEqual(len(time_rows), 4)
        self.assertEqual(
            set(time_rows["feature_matrix_column"]), {"std"})
        self.assertNotIn("energy", set(time_rows["feature_matrix_column"]))
        self.assertEqual(
            matrix.set_index("modality")["concatenated_index"].to_dict(),
            {
                "time_series": 0,
                "optical_image": 5,
                "sonar_image": 11,
            },
        )
        self.assertAlmostEqual(matrix["fixed_weight"].sum(), 1.0)
        self.assertAlmostEqual(
            overall.iloc[0]["overall_risk"], matrix["weighted_contribution"].sum())

    def test_missing_selected_feature_fails_without_fallback(self):
        self.config["modalities"]["optical_image"]["selected_feature"] = "brightness"
        with self.assertRaises(KeyError):
            build_single_feature_risk_matrices(
                self.time,
                self.optical,
                self.sonar,
                self.config,
            )

    def test_second_stage_risk_reads_only_the_provided_reconstructed_tables(self):
        first_stage = {
            "time_series": self.time.copy(),
            "optical_image": self.optical.copy(),
            "sonar_image": self.sonar.copy(),
        }
        second_stage = {
            modality: frame.copy()
            for modality, frame in first_stage.items()
        }
        second_stage["time_series"]["std"] = 2.0
        second_stage["optical_image"]["contrast"] = 2.0
        second_stage["sonar_image"]["entropy"] = 2.0

        _, second_matrix, second_overall = (
            build_second_stage_feature_risk_matrices(
                second_stage,
                self.config,
            )
        )
        first_stage["time_series"]["std"] = -1000.0
        first_stage["optical_image"]["contrast"] = -1000.0
        first_stage["sonar_image"]["entropy"] = -1000.0
        _, repeated_matrix, repeated_overall = (
            build_second_stage_feature_risk_matrices(
                second_stage,
                self.config,
            )
        )

        self.assertEqual(
            set(second_matrix["feature_stage"]),
            {"stage3_transformer_unified_reconstruction"},
        )
        pd.testing.assert_frame_equal(second_matrix, repeated_matrix)
        pd.testing.assert_frame_equal(second_overall, repeated_overall)
        self.assertAlmostEqual(
            second_overall.iloc[0]["overall_risk"],
            1.0 / (1.0 + np.exp(-2.0)),
        )

    def test_second_stage_risk_requires_all_three_modalities(self):
        with self.assertRaisesRegex(KeyError, "missing modalities: sonar_image"):
            build_second_stage_feature_risk_matrices(
                {
                    "time_series": self.time,
                    "optical_image": self.optical,
                },
                self.config,
            )

    def test_second_stage_risk_rejects_first_stage_calibration(self):
        del self.config["calibration_feature_stage"]
        with self.assertRaisesRegex(ValueError, "not calibrated on stage3"):
            build_second_stage_feature_risk_matrices(
                {
                    "time_series": self.time,
                    "optical_image": self.optical,
                    "sonar_image": self.sonar,
                },
                self.config,
            )

    def test_weights_must_be_fixed_and_sum_to_one(self):
        self.config["modalities"]["time_series"]["fixed_weight"] = 0.10
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_fixed_risk_config(self.config)

    def test_final_risk_levels_use_four_equal_intervals(self):
        cases = {
            0.00: "very_low",
            0.249999: "very_low",
            0.25: "low",
            0.499999: "low",
            0.50: "medium",
            0.749999: "medium",
            0.75: "high",
            1.00: "high",
        }
        for score, expected in cases.items():
            with self.subTest(score=score):
                self.assertEqual(risk_level(score), expected)

    def test_both_image_modalities_keep_native_size_and_use_fixed_tiles(self):
        expected_columns = [
            "source",
            "modality",
            "tile_index",
            "x_start",
            "y_start",
            "tile_width",
            "tile_height",
            "input_width",
            "input_height",
            *IMAGE_FEATURE_COLUMNS,
        ]

        cases = (
            ("optical_image", (1080, 1920), 464, 29, 16, (1792, 952)),
            ("sonar_image", (929, 882), 182, 13, 14, (754, 801)),
        )
        for modality, shape, count, columns, rows, last_start in cases:
            with self.subTest(modality=modality):
                image = np.full(shape, 0.25, dtype="float32")
                features = extract_image_tile_first_stage_features(
                    f"{modality}.png", image, modality=modality)
                self.assertEqual(len(features), count)
                self.assertEqual(list(features.columns), expected_columns)
                self.assertEqual(features["x_start"].nunique(), columns)
                self.assertEqual(features["y_start"].nunique(), rows)
                self.assertEqual(
                    features.iloc[0][["x_start", "y_start"]].tolist(),
                    [0, 0],
                )
                self.assertEqual(
                    features.iloc[-1][["x_start", "y_start"]].tolist(),
                    list(last_start),
                )
                self.assertEqual(
                    features.iloc[0][
                        [
                            "input_width",
                            "input_height",
                        ]
                    ].tolist(),
                    [shape[1], shape[0]],
                )
                np.testing.assert_allclose(features["brightness"], 0.25)

    def test_image_risk_averages_all_tile_probabilities(self):
        optical_values = np.linspace(-2.0, 2.0, 464)
        sonar_values = np.linspace(-1.0, 1.0, 182)
        self.optical = pd.DataFrame({
            "source": "optical.png",
            "contrast": optical_values,
        })
        self.sonar = pd.DataFrame({
            "source": "sonar.png",
            "entropy": sonar_values,
        })
        _, matrix, _ = build_single_feature_risk_matrices(
            self.time,
            self.optical,
            self.sonar,
            self.config,
        )

        by_modality = matrix.set_index("modality")
        optical = by_modality.loc["optical_image"]
        sonar = by_modality.loc["sonar_image"]
        optical_probabilities = 1.0 / (1.0 + np.exp(-optical_values))
        sonar_probabilities = 1.0 / (1.0 + np.exp(-sonar_values))
        self.assertEqual(optical["vector_length"], 464)
        self.assertEqual(optical["aggregation_count"], 464)
        self.assertAlmostEqual(
            optical["modality_risk"],
            float(optical_probabilities.mean()),
        )
        self.assertEqual(sonar["vector_length"], 182)
        self.assertEqual(sonar["aggregation_count"], 182)
        self.assertAlmostEqual(
            sonar["modality_risk"],
            float(sonar_probabilities.mean()),
        )

    def test_default_engineering_policy_uses_requested_features_and_weights(self):
        project_root = Path(__file__).resolve().parents[3]
        config = load_fixed_risk_config(
            project_root / "config" / "fixed_single_feature_risk_config.json"
        )
        self.assertEqual(
            {
                name: spec["selected_feature"]
                for name, spec in config["modalities"].items()
            },
            {
                "time_series": "std",
                "optical_image": "edge_density",
                "sonar_image": "edge_density",
            },
        )
        self.assertEqual(
            {
                name: spec["fixed_weight"]
                for name, spec in config["modalities"].items()
            },
            {
                "time_series": 0.20,
                "optical_image": 0.40,
                "sonar_image": 0.40,
            },
        )


if __name__ == "__main__":
    unittest.main()
