"""Risk fusion using one named second-stage feature per modality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from risk_pipeline.feature_schema import FUSION_FEATURE_COLUMNS, FUSION_MODALITIES

MODALITY_ORDER = FUSION_MODALITIES


def load_fixed_risk_config(path: str | Path) -> Dict[str, object]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_fixed_risk_config(config)
    return config


def validate_fixed_risk_config(config: Mapping[str, object]) -> None:
    modalities = config.get("modalities")
    if not isinstance(modalities, Mapping):
        raise ValueError("Risk config must contain a 'modalities' object.")
    missing = [name for name in MODALITY_ORDER if name not in modalities]
    if missing:
        raise ValueError(f"Risk config is missing modalities: {', '.join(missing)}")

    weight_sum = 0.0
    for modality in MODALITY_ORDER:
        spec = modalities[modality]
        for required in (
            "selected_feature",
            "fixed_weight",
            "scaler_mean",
            "scaler_scale",
            "logistic_coefficient",
            "logistic_intercept",
        ):
            if required not in spec:
                raise ValueError(f"Risk config {modality} is missing '{required}'.")
        weight = float(spec["fixed_weight"])
        if weight < 0.0:
            raise ValueError("Fixed modality weights must be non-negative.")
        if float(spec["scaler_scale"]) <= 0.0:
            raise ValueError("Every scaler_scale must be positive.")
        weight_sum += weight
    if not np.isclose(weight_sum, 1.0, atol=1e-9):
        raise ValueError(f"Fixed modality weights must sum to 1.0, got {weight_sum}.")


def build_single_feature_risk_matrices(
    time_features: pd.DataFrame,
    optical_features: pd.DataFrame,
    sonar_features: pd.DataFrame,
    config: Mapping[str, object],
    feature_stage: str = "provided_feature_matrix",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return selected vectors, three-row fixed-weight matrix, and overall risk."""
    validate_fixed_risk_config(config)

    feature_tables = {
        "time_series": time_features,
        "optical_image": optical_features,
        "sonar_image": sonar_features,
    }
    vector_frames = []
    modality_rows = []
    for modality in MODALITY_ORDER:
        spec = config["modalities"][modality]
        selected = str(spec["selected_feature"])
        vector = _select_existing_feature_vector(
            modality, feature_tables[modality], selected)
        probabilities = _logistic_probabilities(vector["feature_value"].to_numpy(), spec)
        vector["predicted_risk"] = probabilities
        vector["modality"] = modality
        vector["feature_stage"] = feature_stage
        vector["selected_feature"] = selected
        vector_frames.append(vector)

        modality_risk = float(np.mean(probabilities))
        aggregation_count = int(len(probabilities))
        fixed_weight = float(spec["fixed_weight"])
        modality_rows.append({
            "modality": modality,
            "feature_stage": feature_stage,
            "selected_feature": selected,
            "concatenated_index": int(vector["concatenated_index"].iloc[0]),
            "transformer_token_index": int(
                vector["transformer_token_index"].iloc[0]
            ),
            "feature_matrix_column_rule": (
                selected
            ),
            "vector_length": int(len(vector)),
            "aggregation_count": aggregation_count,
            "vector_aggregation": "mean of all element-wise calibrated probabilities",
            "risk_model": "standardized univariate logistic regression",
            "validation_dataset": spec.get("validation_dataset", ""),
            "validation_auc": float(spec.get("validation_auc", np.nan)),
            "test_auc": float(spec.get("test_auc", np.nan)),
            "modality_risk": modality_risk,
            "fixed_weight": fixed_weight,
            "weighted_contribution": modality_risk * fixed_weight,
        })

    vectors = pd.concat(vector_frames, ignore_index=True)
    vectors = vectors[[
        "modality",
        "feature_stage",
        "selected_feature",
        "vector_index",
        "source_row",
        "feature_matrix_column",
        "concatenated_index",
        "modality_feature_index",
        "transformer_token_index",
        "feature_value",
        "predicted_risk",
    ]]
    modality_matrix = pd.DataFrame(modality_rows)
    overall_risk = float(modality_matrix["weighted_contribution"].sum())
    dominant = modality_matrix.loc[
        modality_matrix["weighted_contribution"].idxmax(), "modality"]
    overall_matrix = pd.DataFrame([{
        **{
            f"{row.modality}_selected_feature": row.selected_feature
            for row in modality_matrix.itertuples()
        },
        **{
            f"{row.modality}_risk": float(row.modality_risk)
            for row in modality_matrix.itertuples()
        },
        **{
            f"{row.modality}_fixed_weight": float(row.fixed_weight)
            for row in modality_matrix.itertuples()
        },
        **{
            f"{row.modality}_weighted_contribution": float(row.weighted_contribution)
            for row in modality_matrix.itertuples()
        },
        "overall_risk": overall_risk,
        "risk_level": risk_level(overall_risk),
        "dominant_modality": dominant,
        "weight_policy": "fixed configured modality weights; no runtime renormalization",
    }])
    return vectors, modality_matrix, overall_matrix


def build_second_stage_feature_risk_matrices(
    second_stage_features: Mapping[str, pd.DataFrame],
    config: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build risk exclusively from named Transformer reconstruction outputs."""
    expected_stage = "stage3_transformer_unified_reconstruction"
    if config.get("calibration_feature_stage") != expected_stage:
        raise ValueError(
            "Risk config was not calibrated on stage3 Transformer unified "
            "reconstructions. Run calibrate_second_stage_risk_config.py."
        )
    missing = [
        modality
        for modality in MODALITY_ORDER
        if modality not in second_stage_features
    ]
    if missing:
        raise KeyError(
            "Second-stage feature matrices are missing modalities: "
            + ", ".join(missing)
        )
    unexpected = [
        modality
        for modality in second_stage_features
        if modality not in MODALITY_ORDER
    ]
    if unexpected:
        raise KeyError(
            "Unknown second-stage feature modalities: " + ", ".join(unexpected)
        )
    for modality in MODALITY_ORDER:
        if not isinstance(second_stage_features[modality], pd.DataFrame):
            raise TypeError(
                f"Second-stage {modality} features must be a pandas DataFrame."
            )

    return build_single_feature_risk_matrices(
        second_stage_features["time_series"],
        second_stage_features["optical_image"],
        second_stage_features["sonar_image"],
        config,
        feature_stage=expected_stage,
    )


def _select_existing_feature_vector(
    modality: str,
    features: pd.DataFrame,
    selected_feature: str,
) -> pd.DataFrame:
    modality_columns = list(FUSION_FEATURE_COLUMNS[modality])
    if selected_feature not in modality_columns:
        raise KeyError(
            f"Configured feature '{selected_feature}' is not in the production "
            f"schema for {modality}."
        )
    modality_feature_index = modality_columns.index(selected_feature)
    modality_index = list(MODALITY_ORDER).index(modality)
    concatenated_offset = sum(
        len(FUSION_FEATURE_COLUMNS[name])
        for name in MODALITY_ORDER[:modality_index]
    )
    position_metadata = {
        "concatenated_index": concatenated_offset + modality_feature_index,
        "modality_feature_index": modality_feature_index,
        "transformer_token_index": modality_index + 1,
    }
    if modality == "time_series":
        if "sensor_id" not in features.columns:
            raise KeyError(
                "Time-series feature matrix has no 'sensor_id' column.")
        if selected_feature not in features.columns:
            raise KeyError(
                f"Time-series feature matrix has no column '{selected_feature}'.")
        values = pd.to_numeric(features[selected_feature], errors="raise")
        return pd.DataFrame({
            "vector_index": np.arange(len(features), dtype="int64"),
            "source_row": features["sensor_id"].astype(str).to_numpy(),
            "feature_matrix_column": selected_feature,
            **position_metadata,
            "feature_value": values.to_numpy(dtype="float64"),
        })

    if selected_feature not in features.columns:
        raise KeyError(
            f"{modality} feature matrix has no column '{selected_feature}'.")
    values = pd.to_numeric(features[selected_feature], errors="raise")
    if {"source", "tile_index", "x_start", "y_start"}.issubset(features.columns):
        source_rows = [
            f"{row.source}::tile_{int(row.tile_index):04d}"
            f"_x{int(row.x_start)}_y{int(row.y_start)}"
            for row in features.itertuples()
        ]
    elif "source" in features.columns:
        source_rows = features["source"].astype(str).tolist()
    else:
        source_rows = [str(index) for index in features.index]
    return pd.DataFrame({
        "vector_index": np.arange(len(features), dtype="int64"),
        "source_row": source_rows,
        "feature_matrix_column": selected_feature,
        **position_metadata,
        "feature_value": values.to_numpy(dtype="float64"),
    })


def _logistic_probabilities(values: np.ndarray, spec: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("Selected feature vector contains NaN or infinite values.")
    standardized = (
        values - float(spec["scaler_mean"])) / float(spec["scaler_scale"])
    logits = (
        float(spec["logistic_coefficient"]) * standardized
        + float(spec["logistic_intercept"])
    )
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.25:
        return "low"
    return "very_low"
