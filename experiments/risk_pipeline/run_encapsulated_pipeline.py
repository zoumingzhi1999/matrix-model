import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multisource_unified import (
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    load_feature_fusion_checkpoint,
)
from risk_pipeline.artifacts import (
    artifact_path,
    artifact_timestamp,
    copy_file_once,
    latest_file,
    mapping_to_long_frame,
    write_csv_once,
    write_text_once,
)
from risk_pipeline.data_io import load_aligned_sensors, load_grayscale_image
from risk_pipeline.modality_features import (
    extract_image_tile_first_stage_features,
)
from risk_pipeline.model_bundle import (
    DEFAULT_FUSION_CHECKPOINT,
    ProductionModelBundle,
)
from risk_pipeline.single_feature_risk import (
    build_second_stage_feature_risk_matrices,
    load_fixed_risk_config,
    risk_level as _risk_level,
)
from risk_pipeline.time_series_features import (
    extract_first_stage_features,
)
from risk_pipeline.xtf_io import load_xtf_sonar_image


DEFAULT_RISK_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "fixed_single_feature_risk_config.json"
)
def run(args, model_bundle: ProductionModelBundle | None = None):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = getattr(args, "artifact_timestamp", None) or artifact_timestamp()

    imputeformer_output = Path(args.imputeformer_output)
    if not imputeformer_output.exists():
        latest_aligned = latest_file(
            imputeformer_output.parent, "aligned", "stage1_imputeformer_aligned_output_*.csv")
        if latest_aligned is not None:
            imputeformer_output = latest_aligned
    aligned_sensors = load_aligned_sensors(str(imputeformer_output))
    aligned_dir = output_dir / "aligned"
    if imputeformer_output.resolve().parent == aligned_dir.resolve():
        retained_imputeformer = imputeformer_output
    else:
        retained_imputeformer = artifact_path(
            output_dir, "aligned", "stage1_imputeformer_aligned_output", ".csv", timestamp)
        copy_file_once(imputeformer_output, retained_imputeformer)

    time_first_stage, sensor_ids = extract_first_stage_features(aligned_sensors)
    time_first_stage_path = artifact_path(
        output_dir, "features", "stage2a_time_series_first_stage_features", ".csv", timestamp)
    write_csv_once(time_first_stage, time_first_stage_path)

    optical_source_path = Path(args.optical_image)
    sonar_source_path = Path(args.sonar_xtf)
    retained_optical = artifact_path(
        output_dir, "inputs", "stage0_optical_image", optical_source_path.suffix, timestamp)
    retained_sonar_xtf = artifact_path(
        output_dir, "inputs", "stage0_sonar_source", ".xtf", timestamp)
    copy_file_once(optical_source_path, retained_optical)
    copy_file_once(sonar_source_path, retained_sonar_xtf)

    optical_image = load_grayscale_image(optical_source_path)
    sonar_image, sonar_decode_metadata = load_xtf_sonar_image(sonar_source_path)
    sonar_array_path = artifact_path(
        output_dir, "decoded", "stage1_xtf_sonar_intensity", ".npy", timestamp)
    with sonar_array_path.open("xb") as stream:
        np.save(stream, sonar_image, allow_pickle=False)
    sonar_preview_path = artifact_path(
        output_dir, "decoded", "stage1_xtf_sonar_preview", ".png", timestamp)
    with sonar_preview_path.open("xb") as stream:
        Image.fromarray(np.rint(sonar_image * 255.0).astype("uint8")).save(
            stream,
            format="PNG",
        )
    sonar_metadata_path = artifact_path(
        output_dir, "decoded", "stage1_xtf_decode_metadata", ".csv", timestamp)
    write_csv_once(
        mapping_to_long_frame(sonar_decode_metadata),
        sonar_metadata_path,
    )
    optical_first_stage = extract_image_tile_first_stage_features(
        optical_source_path.name,
        optical_image,
        modality="optical_image",
    )
    sonar_first_stage = extract_image_tile_first_stage_features(
        sonar_source_path.name,
        sonar_image,
        modality="sonar_image",
    )
    optical_first_stage_path = artifact_path(
        output_dir, "features", "stage2b_optical_image_first_stage_features", ".csv", timestamp)
    sonar_first_stage_path = artifact_path(
        output_dir, "features", "stage2c_sonar_image_first_stage_features", ".csv", timestamp)
    write_csv_once(optical_first_stage, optical_first_stage_path)
    write_csv_once(sonar_first_stage, sonar_first_stage_path)

    (
        transformer_features,
        second_stage_features,
        token_ranges,
        concatenated_vectors,
        feature_positions,
    ) = run_transformer_stage(
        time_first_stage=time_first_stage,
        optical_first_stage=optical_first_stage,
        sonar_first_stage=sonar_first_stage,
        args=args,
        model_bundle=model_bundle,
    )
    concatenated_path = artifact_path(
        output_dir, "features", "stage3_concatenated_first_stage_vectors", ".csv", timestamp)
    write_csv_once(concatenated_vectors, concatenated_path)
    feature_positions_path = artifact_path(
        output_dir, "features", "stage3_concatenated_feature_positions", ".csv", timestamp)
    write_csv_once(feature_positions, feature_positions_path)
    transformer_path = artifact_path(
        output_dir, "features", "stage3_transformer_unified_features", ".csv", timestamp)
    write_csv_once(transformer_features, transformer_path)
    token_ranges_path = artifact_path(
        output_dir, "features", "stage3_transformer_token_ranges", ".csv", timestamp)
    write_csv_once(token_ranges, token_ranges_path)
    second_stage_paths = {}
    second_stage_stems = {
        "time_series": "stage3a_time_series_second_stage_features",
        "optical_image": "stage3b_optical_image_second_stage_features",
        "sonar_image": "stage3c_sonar_image_second_stage_features",
    }
    for modality in FUSION_MODALITIES:
        path = artifact_path(
            output_dir,
            "features",
            second_stage_stems[modality],
            ".csv",
            timestamp,
        )
        write_csv_once(second_stage_features[modality], path)
        second_stage_paths[modality] = path
    fusion_checkpoint_path = (
        model_bundle.fusion_checkpoint_path
        if model_bundle is not None
        else Path(
            getattr(args, "fusion_checkpoint", DEFAULT_FUSION_CHECKPOINT)
        ).resolve()
    )

    risk_config_path = Path(getattr(args, "risk_config", DEFAULT_RISK_CONFIG))
    risk_config = load_fixed_risk_config(risk_config_path)
    validate_risk_checkpoint_binding(risk_config, fusion_checkpoint_path)

    selected_vectors, feature_risk_matrix, overall_risk_matrix = (
        build_second_stage_feature_risk_matrices(
            second_stage_features,
            risk_config,
        )
    )

    selected_vectors_path = artifact_path(
        output_dir,
        "risk_inputs",
        "selected_second_stage_feature_vectors",
        ".csv",
        timestamp,
    )
    write_csv_once(selected_vectors, selected_vectors_path)

    modality_sources = {
        modality: second_stage_paths[modality].name
        for modality in FUSION_MODALITIES
    }
    modality_risk_result = feature_risk_matrix[[
        "modality",
        "feature_stage",
        "selected_feature",
        "concatenated_index",
        "transformer_token_index",
        "vector_length",
        "aggregation_count",
        "vector_aggregation",
        "modality_risk",
        "fixed_weight",
        "weighted_contribution",
    ]].copy()
    modality_risk_result.insert(
        1,
        "source",
        modality_risk_result["modality"].map(modality_sources),
    )
    modality_risk_result.insert(
        modality_risk_result.columns.get_loc("modality_risk") + 1,
        "risk_level",
        modality_risk_result["modality_risk"].map(
            lambda score: _risk_level(float(score))
        ),
    )
    final_row = overall_risk_matrix.iloc[0]
    final_risk = float(final_row["overall_risk"])
    final_risk_level = str(final_row["risk_level"])
    checkpoint_sha256 = _file_sha256(fusion_checkpoint_path)
    risk_config_sha256 = _file_sha256(risk_config_path)
    risk_result = {
        "schema_version": 1,
        "artifact_timestamp": timestamp,
        "risk_feature_stage": "stage3_transformer_unified_reconstruction",
        "fusion_checkpoint_sha256": checkpoint_sha256,
        "risk_config_sha256": risk_config_sha256,
        "modalities": {
            row.modality: {
                "risk": float(row.modality_risk),
                "risk_level": str(row.risk_level),
                "selected_feature": str(row.selected_feature),
                "concatenated_index": int(row.concatenated_index),
                "transformer_token_index": int(row.transformer_token_index),
                "source": str(row.source),
                "selected_vector_artifact": str(
                    selected_vectors_path.relative_to(output_dir)
                ).replace("\\", "/"),
                "vector_length": int(row.vector_length),
                "feature_value_summary": {
                    "min": float(
                        selected_vectors.loc[
                            selected_vectors["modality"] == row.modality,
                            "feature_value",
                        ].min()
                    ),
                    "mean": float(
                        selected_vectors.loc[
                            selected_vectors["modality"] == row.modality,
                            "feature_value",
                        ].mean()
                    ),
                    "max": float(
                        selected_vectors.loc[
                            selected_vectors["modality"] == row.modality,
                            "feature_value",
                        ].max()
                    ),
                },
                "aggregation": str(row.vector_aggregation),
                "fixed_weight": float(row.fixed_weight),
                "weighted_contribution": float(row.weighted_contribution),
            }
            for row in modality_risk_result.itertuples()
        },
        "total_risk": final_risk,
        "risk_level": final_risk_level,
        "dominant_modality": str(final_row["dominant_modality"]),
        "formula": " + ".join(
            f"{float(risk_config['modalities'][name]['fixed_weight']):.2f}*{name}"
            for name in FUSION_MODALITIES
        ),
    }
    risk_json_path = artifact_path(
        output_dir, "risk", "risk_matrix", ".json", timestamp
    )
    write_text_once(
        risk_json_path,
        json.dumps(risk_result, indent=2, ensure_ascii=False) + "\n",
    )

    summary = {
        "artifact_timestamp": timestamp,
        "stage1_imputeformer_output": str(retained_imputeformer),
        "stage0_optical_image_input": str(retained_optical),
        "stage0_sonar_xtf_input": str(retained_sonar_xtf),
        "stage1_xtf_sonar_intensity": str(sonar_array_path),
        "stage1_xtf_sonar_preview": str(sonar_preview_path),
        "stage1_xtf_decode_metadata": str(sonar_metadata_path),
        "stage2a_time_series_first_stage_features": str(time_first_stage_path),
        "stage2b_optical_image_first_stage_features": str(optical_first_stage_path),
        "stage2c_sonar_image_first_stage_features": str(sonar_first_stage_path),
        "stage3_concatenated_first_stage_vectors": str(concatenated_path),
        "stage3_concatenated_feature_positions": str(feature_positions_path),
        "stage3_transformer_features": str(transformer_path),
        "stage3_transformer_token_ranges": str(token_ranges_path),
        "stage3_second_stage_features": {
            modality: str(second_stage_paths[modality])
            for modality in FUSION_MODALITIES
        },
        "stage3_feature_fusion_checkpoint": str(fusion_checkpoint_path),
        "stage3_feature_fusion_checkpoint_sha256": checkpoint_sha256,
        "selected_second_stage_feature_vectors": str(selected_vectors_path),
        "risk_matrix": str(risk_json_path),
        "final_risk": final_risk,
        "risk_level": final_risk_level,
        "fixed_risk_config": str(risk_config_path.resolve()),
        "fixed_risk_config_sha256": risk_config_sha256,
        "sensor_count": len(sensor_ids),
        "selected_features": {
            name: spec["selected_feature"]
            for name, spec in risk_config["modalities"].items()
        },
        "risk_feature_stage": "stage3_transformer_unified_reconstruction",
        "second_stage_context_policy": (
            "reconstruct every target-modality row through the CLS fusion vector; "
            "represent each other modality by the mean of all its current rows"
        ),
        "fixed_weights": {
            name: spec["fixed_weight"]
            for name, spec in risk_config["modalities"].items()
        },
        "note": (
            "Second-stage Transformer input is strictly the output of "
            "first-stage feature extraction for each modality. Raw time-series, "
            "optical and XTF-decoded sonar arrays are not directly fed into the "
            "second-stage Transformer. The CLS fusion representation is decoded back "
            "to named, original-unit second-stage features while preserving every "
            "sensor and image-tile input. Operational risk reads exactly "
            "one named second-stage feature per modality and applies frozen weights "
            "without runtime renormalization. Optical and sonar inputs keep their "
            "original resolution, are split into 128x128 overlapping tiles with a "
            "nominal 64-pixel stride, and are summarized by averaging all calibrated "
            "tile probabilities. At 1920x1080 this produces 464 tiles. "
            "The three modality risks and final risk are written together in one "
            "JSON result; risk-matrix CSV files are not produced. "
            "The Transformer is loaded from the trained production checkpoint, "
            "has no risk head; it changes the feature values supplied to the fixed "
            "risk formula but never predicts risk itself. Sonar input is required "
            "to be raw XTF and is decoded with pyxtf without image resizing."
        ),
    }
    summary_path = artifact_path(
        output_dir, "summaries", "pipeline_summary", ".csv", timestamp)
    write_csv_once(mapping_to_long_frame(summary), summary_path)

    print(json.dumps(risk_result, indent=2, ensure_ascii=False))

    return {
        "risk_result": risk_result,
        "risk_json_path": risk_json_path,
        "summary_path": summary_path,
    }


def validate_risk_checkpoint_binding(risk_config, fusion_checkpoint_path):
    """Require calibration parameters to match the exact fusion checkpoint."""
    expected_checkpoint_hash = risk_config.get("fusion_checkpoint_sha256")
    if not expected_checkpoint_hash:
        raise ValueError(
            "Risk config is not bound to a trained fusion checkpoint. Run "
            "calibrate_second_stage_risk_config.py."
        )
    actual_checkpoint_hash = _file_sha256(fusion_checkpoint_path)
    if actual_checkpoint_hash.lower() != str(expected_checkpoint_hash).lower():
        raise ValueError(
            "Risk config was calibrated with a different fusion checkpoint. "
            "Re-run calibrate_second_stage_risk_config.py for this checkpoint."
        )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_second_stage_features(
    model,
    normalization,
    feature_frames,
    batch_size=8192,
):
    """Decode each source row through a fused CLS representation.

    The target modality keeps its original row granularity. Other modalities
    contribute their current-table mean, which avoids an arbitrary Cartesian
    product while ensuring every decoded row is cross-modal.
    """
    normalized_values = {}
    for modality in FUSION_MODALITIES:
        frame = feature_frames[modality]
        columns = list(FUSION_FEATURE_COLUMNS[modality])
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(
                f"First-stage {modality} matrix is missing columns: {missing}"
            )
        values = frame.loc[:, columns].to_numpy(dtype="float32", copy=True)
        if len(values) == 0:
            raise ValueError(f"First-stage {modality} feature matrix is empty.")
        if not np.isfinite(values).all():
            raise ValueError(f"First-stage {modality} features contain non-finite values.")
        mean = np.asarray(normalization[modality]["mean"], dtype="float32")
        scale = np.asarray(normalization[modality]["scale"], dtype="float32")
        normalized_values[modality] = (values - mean) / scale

    device = next(model.parameters()).device
    context_vectors = {
        modality: values.mean(axis=0, keepdims=True)
        for modality, values in normalized_values.items()
    }
    second_stage_features = {}
    for target_modality in FUSION_MODALITIES:
        target_values = normalized_values[target_modality]
        reconstructed_batches = []
        for start in range(0, len(target_values), batch_size):
            end = min(start + batch_size, len(target_values))
            current_batch_size = end - start
            batch_inputs = {}
            for modality in FUSION_MODALITIES:
                if modality == target_modality:
                    batch_values = target_values[start:end]
                else:
                    batch_values = np.repeat(
                        context_vectors[modality], current_batch_size, axis=0
                    )
                batch_inputs[modality] = torch.from_numpy(batch_values).to(device)
            with torch.inference_mode():
                reconstructed = model(batch_inputs)["unified_reconstruction"][
                    target_modality
                ]
            reconstructed_batches.append(reconstructed.cpu().numpy())

        reconstructed_normalized = np.concatenate(reconstructed_batches, axis=0)
        mean = np.asarray(normalization[target_modality]["mean"], dtype="float32")
        scale = np.asarray(normalization[target_modality]["scale"], dtype="float32")
        reconstructed_original_units = reconstructed_normalized * scale + mean
        source_frame = feature_frames[target_modality]
        feature_columns = list(FUSION_FEATURE_COLUMNS[target_modality])
        metadata_columns = [
            column for column in source_frame.columns if column not in feature_columns
        ]
        reconstructed_frame = source_frame.loc[:, metadata_columns].reset_index(drop=True)
        for column_index, column in enumerate(feature_columns):
            reconstructed_frame[column] = reconstructed_original_units[:, column_index]
        second_stage_features[target_modality] = reconstructed_frame
    return second_stage_features


def run_transformer_stage(
    time_first_stage,
    optical_first_stage,
    sonar_first_stage,
    args,
    model_bundle: ProductionModelBundle | None = None,
):
    if model_bundle is None:
        checkpoint_path = Path(
            getattr(args, "fusion_checkpoint", DEFAULT_FUSION_CHECKPOINT)
        )
        model, checkpoint = load_feature_fusion_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
    else:
        model = model_bundle.fusion_model
        checkpoint = model_bundle.fusion_checkpoint
    normalization = checkpoint["normalization"]

    feature_frames = {
        "time_series": time_first_stage,
        "optical_image": optical_first_stage,
        "sonar_image": sonar_first_stage,
    }
    normalized_values = {}
    for modality in FUSION_MODALITIES:
        frame = feature_frames[modality]
        columns = list(FUSION_FEATURE_COLUMNS[modality])
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise KeyError(
                f"First-stage {modality} matrix is missing columns: {missing}"
            )
        values = frame.loc[:, columns].to_numpy(dtype="float32", copy=True)
        if len(values) == 0:
            raise ValueError(f"First-stage {modality} feature matrix is empty.")
        if not np.isfinite(values).all():
            raise ValueError(f"First-stage {modality} features contain non-finite values.")
        mean = np.asarray(normalization[modality]["mean"], dtype="float32")
        scale = np.asarray(normalization[modality]["scale"], dtype="float32")
        standardized = (values - mean) / scale
        normalized_values[modality] = standardized

    sensor_count = len(time_first_stage)
    model_device = next(model.parameters()).device
    model_inputs = {
        modality: torch.from_numpy(
            normalized_values[modality]
            if modality == "time_series"
            else np.repeat(
                normalized_values[modality].mean(axis=0, keepdims=True),
                sensor_count,
                axis=0,
            )
        ).to(model_device)
        for modality in FUSION_MODALITIES
    }

    original_model_inputs = {}
    for modality in FUSION_MODALITIES:
        values = feature_frames[modality].loc[
            :, list(FUSION_FEATURE_COLUMNS[modality])
        ].to_numpy(dtype="float32", copy=True)
        original_model_inputs[modality] = (
            values
            if modality == "time_series"
            else np.repeat(values.mean(axis=0, keepdims=True), sensor_count, axis=0)
        )

    with torch.inference_mode():
        output = model(model_inputs)
    vectors = output["unified"].cpu().numpy()
    rows = [
        {
            "sensor_id": sensor_id,
            **{
                f"z_{index:03d}": float(value)
                for index, value in enumerate(vectors[row_index])
            },
        }
        for row_index, sensor_id in enumerate(time_first_stage["sensor_id"])
    ]
    range_rows = []
    concatenated_rows = []
    position_rows = []
    concatenated_offset = 0
    offsets = {}
    for modality in FUSION_MODALITIES:
        width = len(FUSION_FEATURE_COLUMNS[modality])
        offsets[modality] = (concatenated_offset, concatenated_offset + width)
        concatenated_offset += width
    for sensor_id in time_first_stage["sensor_id"]:
        row_index = len(concatenated_rows)
        concatenated_row = {"sensor_id": sensor_id}
        for modality, (start, end, _) in output["source_token_ranges"].items():
            concat_start, concat_end = offsets[modality]
            range_rows.append({
                "sensor_id": sensor_id,
                "source": f"{modality}_first_stage_features",
                "modality": modality,
                "start_token": start,
                "end_token": end,
                "token_count": end - start,
                "concatenated_start": concat_start,
                "concatenated_end_exclusive": concat_end,
            })
            columns = list(FUSION_FEATURE_COLUMNS[modality])
            aggregation = (
                "sensor row" if modality == "time_series" else "mean of all image tiles"
            )
            for feature_index, feature_name in enumerate(columns):
                raw_value = float(original_model_inputs[modality][row_index, feature_index])
                normalized_value = float(model_inputs[modality][row_index, feature_index])
                concatenated_index = concat_start + feature_index
                concatenated_row[
                    f"x_{concatenated_index:02d}_{modality}_{feature_name}"
                ] = raw_value
                position_rows.append({
                    "sensor_id": sensor_id,
                    "concatenated_index": concatenated_index,
                    "modality": modality,
                    "modality_feature_index": feature_index,
                    "feature_name": feature_name,
                    "first_stage_value": raw_value,
                    "normalized_transformer_input": normalized_value,
                    "transformer_token_index": start,
                    "source_row_count": len(feature_frames[modality]),
                    "context_aggregation": aggregation,
                })
        concatenated_rows.append(concatenated_row)
    second_stage_features = reconstruct_second_stage_features(
        model,
        normalization,
        feature_frames,
    )

    return (
        pd.DataFrame(rows),
        second_stage_features,
        pd.DataFrame(range_rows),
        pd.DataFrame(concatenated_rows),
        pd.DataFrame(position_rows),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imputeformer-output",
        required=True,
        help="Aligned sensor CSV produced by the sensor ImputeFormer stage.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/encapsulated_alignment_representation_pipeline",
    )
    parser.add_argument("--optical-image", default="data/public_multisource_images/optical_image.jpg")
    parser.add_argument("--sonar-xtf", required=True, help="Raw sonar XTF file.")
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG))
    parser.add_argument(
        "--fusion-checkpoint",
        default=str(DEFAULT_FUSION_CHECKPOINT),
        help="Trained FeatureFusionTransformer checkpoint; inference fails if absent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
