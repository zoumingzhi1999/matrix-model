import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config.input_paths import resolve_input_paths

from multisource_unified import (
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    load_feature_fusion_checkpoint,
)
from risk_pipeline.artifacts import (
    artifact_path,
    artifact_timestamp,
    configure_inference_logging,
    copy_file_once,
    latest_file,
    log_feature_frame,
    write_csv_once,
    write_text_once,
)
from risk_pipeline.data_io import (
    load_aligned_sensors,
    load_grayscale_image,
    load_sonar_tiff_image,
    resolve_image_source_paths,
    resolve_sonar_source_paths,
)
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
logger = logging.getLogger("f-model.pipeline")


def extract_image_source_features(paths, modality, *, allow_xtf=False):
    """Extract every source sequentially, retaining tile identity in one table."""
    frames = []
    for index, source_path in enumerate(paths, start=1):
        logger.info(
            "Image source started: modality=%s file=%d/%d source=%s",
            modality, index, len(paths), source_path,
        )
        if modality == "sonar_image" and allow_xtf and source_path.suffix.lower() == ".xtf":
            logger.info("Decoding sonar XTF in memory: sonar_source=%s", source_path)
            image, decode_metadata = load_xtf_sonar_image(source_path)
        else:
            logger.info("Loading %s in memory: source=%s", modality, source_path)
            image = (
                load_sonar_tiff_image(source_path)
                if modality == "sonar_image" else load_grayscale_image(source_path)
            )
            decode_metadata = {
                "source_type": "image",
                "format": source_path.suffix.lower().lstrip("."),
                "output_height": int(image.shape[0]),
                "output_width": int(image.shape[1]),
                "resize_applied": False,
            }
        logger.info(
            "Image decoding complete: modality=%s source=%s shape=%s metadata=%s",
            modality, source_path, image.shape,
            json.dumps(decode_metadata, ensure_ascii=False),
        )
        frame = extract_image_tile_first_stage_features(
            source_path.name, image, modality=modality,
        )
        frames.append(frame)
        logger.info(
            "Image source completed: modality=%s file=%d/%d source=%s feature_rows=%d",
            modality, index, len(paths), source_path, len(frame),
        )
        del image
    if not frames:
        raise ValueError(f"No {modality} sources were provided.")
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Image features aggregated: modality=%s source_files=%d feature_rows=%d",
        modality, len(paths), len(combined),
    )
    return combined


def run(args, model_bundle: ProductionModelBundle | None = None):
    configure_inference_logging()
    input_paths = resolve_input_paths(
        optical_image=getattr(args, "optical_image", None),
        sonar_image=getattr(args, "sonar_image", None),
        sonar_xtf=getattr(args, "sonar_xtf", None),
    )
    if input_paths.optical_image is None or input_paths.sonar_source is None:
        raise ValueError(
            "Provide an optical image and a sonar TIFF image, "
            "or set OPTICAL_IMAGE_INPUT_PATH and SONAR_IMAGE_INPUT_PATH in config/input_paths.py."
        )
    optical_source_path = input_paths.optical_image
    sonar_source_path = input_paths.sonar_source
    optical_sources = resolve_image_source_paths(optical_source_path)
    allow_xtf = input_paths.sonar_xtf is not None
    sonar_sources = resolve_sonar_source_paths(sonar_source_path, allow_xtf=allow_xtf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = getattr(args, "artifact_timestamp", None) or artifact_timestamp()

    imputeformer_output = Path(args.imputeformer_output)
    if not imputeformer_output.exists():
        latest_aligned = latest_file(
            imputeformer_output.parent, "aligned", "stage1_imputeformer_aligned_output_*.csv")
        if latest_aligned is not None:
            imputeformer_output = latest_aligned
    logger.info(
        "Representation inference started: aligned_time_series=%s optical_image=%s "
        "sonar_source=%s output_dir=%s",
        imputeformer_output.resolve(),
        optical_source_path,
        sonar_source_path,
        output_dir.resolve(),
    )
    aligned_sensors = load_aligned_sensors(str(imputeformer_output))
    aligned_dir = output_dir / "aligned"
    if imputeformer_output.resolve().parent == aligned_dir.resolve():
        retained_imputeformer = imputeformer_output
    else:
        retained_imputeformer = artifact_path(
            output_dir, "aligned", "stage1_imputeformer_aligned_output", ".csv", timestamp)
        copy_file_once(imputeformer_output, retained_imputeformer)

    logger.info("First-stage extraction started: time_series rows=%d", len(aligned_sensors))
    time_first_stage, sensor_ids = extract_first_stage_features(aligned_sensors)
    time_first_stage_path = artifact_path(
        output_dir, "features", "stage2a_time_series_first_stage_features", ".csv", timestamp)
    write_csv_once(time_first_stage, time_first_stage_path)
    log_feature_frame(
        logger, "First-stage features", "time_series", time_first_stage,
        time_first_stage_path,
    )

    logger.info("First-stage extraction started: optical_image and sonar_image")
    optical_first_stage = extract_image_source_features(optical_sources, "optical_image")
    sonar_first_stage = extract_image_source_features(
        sonar_sources, "sonar_image", allow_xtf=allow_xtf,
    )
    optical_first_stage_path = artifact_path(
        output_dir, "features", "stage2b_optical_image_first_stage_features", ".csv", timestamp)
    sonar_first_stage_path = artifact_path(
        output_dir, "features", "stage2c_sonar_image_first_stage_features", ".csv", timestamp)
    write_csv_once(optical_first_stage, optical_first_stage_path)
    write_csv_once(sonar_first_stage, sonar_first_stage_path)
    log_feature_frame(
        logger, "First-stage features", "optical_image", optical_first_stage,
        optical_first_stage_path,
    )
    log_feature_frame(
        logger, "First-stage features", "sonar_image", sonar_first_stage,
        sonar_first_stage_path,
    )

    logger.info("Second-stage Transformer extraction started: sensor_count=%d", len(sensor_ids))
    (
        _,
        reconstructed_risk_features,
        _,
        _,
        _,
        second_stage_features,
    ) = run_transformer_stage(
        time_first_stage=time_first_stage,
        optical_first_stage=optical_first_stage,
        sonar_first_stage=sonar_first_stage,
        args=args,
        model_bundle=model_bundle,
        include_row_embeddings=True,
    )
    second_stage_paths = {}
    second_stage_stems = {
        "time_series": "stage3a_time_series_second_stage_unified_128d",
        "optical_image": "stage3b_optical_image_second_stage_unified_128d",
        "sonar_image": "stage3c_sonar_image_second_stage_unified_128d",
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
        log_feature_frame(
            logger, "Second-stage features", modality,
            second_stage_features[modality], path,
        )
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
    logger.info(
        "Risk calculation started: fusion_checkpoint=%s risk_config=%s",
        fusion_checkpoint_path, risk_config_path.resolve(),
    )

    selected_vectors, feature_risk_matrix, overall_risk_matrix = (
        build_second_stage_feature_risk_matrices(
            reconstructed_risk_features,
            risk_config,
        )
    )

    modality_sources = {
        modality: second_stage_paths[modality].relative_to(output_dir).as_posix()
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
        "schema_version": 2,
        "artifact_timestamp": timestamp,
        "second_stage_feature_stage": "stage3_transformer_unified_embedding",
        "embedding_dimension": 128,
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
                "source_representation": "transformer_fused_cls",
                "source_feature_columns": "z_000..z_127",
                "source_context": "target_modality_row_with_other_modality_means",
                "risk_cls_token_index": 0,
                "transformer_token_index_role": "input_modality_token",
                "risk_feature_derivation": (
                    "unified_decoder_then_inverse_normalization_then_selected_feature"
                ),
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

    logger.info(
        "Risk calculation complete: risk_matrix=%s result=%s",
        risk_json_path.resolve(), json.dumps(risk_result, ensure_ascii=False),
    )
    if getattr(args, "print_result", True):
        print(json.dumps(risk_result, indent=2, ensure_ascii=False))

    return {
        "risk_result": risk_result,
        "risk_json_path": risk_json_path,
        "artifacts": {
            "aligned": str(retained_imputeformer),
            "first_stage_features": {
                "time_series": str(time_first_stage_path),
                "optical_image": str(optical_first_stage_path),
                "sonar_image": str(sonar_first_stage_path),
            },
            "second_stage_features": {
                modality: str(path) for modality, path in second_stage_paths.items()
            },
            "risk_matrix": str(risk_json_path),
        },
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
    """Return named decoder features for the existing risk calibration contract."""
    _, reconstructed = encode_second_stage_features(
        model, normalization, feature_frames, batch_size,
        include_embeddings=False,
    )
    return reconstructed


def encode_second_stage_features(
    model,
    normalization,
    feature_frames,
    batch_size=8192,
    *,
    include_embeddings=True,
):
    """Return fused CLS vectors and named risk-decoder values from the same pass.

    The target modality keeps its original row granularity. Other modalities
    contribute their current-table mean, which avoids an arbitrary Cartesian
    product while ensuring every decoded row is cross-modal.
    """
    if batch_size < 1:
        raise ValueError("Second-stage batch_size must be positive.")
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
    unified_features = {}
    reconstructed_features = {}
    for target_modality in FUSION_MODALITIES:
        target_values = normalized_values[target_modality]
        logger.info(
            "Second-stage Transformer encoding modality=%s rows=%d embedding_dim=%d "
            "representation=fused_cls device=%s batch_size=%d",
            target_modality, len(target_values), model.d_model, device, batch_size,
        )
        reconstructed_batches = []
        unified_batches = []
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
                output = model(batch_inputs)
                reconstructed = output["unified_reconstruction"][target_modality]
            reconstructed_batches.append(reconstructed.cpu().numpy())
            if include_embeddings:
                unified_batches.append(output["unified"].cpu().numpy())

        reconstructed_normalized = np.concatenate(reconstructed_batches, axis=0)
        mean = np.asarray(normalization[target_modality]["mean"], dtype="float32")
        scale = np.asarray(normalization[target_modality]["scale"], dtype="float32")
        reconstructed_original_units = reconstructed_normalized * scale + mean
        source_frame = feature_frames[target_modality]
        feature_columns = list(FUSION_FEATURE_COLUMNS[target_modality])
        metadata_columns = [
            column for column in source_frame.columns if column not in feature_columns
        ]
        row_metadata = source_frame.loc[:, metadata_columns].reset_index(drop=True)
        if include_embeddings:
            vectors = np.concatenate(unified_batches, axis=0)
            if vectors.shape != (len(source_frame), model.d_model) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid unified Transformer vectors for {target_modality}.")
            vector_frame = pd.DataFrame(
                vectors, columns=[f"z_{index:03d}" for index in range(model.d_model)],
            )
            unified_features[target_modality] = pd.concat([row_metadata, vector_frame], axis=1)
        reconstructed_frame = row_metadata.copy()
        for column_index, column in enumerate(feature_columns):
            reconstructed_frame[column] = reconstructed_original_units[:, column_index]
        reconstructed_features[target_modality] = reconstructed_frame
    return unified_features, reconstructed_features


def run_transformer_stage(
    time_first_stage,
    optical_first_stage,
    sonar_first_stage,
    args,
    model_bundle: ProductionModelBundle | None = None,
    *,
    include_row_embeddings: bool = False,
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
    if include_row_embeddings and model.d_model != 128:
        raise ValueError(
            f"Second-stage output requires a 128-dimensional Transformer checkpoint; "
            f"got d_model={model.d_model}."
        )
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
    unified_features, second_stage_features = encode_second_stage_features(
        model,
        normalization,
        feature_frames,
        include_embeddings=include_row_embeddings,
    )

    result = (
        pd.DataFrame(rows),
        second_stage_features,
        pd.DataFrame(range_rows),
        pd.DataFrame(concatenated_rows),
        pd.DataFrame(position_rows),
    )
    return (*result, unified_features) if include_row_embeddings else result


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
    parser.add_argument("--optical-image", help="Optical image or directory; overrides OPTICAL_IMAGE_INPUT_PATH.")
    sonar = parser.add_mutually_exclusive_group()
    sonar.add_argument("--sonar-image", help="Sonar TIFF (.tif/.tiff) file or directory; overrides configured sonar input.")
    sonar.add_argument("--sonar-xtf", help="Legacy raw sonar XTF input; overrides configured sonar input.")
    parser.add_argument("--risk-config", default=str(DEFAULT_RISK_CONFIG))
    parser.add_argument(
        "--fusion-checkpoint",
        default=str(DEFAULT_FUSION_CHECKPOINT),
        help="Trained FeatureFusionTransformer checkpoint; inference fails if absent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
