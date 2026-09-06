"""Calibrate fixed single-feature risk models on Transformer reconstructions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from multisource_unified import (  # noqa: E402
    FUSION_MODALITIES,
    load_feature_fusion_checkpoint,
)
from risk_pipeline.run_encapsulated_pipeline import (  # noqa: E402
    DEFAULT_FUSION_CHECKPOINT,
    reconstruct_second_stage_features,
)
from risk_pipeline.feature_schema import RISK_FEATURES
from risk_pipeline.artifacts import artifact_timestamp


DEFAULT_CONFIG = ROOT / "config" / "fixed_single_feature_risk_config.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "second_stage_risk_calibration"
SELECTED_FEATURES = RISK_FEATURES
FIXED_WEIGHTS = {
    "time_series": 0.20,
    "optical_image": 0.40,
    "sonar_image": 0.40,
}
DATASET_METADATA = {
    "time_series": {
        "validation_dataset": (
            "USGS MATANZAS five-sensor aligned ocean sequences with controlled "
            "drift and spikes"
        ),
        "dataset_reference": "https://doi.org/10.5066/P9GKB537",
        "positive_label": "controlled sensor-scale drift and sparse spikes",
        "risk_interpretation": "standard-deviation proxy for sensor fluctuation",
    },
    "optical_image": {
        "validation_dataset": "UIEB paired underwater image benchmark",
        "dataset_reference": "https://doi.org/10.1109/TIP.2019.2955241",
        "positive_label": "raw or challenging degraded underwater image",
        "risk_interpretation": (
            "edge-density proxy for optical clarity and structural visibility"
        ),
    },
    "sonar_image": {
        "validation_dataset": (
            "Real XTF side-scan ping image with controlled blur and contrast loss"
        ),
        "dataset_reference": "https://doi.org/10.5281/zenodo.5558794",
        "paper_reference": "https://doi.org/10.1175/JPO-D-21-0230.1",
        "positive_label": "controlled blur and contrast loss on real XTF intensity",
        "risk_interpretation": (
            "edge-density proxy for sonar structural visibility"
        ),
    },
}


def _load_splits(feature_root: Path):
    result = {}
    for split in ("train", "validation", "test"):
        result[split] = {
            "time_series": pd.read_parquet(
                feature_root / f"time_series_{split}_candidate_features.parquet"
            ),
            "optical_image": pd.read_parquet(
                feature_root / f"optical_image_{split}_candidate_features.parquet"
            ),
            "sonar_image": pd.read_parquet(
                feature_root / f"sonar_image_{split}_candidate_features.parquet"
            ),
        }
    return result


def _fit(values, labels):
    values = np.asarray(values, dtype="float64").reshape(-1, 1)
    labels = np.asarray(labels, dtype="int64")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(values)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )
    model.fit(standardized, labels)
    return scaler, model


def _predict(fitted, values):
    scaler, model = fitted
    values = np.asarray(values, dtype="float64").reshape(-1, 1)
    return model.predict_proba(scaler.transform(values))[:, 1]


def _metrics(labels, probabilities):
    labels = np.asarray(labels, dtype="int64")
    probabilities = np.asarray(probabilities, dtype="float64")
    predicted = (probabilities >= 0.5).astype("int64")
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    checkpoint_path = Path(args.fusion_checkpoint).resolve()
    model, checkpoint = load_feature_fusion_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    feature_root = Path(args.feature_root).resolve()
    source_splits = _load_splits(feature_root)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "runs" / artifact_timestamp()
    reconstructed_dir = run_dir / "reconstructed_features"
    reconstructed_dir.mkdir(parents=True, exist_ok=False)

    reconstructed_splits = {}
    manifest = {}
    for split, feature_frames in source_splits.items():
        print(f"Reconstructing complete {split} split on {device}...", flush=True)
        reconstructed = reconstruct_second_stage_features(
            model,
            checkpoint["normalization"],
            feature_frames,
            batch_size=args.batch_size,
        )
        reconstructed_splits[split] = reconstructed
        for modality, frame in reconstructed.items():
            path = reconstructed_dir / f"{modality}_{split}_second_stage.parquet"
            frame.to_parquet(path, index=False)
            manifest[f"{modality}_{split}"] = {
                "path": str(path.resolve()),
                "rows": len(frame),
                "sha256": _sha256(path),
            }

    base_config_path = Path(args.base_config)
    config = deepcopy(json.loads(base_config_path.read_text(encoding="utf-8")))
    checkpoint_hash = _sha256(checkpoint_path)
    config.update({
        "schema_version": 6,
        "generated_by": "calibrate_second_stage_risk_config.py",
        "feature_constraint": (
            "Exactly one named Transformer second-stage reconstructed feature per "
            "modality: " + ", ".join(SELECTED_FEATURES.values())
        ),
        "risk_model": (
            "standardized univariate logistic regression calibrated on named "
            "Transformer second-stage reconstructions"
        ),
        "calibration_feature_stage": "stage3_transformer_unified_reconstruction",
        "calibration_context_policy": (
            "each target-modality row is reconstructed through CLS while every "
            "other modality is represented by its current-split mean"
        ),
        "fusion_checkpoint": str(checkpoint_path),
        "fusion_checkpoint_sha256": checkpoint_hash,
        "runtime_weight_renormalization": False,
        "three_modality_data_root": str(feature_root.parent),
    })
    config.pop("public_data_root", None)
    config["modalities"] = {
        modality: deepcopy(config["modalities"][modality])
        for modality in FUSION_MODALITIES
    }

    metric_rows = []
    for modality in FUSION_MODALITIES:
        feature = SELECTED_FEATURES[modality]
        train = reconstructed_splits["train"][modality]
        validation = reconstructed_splits["validation"][modality]
        test = reconstructed_splits["test"][modality]

        validation_model = _fit(train[feature], train["label"])
        validation_metrics = _metrics(
            validation["label"],
            _predict(validation_model, validation[feature]),
        )
        combined = pd.concat([train, validation], ignore_index=True)
        final_model = _fit(combined[feature], combined["label"])
        test_metrics = _metrics(
            test["label"],
            _predict(final_model, test[feature]),
        )
        scaler, logistic = final_model

        spec = config["modalities"][modality]
        spec.update({
            "selected_feature": feature,
            "fixed_weight": FIXED_WEIGHTS[modality],
            "validation_auc": validation_metrics["roc_auc"],
            "test_auc": test_metrics["roc_auc"],
            "test_average_precision": test_metrics["average_precision"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_f1": test_metrics["f1"],
            "scaler_mean": float(scaler.mean_[0]),
            "scaler_scale": float(scaler.scale_[0]),
            "logistic_coefficient": float(logistic.coef_[0, 0]),
            "logistic_intercept": float(logistic.intercept_[0]),
            "calibration_feature_stage": (
                "stage3_transformer_unified_reconstruction"
            ),
            "calibration_train_rows": len(train),
            "calibration_validation_rows": len(validation),
            "calibration_test_rows": len(test),
            **DATASET_METADATA[modality],
        })
        for phase, metrics in (
            ("validation", validation_metrics),
            ("independent_test", test_metrics),
        ):
            metric_rows.append({
                "modality": modality,
                "selected_feature": feature,
                "phase": phase,
                **metrics,
            })

    config_text = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    immutable_config = run_dir / "fixed_single_feature_risk_config.json"
    immutable_config.write_text(config_text, encoding="utf-8")

    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    temporary_config = output_config.with_suffix(output_config.suffix + ".tmp")
    temporary_config.write_text(config_text, encoding="utf-8")
    temporary_config.replace(output_config)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(
        run_dir / "second_stage_calibration_metrics.csv", index=False
    )
    (run_dir / "second_stage_calibration_metrics.json").write_text(
        json.dumps(metric_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration_manifest.json").write_text(
        json.dumps({
            "feature_root": str(Path(args.feature_root).resolve()),
            "fusion_checkpoint": str(checkpoint_path),
            "fusion_checkpoint_sha256": checkpoint_hash,
            "no_source_row_truncation": True,
            "run_directory": str(run_dir.resolve()),
            "files": manifest,
            "immutable_config": str(immutable_config.resolve()),
            "published_config": str(output_config.resolve()),
        }, indent=2),
        encoding="utf-8",
    )
    print(metric_frame.to_string(index=False), flush=True)
    print(f"Updated risk config: {output_config.resolve()}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        required=True,
        help="Feature directory from one immutable three-modality preparation run.",
    )
    parser.add_argument("--fusion-checkpoint", default=str(DEFAULT_FUSION_CHECKPOINT))
    parser.add_argument("--base-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
