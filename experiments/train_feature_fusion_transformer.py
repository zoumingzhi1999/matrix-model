"""Train the production three-modality feature-fusion Transformer.

The public modality datasets are not synchronized observations of the same
event. This script therefore constructs reproducible label-stratified training
tuples inside each official split. Constructed tuples are explicitly recorded
as such and are never presented as physically synchronized measurements.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from multisource_unified import (  # noqa: E402
    FUSION_FEATURE_COLUMNS,
    FUSION_MODALITIES,
    FeatureFusionTransformer,
    feature_input_dims,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "feature_fusion_transformer_production"
DEFAULT_MODEL_DIR = ROOT / "models" / "feature_fusion_transformer"
CHECKPOINT_NAME = "feature_fusion_transformer.pt"
LABELS = (0, 1)

SPLIT_FILES = {
    "time_series": {
        "train": "time_series_train_candidate_features.parquet",
        "validation": "time_series_validation_candidate_features.parquet",
        "test": "time_series_test_candidate_features.parquet",
    },
    "optical_image": {
        "train": "optical_image_train_candidate_features.parquet",
        "validation": "optical_image_validation_candidate_features.parquet",
        "test": "optical_image_test_candidate_features.parquet",
    },
    "sonar_image": {
        "train": "sonar_image_train_candidate_features.parquet",
        "validation": "sonar_image_validation_candidate_features.parquet",
        "test": "sonar_image_test_candidate_features.parquet",
    },
}

LABEL_SEMANTICS = {
    "time_series": {0: "normal window", 1: "labeled anomalous window"},
    "optical_image": {0: "reference image", 1: "raw degraded underwater image"},
    "sonar_image": {
        0: "original XTF intensity tile",
        1: "controlled degraded view of the same XTF tile",
    },
}


@dataclass(frozen=True)
class FeatureSplit:
    values: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("Feature values must be a two-dimensional matrix.")
        if self.labels.ndim != 1 or len(self.labels) != len(self.values):
            raise ValueError("Feature labels must match the feature row count.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_split(path: Path, modality: str) -> FeatureSplit:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature file does not exist: {path}")
    columns = list(FUSION_FEATURE_COLUMNS[modality])
    frame = pd.read_parquet(path, columns=[*columns, "label"])
    missing = [column for column in [*columns, "label"] if column not in frame]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    values = frame.loc[:, columns].to_numpy(dtype="float32", copy=True)
    labels = frame["label"].to_numpy(dtype="int64", copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite feature values.")
    unexpected = sorted(set(labels.tolist()) - set(LABELS))
    if unexpected:
        raise ValueError(f"{path.name} contains unsupported labels: {unexpected}")
    for label in LABELS:
        if not np.any(labels == label):
            raise ValueError(f"{path.name} has no samples for label {label}.")
    return FeatureSplit(values=values, labels=labels)


def load_raw_corpus(
    feature_root: Path,
) -> tuple[Dict[str, Dict[str, FeatureSplit]], Dict[str, object]]:
    corpus: Dict[str, Dict[str, FeatureSplit]] = {
        modality: {} for modality in FUSION_MODALITIES
    }
    manifest_files: Dict[str, object] = {}

    for modality, split_files in SPLIT_FILES.items():
        for split, filename in split_files.items():
            path = feature_root / filename
            data = _read_split(path, modality)
            corpus[modality][split] = data
            manifest_files[filename] = _file_manifest(path, data)

    manifest = {
        "feature_root": str(feature_root.resolve()),
        "files": manifest_files,
        "splits": corpus_summary(corpus),
    }
    return corpus, manifest


def _file_manifest(path: Path, data: FeatureSplit) -> Dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(len(data.labels)),
        "label_counts": label_counts(data.labels),
    }


def label_counts(labels: np.ndarray) -> Dict[str, int]:
    return {str(label): int(np.sum(labels == label)) for label in LABELS}


def corpus_summary(
    corpus: Mapping[str, Mapping[str, FeatureSplit]],
) -> Dict[str, object]:
    return {
        split: {
            modality: {
                "rows": int(len(corpus[modality][split].labels)),
                "label_counts": label_counts(corpus[modality][split].labels),
            }
            for modality in FUSION_MODALITIES
        }
        for split in ("train", "validation", "test")
    }


def fit_normalization(
    corpus: Mapping[str, Mapping[str, FeatureSplit]],
) -> Dict[str, Dict[str, np.ndarray]]:
    normalization: Dict[str, Dict[str, np.ndarray]] = {}
    for modality in FUSION_MODALITIES:
        train = corpus[modality]["train"].values.astype("float64")
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        normalization[modality] = {
            "mean": mean.astype("float32"),
            "scale": scale.astype("float32"),
        }
    return normalization


def apply_normalization(
    corpus: Mapping[str, Mapping[str, FeatureSplit]],
    normalization: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, FeatureSplit]]:
    normalized: Dict[str, Dict[str, FeatureSplit]] = {
        modality: {} for modality in FUSION_MODALITIES
    }
    for modality in FUSION_MODALITIES:
        mean = normalization[modality]["mean"]
        scale = normalization[modality]["scale"]
        for split, data in corpus[modality].items():
            values = ((data.values - mean) / scale).astype("float32")
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Normalization produced non-finite values for {modality}/{split}."
                )
            normalized[modality][split] = FeatureSplit(
                values=values,
                labels=data.labels.copy(),
            )
    return normalized


def select_split(
    corpus: Mapping[str, Mapping[str, FeatureSplit]], split: str
) -> Dict[str, FeatureSplit]:
    return {modality: corpus[modality][split] for modality in FUSION_MODALITIES}


def combine_splits(
    corpus: Mapping[str, Mapping[str, FeatureSplit]],
    splits: Sequence[str],
) -> Dict[str, FeatureSplit]:
    return {
        modality: FeatureSplit(
            values=np.concatenate(
                [corpus[modality][split].values for split in splits], axis=0
            ),
            labels=np.concatenate(
                [corpus[modality][split].labels for split in splits], axis=0
            ),
        )
        for modality in FUSION_MODALITIES
    }


def construct_pair_indices(
    data: Mapping[str, FeatureSplit],
    seed: int,
    balance_labels: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Create deterministic, label-stratified constructed tuples.

    Each modality's samples are independently permuted in complete cycles.
    Consequently every sample is used at least once before any partial final
    cycle, while tuple members always share the same binary label.
    """
    rng = np.random.default_rng(seed)
    source_indices = {
        modality: {
            label: np.flatnonzero(data[modality].labels == label)
            for label in LABELS
        }
        for modality in FUSION_MODALITIES
    }
    largest_stratum = max(
        len(source_indices[modality][label])
        for modality in FUSION_MODALITIES
        for label in LABELS
    )
    target_by_label = {
        label: (
            largest_stratum
            if balance_labels
            else max(
                len(source_indices[modality][label])
                for modality in FUSION_MODALITIES
            )
        )
        for label in LABELS
    }

    chunks: Dict[str, list[np.ndarray]] = {
        modality: [] for modality in FUSION_MODALITIES
    }
    usage: Dict[str, object] = {modality: {} for modality in FUSION_MODALITIES}
    tuple_labels = []
    for label in LABELS:
        target = target_by_label[label]
        tuple_labels.append(np.full(target, label, dtype="int64"))
        for modality in FUSION_MODALITIES:
            base = source_indices[modality][label]
            sampled_parts = []
            remaining = target
            while remaining:
                permutation = rng.permutation(base)
                take = min(remaining, len(permutation))
                sampled_parts.append(permutation[:take])
                remaining -= take
            sampled = np.concatenate(sampled_parts)
            chunks[modality].append(sampled)
            counts = np.bincount(
                np.searchsorted(np.sort(base), sampled), minlength=len(base)
            )
            usage[modality][str(label)] = {
                "source_samples": int(len(base)),
                "constructed_uses": int(target),
                "minimum_uses_per_source_sample": int(counts.min()),
                "maximum_uses_per_source_sample": int(counts.max()),
            }

    paired = {
        modality: np.concatenate(chunks[modality])
        for modality in FUSION_MODALITIES
    }
    labels = np.concatenate(tuple_labels)
    order = rng.permutation(len(labels))
    paired = {modality: indices[order] for modality, indices in paired.items()}
    labels = labels[order]
    audit = {
        "seed": int(seed),
        "balanced_labels": bool(balance_labels),
        "constructed_tuple_count": int(len(labels)),
        "constructed_label_counts": label_counts(labels),
        "usage": usage,
        "all_source_samples_used": all(
            entry["minimum_uses_per_source_sample"] >= 1
            for modality_usage in usage.values()
            for entry in modality_usage.values()
        ),
    }
    if not audit["all_source_samples_used"]:
        raise RuntimeError("Constructed pairing failed to use every source sample.")
    return paired, audit


def _batch_features(
    data: Mapping[str, FeatureSplit],
    indices: Mapping[str, np.ndarray],
    start: int,
    end: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        modality: torch.from_numpy(
            data[modality].values[indices[modality][start:end]]
        ).to(device=device, non_blocking=True)
        for modality in FUSION_MODALITIES
    }


def corrupt_features(
    clean: Mapping[str, torch.Tensor],
    feature_mask_probability: float,
    modality_drop_probability: float,
    noise_std: float,
) -> Dict[str, torch.Tensor]:
    batch_size = next(iter(clean.values())).shape[0]
    device = next(iter(clean.values())).device
    dropped = torch.rand(
        batch_size, len(FUSION_MODALITIES), device=device
    ) < modality_drop_probability
    all_dropped = dropped.all(dim=1)
    if all_dropped.any():
        keep = torch.randint(
            0, len(FUSION_MODALITIES), (int(all_dropped.sum()),), device=device
        )
        dropped[all_dropped, keep] = False

    corrupted = {}
    for modality_index, modality in enumerate(FUSION_MODALITIES):
        values = clean[modality]
        noisy = values + torch.randn_like(values) * noise_std
        feature_mask = torch.rand_like(values) < feature_mask_probability
        noisy = noisy.masked_fill(feature_mask, 0.0)
        noisy = noisy.masked_fill(dropped[:, modality_index].unsqueeze(1), 0.0)
        corrupted[modality] = noisy
    return corrupted


def _reconstruction_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return torch.stack([
        F.mse_loss(predictions[modality], targets[modality])
        for modality in FUSION_MODALITIES
    ]).mean()


def training_loss(
    model: FeatureFusionTransformer,
    clean: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    view_a = corrupt_features(
        clean,
        args.feature_mask_probability,
        args.modality_drop_probability,
        args.noise_std,
    )
    view_b = corrupt_features(
        clean,
        args.feature_mask_probability,
        args.modality_drop_probability,
        args.noise_std,
    )
    output_a = model(view_a)
    output_b = model(view_b)
    unified = 0.5 * (
        _reconstruction_loss(output_a["unified_reconstruction"], clean)
        + _reconstruction_loss(output_b["unified_reconstruction"], clean)
    )
    token = 0.5 * (
        _reconstruction_loss(output_a["token_reconstruction"], clean)
        + _reconstruction_loss(output_b["token_reconstruction"], clean)
    )
    consistency = (
        1.0
        - F.cosine_similarity(
            output_a["unified"], output_b["unified"], dim=1
        )
    ).mean()
    total = (
        args.unified_loss_weight * unified
        + args.token_loss_weight * token
        + args.consistency_loss_weight * consistency
    )
    return total, {
        "loss": float(total.detach()),
        "unified_reconstruction": float(unified.detach()),
        "token_reconstruction": float(token.detach()),
        "consistency": float(consistency.detach()),
    }


def train_epochs(
    model: FeatureFusionTransformer,
    train_data: Mapping[str, FeatureSplit],
    validation_data: Optional[Mapping[str, FeatureSplit]],
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed_offset: int,
    learning_rate_schedule: Optional[Sequence[float]] = None,
) -> tuple[list[Dict[str, object]], Dict[str, torch.Tensor], int, Dict[str, object]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if validation_data is not None and learning_rate_schedule is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.learning_rate_patience,
            threshold=args.minimum_improvement,
            threshold_mode="abs",
            min_lr=args.minimum_learning_rate,
        )
    if learning_rate_schedule is not None and len(learning_rate_schedule) < epochs:
        raise ValueError("The replay learning-rate schedule is shorter than epochs.")
    validation_indices = None
    validation_audit = None
    if validation_data is not None:
        validation_indices, validation_audit = construct_pair_indices(
            validation_data,
            seed=args.seed + 20_000,
            balance_labels=False,
        )

    history: list[Dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation = math.inf
    epochs_without_improvement = 0
    first_pairing_audit = None
    last_pairing_audit = None

    for epoch in range(epochs):
        epoch_seed = args.seed + seed_offset + epoch
        set_seed(epoch_seed)
        if learning_rate_schedule is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = float(learning_rate_schedule[epoch])
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        pair_indices, pairing_audit = construct_pair_indices(
            train_data,
            seed=epoch_seed,
            balance_labels=False,
        )
        if first_pairing_audit is None:
            first_pairing_audit = pairing_audit
        last_pairing_audit = pairing_audit

        model.train()
        total_examples = len(next(iter(pair_indices.values())))
        sums = {
            "loss": 0.0,
            "unified_reconstruction": 0.0,
            "token_reconstruction": 0.0,
            "consistency": 0.0,
        }
        epoch_start = time.time()
        for start in range(0, total_examples, args.batch_size):
            end = min(start + args.batch_size, total_examples)
            clean = _batch_features(train_data, pair_indices, start, end, device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(model, clean, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_count = end - start
            for name, value in parts.items():
                sums[name] += value * batch_count
        record: Dict[str, object] = {
            "epoch": epoch + 1,
            "learning_rate": epoch_learning_rate,
            "train_examples": int(total_examples),
            "seconds": float(time.time() - epoch_start),
            **{f"train_{name}": value / total_examples for name, value in sums.items()},
        }
        if validation_data is not None:
            validation = evaluate_reconstruction(
                model,
                validation_data,
                validation_indices,
                args,
                device,
                corrupt=True,
                seed=args.seed + 30_000,
            )
            validation_loss = float(validation["overall_unified_mse"])
            record["validation"] = validation
            if scheduler is not None:
                scheduler.step(validation_loss)
            if validation_loss < best_validation - args.minimum_improvement:
                best_validation = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if (
            validation_data is not None
            and epochs_without_improvement >= args.patience
        ):
            break

    audit = {
        "first_epoch": first_pairing_audit,
        "last_epoch": last_pairing_audit,
        "validation": validation_audit,
    }
    return history, best_state, best_epoch, audit


def evaluate_reconstruction(
    model: FeatureFusionTransformer,
    data: Mapping[str, FeatureSplit],
    pair_indices: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    corrupt: bool,
    seed: int,
) -> Dict[str, object]:
    set_seed(seed)
    model.eval()
    total_examples = len(next(iter(pair_indices.values())))
    unified_sse = {modality: 0.0 for modality in FUSION_MODALITIES}
    token_sse = {modality: 0.0 for modality in FUSION_MODALITIES}
    unified_absolute = {modality: 0.0 for modality in FUSION_MODALITIES}
    token_absolute = {modality: 0.0 for modality in FUSION_MODALITIES}
    target_absolute = {modality: 0.0 for modality in FUSION_MODALITIES}
    feature_counts = {modality: 0 for modality in FUSION_MODALITIES}
    representation_sum = torch.zeros(model.d_model, dtype=torch.float64)
    representation_square_sum = torch.zeros(model.d_model, dtype=torch.float64)

    with torch.no_grad():
        for start in range(0, total_examples, args.evaluation_batch_size):
            end = min(start + args.evaluation_batch_size, total_examples)
            clean = _batch_features(data, pair_indices, start, end, device)
            inputs = (
                corrupt_features(
                    clean,
                    args.feature_mask_probability,
                    args.modality_drop_probability,
                    args.noise_std,
                )
                if corrupt
                else clean
            )
            output = model(inputs)
            for modality in FUSION_MODALITIES:
                unified_delta = (
                    output["unified_reconstruction"][modality] - clean[modality]
                )
                token_delta = (
                    output["token_reconstruction"][modality] - clean[modality]
                )
                unified_error = unified_delta.square()
                token_error = token_delta.square()
                unified_sse[modality] += float(unified_error.sum().cpu())
                token_sse[modality] += float(token_error.sum().cpu())
                unified_absolute[modality] += float(
                    unified_delta.abs().sum().cpu()
                )
                token_absolute[modality] += float(token_delta.abs().sum().cpu())
                target_absolute[modality] += float(clean[modality].abs().sum().cpu())
                feature_counts[modality] += int(unified_error.numel())
            representation = output["unified"].double().cpu()
            representation_sum += representation.sum(dim=0)
            representation_square_sum += representation.square().sum(dim=0)

    per_modality = {
        modality: {
            "unified_mae": unified_absolute[modality] / feature_counts[modality],
            "unified_mse": unified_sse[modality] / feature_counts[modality],
            "unified_rmse": math.sqrt(
                unified_sse[modality] / feature_counts[modality]
            ),
            "unified_mre": (
                unified_absolute[modality] / target_absolute[modality]
                if target_absolute[modality] > 1e-12 else None
            ),
            "token_mae": token_absolute[modality] / feature_counts[modality],
            "token_mse": token_sse[modality] / feature_counts[modality],
            "token_rmse": math.sqrt(
                token_sse[modality] / feature_counts[modality]
            ),
            "token_mre": (
                token_absolute[modality] / target_absolute[modality]
                if target_absolute[modality] > 1e-12 else None
            ),
        }
        for modality in FUSION_MODALITIES
    }
    mean = representation_sum / total_examples
    variance = representation_square_sum / total_examples - mean.square()
    representation_mean_std = float(
        torch.sqrt(torch.clamp(variance, min=0.0)).mean()
    )
    overall_unified_mse = float(np.mean([
        per_modality[modality]["unified_mse"]
        for modality in FUSION_MODALITIES
    ]))
    overall_token_mse = float(np.mean([
        per_modality[modality]["token_mse"]
        for modality in FUSION_MODALITIES
    ]))
    return {
        "corrupted_input": bool(corrupt),
        "examples": int(total_examples),
        "overall_unified_mae": float(np.mean([
            per_modality[modality]["unified_mae"]
            for modality in FUSION_MODALITIES
        ])),
        "overall_unified_mse": overall_unified_mse,
        "overall_unified_rmse": math.sqrt(overall_unified_mse),
        "overall_unified_mre": float(np.mean([
            per_modality[modality]["unified_mre"]
            for modality in FUSION_MODALITIES
            if per_modality[modality]["unified_mre"] is not None
        ])),
        "overall_token_mae": float(np.mean([
            per_modality[modality]["token_mae"]
            for modality in FUSION_MODALITIES
        ])),
        "overall_token_mse": overall_token_mse,
        "overall_token_rmse": math.sqrt(overall_token_mse),
        "overall_token_mre": float(np.mean([
            per_modality[modality]["token_mre"]
            for modality in FUSION_MODALITIES
            if per_modality[modality]["token_mre"] is not None
        ])),
        "representation_mean_dimension_std": representation_mean_std,
        "per_modality": per_modality,
    }


def zero_reconstruction_baseline(
    data: Mapping[str, FeatureSplit],
    pair_indices: Mapping[str, np.ndarray],
) -> Dict[str, object]:
    per_modality = {}
    for modality in FUSION_MODALITIES:
        values = data[modality].values[pair_indices[modality]].astype("float64")
        mse = float(np.mean(np.square(values), dtype="float64"))
        absolute = float(np.abs(values).sum(dtype="float64"))
        target_absolute = float(np.abs(values).sum(dtype="float64"))
        per_modality[modality] = {
            "MAE": float(np.mean(np.abs(values), dtype="float64")),
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "MRE": absolute / target_absolute if target_absolute > 1e-12 else None,
        }
    return {
        "definition": "all reconstructed normalized features are zero",
        "overall_unified_mae": float(np.mean([
            values["MAE"] for values in per_modality.values()
        ])),
        "overall_unified_mse": float(np.mean([
            values["MSE"] for values in per_modality.values()
        ])),
        "overall_unified_rmse": math.sqrt(float(np.mean([
            values["MSE"] for values in per_modality.values()
        ]))),
        "overall_unified_mre": float(np.mean([
            values["MRE"] for values in per_modality.values()
            if values["MRE"] is not None
        ])),
        "per_modality": per_modality,
    }


def evaluate_ablation_and_sensitivity(
    model: FeatureFusionTransformer,
    data: Mapping[str, FeatureSplit],
    pair_indices: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    set_seed(args.seed + 50_000)
    model.eval()
    total_examples = len(next(iter(pair_indices.values())))
    accumulators: MutableMapping[str, MutableMapping[str, float]] = {
        modality: {
            "ablation_cosine_distance": 0.0,
            "ablation_l2_per_dimension": 0.0,
            "ablation_reconstruction_mse": 0.0,
            "clean_reconstruction_mse": 0.0,
            "noise_cosine_distance": 0.0,
        }
        for modality in FUSION_MODALITIES
    }

    with torch.no_grad():
        for start in range(0, total_examples, args.evaluation_batch_size):
            end = min(start + args.evaluation_batch_size, total_examples)
            batch_count = end - start
            clean = _batch_features(data, pair_indices, start, end, device)
            baseline = model(clean)
            clean_mse = _reconstruction_loss(
                baseline["unified_reconstruction"], clean
            )
            for modality in FUSION_MODALITIES:
                ablated_inputs = dict(clean)
                ablated_inputs[modality] = torch.zeros_like(clean[modality])
                ablated = model(ablated_inputs)
                cosine = (
                    1.0
                    - F.cosine_similarity(
                        baseline["unified"], ablated["unified"], dim=1
                    )
                ).mean()
                l2 = (
                    baseline["unified"] - ablated["unified"]
                ).square().mean()
                ablated_mse = _reconstruction_loss(
                    ablated["unified_reconstruction"], clean
                )

                noisy_inputs = dict(clean)
                noisy_inputs[modality] = (
                    clean[modality]
                    + torch.randn_like(clean[modality]) * args.sensitivity_noise_std
                )
                noisy = model(noisy_inputs)
                noise_cosine = (
                    1.0
                    - F.cosine_similarity(
                        baseline["unified"], noisy["unified"], dim=1
                    )
                ).mean()
                values = accumulators[modality]
                values["ablation_cosine_distance"] += float(cosine.cpu()) * batch_count
                values["ablation_l2_per_dimension"] += float(l2.cpu()) * batch_count
                values["ablation_reconstruction_mse"] += float(ablated_mse.cpu()) * batch_count
                values["clean_reconstruction_mse"] += float(clean_mse.cpu()) * batch_count
                values["noise_cosine_distance"] += float(noise_cosine.cpu()) * batch_count

    report = {}
    for modality, sums in accumulators.items():
        values = {name: value / total_examples for name, value in sums.items()}
        values["ablation_reconstruction_mse_increase"] = (
            values["ablation_reconstruction_mse"]
            - values["clean_reconstruction_mse"]
        )
        report[modality] = values
    return {
        "examples": int(total_examples),
        "sensitivity_noise_std_in_normalized_units": args.sensitivity_noise_std,
        "modalities": report,
    }


def normalization_for_json(
    normalization: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, object]:
    return {
        modality: {
            "columns": list(FUSION_FEATURE_COLUMNS[modality]),
            "mean": normalization[modality]["mean"].tolist(),
            "scale": normalization[modality]["scale"].tolist(),
        }
        for modality in FUSION_MODALITIES
    }


def model_configuration(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "input_dims": feature_input_dims(),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }


def build_model(args: argparse.Namespace, device: torch.device) -> FeatureFusionTransformer:
    model = FeatureFusionTransformer(**model_configuration(args))
    return model.to(device)


def write_model_card(
    path: Path,
    checkpoint_path: Path,
    metrics: Mapping[str, object],
    manifest: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    path.write_text(
        f"""# Feature Fusion Transformer Model Card

## Intended use

This checkpoint fuses {len(FUSION_FEATURE_COLUMNS['time_series'])} time-series,
{len(FUSION_FEATURE_COLUMNS['optical_image'])} optical-image, and
{len(FUSION_FEATURE_COLUMNS['sonar_image'])} XTF-derived sonar-image first-stage features into a
{args.d_model}-dimensional
representation. It has no risk-prediction head. Named features decoded from
the CLS representation may supply the separately calibrated inputs to the
fixed formula `0.20*time_series + 0.40*optical + 0.40*sonar`;
the Transformer itself never predicts risk.

## Training data and pairing

The three public datasets are not physically synchronized. Training tuples were
constructed inside train/validation/test boundaries by matching the binary
label and independently permuting each modality every epoch. Label meanings
differ by modality and are recorded in the checkpoint. Constructed tuples must
not be described as simultaneous real-world observations.

All source samples in every training epoch were used at least once. Training
used the full feature tables; no row cap or random subset was applied.
Normalization was fitted on training data only.

## Objective

The model was trained with masked/noisy feature reconstruction from the CLS
fusion vector, modality-token reconstruction, and consistency between two
corrupted views. There is no classification or risk loss.

## Independent test result

- Trained corrupted-input unified reconstruction MAE: {metrics['trained_test_corrupted']['overall_unified_mae']:.8f}
- Trained corrupted-input unified reconstruction MSE: {metrics['trained_test_corrupted']['overall_unified_mse']:.8f}
- Trained corrupted-input unified reconstruction RMSE: {metrics['trained_test_corrupted']['overall_unified_rmse']:.8f}
- Trained corrupted-input unified reconstruction MRE: {metrics['trained_test_corrupted']['overall_unified_mre']:.8f}
- Random-model corrupted-input unified reconstruction MSE: {metrics['random_model_test_corrupted']['overall_unified_mse']:.8f}
- Zero-reconstruction baseline MSE: {metrics['zero_reconstruction_test']['overall_unified_mse']:.8f}
- Trained clean-input unified reconstruction MSE: {metrics['trained_test_clean']['overall_unified_mse']:.8f}

## Limitations

Label-stratified constructed tuples can teach label-level cross-modal
regularities, but they cannot establish physical time synchronization or prove
event-level correspondence. A future checkpoint should be compared on a
sufficiently large, licensed, truly synchronized three-modality dataset when
one is available. Sonar quality labels are controlled degradations of real XTF
tiles, not manually annotated operational hazards.

Checkpoint: `{checkpoint_path}`

Feature manifest root: `{manifest['feature_root']}`
""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    feature_root = Path(args.feature_root)
    output_root = Path(args.output_dir)
    model_root = Path(args.model_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = model_root / "runs" / run_id

    raw_corpus, data_manifest = load_raw_corpus(feature_root)
    normalization = fit_normalization(raw_corpus)
    corpus = apply_normalization(raw_corpus, normalization)
    train_data = select_split(corpus, "train")
    validation_data = select_split(corpus, "validation")
    test_data = select_split(corpus, "test")

    validation_model = build_model(args, device)
    history, best_state, selected_epochs, selection_pairing = train_epochs(
        validation_model,
        train_data,
        validation_data,
        args.epochs,
        args,
        device,
        seed_offset=1_000,
    )
    if selected_epochs < 1:
        raise RuntimeError("Validation did not select a usable training epoch.")
    validation_model.load_state_dict(best_state)

    # Refit from scratch on train+validation for the validation-selected epoch
    # count. The independent test split remains untouched until final evaluation.
    set_seed(args.seed + 100_000)
    production_model = build_model(args, device)
    refit_data = combine_splits(corpus, ("train", "validation"))
    refit_history, final_state, _, refit_pairing = train_epochs(
        production_model,
        refit_data,
        validation_data=None,
        epochs=selected_epochs,
        args=args,
        device=device,
        seed_offset=100_000,
        learning_rate_schedule=[
            float(record["learning_rate"])
            for record in history[:selected_epochs]
        ],
    )
    production_model.load_state_dict(final_state)
    production_model.eval()

    test_indices, test_pairing = construct_pair_indices(
        test_data,
        seed=args.seed + 40_000,
        balance_labels=False,
    )
    trained_clean = evaluate_reconstruction(
        production_model,
        test_data,
        test_indices,
        args,
        device,
        corrupt=False,
        seed=args.seed + 60_000,
    )
    trained_corrupted = evaluate_reconstruction(
        production_model,
        test_data,
        test_indices,
        args,
        device,
        corrupt=True,
        seed=args.seed + 60_001,
    )

    set_seed(args.seed + 200_000)
    random_model = build_model(args, device)
    random_model.eval()
    random_corrupted = evaluate_reconstruction(
        random_model,
        test_data,
        test_indices,
        args,
        device,
        corrupt=True,
        seed=args.seed + 60_001,
    )
    zero_baseline = zero_reconstruction_baseline(test_data, test_indices)
    ablation = evaluate_ablation_and_sensitivity(
        production_model, test_data, test_indices, args, device
    )

    if (
        trained_corrupted["overall_unified_mse"]
        >= random_corrupted["overall_unified_mse"]
        or trained_corrupted["overall_unified_mse"]
        >= zero_baseline["overall_unified_mse"]
    ):
        raise RuntimeError(
            "The trained model did not beat both the random-model and zero-output "
            "reconstruction baselines."
        )

    metrics = {
        "selected_epochs": selected_epochs,
        "trained_test_clean": trained_clean,
        "trained_test_corrupted": trained_corrupted,
        "random_model_test_corrupted": random_corrupted,
        "zero_reconstruction_test": zero_baseline,
        "ablation_and_sensitivity": ablation,
    }
    pairing_audit = {
        "pair_type": "constructed_label_stratified_not_physically_synchronized",
        "label_semantics": LABEL_SEMANTICS,
        "split_isolation": "pairing is performed separately inside each split",
        "training_epoch_rule": (
            "natural binary-label counts are retained; inside each label, rows are "
            "expanded only to the largest modality count and each modality is "
            "independently permuted in complete cycles"
        ),
        "validation_and_test_rule": (
            "natural label counts are retained; each label is expanded only to "
            "the largest modality count in that label"
        ),
        "selection_training": selection_pairing,
        "production_refit": refit_pairing,
        "independent_test": test_pairing,
    }

    model_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = model_dir / CHECKPOINT_NAME
    checkpoint = {
        "schema_version": 1,
        "model_class": "FeatureFusionTransformer",
        "has_risk_head": False,
        "model_config": model_configuration(args),
        "feature_columns": {
            modality: list(FUSION_FEATURE_COLUMNS[modality])
            for modality in FUSION_MODALITIES
        },
        "normalization": normalization_for_json(normalization),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in production_model.state_dict().items()
        },
        "training_config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "selected_epochs": selected_epochs,
        "data_manifest": data_manifest,
        "pairing_audit": pairing_audit,
        "metrics": metrics,
        "risk_contract": {
            "transformer_role": "feature_fusion_and_named_reconstruction_only",
            "transformer_has_risk_head": False,
            "risk_feature_stage": "stage3_transformer_unified_reconstruction",
            "direct_unified_vector_used_as_risk_probability": False,
            "named_unified_reconstruction_used_by_fixed_risk_formula": True,
            "fixed_formula": {
                "time_series": 0.20,
                "optical_image": 0.40,
                "sonar_image": 0.40,
            },
        },
        "runtime": {
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
            "elapsed_seconds": time.time() - started,
        },
    }
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "production_refit_history.json").write_text(
        json.dumps(refit_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "independent_test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "pairing_audit.json").write_text(
        json.dumps(pairing_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_model_card(
        output_dir / "MODEL_CARD.md",
        checkpoint_path,
        metrics,
        data_manifest,
        args,
    )
    model_root.mkdir(parents=True, exist_ok=True)
    active_checkpoint = model_root / CHECKPOINT_NAME
    temporary_active = active_checkpoint.with_suffix(".pt.tmp")
    shutil.copyfile(checkpoint_path, temporary_active)
    temporary_active.replace(active_checkpoint)
    print(json.dumps({
        "checkpoint": str(active_checkpoint),
        "immutable_training_checkpoint": str(checkpoint_path),
        "selected_epochs": selected_epochs,
        "trained_test_corrupted_mse": trained_corrupted["overall_unified_mse"],
        "random_test_corrupted_mse": random_corrupted["overall_unified_mse"],
        "zero_test_mse": zero_baseline["overall_unified_mse"],
        "elapsed_seconds": time.time() - started,
    }, indent=2), flush=True)
    return active_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a full three-modality feature-fusion Transformer using all "
            "available public first-stage feature rows."
        )
    )
    parser.add_argument(
        "--feature-root",
        required=True,
        help="Feature directory from one immutable three-modality preparation run.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--evaluation-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-patience", type=int, default=4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-mask-probability", type=float, default=0.30)
    parser.add_argument("--modality-drop-probability", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--sensitivity-noise-std", type=float, default=0.50)
    parser.add_argument("--unified-loss-weight", type=float, default=1.0)
    parser.add_argument("--token-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    args = parser.parse_args()
    for probability_name in (
        "feature_mask_probability",
        "modality_drop_probability",
    ):
        probability = getattr(args, probability_name)
        if not 0.0 <= probability < 1.0:
            parser.error(f"--{probability_name.replace('_', '-')} must be in [0, 1).")
    if args.epochs < 1 or args.patience < 1 or args.learning_rate_patience < 1:
        parser.error("epoch and patience values must be positive.")
    if args.batch_size < 1 or args.evaluation_batch_size < 1:
        parser.error("batch sizes must be positive.")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        parser.error(
            "--minimum-learning-rate must be positive and no greater than "
            "--learning-rate."
        )
    return args


if __name__ == "__main__":
    run(parse_args())
