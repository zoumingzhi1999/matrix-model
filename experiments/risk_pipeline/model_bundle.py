"""Validated, reusable production inference models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

import torch

from multisource_unified import load_feature_fusion_checkpoint
from risk_pipeline.sensor_imputation import (
    DEFAULT_SENSOR_CHECKPOINT,
    load_sensor_imputeformer_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSION_CHECKPOINT = (
    ROOT
    / "outputs"
    / "feature_fusion_transformer_production"
    / "feature_fusion_transformer.pt"
)


def resolve_inference_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class ProductionModelBundle:
    """The two trained models loaded once for forward-only inference."""

    sensor_model: torch.nn.Module
    sensor_checkpoint: dict[str, Any]
    sensor_checkpoint_path: Path
    fusion_model: torch.nn.Module
    fusion_checkpoint: dict[str, Any]
    fusion_checkpoint_path: Path
    device: torch.device
    inference_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @classmethod
    def load(
        cls,
        sensor_checkpoint_path: str | Path = DEFAULT_SENSOR_CHECKPOINT,
        fusion_checkpoint_path: str | Path = DEFAULT_FUSION_CHECKPOINT,
        device: str | torch.device = "auto",
    ) -> "ProductionModelBundle":
        resolved_device = resolve_inference_device(device)
        sensor_path = Path(sensor_checkpoint_path).resolve()
        fusion_path = Path(fusion_checkpoint_path).resolve()
        sensor_model, sensor_checkpoint = load_sensor_imputeformer_checkpoint(
            sensor_path,
            resolved_device,
        )
        fusion_model, fusion_checkpoint = load_feature_fusion_checkpoint(
            fusion_path,
            map_location=resolved_device,
        )
        return cls(
            sensor_model=sensor_model,
            sensor_checkpoint=sensor_checkpoint,
            sensor_checkpoint_path=sensor_path,
            fusion_model=fusion_model,
            fusion_checkpoint=fusion_checkpoint,
            fusion_checkpoint_path=fusion_path,
            device=resolved_device,
        )
