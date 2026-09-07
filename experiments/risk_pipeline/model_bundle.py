"""Validated, reusable production inference models."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
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
    / "models"
    / "feature_fusion_transformer"
    / "feature_fusion_transformer.pt"
)
logger = logging.getLogger("f-model.models")


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
        logger.info(
            "Loading inference models: sensor_checkpoint=%s fusion_checkpoint=%s device=%s",
            sensor_path, fusion_path, resolved_device,
        )
        sensor_model, sensor_checkpoint = load_sensor_imputeformer_checkpoint(
            sensor_path,
            resolved_device,
        )
        logger.info(
            "Sensor model configuration: format=%s time_aware=%s static_sensor_embedding=%s "
            "neural_residual_enabled=%s visible_median_calibration=%s",
            sensor_checkpoint.get("format_version"),
            sensor_checkpoint["model_config"].get("time_aware", False),
            sensor_checkpoint["model_config"].get("static_sensor_embedding", False),
            bool(getattr(sensor_model, "context_residual_gain", torch.ones(1)).any()),
            sensor_checkpoint["model_config"].get("visible_median_calibration", False),
        )
        fusion_model, fusion_checkpoint = load_feature_fusion_checkpoint(
            fusion_path,
            map_location=resolved_device,
        )
        logger.info("Inference models loaded and ready on %s", resolved_device)
        return cls(
            sensor_model=sensor_model,
            sensor_checkpoint=sensor_checkpoint,
            sensor_checkpoint_path=sensor_path,
            fusion_model=fusion_model,
            fusion_checkpoint=fusion_checkpoint,
            fusion_checkpoint_path=fusion_path,
            device=resolved_device,
        )
