"""Trainable Transformer for fusing named first-stage feature vectors."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn

from risk_pipeline.feature_schema import FUSION_FEATURE_COLUMNS, FUSION_MODALITIES


class FeatureFusionTransformer(nn.Module):
    """Fuse three first-stage feature vectors without a risk-prediction head.

    Each modality has its own projection and learned modality embedding. The
    Transformer CLS token is trained to reconstruct all clean input feature
    vectors from corrupted views, making the unified representation useful for
    feature fusion while keeping operational risk outside this model.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        missing = [name for name in FUSION_MODALITIES if name not in input_dims]
        if missing:
            raise ValueError(f"Missing fusion input dimensions: {missing}")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.input_dims = {
            name: int(input_dims[name]) for name in FUSION_MODALITIES
        }
        self.d_model = int(d_model)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(self.input_dims[name]),
                nn.Linear(self.input_dims[name], d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for name in FUSION_MODALITIES
        })
        self.modality_embedding = nn.Embedding(len(FUSION_MODALITIES), d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.randn(1, len(FUSION_MODALITIES) + 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

        self.unified_decoders = nn.ModuleDict({
            name: self._decoder(d_model, ff_dim, self.input_dims[name])
            for name in FUSION_MODALITIES
        })
        self.token_decoders = nn.ModuleDict({
            name: self._decoder(d_model, ff_dim, self.input_dims[name])
            for name in FUSION_MODALITIES
        })

        nn.init.normal_(self.cls_token, std=0.02)

    @staticmethod
    def _decoder(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, object]:
        tokens = []
        for modality_index, modality in enumerate(FUSION_MODALITIES):
            if modality not in features:
                raise KeyError(f"Fusion input is missing modality '{modality}'.")
            values = features[modality].float()
            if values.dim() != 2 or values.shape[1] != self.input_dims[modality]:
                raise ValueError(
                    f"{modality} expects [batch, {self.input_dims[modality]}], "
                    f"got {tuple(values.shape)}."
                )
            token = self.projections[modality](values)
            token = token + self.modality_embedding.weight[modality_index]
            tokens.append(token)

        token_tensor = torch.stack(tokens, dim=1)
        batch_size = token_tensor.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = torch.cat([cls, token_tensor], dim=1)
        encoded = encoded + self.position_embedding
        encoded = self.norm(self.encoder(encoded))

        unified = encoded[:, 0]
        modality_tokens = encoded[:, 1:]
        unified_reconstruction = {
            modality: self.unified_decoders[modality](unified)
            for modality in FUSION_MODALITIES
        }
        token_reconstruction = {
            modality: self.token_decoders[modality](
                modality_tokens[:, modality_index]
            )
            for modality_index, modality in enumerate(FUSION_MODALITIES)
        }
        return {
            "unified": unified,
            "tokens": modality_tokens,
            "unified_reconstruction": unified_reconstruction,
            "token_reconstruction": token_reconstruction,
            "source_token_ranges": {
                modality: (index + 1, index + 2, modality)
                for index, modality in enumerate(FUSION_MODALITIES)
            },
        }


def feature_input_dims(
    feature_columns: Mapping[str, Sequence[str]] = FUSION_FEATURE_COLUMNS,
) -> Dict[str, int]:
    return {
        modality: len(feature_columns[modality])
        for modality in FUSION_MODALITIES
    }


def load_feature_fusion_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[FeatureFusionTransformer, Dict[str, object]]:
    """Load and validate a production feature-fusion checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Trained feature-fusion checkpoint not found: "
            f"{checkpoint_path}. Run experiments/train_feature_fusion_transformer.py "
            "before inference."
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if checkpoint.get("model_class") != "FeatureFusionTransformer":
        raise ValueError("Checkpoint model_class is not FeatureFusionTransformer.")
    if checkpoint.get("has_risk_head") is not False:
        raise ValueError("Feature-fusion checkpoint must explicitly have no risk head.")

    expected_columns = {
        modality: list(FUSION_FEATURE_COLUMNS[modality])
        for modality in FUSION_MODALITIES
    }
    if checkpoint.get("feature_columns") != expected_columns:
        raise ValueError("Checkpoint feature columns do not match the production schema.")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Checkpoint is missing training-set normalization metadata.")
    for modality in FUSION_MODALITIES:
        spec = normalization.get(modality)
        if not isinstance(spec, Mapping):
            raise ValueError(f"Checkpoint normalization is missing {modality}.")
        if spec.get("columns") != expected_columns[modality]:
            raise ValueError(
                f"Checkpoint normalization columns do not match {modality}."
            )
        expected_dim = len(expected_columns[modality])
        if len(spec.get("mean", ())) != expected_dim:
            raise ValueError(f"Checkpoint normalization mean is invalid for {modality}.")
        if len(spec.get("scale", ())) != expected_dim:
            raise ValueError(f"Checkpoint normalization scale is invalid for {modality}.")
        means = [float(value) for value in spec["mean"]]
        scales = [float(value) for value in spec["scale"]]
        if not all(math.isfinite(value) for value in means):
            raise ValueError(f"Checkpoint normalization mean is invalid for {modality}.")
        if not all(math.isfinite(value) and value > 0.0 for value in scales):
            raise ValueError(f"Checkpoint normalization scale is invalid for {modality}.")

    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint is missing model_config.")
    if config.get("input_dims") != feature_input_dims():
        raise ValueError("Checkpoint input dimensions do not match the feature schema.")
    model = FeatureFusionTransformer(**dict(config))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(map_location)
    model.eval()
    return model, checkpoint
