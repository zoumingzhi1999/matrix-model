from typing import Dict

import torch


def image_risk_features(image: torch.Tensor) -> Dict[str, float]:
    x = image.float()
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    brightness = float(x.mean())
    contrast = float(x.std())
    edge_density = float((gx.abs().mean() + gy.abs().mean()) * 0.5)
    entropy = _entropy(x)
    return {
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
        "entropy": entropy,
    }


def _entropy(image: torch.Tensor) -> float:
    hist = torch.histc(image.flatten(), bins=64, min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(1.0)
    prob = prob[prob > 0]
    return float(-(prob * torch.log2(prob)).sum())
