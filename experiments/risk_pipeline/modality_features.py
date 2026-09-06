import numpy as np
import pandas as pd
import torch

from risk_pipeline.feature_schema import IMAGE_FEATURE_COLUMNS
from risk_pipeline.risk import image_risk_features


IMAGE_TILE_SIZE = 128
IMAGE_TILE_STRIDE = 64


def extract_image_tile_first_stage_features(
    name: str,
    image: np.ndarray,
    modality: str,
) -> pd.DataFrame:
    """Extract overlapping tiles without resizing the optical/sonar image."""
    if modality not in {"optical_image", "sonar_image"}:
        raise ValueError(
            "modality must be either 'optical_image' or 'sonar_image'.")
    array = np.asarray(image, dtype="float32")
    if array.ndim != 2:
        raise ValueError(
            "Image tile extraction expects a two-dimensional grayscale image.")
    if 0 in array.shape:
        raise ValueError("Image width and height must both be positive.")
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite pixel values.")

    input_height, input_width = array.shape
    pad_bottom = max(IMAGE_TILE_SIZE - input_height, 0)
    pad_right = max(IMAGE_TILE_SIZE - input_width, 0)
    if pad_bottom or pad_right:
        array = np.pad(
            array,
            ((0, pad_bottom), (0, pad_right)),
            mode="edge",
        )

    x_starts = _tile_starts(array.shape[1])
    y_starts = _tile_starts(array.shape[0])
    rows = []
    tile_index = 0
    for y0 in y_starts:
        for x0 in x_starts:
            tile = array[
                y0:y0 + IMAGE_TILE_SIZE,
                x0:x0 + IMAGE_TILE_SIZE,
            ]
            tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
            rows.append({
                "source": name,
                "modality": modality,
                "tile_index": tile_index,
                "x_start": x0,
                "y_start": y0,
                "tile_width": IMAGE_TILE_SIZE,
                "tile_height": IMAGE_TILE_SIZE,
                "input_width": input_width,
                "input_height": input_height,
                **image_risk_features(tensor),
            })
            tile_index += 1
    return pd.DataFrame(rows)


def _tile_starts(length: int) -> list[int]:
    """Return stride-64 starts and anchor the final tile to the far edge."""
    if length < IMAGE_TILE_SIZE:
        raise ValueError("Tile axis length must be at least IMAGE_TILE_SIZE.")
    last_start = length - IMAGE_TILE_SIZE
    starts = list(range(0, last_start + 1, IMAGE_TILE_STRIDE))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
