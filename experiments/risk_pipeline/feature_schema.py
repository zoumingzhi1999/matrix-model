"""Single source of truth for production first-stage feature columns."""

TIME_SERIES_FEATURE_COLUMNS = (
    "std",
    "min",
    "energy",
    "fft_1",
)

IMAGE_FEATURE_COLUMNS = (
    "brightness",
    "contrast",
    "edge_density",
    "entropy",
)

FUSION_MODALITIES = (
    "time_series",
    "optical_image",
    "sonar_image",
)

FUSION_FEATURE_COLUMNS = {
    "time_series": TIME_SERIES_FEATURE_COLUMNS,
    "optical_image": IMAGE_FEATURE_COLUMNS,
    "sonar_image": IMAGE_FEATURE_COLUMNS,
}

RISK_FEATURES = {
    "time_series": "std",
    "optical_image": "edge_density",
    "sonar_image": "edge_density",
}
