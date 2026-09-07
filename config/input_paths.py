"""Edit these three paths for local inference; request/CLI paths take priority.

Relative paths are resolved from the repository root. Leave a path as None
when that source is supplied by the HTTP request or command line instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Timestamped five-sensor records: long/wide CSV, or long-format JSONL.
# Example: Path("D:/sensor_data/five_sensors.csv")
TIME_SERIES_INPUT_PATH: str | Path | None = None
# An original optical image or a directory of images, at native resolution.
# Example: Path("D:/sensor_data/optical.png")
OPTICAL_IMAGE_INPUT_PATH: str | Path | None = None
# Single-page TIFF sonar image or a directory of .tif/.tiff images.
# Example: Path("D:/sensor_data/sonar.tif")
SONAR_IMAGE_INPUT_PATH: str | Path | None = None
# Legacy fallback when SONAR_IMAGE_INPUT_PATH is unset.
# Example: Path("D:/sensor_data/sonar.xtf")
SONAR_XTF_INPUT_PATH: str | Path | None = None


@dataclass(frozen=True)
class InputPaths:
    time_series: Path | None
    optical_image: Path | None
    sonar_xtf: Path | None
    sonar_image: Path | None = None

    @property
    def sonar_source(self) -> Path | None:
        return self.sonar_image or self.sonar_xtf


def resolve_input_paths(
    *,
    time_series: str | Path | None = None,
    optical_image: str | Path | None = None,
    sonar_xtf: str | Path | None = None,
    sonar_image: str | Path | None = None,
) -> InputPaths:
    def resolve(override: str | Path | None, configured: str | Path | None):
        value = override if override is not None else configured
        if value is None or not str(value).strip():
            return None
        path = Path(value).expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    if sonar_image is not None and sonar_xtf is not None:
        raise ValueError("Provide only one sonar source: sonar_image or sonar_xtf.")
    if sonar_image is not None:
        resolved_sonar_image, resolved_sonar_xtf = resolve(sonar_image, None), None
    elif sonar_xtf is not None:
        resolved_sonar_image, resolved_sonar_xtf = None, resolve(sonar_xtf, None)
    else:
        resolved_sonar_image = resolve(None, SONAR_IMAGE_INPUT_PATH)
        resolved_sonar_xtf = (
            None if resolved_sonar_image is not None else resolve(None, SONAR_XTF_INPUT_PATH)
        )

    return InputPaths(
        time_series=resolve(time_series, TIME_SERIES_INPUT_PATH),
        optical_image=resolve(optical_image, OPTICAL_IMAGE_INPUT_PATH),
        sonar_xtf=resolved_sonar_xtf,
        sonar_image=resolved_sonar_image,
    )
