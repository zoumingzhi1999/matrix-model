"""Export the existing XTF intensity images as lossless float32 TIFF inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from risk_pipeline.data_io import load_grayscale_image
from risk_pipeline.xtf_io import load_xtf_sonar_image


DATA_ROOT = ROOT / "data" / "\u6307\u68071"


def convert_source(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite TIFF: {destination}")
    image, metadata = load_xtf_sonar_image(source)
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata.update({
        "source_sha256": source_hash,
        "storage_format": "TIFF",
        "storage_dtype": "float32",
        "storage_compression": "tiff_adobe_deflate",
        "pixel_values": "Existing normalized XTF intensity, unchanged in [0, 1]",
    })
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    tags[270] = json.dumps(metadata, ensure_ascii=True)
    with destination.open("xb") as handle:
        Image.fromarray(image).save(
            handle, format="TIFF", compression="tiff_adobe_deflate", tiffinfo=tags,
        )
    restored = load_grayscale_image(destination)
    if not np.array_equal(image, restored):
        raise ValueError(f"TIFF readback differs from decoded XTF pixels: {destination}")
    return {
        "source": source.name,
        "output": str(destination),
        "shape": list(image.shape),
        "dtype": str(restored.dtype),
        "pixel_exact": True,
        "source_sha256": source_hash,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DATA_ROOT / "sonar_images",
        help="Original XTF file or directory of XTF files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_ROOT / "sonar_tif_images",
        help="TIFF output directory; existing output files are never overwritten.",
    )
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    if source.is_file() and source.suffix.lower() == ".xtf":
        paths = [source]
    elif source.is_dir():
        paths = sorted(
            (path for path in source.iterdir()
             if path.is_file() and path.suffix.lower() == ".xtf"),
            key=lambda path: path.name,
        )
    else:
        parser.error(f"Expected an existing XTF file or directory: {source}")
    if not paths:
        parser.error(f"No XTF files found: {source}")
    output_dir = args.output_dir.expanduser().resolve()
    destinations = [output_dir / f"{path.stem}.tif" for path in paths]
    if len({str(path).casefold() for path in destinations}) != len(destinations):
        parser.error("Source filenames would map to duplicate TIFF outputs.")
    for destination in destinations:
        if destination.exists():
            parser.error(f"Refusing to overwrite TIFF: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (path, destination) in enumerate(zip(paths, destinations), start=1):
        print(f"Converting {index}/{len(paths)}: {path.name}", flush=True)
        print(json.dumps(convert_source(path, destination)), flush=True)


if __name__ == "__main__":
    main()
