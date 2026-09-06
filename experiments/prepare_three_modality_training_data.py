"""Download, split, and featurize the three production modalities.

The Transformer is trained on first-stage feature rows, never on raw tensors.
This script retains every downloaded source file, every split, every extracted
feature row, and an auditable source/split manifest under one immutable run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from risk_pipeline.data_io import load_grayscale_image  # noqa: E402
from risk_pipeline.feature_schema import (  # noqa: E402
    IMAGE_FEATURE_COLUMNS,
    TIME_SERIES_FEATURE_COLUMNS,
)
from risk_pipeline.modality_features import (  # noqa: E402
    extract_image_tile_first_stage_features,
)
from risk_pipeline.time_series_features import extract_first_stage_features  # noqa: E402
from risk_pipeline.xtf_io import load_xtf_sonar_image  # noqa: E402


UIEB_ARCHIVE_URL = (
    "https://codeload.github.com/JJsnowx/UIEB_Dataset/zip/refs/heads/main"
)
UIEB_REPOSITORY = "https://github.com/JJsnowx/UIEB_Dataset"
UIEB_REFERENCE = "https://doi.org/10.1109/TIP.2019.2955241"
XTF_DECODER_TEST_URL = (
    "https://raw.githubusercontent.com/MichalMotylinski/"
    "SideScanSonarEditor/master/tests/data/test_sample.xtf"
)
XTF_DECODER_TEST_LICENSE_URL = (
    "https://raw.githubusercontent.com/MichalMotylinski/"
    "SideScanSonarEditor/master/LICENSE"
)
XTF_DATASET_REFERENCE = "https://doi.org/10.5281/zenodo.5558794"
XTF_PAPER_REFERENCE = "https://doi.org/10.1175/JPO-D-21-0230.1"
XTF_DATASET_LICENSE = "CC-BY-4.0"
XTF_DATASET_FILES = {
    "2018_10_02_11_25_53.xtf": {
        "bytes": 81_204_224,
        "md5": "7fcaadd9b170e990183a5b9e0c8a2384",
    },
    "2018_10_02_11_59_08.xtf": {
        "bytes": 81_224_320,
        "md5": "bc39703a85853f2ee9e7496004ac9546",
    },
    "2018_10_02_15_57_32.xtf": {
        "bytes": 81_213_824,
        "md5": "4151a87cb8f8e9cfff67353ee1f09205",
    },
    "2018_10_02_20_27_54.xtf": {
        "bytes": 81_173_760,
        "md5": "c0944b5ea023c70ec68ee21083ead688",
    },
    "2018_10_02_21_01_07.xtf": {
        "bytes": 81_204_096,
        "md5": "b9ddd102847e8bc1ef039b0fa0e1070e",
    },
    "2018_10_03_01_02_16.xtf": {
        "bytes": 81_284_608,
        "md5": "673302d981a24679f35afef675d06e08",
    },
}
XTF_SPLIT_FILES = {
    "train": tuple(list(XTF_DATASET_FILES)[:4]),
    "validation": (list(XTF_DATASET_FILES)[4],),
    "test": (list(XTF_DATASET_FILES)[5],),
}
SPLITS = ("train", "validation", "test")
TIME_WINDOW_ROWS = 3600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def prepare_or_load_report(run_dir: Path, modality: str, prepare) -> dict[str, object]:
    """Resume a completed modality without rewriting any retained artifact."""
    report_path = run_dir / "preparation_reports" / f"{modality}.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    report = prepare()
    write_json_once(report_path, report)
    return report


def download_once(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Matrix-model"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(destination)


def safe_extract_zip_once(archive: Path, destination: Path) -> None:
    if destination.exists():
        return
    temporary = destination.with_name(destination.name + ".extracting")
    temporary.mkdir(parents=True, exist_ok=False)
    root = temporary.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (temporary / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe path in UIEB archive: {member.filename}")
        bundle.extractall(temporary)
    temporary.replace(destination)


def deterministic_split(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _save_frame_once(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _hardlink_once(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite split member: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def prepare_optical(uieb_root: Path, run_dir: Path) -> dict[str, object]:
    dataset_roots = list(uieb_root.rglob("UIEB Dataset"))
    if len(dataset_roots) != 1:
        raise FileNotFoundError("UIEB archive does not contain one 'UIEB Dataset' root.")
    dataset = dataset_roots[0]
    raw_by_name = {
        path.name: path for path in (dataset / "raw" / "UIEB_raw_reName").iterdir()
        if path.is_file()
    }
    reference_by_name = {
        path.name: path
        for path in (dataset / "reference" / "UIEB_reference_reName").iterdir()
        if path.is_file()
    }
    if set(raw_by_name) != set(reference_by_name):
        raise ValueError("UIEB raw/reference filenames are not exactly paired.")
    challenge = sorted(
        path for path in (dataset / "challenge" / "challenging-60").iterdir()
        if path.is_file()
    )

    rows_by_split = {split: [] for split in SPLITS}
    manifest_rows = []
    sources = []
    for name in sorted(raw_by_name):
        split = deterministic_split(f"paired/{name}")
        sources.extend(((raw_by_name[name], 1, "raw_degraded"),
                        (reference_by_name[name], 0, "reference")))
        for source, label, quality in sources[-2:]:
            manifest_rows.append({
                "image_id": name,
                "source_path": str(source.resolve()),
                "split": split,
                "label": label,
                "quality_role": quality,
                "sha256": sha256_file(source),
            })
    for source in challenge:
        manifest_rows.append({
            "image_id": f"challenge/{source.name}",
            "source_path": str(source.resolve()),
            "split": deterministic_split(f"challenge/{source.name}"),
            "label": 1,
            "quality_role": "challenge_degraded",
            "sha256": sha256_file(source),
        })

    for item_index, item in enumerate(manifest_rows, start=1):
        source = Path(item["source_path"])
        split = str(item["split"])
        role = str(item["quality_role"])
        _hardlink_once(
            source,
            run_dir / "splits" / "optical" / split / role / source.name,
        )
        features = extract_image_tile_first_stage_features(
            source.name,
            load_grayscale_image(source),
            modality="optical_image",
        )
        features.insert(0, "image_id", item["image_id"])
        features["quality_role"] = role
        features["label"] = int(item["label"])
        rows_by_split[split].append(features)
        if item_index % 100 == 0 or item_index == len(manifest_rows):
            print(f"optical {item_index}/{len(manifest_rows)}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = run_dir / "splits" / "optical_split_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {manifest_path}")
    manifest.to_csv(manifest_path, index=False)
    counts = {}
    for split in SPLITS:
        frame = pd.concat(rows_by_split[split], ignore_index=True)
        path = run_dir / "features" / f"optical_image_{split}_candidate_features.parquet"
        _save_frame_once(frame, path)
        counts[split] = {
            "images": int((manifest["split"] == split).sum()),
            "feature_rows": len(frame),
            "label_counts": frame["label"].value_counts().sort_index().to_dict(),
            "feature_file": str(path.resolve()),
            "feature_sha256": sha256_file(path),
        }
    return {
        "source_images": len(manifest),
        "paired_raw_reference_ids": len(raw_by_name),
        "challenge_images": len(challenge),
        "split_rule": "SHA-256 image-id buckets 0-69/70-84/85-99; pairs stay together",
        "features": list(IMAGE_FEATURE_COLUMNS),
        "splits": counts,
    }


def _degrade_sonar(image: np.ndarray) -> np.ndarray:
    blurred = gaussian_filter(image, sigma=(1.5, 2.5), mode="nearest")
    compressed = 0.5 + (blurred - 0.5) * 0.45
    return np.clip(compressed, 0.0, 1.0).astype("float32")


def prepare_sonar(xtf_paths: dict[str, Path], run_dir: Path) -> dict[str, object]:
    manifest_rows = []
    decode_reports = {}
    counts = {
        split: {
            "source_files": 0,
            "ping_rows": 0,
            "feature_rows": 0,
            "label_counts": {0: 0, 1: 0},
        }
        for split in SPLITS
    }
    frames_by_split = {split: [] for split in SPLITS}
    for split in SPLITS:
        split_dir = run_dir / "splits" / "sonar" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for filename in XTF_SPLIT_FILES[split]:
            xtf_path = xtf_paths[filename]
            image, decode_metadata = load_xtf_sonar_image(xtf_path)
            decode_reports[filename] = decode_metadata
            original = image
            degraded = _degrade_sonar(original)
            stem = xtf_path.stem
            original_path = split_dir / f"{stem}_original_xtf_intensity.npy"
            degraded_path = split_dir / f"{stem}_controlled_degraded_intensity.npy"
            if original_path.exists() or degraded_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite sonar split member: {filename}"
                )
            np.save(original_path, original, allow_pickle=False)
            np.save(degraded_path, degraded, allow_pickle=False)

            frames = []
            for label, quality, array in (
                (0, "original_xtf_intensity", original),
                (1, "controlled_blur_contrast_loss", degraded),
            ):
                frame = extract_image_tile_first_stage_features(
                    xtf_path.name,
                    array,
                    modality="sonar_image",
                )
                frame["xtf_filename"] = filename
                frame["ping_start"] = 0
                frame["ping_end_exclusive"] = len(original)
                frame["quality_role"] = quality
                frame["label"] = label
                frames.append(frame)
                manifest_rows.append({
                    "split": split,
                    "xtf_filename": filename,
                    "source_xtf": str(xtf_path.resolve()),
                    "source_sha256": sha256_file(xtf_path),
                    "ping_start": 0,
                    "ping_end_exclusive": len(original),
                    "ping_count": len(original),
                    "label": label,
                    "quality_role": quality,
                    "array_path": str(
                        (original_path if label == 0 else degraded_path).resolve()
                    ),
                })
            frames_by_split[split].extend(frames)
            counts[split]["source_files"] += 1
            counts[split]["ping_rows"] += len(original)
            for frame in frames:
                label = int(frame["label"].iloc[0])
                counts[split]["feature_rows"] += len(frame)
                counts[split]["label_counts"][label] += len(frame)
            print(
                f"sonar {split} {filename}: {len(original)} pings",
                flush=True,
            )

    for split in SPLITS:
        combined = pd.concat(frames_by_split[split], ignore_index=True)
        path = run_dir / "features" / f"sonar_image_{split}_candidate_features.parquet"
        _save_frame_once(combined, path)
        counts[split]["feature_file"] = str(path.resolve())
        counts[split]["feature_sha256"] = sha256_file(path)
    sonar_manifest_path = run_dir / "splits" / "sonar_split_manifest.csv"
    if sonar_manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite artifact: {sonar_manifest_path}"
        )
    pd.DataFrame(manifest_rows).to_csv(sonar_manifest_path, index=False)
    write_json_once(
        run_dir / "splits" / "sonar_decode_metadata.json", decode_reports
    )
    return {
        "source_xtf_files": [
            {
                "filename": filename,
                "path": str(xtf_paths[filename].resolve()),
                "bytes": xtf_paths[filename].stat().st_size,
                "sha256": sha256_file(xtf_paths[filename]),
                "md5": md5_file(xtf_paths[filename]),
            }
            for filename in XTF_DATASET_FILES
        ],
        "all_six_source_files_used": sum(
            value["source_files"] for value in counts.values()
        ) == len(XTF_DATASET_FILES),
        "split_rule": (
            "chronological whole-XTF-file split: first four train, fifth "
            "validation, sixth independent test; no source ping is split across sets"
        ),
        "controlled_positive": "Gaussian blur plus contrast compression",
        "features": list(IMAGE_FEATURE_COLUMNS),
        "decode": decode_reports,
        "splits": counts,
    }


def _controlled_time_series_anomaly(frame: pd.DataFrame) -> pd.DataFrame:
    corrupted = frame.copy()
    position = np.linspace(-1.0, 1.0, len(frame), dtype="float64")
    for column_index, column in enumerate(frame.columns):
        values = frame[column].to_numpy(dtype="float64")
        scale = float(np.std(values))
        if scale < 1e-8:
            scale = max(abs(float(np.mean(values))), 1.0) * 0.01
        drift = position * scale * (0.7 + 0.1 * column_index)
        spikes = np.zeros(len(values), dtype="float64")
        spikes[column_index :: 257] = scale * 4.0
        corrupted[column] = values + drift + spikes
    return corrupted


def prepare_time_series(
    aligned_path: Path,
    ocean_dataset_run: Path,
    run_dir: Path,
) -> dict[str, object]:
    aligned = pd.read_parquet(aligned_path)
    if not isinstance(aligned.index, pd.DatetimeIndex):
        raise ValueError("Aligned time-series parquet must use a DatetimeIndex.")
    if not np.isfinite(aligned.to_numpy(dtype="float64")).all():
        raise ValueError("Aligned time-series input contains missing values.")
    source_splits = {
        split: pd.read_parquet(ocean_dataset_run / "splits" / f"{split}.parquet")
        for split in SPLITS
    }
    expected_rows = sum(len(frame) for frame in source_splits.values())
    if len(aligned) != expected_rows:
        raise ValueError(
            f"Aligned rows ({len(aligned)}) do not match source splits ({expected_rows})."
        )

    cursor = 0
    counts = {}
    window_manifest = []
    for split in SPLITS:
        row_count = len(source_splits[split])
        split_frame = aligned.iloc[cursor:cursor + row_count].copy()
        cursor += row_count
        retained_path = run_dir / "splits" / "time_series" / f"{split}.parquet"
        if retained_path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {retained_path}")
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        split_frame.to_parquet(retained_path, compression="zstd")

        feature_frames = []
        for window_index, start in enumerate(range(0, len(split_frame), TIME_WINDOW_ROWS)):
            window = split_frame.iloc[start:start + TIME_WINDOW_ROWS]
            if window.empty:
                continue
            for label, quality, values in (
                (0, "aligned_observation", window),
                (1, "controlled_drift_and_spikes", _controlled_time_series_anomaly(window)),
            ):
                input_frame = values.reset_index(names="timestamp")
                features, _ = extract_first_stage_features(input_frame)
                features.insert(0, "window_index", window_index)
                features["window_start"] = window.index[0]
                features["window_end"] = window.index[-1]
                features["source_row_start"] = start
                features["source_row_end_exclusive"] = start + len(window)
                features["quality_role"] = quality
                features["label"] = label
                feature_frames.append(features)
            window_manifest.append({
                "split": split,
                "window_index": window_index,
                "source_row_start": start,
                "source_row_end_exclusive": start + len(window),
                "source_row_count": len(window),
                "timestamp_start": window.index[0].isoformat(),
                "timestamp_end": window.index[-1].isoformat(),
            })
        combined = pd.concat(feature_frames, ignore_index=True)
        path = run_dir / "features" / f"time_series_{split}_candidate_features.parquet"
        _save_frame_once(combined, path)
        counts[split] = {
            "aligned_rows": len(split_frame),
            "windows": len(feature_frames) // 2,
            "feature_rows": len(combined),
            "label_counts": combined["label"].value_counts().sort_index().to_dict(),
            "retained_split": str(retained_path.resolve()),
            "feature_file": str(path.resolve()),
            "feature_sha256": sha256_file(path),
        }
    pd.DataFrame(window_manifest).to_csv(
        run_dir / "splits" / "time_series_window_manifest.csv", index=False
    )
    return {
        "aligned_source": str(aligned_path.resolve()),
        "aligned_source_sha256": sha256_file(aligned_path),
        "all_aligned_rows_used": cursor == len(aligned),
        "window_rows": TIME_WINDOW_ROWS,
        "final_partial_window_retained": True,
        "controlled_positive": "sensor-scale drift plus deterministic sparse spikes",
        "features": list(TIME_SERIES_FEATURE_COLUMNS),
        "splits": counts,
    }


def run(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_optical = run_dir / "raw" / "optical"
    raw_sonar = run_dir / "raw" / "sonar"
    archive = raw_optical / "UIEB_Dataset-main.zip"
    download_once(UIEB_ARCHIVE_URL, archive)
    decoder_test_path = raw_sonar / "decoder_test" / "test_sample.xtf"
    download_once(XTF_DECODER_TEST_URL, decoder_test_path)
    decoder_test_license = raw_sonar / "decoder_test" / "LICENSE_GPL-3.0.txt"
    download_once(XTF_DECODER_TEST_LICENSE_URL, decoder_test_license)
    xtf_paths = {}
    for filename, expected in XTF_DATASET_FILES.items():
        xtf_path = raw_sonar / "zenodo_5558794" / filename
        download_once(
            f"https://zenodo.org/api/records/5558794/files/{filename}/content",
            xtf_path,
        )
        if xtf_path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Downloaded XTF size mismatch: {xtf_path}")
        if md5_file(xtf_path) != expected["md5"]:
            raise ValueError(f"Downloaded XTF checksum mismatch: {xtf_path}")
        xtf_paths[filename] = xtf_path
    extracted = raw_optical / "UIEB_Dataset-main"
    safe_extract_zip_once(archive, extracted)

    source_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "no_audio": True,
        "sources": {
            "optical_image": {
                "dataset": "UIEB",
                "repository_mirror": UIEB_REPOSITORY,
                "paper_doi": UIEB_REFERENCE,
                "download_url": UIEB_ARCHIVE_URL,
                "archive": str(archive.resolve()),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": sha256_file(archive),
                "rights_note": "Public academic mirror; no explicit license file in mirror.",
            },
            "sonar_image": {
                "dataset": "Echosounder images used in a Journal of Physical Oceanography manuscript",
                "zenodo_doi": XTF_DATASET_REFERENCE,
                "paper_doi": XTF_PAPER_REFERENCE,
                "zenodo_record": "https://zenodo.org/records/5558794",
                "license": XTF_DATASET_LICENSE,
                "xtf_files": [str(path.resolve()) for path in xtf_paths.values()],
                "decoder_regression_sample": str(decoder_test_path.resolve()),
                "decoder_regression_sample_license": "GPL-3.0",
                "decoder_regression_sample_license_file": str(
                    decoder_test_license.resolve()
                ),
            },
            "time_series": {
                "aligned_imputeformer_output": str(Path(args.aligned_time_series).resolve()),
                "ocean_dataset_run": str(Path(args.ocean_dataset_run).resolve()),
            },
        },
    }
    source_manifest_path = run_dir / "source_manifest.json"
    if not source_manifest_path.exists():
        write_json_once(source_manifest_path, source_manifest)
    reports = {
        "time_series": prepare_or_load_report(
            run_dir,
            "time_series",
            lambda: prepare_time_series(
                Path(args.aligned_time_series), Path(args.ocean_dataset_run), run_dir
            ),
        ),
        "optical_image": prepare_or_load_report(
            run_dir,
            "optical_image",
            lambda: prepare_optical(extracted, run_dir),
        ),
        "sonar_image": prepare_or_load_report(
            run_dir,
            "sonar_image",
            lambda: prepare_sonar(xtf_paths, run_dir),
        ),
    }
    preparation_report_path = run_dir / "preparation_report.json"
    if not preparation_report_path.exists():
        write_json_once(preparation_report_path, reports)
    print(json.dumps({"run_dir": str(run_dir), **reports}, indent=2), flush=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(
            ROOT / "data" / "multimodal_training" / "runs"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ),
    )
    parser.add_argument("--aligned-time-series", required=True)
    parser.add_argument("--ocean-dataset-run", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
