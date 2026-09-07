from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import uuid

import pandas as pd


def configure_inference_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def log_feature_frame(
    logger: logging.Logger,
    stage: str,
    modality: str,
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Log every feature row and column without pandas display truncation."""
    logger.info(
        "%s output modality=%s rows=%d columns=%d artifact=%s\n%s",
        stage,
        modality,
        len(frame),
        len(frame.columns),
        path.resolve(),
        frame.to_csv(index=False, lineterminator="\n"),
    )


def artifact_timestamp() -> str:
    local_time = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    return f"{local_time}_{uuid.uuid4().hex[:8]}"


def artifact_path(
    output_dir: Path,
    category: str,
    stem: str,
    suffix: str,
    timestamp: str | None = None,
) -> Path:
    target_dir = output_dir / category
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or artifact_timestamp()
    path = target_dir / f"{stem}_{ts}{suffix}"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    return path


def latest_file(output_dir: Path, category: str, pattern: str) -> Path | None:
    target_dir = output_dir / category
    if not target_dir.exists():
        return None
    matches = sorted(target_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def write_csv_once(
    frame: pd.DataFrame,
    path: Path,
    *,
    index: bool = False,
    index_label: str | None = None,
    float_format=None,
) -> None:
    """Write an Excel-readable CSV while refusing any existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        mode="x",
        encoding="utf-8-sig",
        lineterminator="\n",
        index=index,
        index_label=index_label,
        float_format=float_format,
    )


def write_text_once(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding=encoding, newline="\n") as stream:
        stream.write(content)


def copy_file_once(source: Path, destination: Path) -> None:
    """Copy a retained input without allowing a race to overwrite it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def mapping_to_long_frame(mapping: Mapping[str, object]) -> pd.DataFrame:
    """Flatten nested metadata into a two-column human-readable table."""
    rows: list[dict[str, object]] = []

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), nested)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            rows.append({"field": prefix, "value": " | ".join(map(str, value))})
            return
        rows.append({"field": prefix, "value": value})

    visit("", mapping)
    return pd.DataFrame(rows, columns=("field", "value"))
