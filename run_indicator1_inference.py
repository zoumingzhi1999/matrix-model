"""Run all three indicator-1 data sources as one inference batch."""

from __future__ import annotations

import argparse
from pathlib import Path

from f_model import RiskPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "\u6307\u68071"
TIME_SERIES_INPUT_PATH = DATA_ROOT / "time_series" / "time_series_100_native_records.csv"
OPTICAL_IMAGE_INPUT_PATH = DATA_ROOT / "optical_images"
SONAR_IMAGE_INPUT_PATH = DATA_ROOT / "sonar_tif_images"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "indicator1_inference"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run indicator-1 sensors and all optical/sonar files in one batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--time-series-input", type=Path, default=TIME_SERIES_INPUT_PATH,
        help="Time-series CSV or JSONL input.",
    )
    parser.add_argument(
        "--optical-image", type=Path, default=OPTICAL_IMAGE_INPUT_PATH,
        help="Optical image file or directory of images.",
    )
    parser.add_argument(
        "--sonar-source", type=Path, default=SONAR_IMAGE_INPUT_PATH,
        help="Single-page sonar .tif/.tiff file or directory of TIFF images.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=OUTPUT_ROOT,
        help="Parent directory for newly created, non-overwriting run directories.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    pipeline = RiskPipeline(
        device=args.device,
        freq="auto",
        batch_size=args.batch_size,
        output_root=args.output_root,
    )
    return pipeline.run(
        time_series_input=args.time_series_input,
        optical_image=args.optical_image,
        sonar_image=args.sonar_source,
    )


def main(argv: list[str] | None = None) -> None:
    try:
        result = run(parse_args(argv))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Inference failed: {exc}") from None
    print(f"Risk matrix JSON: {result['risk_matrix_path']}")


if __name__ == "__main__":
    main()
