"""Run the complete pipeline with configured paths or command-line overrides."""

from experiments.full_pipeline.run_streaming_pipeline import parse_args, run


def main() -> None:
    try:
        run(parse_args())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Inference failed: {exc}") from None


if __name__ == "__main__":
    main()
