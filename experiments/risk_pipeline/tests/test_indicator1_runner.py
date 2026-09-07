import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_indicator1_inference as runner


class Indicator1RunnerTests(unittest.TestCase):
    def test_defaults_run_all_sources_once_with_active_model_defaults(self):
        args = runner.parse_args([])
        data_root = ROOT / "data" / "\u6307\u68071"
        with patch.object(runner, "RiskPipeline") as pipeline_class:
            expected = {"risk_matrix_path": "new_job/risk.json"}
            pipeline_class.return_value.run.return_value = expected
            self.assertIs(runner.run(args), expected)

        pipeline_class.assert_called_once_with(
            device="auto", freq="auto", batch_size=8,
            output_root=ROOT / "outputs" / "indicator1_inference",
        )
        pipeline_class.return_value.run.assert_called_once_with(
            time_series_input=data_root / "time_series" / "time_series_100_native_records.csv",
            optical_image=data_root / "optical_images",
            sonar_image=data_root / "sonar_tif_images",
        )

    def test_explicit_paths_and_compute_options_override_defaults(self):
        args = runner.parse_args([
            "--time-series-input", "custom/measurements.jsonl",
            "--optical-image", "custom/optical.png",
            "--sonar-source", "custom/sonar.tif",
            "--output-root", "custom/results",
            "--device", "cpu", "--batch-size", "3",
        ])
        with patch.object(runner, "RiskPipeline") as pipeline_class:
            runner.run(args)

        pipeline_class.assert_called_once_with(
            device="cpu", freq="auto", batch_size=3,
            output_root=Path("custom/results"),
        )
        pipeline_class.return_value.run.assert_called_once_with(
            time_series_input=Path("custom/measurements.jsonl"),
            optical_image=Path("custom/optical.png"),
            sonar_image=Path("custom/sonar.tif"),
        )

    def test_invalid_batch_sizes_fail_before_pipeline_initialization(self):
        for value in ("0", "-1", "1.5", "abc"):
            with self.subTest(value=value):
                with patch.object(runner, "RiskPipeline") as pipeline_class:
                    with contextlib.redirect_stderr(io.StringIO()) as stderr:
                        with self.assertRaises(SystemExit) as raised:
                            runner.main(["--batch-size", value])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("must be a positive integer", stderr.getvalue())
                pipeline_class.assert_not_called()

    def test_main_prints_saved_risk_path(self):
        with patch.object(runner, "RiskPipeline") as pipeline_class:
            pipeline_class.return_value.run.return_value = {
                "risk_matrix_path": "new_job/risk.json",
            }
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                runner.main([])
        self.assertEqual(stdout.getvalue(), "Risk matrix JSON: new_job/risk.json\n")

    def test_input_failures_exit_with_clear_message(self):
        for failure in (FileNotFoundError("missing input"), ValueError("invalid records")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(runner, "RiskPipeline") as pipeline_class:
                    pipeline_class.return_value.run.side_effect = failure
                    with self.assertRaises(SystemExit) as raised:
                        runner.main([])
                self.assertEqual(raised.exception.code, f"Inference failed: {failure}")


if __name__ == "__main__":
    unittest.main()
