import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.service import Record, RecordBatch, create_app  # noqa: E402
from experiments.full_pipeline import run_streaming_pipeline as streaming  # noqa: E402
from risk_pipeline.data_io import SensorBuffer, load_streaming_records  # noqa: E402
from risk_pipeline.sensor_imputation import EXPECTED_SENSOR_IDS  # noqa: E402


BASE = pd.Timestamp("2026-01-01T00:00:00Z")


def records_at(sensor, seconds):
    return [
        {"timestamp": (BASE + pd.Timedelta(seconds=second)).isoformat(),
         "sensor_id": sensor, "value": float(step + 1)}
        for step, second in enumerate(seconds)
    ]


def five_sensor_records():
    records = records_at(EXPECTED_SENSOR_IDS[0], (0, 2, 4, 8, 10))
    for sensor in EXPECTED_SENSOR_IDS[1:]:
        records.extend(records_at(sensor, (0, 10)))
    return records


def fake_bundle(time_aware):
    return SimpleNamespace(
        sensor_checkpoint_path=Path("unused-sensor.pt"),
        sensor_model=object(),
        sensor_checkpoint={"model_config": {"time_aware": time_aware}},
    )


def endpoints_for(app):
    return {route.path: route.endpoint for route in app.routes
            if hasattr(route, "endpoint")}


class CheckpointGridPolicyTests(unittest.TestCase):
    def test_regular_auto_uses_most_valid_samples_instead_of_short_faster_burst(self):
        records = records_at("reference", (2, 4, 6, 8, 10))
        records += records_at("short_fast_burst", (0, 1, 2))
        records += records_at("late_sensor", (3, 12))
        records.append({"timestamp": (BASE + pd.Timedelta(seconds=14)).isoformat(),
                        "sensor_id": "late_sensor", "value": None})

        wide, _, _, _, metadata = load_streaming_records(
            records, preserve_actual_timestamps=False,
        )

        expected = pd.date_range(BASE, periods=8, freq="2s", name="timestamp")
        np.testing.assert_array_equal(wide.index.as_unit("ns").asi8, expected.as_unit("ns").asi8)
        self.assertEqual(metadata["reference_sensor_id"], "reference")
        self.assertEqual(metadata["observation_counts"]["reference"], 5)
        self.assertEqual(metadata["fastest_sensor_ids"], ["short_fast_burst"])
        self.assertEqual(metadata["alignment_frequency_seconds"], 2.0)
        self.assertEqual(metadata["frequency_source"], "most_sampled_sensor")
        self.assertEqual(metadata["grid_policy"],
                         "reference_cadence_covering_all_input_timestamps")
        self.assertEqual(metadata["alignment_interval_role"], "fixed_grid_interval")
        self.assertFalse(metadata["irregular_timestamps_preserved"])
        self.assertEqual(wide["reference"].notna().sum(), 5)
        self.assertEqual(wide.loc[BASE + pd.Timedelta(seconds=12), "late_sensor"], 2.0)
        self.assertTrue(wide.iloc[-1].isna().all())

    def test_regular_auto_restores_dropped_grid_rows_and_maps_jitter(self):
        for seconds, periods in (((0, 2, 4, 8, 10), 6),
                                 ((0, 2.02, 4, 6.04, 8), 5)):
            with self.subTest(seconds=seconds):
                records = records_at("reference", seconds) + records_at("slow", (0,))
                wide, _, mask, sensors, metadata = load_streaming_records(
                    records, preserve_actual_timestamps=False,
                )
                expected = pd.date_range(BASE, periods=periods, freq="2s")
                np.testing.assert_array_equal(wide.index.as_unit("ns").asi8, expected.as_unit("ns").asi8)
                np.testing.assert_array_equal(
                    wide["reference"].dropna().to_numpy(), np.arange(1.0, 6.0),
                )
                self.assertEqual(metadata["alignment_frequency_seconds"], 2.0)
                self.assertEqual(int(mask[:, sensors.index("reference"), 0].sum()), 5)
                if periods == 6:
                    self.assertTrue(pd.isna(wide.loc[BASE + pd.Timedelta(seconds=6),
                                                   "reference"]))

    def test_default_auto_keeps_actual_reference_timestamps(self):
        seconds = (0, 2.02, 4, 8.04, 10)
        records = records_at("reference", seconds) + records_at("slow", (0,))

        wide, _, _, _, metadata = load_streaming_records(records)

        expected = pd.DatetimeIndex([BASE + pd.Timedelta(seconds=s) for s in seconds])
        np.testing.assert_array_equal(wide.index.as_unit("ns").asi8, expected.as_unit("ns").asi8)
        self.assertTrue(metadata["irregular_timestamps_preserved"])
        self.assertEqual(metadata["grid_policy"],
                         "reference_actual_timestamps_with_outer_coverage")
        self.assertEqual(metadata["alignment_interval_role"],
                         "representative_cadence_estimate")

    def test_regular_auto_preserves_submillisecond_and_hourly_cadence(self):
        for frequency in ("200us", "2h"):
            with self.subTest(frequency=frequency):
                index = pd.date_range(BASE, periods=9, freq=frequency)
                records = [
                    {"timestamp": timestamp.isoformat(), "sensor_id": sensor,
                     "value": float(step)}
                    for sensor, timestamps in (("reference", index), ("slow", index[::4]))
                    for step, timestamp in enumerate(timestamps)
                ]
                wide, _, _, _, metadata = load_streaming_records(
                    records, preserve_actual_timestamps=False,
                )
                np.testing.assert_array_equal(wide.index.as_unit("ns").asi8, index.as_unit("ns").asi8)
                self.assertEqual(metadata["alignment_frequency_seconds"],
                                 pd.Timedelta(frequency).value / 1e9)

    def test_single_timestamp_has_one_row_and_no_invented_cadence(self):
        records = [record for sensor in EXPECTED_SENSOR_IDS
                   for record in records_at(sensor, (0,))]
        wide, _, _, _, metadata = load_streaming_records(
            records, preserve_actual_timestamps=False,
        )
        self.assertEqual(len(wide), 1)
        self.assertEqual(wide.index[0], BASE)
        self.assertIsNone(metadata["alignment_frequency_seconds"])
        self.assertEqual(metadata["interval_inference_source"], "undetermined")

    def test_explicit_frequency_uses_regular_grid_under_both_checkpoint_policies(self):
        for preserve in (True, False):
            with self.subTest(preserve=preserve):
                wide, _, _, _, metadata = load_streaming_records(
                    five_sensor_records(), freq="5s", preserve_actual_timestamps=preserve,
                )
                expected = pd.date_range(BASE, periods=3, freq="5s")
                np.testing.assert_array_equal(wide.index.as_unit("ns").asi8, expected.as_unit("ns").asi8)
                self.assertEqual(metadata["frequency_source"], "explicit")
                self.assertFalse(metadata["irregular_timestamps_preserved"])

    def test_buffer_readiness_and_flush_use_same_checkpoint_grid(self):
        records = five_sensor_records()
        for preserve, expected_rows in ((False, 6), (True, 5)):
            with self.subTest(preserve=preserve):
                buffer = SensorBuffer(
                    freq="auto", min_window=6, preserve_actual_timestamps=preserve,
                )
                buffer.extend(records)
                progress = buffer.ready_progress()
                self.assertEqual(progress["resampled_rows"], expected_rows)
                self.assertEqual(buffer.is_ready(), not preserve)
                if preserve:
                    with self.assertRaisesRegex(ValueError, "need at least 6"):
                        buffer.flush_with_records()
                    self.assertEqual(len(buffer), len(records))
                    buffer.min_window = 5
                retained, wide, _, _, _, _ = buffer.flush_with_records()
                self.assertEqual(len(wide), expected_rows)
                self.assertEqual(retained, records)
                self.assertEqual(len(buffer), 0)

    def test_pipeline_selects_checkpoint_grid_and_reuses_preloaded_weights(self):
        for time_aware in (False, True):
            for supplied_bundle in (False, True):
                with self.subTest(time_aware=time_aware, supplied_bundle=supplied_bundle):
                    bundle = fake_bundle(time_aware)
                    args = SimpleNamespace(
                        freq="auto", sensor_checkpoint="unused-sensor.pt",
                        batch_size=8, device="cpu",
                    )

                    def align(frame, **kwargs):
                        self.assertIs(kwargs["preloaded_model"], bundle.sensor_model)
                        self.assertIs(kwargs["preloaded_checkpoint"], bundle.sensor_checkpoint)
                        expected_seconds = (0, 2, 4, 8, 10) if time_aware else (0, 2, 4, 6, 8, 10)
                        expected = pd.DatetimeIndex(
                            [BASE + pd.Timedelta(seconds=s) for s in expected_seconds],
                        )
                        np.testing.assert_array_equal(frame.index.as_unit("ns").asi8,
                                                      expected.as_unit("ns").asi8)
                        return frame.fillna(0.0), {
                            "sensor_ids": list(EXPECTED_SENSOR_IDS), "evaluation_metrics": [],
                        }

                    with tempfile.TemporaryDirectory() as directory, patch.object(
                        streaming, "load_sensor_imputeformer_checkpoint",
                        return_value=(bundle.sensor_model, bundle.sensor_checkpoint),
                    ) as load, patch.object(
                        streaming, "align_with_pretrained_imputeformer", side_effect=align,
                    ) as infer:
                        path, _, mask, metadata = streaming.run_imputeformer_alignment_from_records(
                            five_sensor_records(), args, Path(directory),
                            model_bundle=bundle if supplied_bundle else None,
                        )
                        self.assertEqual(len(pd.read_csv(path)), 5 if time_aware else 6)
                        self.assertEqual(mask.shape[0], 5 if time_aware else 6)
                        self.assertEqual(metadata["grid_alignment"]["irregular_timestamps_preserved"],
                                         time_aware)
                        self.assertEqual(load.call_count, 0 if supplied_bundle else 1)
                        infer.assert_called_once()


class ApiCheckpointGridPolicyTests(unittest.TestCase):
    def test_loaded_checkpoint_controls_auto_buffer_row_count_and_readiness(self):
        for time_aware in (False, True):
            with self.subTest(time_aware=time_aware), tempfile.TemporaryDirectory() as directory:
                app = create_app("auto", 6, Path(directory), model_bundle=fake_bundle(time_aware))
                endpoints = endpoints_for(app)
                with patch("backend.service.ProductionModelBundle.load") as load:
                    status = endpoints["/v1/buffer/records"](RecordBatch(
                        records=[Record(**record) for record in five_sensor_records()],
                    ))
                    later_status = endpoints["/v1/buffer/status"]()
                    load.assert_not_called()
                self.assertEqual(status.resampled_rows, 5 if time_aware else 6)
                self.assertEqual(status.is_ready, not time_aware)
                self.assertEqual(status.alignment_frequency_seconds, 2.0)
                self.assertEqual(later_status.model_dump(), status.model_dump())

    def test_lazy_auto_loads_once_on_valid_append_and_updates_existing_empty_buffer(self):
        for time_aware in (False, True):
            with self.subTest(time_aware=time_aware), tempfile.TemporaryDirectory() as directory:
                with patch("backend.service.ProductionModelBundle.load",
                           return_value=fake_bundle(time_aware)) as load:
                    endpoints = endpoints_for(create_app("auto", 6, Path(directory)))
                    self.assertFalse(endpoints["/v1/health"]()["models_loaded"])
                    self.assertEqual(endpoints["/v1/buffer/status"]().raw_records, 0)
                    endpoints["/v1/buffer/records"](RecordBatch(records=[]))
                    with self.assertRaises(HTTPException):
                        endpoints["/v1/buffer/records"](RecordBatch(records=[Record(
                            timestamp=BASE.isoformat(), sensor_id="invalid", value=1.0,
                        )]))
                    load.assert_not_called()

                    batch = [Record(**record) for record in five_sensor_records()]
                    first = endpoints["/v1/buffer/records"](RecordBatch(records=batch))
                    second = endpoints["/v1/buffer/records"](RecordBatch(
                        records=batch, buffer_key="another-stream",
                    ))
                    load.assert_called_once()
                    self.assertTrue(endpoints["/v1/health"]()["models_loaded"])
                    self.assertEqual(first.resampled_rows, 5 if time_aware else 6)
                    self.assertEqual(second.resampled_rows, first.resampled_rows)
                    self.assertEqual(endpoints["/v1/buffer/status"]().is_ready, not time_aware)


if __name__ == "__main__":
    unittest.main()
