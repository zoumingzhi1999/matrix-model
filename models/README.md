# Model files

`D:\Matrix-model\models` holds trained model weights used by production
inference. The retained legacy `outputs/` directory is not the active
model location.

The active production files are:

- `sensor_imputeformer/imputeformer_ocean_five_sensor.pt`
- `feature_fusion_transformer/feature_fusion_transformer.pt`

Training-run checkpoints are immutable and live below the corresponding
`runs/<timestamp>/` directory. Generated reports, metrics, aligned data, and
inference artifacts belong in `outputs/`, not here.

`experiments/train_sensor_imputeformer.py` starts fresh adaptive ImputeFormer
training and publishes the checkpoint selected on chronological validation
data after evaluating it on the held-out test split. The final weights are not
refit on validation/test data. The active alias is replaced atomically; older
weights are not loaded to initialize this workflow.

The adaptive checkpoint records variable-length inference, interpolation
residuals, sensor normalization, and bounded sampling-interval encoding.
The default 3,600-row context is not a minimum input length or a fixed physical
duration. Both short sequences and arbitrarily long windowed sequences are
accepted, including hourly input. Accuracy depends on the measured evaluation
conditions and the deployment data; the baseline report provides comparisons
rather than an unconditional accuracy guarantee.

Sensor training history and baseline metrics are in
`outputs/sensor_imputeformer_production/runs/<timestamp>/evaluation_metrics.json`.
Production inference keeps one aligned CSV, one metrics CSV, six feature CSVs,
and one risk JSON under its output directory. Input paths are configured in
`config/input_paths.py` through `TIME_SERIES_INPUT_PATH`,
`OPTICAL_IMAGE_INPUT_PATH`, and `SONAR_IMAGE_INPUT_PATH`, or supplied by the
API, Python calls, or CLI. All input-path settings, including the legacy
`SONAR_XTF_INPUT_PATH`, default to `None`; integrations supply their input
paths explicitly. The primary sonar input accepts single-page
`.tif` / `.tiff` files (case-insensitive) or a directory of those TIFFs.
The explicit `sonar_xtf` / `--sonar-xtf` entry preserves raw XTF compatibility.
The production checkpoints and their original training sources are retained.
Each second-stage feature CSV contains 128 unified coordinates, `z_000`
through `z_127`. Production responses do not include `input_records_path`.
