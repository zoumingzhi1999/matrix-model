# Matrix-model: Mobile-Cable Three-Modality Risk Pipeline

The TIF pipeline is installed directly in `D:\Matrix-model`, reusing its
existing project directories. Integrate through the HTTP service or the
Python `RiskPipeline` class. `run_indicator1_inference.py` is an optional
bundled-data example and is not required by either integration.

Install the locked dependencies into the existing `.venv` and start HTTP:

```powershell
Set-Location D:\Matrix-model
uv sync --frozen
.\.venv\Scripts\python.exe -m backend.service --host 0.0.0.0 --port 8000
```

From another PowerShell terminal, submit paths accessible to the service
process (replace the measurement paths with your input files):

```powershell
$request = @{
  time_series_input = 'D:\measurements\sensors.csv'
  optical_image = 'D:\measurements\underwater.png'
  sonar_image = 'D:\measurements\sonar.tif'
} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/v1/risk/run' `
  -Method Post -ContentType 'application/json' -Body $request
```

For direct Python integration, run from `D:\Matrix-model` with this environment:

```python
from f_model import RiskPipeline

pipeline = RiskPipeline(device="auto", freq="auto")
result = pipeline.run(
    time_series_input=r"D:\measurements\sensors.csv",
    optical_image=r"D:\measurements\underwater.png",
    sonar_image=r"D:\measurements\sonar.tif",
)
risk = result["risk_matrix"]
```

The primary sonar field is `sonar_image`: a single-page `.tif` / `.tiff` file
or a directory containing those files, with case-insensitive extensions.
Multipage TIFFs are rejected. All configured input defaults are `None`; callers
provide inputs explicitly or set `config/input_paths.py`. Relative input paths
resolve from `D:\Matrix-model`. Optical input formats are unchanged.

Each production run retains exactly nine files: one aligned time-series CSV,
one alignment metrics CSV, three first-stage CSVs, three second-stage CSVs
with 128 feature columns (`z_000` through `z_127`), and one risk JSON. The API
returns `risk_matrix`, `risk_matrix_path`, and `artifacts`; it no longer returns
`input_records_path`.

The pre-migration project is backed up at
`D:\Matrix-model-backup-20260907_094031`. Existing files in `data/` and
historical results in `outputs/` are retained. Active model weights are loaded from
`models/`, with current inference results written under `outputs/`.

The six sonar TIFFs in `data/指标1/sonar_tif_images` are converted from the
retained original XTF sources in `data/指标1/sonar_images`. Each TIFF is a
single-page, 32-bit floating-point image with the native decoded dimensions
and the existing XTF normalization. Its pixel values match the decoded XTF
array exactly when read back. To convert another XTF file or directory into
a new output directory:

```powershell
.\.venv\Scripts\python.exe experiments\convert_sonar_xtf_to_tif.py `
  --source D:\measurements\sonar_xtf `
  --output-dir D:\measurements\sonar_tif
```

Without arguments, the converter uses the bundled XTF and TIFF directories
above. It preserves original XTF files and refuses to overwrite output files.

This project processes one mobile cable carrying five independent ocean
sensors. It has no site, station, latitude, longitude, or coordinate input.
Audio is not part of training, inference, feature extraction, or risk output.

## Production flow

The API and command-line production paths never train a model. The API loads
the existing ImputeFormer and feature-fusion Transformer checkpoints once when
the service starts and reuses that validated model bundle for every request.
The command-line entry loads the same bundle once per process. Both paths use
forward-only inference and write a new immutable set of artifacts.

The active sensor model is `20260906_143522_756272`, selected for its lower
overall error in the comparison below. Its SHA-256 is
`7e510ac344af67afd05dc382ab7382db72cc3f0d3c91a576e64db6a030d01411`.
This checkpoint uses a regular common time grid and variable-length inference.

1. Accept sensor records with `timestamp`, `sensor_id`, and `value`.
2. Select the sensor with the most distinct valid observations in the current
   batch. Infer its sampling interval and construct a regular common grid
   covering the input range. Map observations to that grid, then use
   ImputeFormer to fill missing values. The interval is not fixed to one second.
3. Retain the aligned time series and extract four first-stage features from
   each sensor, the optical image, and the sonar image.
4. Concatenate the 12 named feature positions in this fixed order:
   `time_series[std,min,energy,fft_1]`,
   `optical_image[brightness,contrast,edge_density,entropy]`, and
   `sonar_image[brightness,contrast,edge_density,entropy]`.
5. Use the trained Transformer to create 128-dimensional unified CLS
   representations. Save these second-stage features in three CSVs, retaining
   each sensor or image tile's row identifiers followed by `z_000` through
   `z_127`. For each target row, the other two modalities provide their current
   feature-table means as context. All three CSVs therefore contain cross-modal
   representations. The model has no risk head.
6. Decode each representation with its modality's trained unified
   decoder, then reverse the training normalization. Keep these four-dimensional
   reconstructions in memory to read `std`, optical `edge_density`, and sonar
   `edge_density` for the existing calibrated risk calculation:
   `0.20*time_series + 0.40*optical_image + 0.40*sonar_image`.
7. Return the three modality risks and total risk as JSON. No risk CSV is
   generated.

Optical and sonar images keep their native dimensions. Both use 128 by 128
pixel tiles with a nominal 64-pixel stride and edge-anchored final tiles. A
1920 by 1080 input produces 29 by 16, or 464, overlapping tiles. Modality risk
is the mean of every tile's calibrated probability. The primary sonar API and
CLI input (`sonar_image` / `--sonar-image`) accepts single-page `.tif` and
`.tiff` images or a directory of those images. Multipage TIFFs are rejected.
The explicit `sonar_xtf` / `--sonar-xtf` compatibility entry still accepts
original XTF sources. XTF decoding and intermediate arrays are kept in memory
during inference; only the requested feature matrices are written.

## Sensor contract

The five supported sensor identifiers are:

| `sensor_id` | Unit | Meaning |
| --- | --- | --- |
| `water_temperature_c` | degree C | Seawater temperature |
| `water_depth_m` | m | Direct immersion depth, positive downward |
| `turbidity_ntu` | NTU | Turbidity |
| `significant_wave_height_m` | m | Significant wave height |
| `current_speed_mps` | m/s | Current-speed magnitude |

Runtime callers supply `water_depth_m` directly. Pressure-to-depth conversion
is only part of preparation for the pressure-based training source and is not
applied to API depth values.

The time-aware training entry point uses the following default root for the
USGS `MATANZAS_18-1110` source, native observations and chronological splits:

```text
data/ocean_adaptive_v2/runs/<timestamp>/
```

The experimental dataset run is `20260906_150000_native_current`. Its source DOI is
<https://doi.org/10.5066/P9GKB537>. It adds 3,151,259 quality-controlled
one-second current samples from the original 16 Hz Vector bursts to the
unchanged temperature, depth, turbidity and significant-wave-height records.
The current magnitude now describes near-bottom instantaneous horizontal
velocity; the older dataset used hourly depth-averaged velocity. Absolute
errors across those datasets are therefore not directly comparable.
Only depth and current have genuine second-scale measurements; temperature,
turbidity and significant wave height remain sparse. Significant wave height
is a window statistic, not instantaneous surface elevation. Empty cells remain
unobserved and are never used as training targets. The dataset manifest records
source URLs, checksums, quality filtering and the original split boundaries.
Reproduction uses `experiments/prepare_adaptive_ocean_dataset.py`; install the
optional `ocean-data` dependencies for its DAP2 reader.

Inference accepts 100, 1,000, 3,600, or any positive number of aligned time
points, subject to available memory. A single timestamp is also accepted when
the five sensor identifiers are declared and at least one value is observed.
The model's default 3,600-row context is a computation window, not a required
input length or physical duration. Short inputs use native variable-length
inference; long inputs use overlapping windows with complete tail coverage.

Reference selection counts distinct finite observations, breaks ties by shorter
inferred cadence, then by sensor identifier. With `--freq auto`, the reference
sensor's inferred cadence determines the active model's regular common grid.
Missing or jittered reference timestamps are regularized; asynchronous
observations map to the nearest grid point. Values in the same sensor/time
cell are averaged, and the merge count is logged. An explicit `--freq` overrides
the inferred cadence. Imputation preserves these
mapped observed values. A declared sensor may be missing throughout the batch;
the log identifies it and its unavailable evaluation metrics remain null.

There is no fixed seconds/minutes/hours admission limit. The active model
receives calendar features and an encoding of the common grid interval; it
does not receive a separate true elapsed interval for each original step.
Selecting a time-aware experimental checkpoint instead preserves actual
reference timestamps under `--freq auto` and supplies elapsed/per-step times.
Accepting submillisecond, hourly
or other cadences does not establish accuracy for unseen field distributions;
the measured held-out results and target-cable validation determine accuracy.

## Training

The following command trains a new experimental time-aware ImputeFormer from
random initialization; it does not reproduce the active regular-grid model:

```powershell
D:\python\3.11.14\python.exe experiments\train_sensor_imputeformer.py `
  --dataset-run data\ocean_adaptive_v2\runs\20260906_150000_native_current `
  --training-lengths 100 256 500 1000 2000 3600 `
  --sampling-intervals-seconds 1 2 5 10 30 60 300 3600 `
  --max-epochs 48 --min-epochs 24 --patience 10 --steps-per-epoch 160 `
  --input-embedding-dim 24 --learnable-embedding-dim 24 --feed-forward-dim 96 `
  --lr 0.001 --min-delta 0.00001 --dropout 0.05 --device cuda
```

This entry point invokes `train_time_aware_imputeformer.py`. When validation
disables all neural residuals, it then invokes `calibrate_time_aware_baseline.py`.
Otherwise it retains the evaluated neural model without adding an independently
calibrated median correction. The command above
reproduces the new training configuration. The model uses two Transformer
layers, learning rate 0.001 and dropout 0.05. Five static sensor embeddings are
shared across time, giving 84,630 trainable parameters. Physical time features
provide temporal position instead of a learned parameter for every row.
The maximum batch size is eight;
`--batch-row-budget 4096` reduces batch size for long contexts. The context is
3,600 rows with inference stride 1,800. `--evaluation-repeats` controls seeded
evaluation repeats (default two). Use `--help` for supported options.

Each batch samples a sensor-balanced window around observed measurements.
Training lengths are selected from 100, 256, 500, 1,000, 2,000 and 3,600 rows;
15 percent of batches draw additional lengths between 32 and the configured
context. Lengths are capped by the available split length. Every training row is available
for sampling, but an epoch does not visit every row and windows may repeat.
Scales are sampled from the native timestamps plus absolute intervals of
1, 2, 5, 10, 30, 60, 300 and 3,600 seconds. These are measured-bin means built
separately within each split. Twenty percent of training windows randomly
subsample actual rows to introduce additional irregular gaps. Missingness
includes 20/50 percent point loss, temporal blocks, temporary single-sensor
outages, whole-window sensor outages, and differing sensor sampling rates.
Multirate masks sample each sensor's actual observation sequence and preserve
the densest reference sensor, avoiding aliases caused by global row indices.

Normalization uses only the chronological training split. Model selection uses
fixed seeded windows from the validation split, and the selected checkpoint is
then evaluated on the chronological test split. Evaluation compares ImputeFormer
with linear interpolation, forward/backward hold, and training sensor means on
identical hidden measured values. It reports MAE, MSE, RMSE, MRE, and unique
target counts per sensor and length/interval/missingness condition. Targets are
also separated into no same-sensor context, one-sided context and two-sided
context. Requested and actual lengths are both reported: the hourly held-out
split cannot provide 3,600 hours when its recorded duration is shorter.
Sampled evaluation windows may overlap;
reported target occurrences are not necessarily independent unique observations.

Each run writes its history and evaluation report to
`outputs/sensor_imputeformer_production/runs/<timestamp>/evaluation_metrics.json`
and stores weights under `models/sensor_imputeformer/runs/<timestamp>/`.
The experimental training output is the validation-selected checkpoint, with no subsequent
refit on the validation or test data. Residual gains in [0, 1] are calibrated
per sensor and context class using validation only; interpolation is retained
when validation provides no useful residual improvement. A separate validation
calibration can blend the time-linear estimate with the median of the same
sensor's currently visible values in the window. It requires at least three
visible values, a supported validation group, lower MAE and no RMSE regression.
These robust-estimation gains are separate from neural residual gains and are
printed in inference logs. They must not be reported as Transformer learning
improvements. The whole-input baseline still handles gaps beyond the neural
window, and original observations are preserved exactly. Baseline results
are reported for inspection; publication does not imply superiority to every
baseline or a guaranteed field accuracy. No aligned training-data copy is
generated by this training command.

The inactive experimental run is `20260906_161927_315820`, calibrated from fresh neural run
`20260906_155236_644869` (24 epochs completed, epoch 1 selected). Its SHA-256 is
`861b042fb1345240713004e485af46137e88acc9bea4f5234a65fe7cf677e7f5`.
Its full results are recorded in
`outputs/sensor_imputeformer_production/runs/20260906_161927_315820/evaluation_metrics.json`.
All neural residual gains were rejected by validation. This experimental pipeline
therefore uses physical-time interpolation and separately calibrated visible
medians; its improvement over linear interpolation is not a Transformer gain.
Without any same-sensor context, a separate train-only median candidate is
validated against the training mean. Only temperature passed both MAE and RMSE
criteria, so its default is 15.100550 degrees C; the other sensors keep their
training means. These defaults are estimates, not recovered observations.

Across 324 identical length/interval/missingness test cases, equally averaging
each case's sensor-macro standardized errors gives:

| Method | MAE | RMSE |
| --- | --- | --- |
| Active checkpoint `20260906_143522_756272`, on the same new targets | 0.419466 | 0.666495 |
| Inactive time-aware experiment `20260906_161927_315820` | 0.458440 | 0.713127 |
| Physical-time linear interpolation | 0.464046 | 0.723592 |

The time-aware experiment's overall MAE is 9.29% worse than the active checkpoint.
It is therefore not enabled for production. With two-sided visible
context its MAE improves 5.41%, and the multirate case average improves 5.94%;
No-same-sensor-context MAE regresses 13.78%. The no-context prediction still
relies on training statistics, not recovered measurements. See
`previous_model_comparison.json` in the same output directory.

Pooling target errors per sensor before averaging sensors, a different weighting,
gives MAE 0.447788 to 0.437334 (2.33% improvement over time-linear interpolation)
and RMSE 0.979009 to 0.932964 (4.70%). Temperature and depth MAE nevertheless
regress 0.14% and 0.21% against that baseline; these regressions are retained in
the report. Neither weighting supports a claim of uniformly high accuracy.

The experiment's additional `unseen_time_scales_evaluation.json` covers 3/15/120/1800-second
scales, 137/777/2501-row requests and deletion of actual rows. Overall MAE is
0.407139 versus linear 0.408238 on the original timelines, and 0.414512 versus
0.419602 after irregular row deletion. Longer requested hourly-scale windows
are reported with their available actual length, not fabricated.

Historical verification of the inactive time-aware experiment passed
1/2/100/256/500/1000/2000/3600/7217-row inputs,
0.0002-second through hourly timestamps and irregular timestamps. A complete
real optical/XTF run retained exactly nine files; all six feature logs matched
their CSV files and observed values were preserved. Automated tests report
120 passes and two skipped tests whose repository-local XTF fixtures are absent;
the separate full pipeline used the retained external XTF file successfully.
The prior model's test dates have already been inspected. These chronological
test results are a development benchmark, not a new blind field evaluation.
Comparisons with the previous checkpoint must use the same new dataset,
normalization, windows and hidden targets. Timestamp-construction checks for
submillisecond inputs establish functionality, not field accuracy at that rate.

Prepare all three modality splits and first-stage features after alignment:

```powershell
D:\python\3.11.14\python.exe experiments\prepare_three_modality_training_data.py `
  --run-dir data\multimodal_training\runs\<new_timestamp> `
  --aligned-time-series path\to\complete_aligned_ocean_timeseries.parquet `
  --ocean-dataset-run data\ocean_time_series_matanzas_18_second\runs\<run>
```

The preparation command requires an already aligned Parquet table with a
DatetimeIndex covering the complete referenced ocean dataset and its split
order. This offline dataset preparation is separate from the production output
retention policy below.

The preparation run retains the complete UIEB archive and extracted images,
all six CC BY 4.0 echosounder XTF files from Zenodo
<https://doi.org/10.5281/zenodo.5558794>, deterministic
train/validation/test splits, decoded sonar arrays, every feature row,
checksums, and manifests. UIEB raw/reference pairs remain in the same split.
The six XTF files use a chronological 4/1/1 whole-file split so pings from one
source file cannot leak across splits. A separate small GPL-3.0 side-scan XTF is
retained only as a decoder regression sample.

The XTF data support the Journal of Physical Oceanography paper "Shear
Instabilities and Stratified Turbulence in an Estuarine Fluid Mud"
(<https://doi.org/10.1175/JPO-D-21-0230.1>).

Train and independently evaluate the feature-fusion Transformer, then calibrate
the fixed risk formula against that exact checkpoint:

```powershell
D:\python\3.11.14\python.exe experiments\train_feature_fusion_transformer.py `
  --feature-root data\multimodal_training\runs\<run>\features `
  --epochs 200

D:\python\3.11.14\python.exe experiments\risk_pipeline\calibrate_second_stage_risk_config.py `
  --feature-root data\multimodal_training\runs\<run>\features
```

The public sources are not physically synchronized observations of one event.
Transformer tuples are therefore reproducible, split-isolated, label-matched
constructed pairs, and the pairing audit says so explicitly. This is suitable
for pretrained feature fusion, but field acceptance still requires validation
and preferably fine-tuning on truly paired measurements from the target cable.

## Model storage

Production inference loads the published checkpoints from:

```text
models/
  sensor_imputeformer/imputeformer_ocean_five_sensor.pt
  feature_fusion_transformer/feature_fusion_transformer.pt
```

The active sensor file is a copy of
`models/sensor_imputeformer/runs/20260906_143522_756272/imputeformer_ocean_five_sensor.pt`.
The time-aware experiment remains available under its versioned run directory.

Training histories and metrics remain under `outputs/`. Versioned training
checkpoints stay under the corresponding `models/.../runs/<timestamp>/`
directory. Fresh ImputeFormer training evaluates the validation-selected model
on the held-out test split before atomically replacing its active checkpoint.

## Run the complete pipeline

`run_pipeline.py`, the public Python `RiskPipeline` class, and the HTTP API
share the same production pipeline: automatic sensor alignment, both feature
extraction stages, and the final three-modality risk JSON. They use the active
`20260906_143522_756272` sensor checkpoint described above.

The three local input paths are explicitly configured in
`config/input_paths.py`:

```python
TIME_SERIES_INPUT_PATH = r"D:\measurements\sensors.csv"
OPTICAL_IMAGE_INPUT_PATH = r"D:\measurements\underwater.png"
SONAR_IMAGE_INPUT_PATH = r"D:\measurements\sonar.tif"
```

`TIME_SERIES_INPUT_PATH`, `OPTICAL_IMAGE_INPUT_PATH`,
`SONAR_IMAGE_INPUT_PATH`, and `SONAR_XTF_INPUT_PATH` all default to `None`.
Relative configured paths resolve from the `D:\Matrix-model` repository root.
Explicit Python, API request, or command-line paths
override them and also resolve from the repository root when relative.
The time-series file accepts long/wide CSV or long-format JSONL.
The primary sonar path accepts a single-page TIFF or a directory of TIFFs.
The legacy `SONAR_XTF_INPUT_PATH` setting remains a fallback when
`SONAR_IMAGE_INPUT_PATH` is unset; raw XTF overrides use the explicit
`sonar_xtf` / `--sonar-xtf` entry.

After setting the three paths, run the entire workflow from the project root:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

Use `D:\Matrix-model\.venv\Scripts\python.exe` after `uv sync --frozen`.
Paths can also be supplied directly:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --time-series-input D:\measurements\sensors.csv `
  --optical-image D:\measurements\underwater.png `
  --sonar-image D:\measurements\sonar.tif
```

For raw sonar, replace `--sonar-image` with `--sonar-xtf` and its XTF path.
The entry point also accepts `--records-csv`, `--records-jsonl`,
`--records-json`, and `--stdin`. Use `python run_pipeline.py --help` for all
options. The previous entry script remains compatible.

### Optional Indicator-1 example

`run_indicator1_inference.py` is optional and is not used by HTTP or Python
integration. Run the complete `data/指标1` batch without changing the general
input config:

```powershell
Set-Location D:\Matrix-model
.\.venv\Scripts\python.exe run_indicator1_inference.py
```

The script explicitly sets all three input paths at its top:

- `data/指标1/time_series/time_series_100_native_records.csv`
- `data/指标1/optical_images`, currently all 30 optical images.
- `data/指标1/sonar_tif_images`, containing the six converted TIFF files.

It reads the time-series source once and processes the supported files directly
inside each image directory in filename order. All inputs form one batch,
producing one risk JSON with the three modality risks and the combined risk.
The optical and sonar risks each average all of their respective tiles across
all files. Feature CSVs retain source-file and tile identifiers. Both extraction
stages use the existing pipeline; the saved second-stage outputs are the
Transformer's 128-dimensional unified representations.

Each invocation creates a new directory under `outputs/indicator1_inference`,
retains the same nine result files, and prints the saved risk JSON path. Existing
results are never overwritten. Complete feature matrices and inference progress
are emitted to the process log without adding log or manifest files.

Override inputs or execution options with `--time-series-input`,
`--optical-image`, `--sonar-source`, `--output-root`, `--device auto|cpu|cuda`,
or `--batch-size` (a positive integer, default 8). Image source arguments accept
either a file or a directory. `--sonar-source` accepts single-page `.tif` and
`.tiff` files only, including uppercase extensions; sonar directories select
only these TIFF files. The active checkpoints remain the production
defaults, and alignment uses `freq="auto"`, with no fixed one-second interval.

The bundled CSV contains 100 long-format records: 20 observations for each of
the five sensors. Their time coverage differs, so automatic alignment can
produce 71,827 common-grid rows across the complete input time range. Those
filled rows are estimates; this sample run is not evidence of reconstruction
accuracy throughout that time range.

### Python calls

```python
from f_model import RiskPipeline

pipeline = RiskPipeline(device="auto", freq="auto")
result = pipeline.run(
    time_series_input=r"D:\measurements\sensors.csv",
    optical_image=r"D:\measurements\underwater.png",
    sonar_image=r"D:\measurements\sonar.tif",
)

risk = result["risk_matrix"]
risk_json_path = result["risk_matrix_path"]
aligned_csv = result["artifacts"]["aligned"]
metrics_csv = result["artifacts"]["alignment_metrics"]
first_stage_csvs = result["artifacts"]["first_stage_features"]
second_stage_csvs = result["artifacts"]["second_stage_features"]
```

`pipeline.run()` uses the configured paths when overrides are omitted; the
default configuration has no input paths. For
records already in memory, supply a list of `timestamp`, `sensor_id`, and
`value` dictionaries instead of a time-series file. Declare all five sensor
identifiers; missing values use `None`:

```python
result = pipeline.run(
    records=records,
    optical_image=r"D:\measurements\underwater.png",
    sonar_image=r"D:\measurements\sonar.tif",
)
```

The same `pipeline` instance loads its models on the first run and reuses them
for subsequent calls. `freq="auto"` infers the reference sensor cadence for
each batch, with the variable-length and time-axis contract described above.
Each call creates a unique task directory under `outputs/streaming_jobs` by
default. The constructor's `output_root` changes this parent directory;
`run(output_dir=...)` selects an exact task directory, which must not already
exist. The CLI's `--output-dir` follows the same rule. Result files are opened
with exclusive creation (`x` or `xb`); an existing destination is rejected even
if another process creates it during a run. Earlier job directories and their
results are retained, and new runs never overwrite them.

### HTTP service

Start the service:

```powershell
Set-Location D:\Matrix-model
.\.venv\Scripts\python.exe -m backend.service --host 0.0.0.0 --port 8000
```

`POST /v1/risk/run` accepts `time_series_input`, `optical_image`, and either
`sonar_image` or `sonar_xtf` paths, with configured paths as fallbacks.
`sonar_image` accepts single-page TIFF input; `sonar_xtf` is the explicit
legacy XTF entry. Providing both sonar fields is rejected. Sensor records can
also be supplied through the buffer endpoints. It returns the complete saved object in
`risk_matrix`; `GET /v1/risk/{job_id}/matrix` downloads the same JSON document.
Input paths must be readable by the server process. The response does not
include the former `input_records_path` field.
Add `?download=true` to `POST /v1/risk/run` to download its saved JSON directly.
Each modality's JSON `source` identifies its 128-dimensional second-stage CSV.
The JSON also describes how the unified decoder, inverse normalization, and
selected reconstructed feature produce the risk input; the named risk feature
is not a column in the saved embedding CSV.
The risk values below illustrate the response structure, not measured results:

```json
{
  "modalities": {
    "time_series": {"risk": 0.0, "fixed_weight": 0.20},
    "optical_image": {"risk": 0.0, "fixed_weight": 0.40},
    "sonar_image": {"risk": 0.0, "fixed_weight": 0.40}
  },
  "total_risk": 0.0,
  "risk_level": "very_low",
  "formula": "0.20*time_series + 0.40*optical_image + 0.40*sonar_image"
}
```

Run the CSV command-line pipeline. CSV may use the long-format columns
`timestamp,sensor_id,value`, or sparse wide format with `timestamp` followed by
the five canonical sensor columns. Empty sensor cells are treated as missing:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --records-csv path\to\sensors.csv `
  --optical-image path\to\underwater.png `
  --sonar-image path\to\sonar.tif
```

JSONL remains supported:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py `
  --records-jsonl path\to\sensors.jsonl `
  --optical-image path\to\underwater.png `
  --sonar-image path\to\sonar.tif
```

Once all three configured paths are set, the CSV/JSONL source and image flags
can be omitted. Python calls, API, and CLI log resolved input paths, alignment and model
inference progress, all rows and columns of each first-stage feature matrix,
and all rows and all 128 embedding columns of each second-stage feature matrix.
Feature values in the logs match the saved CSV contents without truncation.

Each production run retains exactly nine result files:

- One aligned time-series CSV.
- One alignment evaluation metrics CSV.
- Three first-stage feature CSVs: time series, optical image, and sonar image.
- Three second-stage feature CSVs for the same modalities, each containing
  source-row identifiers and the 128 unified CLS coordinates `z_000` to `z_127`.
- One risk JSON containing all three modality risks and the combined risk.

Raw input copies, decoded sonar images/previews, concatenation maps, the
four-dimensional decoder reconstructions, selected risk-input vectors, and
extra summary files are not persisted by production inference. Logs are emitted
to the process logging stream.
Alignment evaluation deterministically hides 10 percent of known input values
and reports MAE, MSE, and RMSE without changing the formal aligned output.
When too few observations exist for holdout evaluation, the report marks that
condition and leaves the unavailable metrics null. Timestamped artifact names
are never overwritten. Offline dataset preparation and model training retain
their separate source data, weights, and evaluation reports.
