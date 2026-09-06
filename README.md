# Mobile-Cable Three-Modality Risk Pipeline

This project processes one mobile cable carrying five independent ocean
sensors. It has no site, station, latitude, longitude, or coordinate input.
Audio is not part of training, inference, feature extraction, or risk output.

## Production flow

The API and command-line production paths never train a model. The API loads
the existing ImputeFormer and feature-fusion Transformer checkpoints once when
the service starts and reuses that validated model bundle for every request.
The command-line entry loads the same bundle once per process. Both paths use
forward-only inference and write a new immutable set of artifacts.

1. Accept sensor records with `timestamp`, `sensor_id`, and `value`.
2. Infer each sensor cadence and align all five streams to the fastest sensor
   in the current batch. ImputeFormer fills values missing on that grid.
3. Retain the aligned time series and extract four first-stage features from
   each sensor, the optical image, and the XTF-decoded sonar image.
4. Concatenate the 12 named feature positions in this fixed order:
   `time_series[std,min,energy,fft_1]`,
   `optical_image[brightness,contrast,edge_density,entropy]`, and
   `sonar_image[brightness,contrast,edge_density,entropy]`.
5. Use a trained Transformer only to create a unified representation and
   reconstruct named second-stage features in their original units. The model
   has no risk head.
6. Read `std`, optical `edge_density`, and sonar `edge_density` directly from
   the three second-stage matrices. Calibrate their risks independently and
   calculate `0.20*time_series + 0.40*optical_image + 0.40*sonar_image`.
7. Return the three modality risks and total risk as JSON. No risk CSV is
   generated.

Optical and sonar images keep their native dimensions. Both use 128 by 128
pixel tiles with a nominal 64-pixel stride and edge-anchored final tiles. A
1920 by 1080 input produces 29 by 16, or 464, overlapping tiles. Modality risk
is the mean of every tile's calibrated probability. The sonar API and CLI input
must be the original `.xtf` file; its full ping image, preview, and decoder
metadata are retained before feature extraction.

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

The retained USGS `MATANZAS_18-1110` source, native observations, complete
one-second grid, and chronological train/validation/test splits are under:

```text
data/ocean_time_series_matanzas_18_second/runs/<timestamp>/
```

The source DOI is <https://doi.org/10.5066/P9GKB537>. The model uses 3,600-row
windows, corresponding to one hour only when the inferred reference cadence is
one second. No source row is randomly subsampled or discarded.

## Training

Train the full-size sensor-only ImputeFormer:

```powershell
D:\python\3.11.14\python.exe experiments\train_sensor_imputeformer.py --max-epochs 200
```

If model selection was intentionally stopped after complete epochs, continue
from its saved best validation checkpoint without repeating those epochs:

```powershell
D:\python\3.11.14\python.exe experiments\train_sensor_imputeformer.py `
  --resume-selection-run outputs\sensor_imputeformer_production\runs\<run>
```

For a time-bounded continuation, a saved validation checkpoint can warm-start a
new immutable run. This still covers the complete training split and complete
timeline; it changes the optimization schedule, not the model or dataset size:

```powershell
D:\python\3.11.14\python.exe experiments\train_sensor_imputeformer.py `
  --initial-selection-state outputs\sensor_imputeformer_production\runs\<run>\best_selection_state.pt `
  --training-mask-pattern single_sensor_outage `
  --validation-mask-pattern single_sensor_outage `
  --max-epochs 3 --min-epochs 3 --patience 3 --lr 3e-4 `
  --final-finetune-epochs 1
```

For a saved state whose validation performance is good but whose normalized
outputs have systematic per-sensor bias, fit an affine calibration using only
validation outages and retain at least 10 percent model contribution:

```powershell
D:\python\3.11.14\python.exe experiments\train_sensor_imputeformer.py `
  --initial-selection-state outputs\sensor_imputeformer_production\runs\<run>\best_selection_state.pt `
  --selection-evaluation-only --validation-mask-pattern single_sensor_outage `
  --calibrate-output --minimum-calibration-scale 0.10 `
  --final-finetune-epochs 1
```

Each immutable run under
`outputs/sensor_imputeformer_production/runs/<timestamp>/` retains model
selection history, production refit history, MAE, MSE, RMSE, and MRE metrics,
baseline comparisons, the checkpoint, and the complete aligned five-sensor
time series. Publishing is allowed only when the model beats the strongest
configured baseline on the required outage metrics.

Prepare all three modality splits and first-stage features after alignment:

```powershell
D:\python\3.11.14\python.exe experiments\prepare_three_modality_training_data.py `
  --run-dir data\multimodal_training\runs\<new_timestamp> `
  --aligned-time-series outputs\sensor_imputeformer_production\runs\<run>\aligned_ocean_five_sensor_timeseries.parquet `
  --ocean-dataset-run data\ocean_time_series_matanzas_18_second\runs\<run>
```

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

## API and CLI

Start the service:

```powershell
D:\python\3.11.14\python.exe -m backend.service --host 0.0.0.0 --port 8000
```

`POST /v1/risk/run` requires `optical_image` and `sonar_xtf`. It returns the
complete saved object in `risk_matrix`; `GET /v1/risk/{job_id}/matrix` returns
the same JSON document:

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

Run the CSV command-line pipeline. The CSV must contain the long-format columns
`timestamp,sensor_id,value`; an empty `value` is treated as missing:

```powershell
D:\python\3.11.14\python.exe experiments\full_pipeline\run_streaming_pipeline.py `
  --records-csv path\to\sensors.csv `
  --optical-image path\to\underwater.png `
  --sonar-xtf path\to\sidescan.xtf
```

JSONL remains supported:

```powershell
D:\python\3.11.14\python.exe experiments\full_pipeline\run_streaming_pipeline.py `
  --records-jsonl path\to\sensors.jsonl `
  --optical-image path\to\underwater.png `
  --sonar-xtf path\to\sidescan.xtf
```

Every run uses a timestamped relative output path and rejects an existing
artifact name. Retained intermediate results include original inputs, aligned
time series, XTF decoding, three first-stage matrices, the 12-position
concatenation map with original and normalized values, unified Transformer
vectors, three named second-stage matrices, and the exact selected risk-input
vectors. Alignment evaluation deterministically hides 10 percent of known
input values and reports MAE, MSE, and RMSE without changing the formal aligned
output. Only the active model aliases are atomically replaced after a new run
passes its publication checks; immutable run directories are never overwritten.
