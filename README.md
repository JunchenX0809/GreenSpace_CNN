# GreenSpace_CNN

GreenSpace_CNN is a multi-task PyTorch/TorchGeo pipeline for learning perceived
green-space characteristics from aerial imagery and multi-rater survey labels.
The active model uses a Satlas-pretrained Swin V2 B backbone with shared image
features and four output heads:

- seven binary features (gardens is currently excluded);
- path shade (two classes);
- structured/unstructured score (1–5 regression);
- vegetation distribution score (1–5 regression).

TensorFlow/Keras notebooks remain as historical references. New training,
evaluation, packaging, and demonstration work is PyTorch-first.

## Current status (July 2026)

The approved raw survey is now fixed because rating has commenced:

| Stage | Count |
|---|---:|
| Raw survey response rows | 8,651 |
| Included ratings (`include_tile == yes`) | 8,297 |
| Aggregated labeled images | 5,518 |
| Training images | 3,311 |
| Validation images | 1,104 |
| Test images | 1,103 |

The 34 exact duplicate raw-row copies are intentionally retained. The
preprocessing pipeline aggregates every included row by image and does not edit
the raw survey.

The latest completed full run is
`PyTorch_20260719_full_windows`. It used 5 frozen-backbone warm-up epochs and
24 end-to-end fine-tuning epochs (29 total, about 6.56 hours) before early
stopping. The configured full-run ceiling is 100 fine-tuning epochs.

Best-MCMAE checkpoint test results:

| Metric | Test value |
|---|---:|
| Macro PR-AUC | 0.8770 |
| Macro ROC-AUC | 0.9354 |
| Macro F1 at validation-tuned thresholds | 0.8057 |
| Shade accuracy | 0.6582 |
| Structured-score MAE vs. rater mean | 0.5584 |
| Vegetation-score MAE vs. rater mean | 0.4172 |

These values come from
`report_outputs/runs/PyTorch_20260719_full_windows/overall_metrics_by_split_best_mcmae.csv`.
Generated reports and model files are ignored by Git.

## Packaged workflow

| Step | Interface | Status |
|---|---|---|
| Google Drive image download | `scripts/download_drive_images.py` | Ready |
| Survey preprocessing and splitting | `scripts/preprocess.py` | Ready |
| Resumable PyTorch training | `scripts/train_torch.py` | Ready |
| Saved-history epoch visual | `scripts/plot_training_curves.py` | Ready |
| Evaluation and validation threshold tuning | `scripts/evaluate_torch.py` | Ready |
| Standalone prediction CLI | `scripts/predict_torch.py` | Ready |
| Read-only pipeline validation CLI | `scripts/validate_pipeline.py` | Ready |
| End-to-end 50-image showcase | `notebooks/CORE_pipeline_v1.ipynb` | Ready |

Prediction loads one explicit portable run bundle: the selected checkpoint,
its saved model configuration, and the matching validation-tuned threshold
CSV. The validation command checks readiness without launching training,
evaluation, or prediction.

## Install

Python 3.11 is the tested environment.

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_python_version.py
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_python_version.py
```

TensorFlow is not required for the active PyTorch workflow.

## Prediction-only clean-clone trial: 50 images

This reviewer path does not require Google Drive authentication, survey data,
split manifests, preprocessing, training, or evaluation. The current
`requirements.txt` installs the full project environment; a smaller
inference-only dependency file is not yet provided.

Download the July 19 BestMCMAE model bundle and 50-image inference sample from
the [GreenSpace_CNN external-review files](https://drive.google.com/drive/folders/1tlsfN30WkAFBkwEmtTJmGt-uA6KbKXZA).
The Drive folder also contains raw survey samples, but they are not needed for
this prediction trial.

Extract the model bundle without renaming its directory or internal files:

```text
models/runs/PyTorch_20260719_full_windows/
├── best_mcmae_PyTorch_20260719_full_windows.pt
├── model_config_PyTorch_20260719_full_windows.json
└── thresholds_best_mcmae.csv
```

See [`models/README.md`](models/README.md) for the external-artifact contract
and reference checksums. Extract the image sample to:

```text
data/cache/inference_images/review50/
```

That directory must contain exactly 50 top-level JPG, JPEG, or PNG files;
subdirectories are not scanned. The model and image input directories must
exist before running the commands below. The prediction output directory is
created automatically.

First validate the real bundle and all 50 input filenames without requiring
labeled data:

```console
python scripts/validate_pipeline.py --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt --skip-data --inference-dir data/cache/inference_images/review50 --output-dir predictions/review_trial --device cpu
```

Every validation check should report `PASS`, including `Inference images: 50`.
Validation checks supported filenames but does not fully decode every image, so
run a five-image inference smoke test next:

```console
python scripts/predict_torch.py --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt --image-dir data/cache/inference_images/review50 --dataset-tag review50 --limit 5 --output-dir predictions/review_trial --device cpu
```

Then predict all 50 images by omitting `--limit`:

```console
python scripts/predict_torch.py --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt --image-dir data/cache/inference_images/review50 --dataset-tag review50 --output-dir predictions/review_trial --device cpu
```

The two outputs are:

```text
predictions/review_trial/predictions_PyTorch_20260719_full_windows_review50_sample5.csv
predictions/review_trial/predictions_PyTorch_20260719_full_windows_review50.csv
```

The full command should report `Images: 50`. Its CSV contains 50 unique image
rows and 19 columns for the current seven-binary-label model. Existing output
files are protected; use a new dataset tag or pass `--overwrite` explicitly
when a replacement is intended.

## Data contract

Local data is intentionally not versioned:

```text
data/
  raw/                              raw survey CSV
  cache/images/                     cached rated images
  interim/                          cleaned, run-tagged survey CSV
  processed/
    labels_soft_<run-tag>.csv
    labels_hard_<run-tag>.csv
    preprocessing_summary_<run-tag>.json
    splits/
      train.csv
      val.csv
      test.csv
```

The preprocessing summary is the provenance record for a prepared dataset. It
stores input hashes, explicit input/output paths, seeds, counts, parameters,
and duplicate-row policy. Split manifests use the established two-stage
60/20/20 split with seeds 123 and 456.

For data stored outside the repository, set an explicit root:

```powershell
$env:GREENSPACE_DATA_ROOT = "D:\approved\GreenSpace\data"
$env:GREENSPACE_IMAGE_ROOT = "D:\approved\GreenSpace\data\cache\images"
```

## Download rated images

Google Drive download remains separate from preprocessing. Follow
[`instruction_docs/google_drive_auth.md`](instruction_docs/google_drive_auth.md)
to configure OAuth credentials, then inspect the command:

```powershell
python scripts/download_drive_images.py --help
```

Credential files belong under `secrets/` and are ignored by Git.

## Preprocess

Example for the approved survey:

```powershell
python scripts/preprocess.py `
  --survey-csv data/raw/0718_survey_response.csv.csv `
  --filelist-csv data/interim/filelist_with_drive_ids.csv `
  --image-dir data/cache/images `
  --run-tag 0719_windows_full `
  --interim-dir data/interim `
  --processed-dir data/processed `
  --fail-on-missing-images
```

Omit `--filelist-csv` only when Drive identifiers are not required in the
manifests. Use `--sample-size 50` for the deterministic wiring demonstration;
that does not limit full label aggregation.

## Train or resume

Active task, model, data, and training settings live in
`src_torch/config.py`. This is the only source for the active PyTorch training
schedule and optimization controls. Shared label/task choices originate in
`src/label_schema.py`; its separate legacy TensorFlow block exists only to keep
historical notebooks reproducible.

Smoke run:

```powershell
python scripts/train_torch.py `
  --mode smoke `
  --split-dir data/smoke_50/processed/splits `
  --image-root data/cache/images `
  --device auto
```

Full run:

```powershell
python scripts/train_torch.py `
  --mode full `
  --split-dir data/processed/splits `
  --image-root data/cache/images `
  --run-tag PyTorch_<descriptive-tag> `
  --device auto
```

Resume an interrupted run:

```powershell
python scripts/train_torch.py `
  --mode full `
  --split-dir data/processed/splits `
  --image-root data/cache/images `
  --resume models/runs/<run-tag>/last_<run-tag>.pt `
  --device auto
```

Each run saves resumable state, best-MCMAE, best-PR-AUC, final inference
checkpoint, effective configuration, history, standard loss curves, the
presentation-style PR-AUC/MAE epoch visual, and split fingerprints.

Regenerate the presentation-style visual for an existing run:

```powershell
python scripts/plot_training_curves.py `
  --run-dir models/runs/<run-tag>
```

By default, this writes `training_metric_curves.png` inside the run directory.
Use `--output <path>` for an additional presentation copy. The script never
writes to `presentation_visuals_only/` unless that location is requested.

## Evaluate

Evaluation predicts train/validation/test, tunes per-label F1 thresholds only
on validation, evaluates all splits with that fixed threshold map, and writes:

- loss monitoring under `monitoring_output/runs/<run-tag>/`;
- overall and per-label reports under `report_outputs/runs/<run-tag>/`;
- the portable threshold CSV beside the selected checkpoint.

Use an explicit checkpoint:

```powershell
python scripts/evaluate_torch.py `
  --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt `
  --split-dir data/processed/splits `
  --image-root data/cache/images `
  --device auto
```

Or an explicit run directory and variant:

```powershell
python scripts/evaluate_torch.py `
  --run-dir models/runs/PyTorch_20260719_full_windows `
  --preferred-variant best_mcmae `
  --split-dir data/processed/splits `
  --image-root data/cache/images
```

The command never selects a different run implicitly.

## Predict an unlabeled image folder

Prediction accepts JPG, JPEG, and PNG files, sorts them deterministically by
filename, and writes one row per image. It uses the label order saved with the
checkpoint and the selected checkpoint variant's validation-tuned thresholds.

Run a small smoke prediction first:

```powershell
python scripts/predict_torch.py `
  --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt `
  --image-dir D:\approved\unseen_images `
  --dataset-tag july_review `
  --limit 100 `
  --device auto
```

Then omit `--limit` for the complete folder. The default output is:

```text
predictions/predictions_<run-tag>_<dataset-tag>[_sampleN].csv
```

Use `--output` for an exact CSV path or `--output-dir` for another output
folder. Existing files are protected unless `--overwrite` is supplied
explicitly. The stable output schema contains the image filename, probability
and tuned hard prediction for each of seven binary labels, shade class and
confidence, and the two bounded continuous predictions (`score_ev`, `veg_ev`).

## Validate readiness

Before allocating a long training/evaluation job, validate the environment,
all three manifests, every labeled image, the checkpoint/config/threshold
bundle, checkpoint-to-manifest label order, selected device, split isolation,
and output location:

```powershell
python scripts/validate_pipeline.py `
  --checkpoint models/runs/PyTorch_20260719_full_windows/best_mcmae_PyTorch_20260719_full_windows.pt `
  --split-dir data/processed/splits `
  --image-root data/cache/images `
  --device auto
```

For prediction-only handoff checks, omit labeled data and inspect an unlabeled
folder:

```powershell
python scripts/validate_pipeline.py `
  --checkpoint models/runs/<run-tag>/best_mcmae_<run-tag>.pt `
  --skip-data `
  --inference-dir D:\approved\unseen_images `
  --output-dir D:\approved\prediction_outputs `
  --device cpu
```

Validation is read-only: it reports all checks and exits nonzero when any item
needs attention.

## Core 50-image showcase

Open `notebooks/CORE_pipeline_v1.ipynb` for the clean handoff narrative:
OAuth prerequisites, Drive download interface, preprocessing, deterministic
50-image selection, 30/10/10 split, model construction, 1+1 smoke training,
evaluation, and threshold calibration.

Expensive or authenticated actions are guarded by `RUN_*` toggles. The
50-image, two-epoch result is wiring evidence only and must not be interpreted
as model-performance evidence.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src src_torch scripts
python -m pip check
python scripts/check_offline_checkpoint_load.py
python scripts/predict_torch.py --help
python scripts/validate_pipeline.py --help
```

The offline check constructs a tiny synthetic bundle and does not need the
full checkpoint. Production checkpoints are too large for this Git repository.

## Git and artifact boundaries

Git tracks source, scripts, notebooks, tests, documentation, and selected
directory-contract README files. It must not track:

- raw/processed data or cached imagery;
- `.env` files or OAuth credentials;
- model checkpoints;
- evaluation reports, monitoring outputs, or predictions;
- virtual environments and Python caches.

The target remote is:

```text
git@github.com:JunchenX0809/GreenSpace_CNN.git
```

Local commits can be created without pushing. SSH authentication is required
only when communicating with that SSH remote.
