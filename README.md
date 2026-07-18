# GreenSpace_CNN

## Overview
Train a CNN to reproduce greenspace ratings at scale using human judgments from photos/satellite images.

## Reviewer Quickstart (PyTorch)

This is the current review path. The active model code is the PyTorch/TorchGeo
pipeline on the `PyTorch_test` branch. If this README is already on `main` after
the PyTorch merge, you do not need to switch branches.

The Git repository contains code and notebooks. Large artifacts are shared
separately: the trained model bundle, raw data, cached images, and prediction
outputs are intentionally not committed.

### 1) Clone and install

```bash
git clone https://github.com/JunchenX0809/GreenSpace_CNN.git
cd GreenSpace_CNN
git checkout PyTorch_test  # skip this if main already contains the PyTorch README/code

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# Optional, avoids Matplotlib cache warnings on some locked-down machines.
export MPLCONFIGDIR="$PWD/.matplotlib"
```

On Apple Silicon Macs, use an arm64 Python 3.11 build. Check with:

```bash
python -c "import platform; print(platform.machine())"
```

### 2) Download the model bundle

Download the `PyTorch_20260614_220926` best-MC-MAE bundle from the shared Drive
folder:

[Model bundle Google Drive folder](https://drive.google.com/drive/folders/1tlsfN30WkAFBkwEmtTJmGt-uA6KbKXZA?usp=sharing)

Unzip/place it so the project contains this folder:

```text
models/runs/PyTorch_20260614_220926/
├── best_mcmae_PyTorch_20260614_220926.pt
├── model_config_PyTorch_20260614_220926.json
└── thresholds_best_mcmae.csv
```

Those three files are the portable model unit: checkpoint, architecture/config,
and prediction thresholds. They are enough to reuse the trained model without
retraining.

### 3) Run the quick checks

```bash
python scripts/check_python_version.py
python scripts/check_offline_checkpoint_load.py
```

`check_offline_checkpoint_load.py` uses a tiny synthetic checkpoint to verify
that the environment and bundle-loading path work offline. It does not require
the large real checkpoint.

### 4) Run sample image inference

Put a small folder of `.jpg`, `.jpeg`, or `.png` images at
`data/cache/inference_images/`, or point to any image folder:

```bash
export GREENSPACE_INFERENCE_IMAGE_ROOT=/path/to/sample-images
```

Then run this small inference smoke test:

```bash
python - <<'PY'
import os
from pathlib import Path

from src_torch.config import TORCH_DATA_CONFIG, resolve_prediction_output_root
from src_torch.inference import (
    build_prediction_dataframe,
    list_inference_image_paths,
    predict_image_paths,
)
from src_torch.run_bundle import load_run_bundle
from src_torch.training import resolve_device

model_path = Path("models/runs/PyTorch_20260614_220926/best_mcmae_PyTorch_20260614_220926.pt")
image_dir = Path(os.getenv("GREENSPACE_INFERENCE_IMAGE_ROOT", "data/cache/inference_images"))
device = resolve_device("auto")

bundle = load_run_bundle(model_path, device=device)
image_paths = list_inference_image_paths(image_dir, limit=10)
preds = predict_image_paths(bundle.model, image_paths, device=device, batch_size=int(TORCH_DATA_CONFIG["batch_size"]))
pred_df = build_prediction_dataframe(image_paths, preds, bundle.bin_names, bundle.thresholds)

out_dir = resolve_prediction_output_root()
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "predictions_reviewer_smoke.csv"
pred_df.to_csv(out_path, index=False)
print(f"Saved {len(pred_df)} rows to {out_path}")
print(pred_df.head())
PY
```

For a notebook workflow, use `notebooks/05_pyTorch_prediction_demo.ipynb`.

## Survey Design
Human raters evaluate images across standardized criteria:

**Data structure (raw survey):**
- One row per **(rater × image)**
- The raw CSV uses human-readable headers; our cleaner converts them to normalized `snake_case`

**Row inclusion (required):**
- `include_tile`: **Yes/No**
  - We only keep rows where `include_tile == "Yes"` (the pipeline filters out `"No"` early, before any joins/labels)

**Multi-task label schema (what the model trains on):**
- **Binary presence items (Yes/No)**:
  - `sports_field`, `multipurpose_open_area`, `children_s_playground`,
    `water_feature`, `gardens`, `walking_paths`, `built_structures`, `parking_lots`
- **Shade along paths (2-class)**:
  - `shade_along_paths`: **Minimal / Abundant**
    - Encoded as `shade_class ∈ {0=minimal, 1=abundant}`
- **Structured–Unstructured (5-class)**:
  - `structured_unstructured`: **1–5** (stored as `score_class ∈ {1..5}`)
- **Vegetation cover distribution (5-class)**:
  - `vegetation_cover_distribution`: mapped to **1–5** (stored as `veg_class ∈ {1..5}`)

**Processed training tables:**
- `data/processed/labels_soft.csv`: per-image soft probabilities (`*_p`) + `score_mean`, `veg_mean`
- `data/processed/labels_hard.csv`: per-image hard labels + `shade_class`, `score_class`, `veg_class`

## CNN Development
This repository contains the machine learning pipeline to:
1. Process survey response data into training labels
2. Train multi-task CNN models on satellite images
3. Predict green space characteristics at scale
4. Validate model performance against human annotations

## DL Choice: PyTorch + TorchGeo (current) — TensorFlow/Keras (legacy)

The model is now a **multi-task PyTorch model built on TorchGeo**. The current
backbone is **Swin V2 Base with NAIP RGB Satlas pretrained weights**, which
evaluated slightly better than the earlier TensorFlow/Keras EfficientNet model
on our priority labels (see `skills/0615_process_update.md`). The PyTorch
pipeline is the canonical path: it lives in `src_torch/` and is exercised by the
`*_pyTorch_*` / `*_torch_*` notebooks below.

The original **TensorFlow/Keras implementation** (`src/` and the unprefixed
`03_model_training.ipynb` / `04_model_evaluation.ipynb`) is **retained as
legacy reference only**. It is not the path under active development. New work,
cluster runs, and the code under review all use the PyTorch pipeline.

**Active entry points (PyTorch):**
- Train: `notebooks/03_torch_model_training.ipynb` (model builders in `src_torch/models.py`)
- Evaluate + calibrate thresholds: `notebooks/04_pyTorch_model_evaluation_v1.ipynb`
- Predict on new images: `notebooks/05_pyTorch_prediction_demo.ipynb`

**Why a multi-task model:** one backbone feeds four heads matching our label
schema — a multi-label binary head, a 2-class shade head, and two 1–5 ordinal
heads (structure, vegetation):
- Multi-label binary: `sports_field`, `multipurpose_open_area`, `children_s_playground`, `water_feature`, `gardens`, `walking_paths`, `built_structures`, `parking_lots`
- Shade (2-class): `shade_class ∈ {0=minimal, 1=abundant}`
- Structured (1–5): `score_class ∈ {1..5}`
- Vegetation (1–5): `veg_class ∈ {1..5}`

**Training Step:**
1. Pick a backbone (e.g., EfficientNet-B0/B3 or ResNet-50)
    - These are well-tested, fast, and have pretrained weights.
2. Warm-up phase (frozen backbone)
    - Freeze the backbone (don’t change its weights) and train only the heads.
    - This lets the heads quickly learn our label definitions using the backbone’s generic features.
3. Finetuning
    - Unfreeze the top few blocks of the backbone and train at a lower learning rate.
    - This adapts the generic features to the specifics of our own image set.
4. Evaluate and calibrate
    - We check performance on holdout images and set thresholds for the binary labels to meet our priorities.

## Full Setup and Training Workflow

### 0) Prerequisites (recommended)
- Python **3.11+**
- On Apple Silicon Macs, make sure you are using an **arm64** Python (not an Intel/Rosetta one).

Quick checks:
```bash
uname -m                      # should be arm64 on Apple Silicon
python3 -c "import platform; print(platform.machine())"
```

### 1) Create a virtual environment
```bash
# Recommended: be explicit about Python 3.11 if available
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 1b) (Optional) Register the venv as a Jupyter kernel
This makes it easy to select the correct interpreter inside notebooks.
```bash
python -m ipykernel install --user --name GreenSpace_CNN --display-name "GreenSpace_CNN (.venv)"
```

### 2) Install dependencies
- Base packages (shared across platforms):
```bash
pip install -r requirements.txt
```

### 2b) Optional: install TensorFlow for legacy notebooks only

The current review path does not require TensorFlow. Install it only if you plan
to run the legacy `src/`, `03_model_training.ipynb`, or
`04_model_evaluation.ipynb` workflow.

```bash
pip install "tensorflow>=2.16"
```

Note (Apple Silicon):
- If you previously used Intel Homebrew under `/usr/local` (x86_64), install an arm64 Python (Homebrew under `/opt/homebrew`) before creating your venv.

Recommended (avoids cross-machine cache issues): use a project-local Keras cache:
```bash
export KERAS_HOME="$PWD/.keras"
```

### 2c) Quick sanity checks (recommended)
```bash
python scripts/check_python_version.py
```

For legacy TensorFlow work, also run:

```bash
python scripts/diagnose_tf_env.py
```


### 3) Place data
- Raw survey CSV: `data/raw/0103_survey_response.csv` (or your latest dated file)
- Images are downloaded/cached locally under `data/cache/images/` by the preprocessing notebook

### 3a) Use an approved data location on another machine or cluster

Do not edit split CSVs to replace another user's absolute paths. Set these
environment variables before running PyTorch training or evaluation instead:

```bash
# Directory containing processed/splits/{train,val,test}.csv and cache/images/
export GREENSPACE_DATA_ROOT=/approved/greenspace-data

# Optional explicit directory containing the training/validation image files.
# When set, every split-manifest image must be present here by image filename.
export GREENSPACE_IMAGE_ROOT=/approved/greenspace-images

# Optional directory of unseen images for the PyTorch prediction notebook.
export GREENSPACE_INFERENCE_IMAGE_ROOT=/approved/greenspace-inference-images

# Optional destination for PyTorch inference CSVs and diagnostic plots.
export GREENSPACE_PREDICTION_OUTPUT_ROOT=/approved/greenspace-predictions

# Optional loader throughput knobs for a GPU node (defaults: 0 workers, no pin).
# num_workers parallelizes image decode; pin_memory speeds host->GPU copies.
export GREENSPACE_NUM_WORKERS=8
export GREENSPACE_PIN_MEMORY=true
```

If the variables are not set, local development continues to use this
repository's `data/` directory. The cluster image folder does not need to match
anyone's laptop path; it only needs to contain the filenames named by the split
manifest.

**Input versus output:** users must point the code to approved input data and
images; the code never creates or guesses input files. Prediction outputs are
different: if `GREENSPACE_PREDICTION_OUTPUT_ROOT` is omitted, PyTorch inference
creates and uses `predictions/` in the repository. If a cluster checkout is
read-only or temporary, set that variable to a permitted persistent location;
the output directory and any missing parent directories are created
automatically.

### 3b) Reuse a saved PyTorch model

For image-only prediction, download the shared model bundle and place the run
folder under `models/runs/`. The selected checkpoint, its
`model_config_<tag>.json`, and matching `thresholds_<variant>.csv` must stay in
that same folder. This is the portable model unit; no retraining is required to
use it on a new image folder.

Current review bundle:
[PyTorch_20260614_220926 best-MC-MAE bundle](https://drive.google.com/drive/folders/1tlsfN30WkAFBkwEmtTJmGt-uA6KbKXZA?usp=sharing)

### 3c) Google Drive authentication (for image download)
To download/cache the rated images from the team Google Drive folder:
- Put OAuth secrets at `secrets/client_secrets.json`
- Create a project-root `.env` with `GOOGLE_DRIVE_FOLDER_ID="..."` (required)

```bash
python scripts/download_drive_images.py \
  --survey-csv data/raw/0614_survey_response.csv \
  --filelist-csv data/filelist_0103.csv \
  --cache-dir data/cache/images
```

Replace the CSV paths for each run. Start with `--manifest-only` to inspect the
joins without downloading, or `--limit 50` for a small cache test.

Full walkthrough + troubleshooting: see [Google Drive authentication guide](instruction_docs/google_drive_auth.md).

### 4) Preprocess the survey and build train/validation/test splits

Run the established cleaning, inclusion filtering, rater aggregation, and
deterministic 60/20/20 split through one command:

```bash
python scripts/preprocess.py \
  --survey-csv data/raw/0614_survey_response.csv \
  --filelist-csv data/interim/filelist_0418_215415_with_drive_fileid.csv
```

Replace both paths for the current run. `--filelist-csv` is optional, but when
provided it preserves `drive_file_id` in the split manifests. Cached rated
images default to `data/cache/images`; outputs are written to `data/interim/`
and `data/processed/`. The established split seeds remain 123 and 456.

For a deterministic 50-image smoke dataset, add `--sample-size 50`. Add
`--fail-on-missing-images` for a full production run that must not silently
filter uncached labeled images.

The lower-level `scripts/clean_survey.py` remains available when only header and
filename normalization is needed.

### 5) Run preprocessing notebook
Open `notebooks/02_data_preprocessing.ipynb` and run cells:
- Step 1: validate CSV ↔ images (prints counts)
- Step 2: aggregate raters → writes `data/processed/labels_soft.csv` and `labels_hard.csv`
- Step 3: print label prevalence
- Step 4: build oversampled + augmented preview stream (in-memory)
- Step 5: dynamic 60/20/20 split → writes `data/processed/splits/{train,val,test}.csv`

### 6) Train + evaluate (PyTorch — current)

Run the deterministic 1+1 smoke schedule:

```bash
python scripts/train_torch.py --mode smoke --data-root data/core_pipeline_demo
```

Run the current full 5-warm-up plus 15-fine-tuning schedule:

```bash
python scripts/train_torch.py --mode full --data-root data
```

Each completed epoch atomically updates
`models/runs/<run-tag>/last_<run-tag>.pt`. Resume an interrupted run with the
same mode and data:

```bash
python scripts/train_torch.py \
  --mode full \
  --data-root data \
  --resume models/runs/<run-tag>/last_<run-tag>.pt
```

Resume validates that train/validation rows, ordering, filenames, labels, model
settings, batch size, and epoch targets match the saved run. Device, worker
count, and pinned-memory settings may change. Checkpoints are written after
completed epochs, so an interruption inside an epoch repeats that epoch.
Because resumable checkpoints include optimizer and training-control state,
they can be substantially larger than inference-only checkpoints; check free
disk space before a long run.

- Historical/interactive train entry: `notebooks/03_torch_model_training.ipynb`
- Evaluate: `notebooks/04_pyTorch_model_evaluation_v1.ipynb` (uses `data/processed/splits/test.csv`)
- Predict on new images: `notebooks/05_pyTorch_prediction_demo.ipynb`

> Legacy TensorFlow/Keras equivalents (`03_model_training.ipynb`,
> `04_model_evaluation.ipynb`) remain in the repo for reference but are no longer
> the active path.

Notes:
- Raw data and images are ignored by Git; manifests under `data/processed/` can be committed for reproducibility.

## Troubleshooting (common onboarding issue)

### EfficientNet ImageNet weights fail to load (shape mismatch)
Symptom (example):
- `ValueError: Shape mismatch ... stem_conv/kernel ... expects (3, 3, 1, 32) received (3, 3, 3, 32)`

Root causes:
- Installing the wrong TensorFlow build for the machine (Apple Silicon vs x86_64)
- Creating the venv with an **x86_64 Python** on an Apple Silicon Mac (common if using Intel Homebrew under `/usr/local`)
- Stale cached Keras application weights from a different environment/architecture

Fast fix:
```bash
export KERAS_HOME="$PWD/.keras"
python scripts/clear_keras_cache.py
python scripts/diagnose_tf_env.py
```

If `diagnose_tf_env.py` still fails, re-check that you installed TensorFlow using the correct command in **Setup 2b**.

### Alternative: Download prepackaged archive (shared drive)
If you prefer, download a zipped project snapshot from the team drive and set up locally:

1. Download from the team folder: [Shared Google Drive](https://drive.google.com/drive/folders/1u_32rK3jT_CVv2ycraPY31Gib5RwUjdb)
2. Unzip to your workspace directory
3. Create and activate a virtual environment, then install deps:
   - macOS (Apple Silicon):
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     python -m pip install -U pip
     pip install -r requirements.txt
     pip install "tensorflow>=2.16"
     ```
   - Linux/Windows:
     ```bash
     python -m venv .venv
     .venv\\Scripts\\activate  # Windows
     # or: source .venv/bin/activate  # Linux
     python -m pip install -U pip
     pip install -r requirements.txt
     pip install "tensorflow>=2.16"
     ```
4. (Optional) Register the Jupyter kernel as above
5. Place raw CSV/images under `data/raw/`, run `scripts/clean_survey.py`, then the 02 notebook
