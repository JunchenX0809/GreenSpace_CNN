# GreenSpace_CNN

## Overview
Train a CNN to reproduce greenspace ratings at scale using human judgments from photos/satellite images.

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

## DL Choice: Tensorflow - Keras Integration
**Why Tensorflow**
1. Keras API
High-level Tensorflow API. It takes care of the training loop, metrics, logging, and saving models.
    - Concise and User-friendly
2. Multi-task models
Keras makes it straightforward to build one CNN with multiple outputs. It suits our usecase because:
    - we have multilabel binary, ordinal, and 1-5 features. Keras can have a binary head for features, a categorical head for shade, and a 5-class head for structure. 

**Our Label:**
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

## Setup and Quick Start

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

### 2b) Install TensorFlow (platform-specific)
- macOS (Apple Silicon / Intel), Linux, Windows:
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
python scripts/diagnose_tf_env.py
```


### 3) Place data
- Raw survey CSV: `data/raw/0103_survey_response.csv` (or your latest dated file)
- Images are downloaded/cached locally under `data/cache/images/` by the preprocessing notebook

### 4) Clean the survey CSV (reproducible script)
```bash
python scripts/clean_survey.py \
  --in-csv data/raw/0103_survey_response.csv \
  --out-csv data/raw/0103_survey_response_clean.csv \
  --image-col "Image Name"
```

### 5) Run preprocessing notebook
Open `notebooks/02_data_preprocessing.ipynb` and run cells:
- Step 1: validate CSV ↔ images (prints counts)
- Step 2: aggregate raters → writes `data/processed/labels_soft.csv` and `labels_hard.csv`
- Step 3: print label prevalence
- Step 4: build oversampled + augmented preview stream (in-memory)
- Step 5: dynamic 60/20/20 split → writes `data/processed/splits/{train,val,test}.csv`

### 6) Train + evaluate
- Train: `notebooks/03_model_training.ipynb`
- Evaluate: `notebooks/04_model_evaluation.ipynb` (uses `data/processed/splits/test.csv`)

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
