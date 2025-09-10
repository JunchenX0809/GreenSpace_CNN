# GreenSpace_CNN

## Overview
Train a CNN to reproduce greenspace ratings at scale using human judgments from photos/satellite images.

## Survey Design
Human raters evaluate images across standardized criteria:

**Core Rating:**
- `structured_unstructured_rating`: 1-5 scale (1=very structured, 5=very unstructured)

**Binary Presence Items:** (Yes/No/Can't answer)
- Sports fields, multipurpose areas, playgrounds, water features, gardens， walking paths, built structures

**Ordinal Scale:**
- `shade_along_paths`: None / Some (<50%) / Abundant (>50%)

**Data Structure:** One row per (rater × image) with normalized snake_case fields, ISO8601 timestamps, and anonymized rater codes.

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
- Multi-label binary: sports_field, multipurpose_open_area, childrens_playground, water_feature, gardens, walking_paths (Round 5)
- Ordinal: shade_along_paths ∈ {None, Some (<50%), Abundant (>50%)}.
- Global 1–5 score: structured_unstructured_rating

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

### 1) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies
- Base packages (shared across platforms):
```bash
pip install -r requirements.txt
```
- Apple Silicon (macOS, Metal acceleration):
```bash
pip install "tensorflow-macos>=2.16" tensorflow-metal
```
- Linux/Windows: install `tensorflow>=2.16` instead of the Apple Silicon packages.


### 3) Place data
- Raw CSV: `data/raw/survey_responses.csv`
- Images: `data/raw/images/` (filenames must match the CSV after cleaning)

### 4) Clean the survey CSV (reproducible script)
```bash
python scripts/clean_survey.py \
  --in-csv data/raw/survey_responses.csv \
  --out-csv data/raw/survey_responses_clean.csv
```

### 5) Run preprocessing notebook
Open `notebooks/02_data_preprocessing.ipynb` and run cells:
- Step 1: validate CSV ↔ images (prints counts)
- Step 2: aggregate raters → writes `data/processed/labels_soft.csv` and `labels_hard.csv`
- Step 3: print label prevalence
- Step 4: build oversampled + augmented preview stream (in-memory)
- Step 5: dynamic 60/20/20 split → writes `data/processed/splits/{train,val,test}.csv`

Notes:
- Raw data and images are ignored by Git; manifests under `data/processed/` can be committed for reproducibility.

### Alternative: Download prepackaged archive (shared drive)
If you prefer, download a zipped project snapshot from the team drive and set up locally:

1. Download from the team folder: [Shared Google Drive](https://drive.google.com/drive/folders/1u_32rK3jT_CVv2ycraPY31Gib5RwUjdb)
2. Unzip to your workspace directory
3. Create and activate a virtual environment, then install deps:
   - macOS (Apple Silicon):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     python -m pip install -U pip
     pip install -r requirements.txt
     pip install "tensorflow-macos>=2.16" tensorflow-metal
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
