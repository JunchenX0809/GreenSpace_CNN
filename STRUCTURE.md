# Repository structure

This document describes the July 2026 PyTorch-first repository. Generated
data, checkpoints, reports, credentials, and local environments are ignored by
Git even when they exist in a working copy.

```text
GreenSpace_CNN/
├── README.md
├── STRUCTURE.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/                       # immutable survey inputs (local)
│   ├── cache/images/              # downloaded rated images (local)
│   ├── interim/                   # cleaned run-tagged surveys (generated)
│   ├── processed/                 # labels, provenance summary, active splits
│   ├── smoke_50/                  # small test dataset (local)
│   └── core_pipeline_demo/        # notebook demonstration outputs (local)
├── instruction_docs/
│   └── google_drive_auth.md       # OAuth and Drive setup
├── notebooks/
│   ├── CORE_pipeline_v1.ipynb     # canonical 50-image handoff showcase
│   ├── 03_pyTorch_training_v1.ipynb
│   ├── 04_pyTorch_model_evaluation_v1.ipynb
│   ├── 05_pyTorch_prediction_demo.ipynb
│   └── ...                        # historical/analysis notebooks
├── scripts/
│   ├── download_drive_images.py   # explicit Google Drive download
│   ├── preprocess.py              # survey → labels/splits/summary
│   ├── train_torch.py             # new/resumed smoke or full training
│   ├── plot_training_curves.py    # saved history → PR-AUC/MAE visual
│   ├── evaluate_torch.py          # reports + validation thresholds
│   ├── predict_torch.py           # deterministic image-folder prediction
│   ├── validate_pipeline.py       # read-only workflow readiness checks
│   ├── check_python_version.py
│   ├── check_offline_checkpoint_load.py
│   └── ...                        # historical diagnostics/report utilities
├── src/
│   ├── preprocessing.py           # reusable survey/split orchestration
│   ├── drive_download.py          # reusable Drive download orchestration
│   ├── drive_utils.py
│   ├── label_schema.py            # shared task + legacy TF compatibility
│   ├── augmentation.py
│   └── ordinal_targets.py
├── src_torch/
│   ├── config.py                  # active PyTorch source of truth
│   ├── data.py                    # manifests, datasets, loaders
│   ├── transforms.py
│   ├── sampling.py
│   ├── models.py
│   ├── losses.py
│   ├── training.py                # resumable warm-up/fine-tune loop
│   ├── evaluation.py              # metrics and threshold calibration
│   ├── inference.py               # reusable inference functions
│   ├── run_bundle.py              # portable checkpoint bundle loading
│   └── artifacts.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_drive_download.py
│   ├── test_training_resume.py
│   ├── test_training_artifacts.py
│   ├── test_evaluation_cli.py
│   ├── test_prediction_cli.py
│   └── test_config_contract.py
├── models/runs/                   # local checkpoints/config/history
├── monitoring_output/runs/        # generated loss/threshold tables
├── report_outputs/runs/           # generated evaluation reports
├── secrets/                       # local OAuth files; never commit
└── skills/                        # historical progress notes and side plans
```

## Active artifact contracts

### Prepared dataset

```text
data/interim/survey_response_clean_<run-tag>.csv
data/processed/labels_soft_<run-tag>.csv
data/processed/labels_hard_<run-tag>.csv
data/processed/preprocessing_summary_<run-tag>.json
data/processed/splits/{train,val,test}.csv
```

The JSON summary is the dataset provenance record. It includes the survey and
optional filelist hashes, explicit paths, seeds, row/image counts, split counts,
and duplicate-row policy.

### PyTorch run

```text
models/runs/<run-tag>/
├── last_<run-tag>.pt              # resumable optimizer/training state
├── best_mcmae_<run-tag>.pt
├── best_prauc_<run-tag>.pt
├── final_<run-tag>.pt
├── model_config_<run-tag>.json
├── training_history_<run-tag>.json
├── training_curves.png
├── training_metric_curves.png     # presentation-style PR-AUC/MAE visual
└── thresholds_<variant>.csv       # added by evaluation
```

### Evaluation

```text
monitoring_output/runs/<run-tag>/
├── loss_monitor_<variant>.csv
└── thresholds_<variant>.csv       # historical monitoring copy

report_outputs/runs/<run-tag>/
├── overall_metrics_by_split_<variant>.csv
└── per_label_metrics_by_split_<variant>.csv
```

### Prediction

```text
predictions/
└── predictions_<run-tag>_<dataset-tag>[_sampleN].csv
```

Prediction output never enters the run bundle and remains ignored by Git.
Each CSV contains one unique `image_filename`, two columns per active binary
label (`*_prob`, `*_pred`), shade class/confidence, `score_ev`, and `veg_ev`.
The command validates schema, bounds, finite values, row count, and filename
uniqueness before publishing the CSV atomically.

## Configuration ownership

- `src_torch/config.py` owns the active PyTorch model, loader, loss, training,
  and stopping defaults.
- `src/label_schema.py::MODEL_TASK_CONFIG` owns shared task choices such as
  label exclusion, head modes, and oversampling targets.
- `src/label_schema.py::LEGACY_TF_TRAINING_CONFIG` exists only for historical
  TensorFlow notebook compatibility and is not read by active PyTorch code.
- Per-run `model_config_<run-tag>.json` records the effective configuration and
  split fingerprints actually used.

## Packaged interfaces

Drive download, preprocessing, training, saved-history visualization,
evaluation, image-folder prediction, and read-only pipeline validation are
packaged terminal interfaces. The canonical notebook demonstrates the same
reusable preprocessing, training, and evaluation functions on 50 images.
