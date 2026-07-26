# Process Update — 2026-07-26

## Objective

Finish the July Windows preprocessing/training/evaluation handoff, package the
evaluation interface, correct the 50-image showcase, clarify configuration
ownership, update documentation, and connect the extracted working folder to
GitHub without committing large or sensitive artifacts.

## Approved dataset

- Raw survey: 8,651 rows and 5,711 unique image names.
- Included ratings: 8,297.
- Aggregated/cached labeled images: 5,518; zero missing.
- Split seeds: 123 and 456.
- Splits: 3,311 train, 1,104 validation, 1,103 test.
- The 34 exact duplicate raw-row copies are intentionally retained.
- Raw survey remains unchanged.
- Provenance is saved locally at
  `data/processed/preprocessing_summary_0719_windows_full.json`, including
  input hashes, paths, parameters, counts, seeds, outputs, and duplicate policy.
- Regenerating preprocessing left all three split SHA-256 hashes unchanged.

## Full PyTorch run

Run: `PyTorch_20260719_full_windows`

- TorchGeo Swin V2 B with Satlas NAIP RGB weights.
- Python 3.11.9, PyTorch 2.10.0+cu128, TorchGeo 0.8.1.
- Windows CUDA on RTX 2070 Super Max-Q, batch size 4.
- Completed 5 warm-up plus 24 fine-tuning epochs in about 6.56 hours.
- Early stopping ended the run before the configured 100-epoch fine-tuning
  ceiling.

Best-MCMAE test results:

| Metric | Value |
|---|---:|
| Macro PR-AUC | 0.8770 |
| Macro ROC-AUC | 0.9354 |
| Tuned macro F1 | 0.8057 |
| Score MAE vs. rater mean | 0.5584 |
| Vegetation MAE vs. rater mean | 0.4172 |

Validation-tuned thresholds and train/validation/test reports are saved under
the run, monitoring, and report-output directories.

## Packaging completed

- Hardened `scripts/evaluate_torch.py`.
- Evaluation now requires an explicit checkpoint or run directory, accepts
  explicit split/image/output roots and loader controls, validates checkpoint
  label compatibility, tunes thresholds on validation only, and saves the
  portable threshold artifact beside the checkpoint.
- Preserved the earlier `tune_val_thresholds` API for compatibility.
- Fixed `notebooks/CORE_pipeline_v1.ipynb` so demo evaluation reads
  `DEMO_SPLIT_DIR` and `IMAGE_CACHE_DIR`; it no longer falls back to production
  splits.
- Added persistent preprocessing-summary JSON output.
- Established `src_torch/config.py` as the active PyTorch training source of
  truth and separated shared task settings from legacy TensorFlow controls.
- Updated `README.md`, `STRUCTURE.md`, tests, and smoke-data ignore rules.

Current packaged commands:

```text
scripts/download_drive_images.py
scripts/preprocess.py
scripts/train_torch.py
scripts/evaluate_torch.py
```

Standalone prediction and broad validation CLIs remain deferred.
