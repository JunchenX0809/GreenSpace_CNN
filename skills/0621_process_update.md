# Process Update — 2026-06-21

## Inference Update

_Scaffold for the accompanying inference update. Add model choice, input image
population, output location, and result summary here._

## Code Review Preparation: Portable PyTorch Orchestration

### Goal

Prepare the PyTorch pipeline for external review and eventual use at RPI and
Mount Sinai. A new user should not need this developer's laptop paths, local
cache, or device type to run an approved model.

### Review Method

The review followed the full execution path:

```text
split manifests -> image loading -> transforms -> model -> checkpoints
-> evaluation/thresholds -> inference output
```

The first focus is portability without changing scientific model behavior or
training results.

### Changes Made

- **Portable data locations:**
  - `GREENSPACE_DATA_ROOT` selects a directory containing the split manifests.
  - `GREENSPACE_IMAGE_ROOT` selects the folder containing training/evaluation
    images by filename.
  - `GREENSPACE_INFERENCE_IMAGE_ROOT` selects an unseen-image folder for
    PyTorch inference.
  - An explicit image folder must contain all requested files; the code stops
    clearly instead of mixing a cluster folder with a laptop cache.
- **Portable model loading:** saved checkpoints rebuild without deliberately
  re-downloading pretrained weights, and inference uses the label order saved
  with the checkpoint.
- **Portable defaults:** PyTorch now selects the available device automatically
  (`CUDA`, Apple Silicon, or CPU); evaluation no longer contains one absolute
  laptop checkpoint path.
- **Documentation:** the root README describes the three environment variables.

### Simple Cluster Setup

```bash
export GREENSPACE_DATA_ROOT=/approved/greenspace-data
export GREENSPACE_IMAGE_ROOT=/approved/greenspace-images
export GREENSPACE_INFERENCE_IMAGE_ROOT=/approved/inference-images
```

The image folders can be different from the original laptop folders. They only
need to contain files with the filenames recorded in the selected manifest.

### Verification Completed

- Existing local split manifests still find all expected training images.
- A simulated alternate image folder successfully resolved images despite a
  different legacy laptop path.
- Python source compiles and the edited PyTorch notebooks remain valid.

### Remaining Verification

Run one real PyTorch/TorchGeo offline checkpoint-load test in an environment
with those packages installed. It must load a saved run and produce the same
small-sample predictions without attempting a network download.

## Portability Smoke Test

### Rationale

The prior pipeline expected one developer's image paths. This test checks that
inference can read images from a separate folder and write results outside the
normal project output location.

### Change Tested

The PyTorch input/output location settings were used:

```bash
export GREENSPACE_INFERENCE_IMAGE_ROOT="$PWD/clusters/inference"
export GREENSPACE_PREDICTION_OUTPUT_ROOT="$PWD/clusters/outputs"
```

The reusable checkpoint loader and image-only inference helpers were then run
with the project virtual environment (`.venv/bin/python`) and the saved
`PyTorch_20260614_220926` best-MC-MAE checkpoint.

### Result

- The simulated cluster folder supplied 10 RGB `512 x 512` JPEG inputs.
- Inference ran successfully on CPU and produced 10 rows with the expected 19
  prediction columns.
- All numeric outputs were finite; score and vegetation predictions remained
  within the expected 1–5 range.
- The output was created at:

```text
clusters/outputs/predictions_portability_smoke.csv
```

This demonstrates that inference input and output locations no longer depend
on the original laptop cache/output paths. The selected model checkpoint and
threshold file still reside in the repository for this first simulation;
packaging them together as a movable model run bundle remains the next step.

