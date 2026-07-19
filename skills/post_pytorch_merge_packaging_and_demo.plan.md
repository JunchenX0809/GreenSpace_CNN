# Tentative Post-Merge Packaging and Demonstration Plan

## Status

**Direction confirmed for the initial implementation shape; detailed interfaces
still require review before coding.**

This plan begins after `PyTorch_test` has been merged into `main`. It does not
authorize the merge, changes to model behavior, a new full training run, or the
removal of the legacy TensorFlow workflow.

## Confirmed User Decisions - 2026-07-18

1. The new survey target is approximately **10,000 raw survey-response rows**,
   not 10,000 unique images. The number of unique included images will be lower
   after `include_tile` filtering and per-image aggregation.
2. The next dataset will be split again from the complete newly processed
   image set. The previous test set does not need to remain fixed.
3. Split reproducibility should follow the existing Notebook 02 behavior:
   - master/preprocessing seed: `123`;
   - first train/remainder split seed: `123`;
   - second validation/test split seed: `456`;
   - current split ratios: 60% train, 20% validation, 20% test;
   - no rare-label stratification unless separately approved.
4. The current June 2026 checkpoint is not the long-term default. A new model
   will be trained after the 10,000-row survey is processed. That new approved
   model will become the documented default only after evaluation and threshold
   calibration.
5. Google Drive download will remain a separate command from preprocessing.
6. Use several clearly named scripts rather than hiding all workflow behavior
   inside notebooks. These scripts should be thin adapters over the already
   tested functions/orchestration.
7. TensorFlow is historical reference only. The clean workflow and new
   demonstration notebook should be PyTorch-first; legacy TensorFlow notebooks
   should be clearly labeled and should not drive the new orchestration.
8. The demonstration notebook must cover:
   - Google OAuth setup/authentication;
   - Google Drive image download;
   - raw survey cleaning, preparation, aggregation, and preprocessing;
   - a new 60/20/20 train/validation/test split;
   - PyTorch, selected TorchGeo backbone, preprocessing, and GreenSpace heads;
   - one frozen-backbone warm-up epoch;
   - one unfrozen fine-tuning epoch.
9. Existing behavior is the implementation reference. Do not redesign labels,
   aggregation, split logic, architecture, losses, augmentation, oversampling,
   shade masking, or artifact conventions without documenting and approving a
   deliberate change.
10. In the demonstration notebook, "testing" means the existing PyTorch
    showcase schedule of one frozen-backbone warm-up epoch followed by one
    unfrozen fine-tuning epoch.
11. The demonstration training subset will contain **50 unique prepared
    images**. The notebook must label this as a wiring/usability demonstration,
    not model-performance evidence.
12. The expected new survey schema and rated-image Drive layout are unchanged
    from the current Notebook 02 workflow. The user will obtain/place the new
    survey and images using that existing arrangement.

Current clarification: the repository has a Google Drive implementation using
PyDrive2 and OAuth credentials configured through Google Cloud Console. It does
not currently implement Google Cloud Storage bucket download. The initial plan
therefore covers Google Drive only unless Cloud Storage is explicitly added to
scope.

No new cloud storage path is planned. References to "Google Cloud" in the
demonstration describe configuring Google OAuth credentials for the existing
Google Drive API flow.

## Implementation Progress - First Preprocessing Increment

Completed on 2026-07-18:

- added `src/preprocessing.py` with reusable cleaning, inclusion filtering,
  rater aggregation, label writing, split construction, and split writing;
- changed `scripts/clean_survey.py` to reuse the extracted cleaning functions;
- added synthetic regression tests under `tests/test_preprocessing.py`;
- verified the extracted cleaner against all 7,991 June survey rows;
- verified the extracted soft/hard aggregation against all 5,242 saved June
  image-level label rows;
- verified exact ordered filename parity for the existing 3,145/1,048/1,049
  train/validation/test manifests using split seeds `123` and `456`.

Notebook 02 has intentionally not been switched to the extracted functions.
It remains a historical/proven orchestration reference. The extracted
functions will instead be consumed by the new handoff notebook,
`notebooks/CORE_pipeline_v1.ipynb`, and by the separate command-line scripts.

Initial notebook scaffold completed on 2026-07-18:

- created the versioned `notebooks/CORE_pipeline_v1.ipynb` file;
- wired setup, OAuth authentication, extracted preprocessing, deterministic
  50-image selection, 60/20/20 splitting, current model inspection, existing
  1+1 test training, and existing evaluation calls;
- left the rated-image download section explicitly pending until its current
  Notebook 02 orchestration is packaged as the approved separate function and
  script;
- kept all expensive/authenticated actions behind explicit `RUN_*` toggles;
- validated notebook JSON and parsed every code cell.

## Purpose

Make the current PyTorch/TorchGeo pipeline usable by another person who clones
the repository on macOS, Windows, or Linux without needing to reconstruct the
workflow from notebook cells or edit developer-specific paths.

The target experience is:

```text
clone repository
  -> create environment
  -> configure approved data/model locations
  -> validate inputs
  -> run a small smoke workflow
  -> run preprocessing, training, evaluation, or inference through scripts
  -> use one clean notebook to understand the same end-to-end workflow
```

## Current Baseline

The following pieces already exist and should be reused rather than rewritten:

- `scripts/clean_survey.py` for raw survey column/filename normalization.
- `scripts/cache_drive_folder.py` for bulk inference-image caching.
- `src_torch/` modules for data loading, transforms, sampling, model creation,
  training, evaluation, inference, artifacts, and run-bundle loading.
- `notebooks/03_pyTorch_training_v1.ipynb` as the current training record.
- `notebooks/04_pyTorch_model_evaluation_v1.ipynb` as the current evaluation
  and threshold-calibration record.
- `notebooks/05_pyTorch_prediction_demo.ipynb` as the current inference record.
- `scripts/check_offline_checkpoint_load.py` as the current portable bundle
  smoke test.

Known gaps at the start of this plan:

- label aggregation, cache orchestration, and split creation still live mainly
  in Notebook 02;
- training has a reusable Python orchestrator but no stable command-line entry;
- evaluation and inference have reusable helpers but no end-user scripts;
- there is no single clean end-to-end demonstration notebook;
- documentation and some notebook paths remain macOS-oriented;
- long training runs cannot be resumed robustly after interruption;
- the current production training loop still performs expensive smoke-style
  diagnostics on every batch;
- there is no automated Windows/macOS/Linux verification;
- `STRUCTURE.md` and some unversioned processed-data names are stale.

## Phase 0 - Merge and Baseline Gate

Goal: establish `main` as the canonical PyTorch baseline before new packaging
work begins.

Planned checks after the user completes the merge:

1. Confirm the active branch is `main` and it contains the PyTorch commits.
2. Confirm the working tree is clean and `main` matches the intended remote.
3. Run the existing checks:
   - Python version check;
   - offline run-bundle load and prediction check;
   - Python source parse/import check;
   - dependency consistency check.
4. Record the post-merge commit as the baseline for later comparisons.
5. Confirm that ignored local data, credentials, models, and predictions were
   not accidentally added during the merge.

Acceptance gate:

```text
main contains the current PyTorch workflow;
existing smoke checks pass;
no protected or large local artifacts are tracked;
no model behavior has changed during the branch transition.
```

## Phase 1 - Define Stable Pipeline Contracts

Goal: agree on inputs and outputs before adding command-line wrappers.

### 1.1 Preprocessing contract

Define explicit inputs for:

- raw survey CSV;
- survey image-name column;
- Google Drive file manifest or Drive folder;
- rated-image cache directory;
- output directory/run tag;
- split seed and split policy.

Define explicit outputs for:

- cleaned survey CSV;
- survey-to-Drive manifest;
- soft and hard label tables;
- train/validation/test manifests;
- preprocessing summary and validation report.

Avoid selecting the "newest file" implicitly. Every production run should
record exactly which input files and split policy it used.

### 1.2 Training contract

Define explicit inputs for:

- split directory and image root;
- model/backbone configuration;
- smoke versus full-run mode;
- device, batch size, worker count, and epoch limits;
- run/output directory;
- optional resume checkpoint.

Define the minimum durable outputs:

- last resumable checkpoint;
- best MC-MAE checkpoint;
- best PR-AUC checkpoint;
- inference-only model artifact;
- effective configuration;
- training history and timing;
- data/split fingerprints and Git revision.

### 1.3 Evaluation and inference contracts

Evaluation must bind a checkpoint to its saved label order, preprocessing
contract, and matching thresholds. Inference must accept an image directory
and write a uniquely named output without requiring training labels unless the
optional comparison diagnostics are requested.

## Phase 2 - Extract the Remaining Notebook-Only Logic

Goal: notebooks demonstrate reusable functions rather than serving as the only
implementation.

Planned extraction from Notebook 02:

1. Clean and validate a survey table.
2. Filter `include_tile == yes`.
3. Attach Drive identifiers.
4. Aggregate rater rows into soft/hard per-image labels.
5. Validate/cache rated images.
6. Build reproducible train/validation/test manifests.
7. Write a machine-readable preprocessing summary.

The exact module layout will be selected after reviewing the preferred public
API. A likely home is a small `src/preprocessing.py` module plus existing Drive
utilities; this is not final.

Important split decision:

- use the existing two-stage random 60/20/20 split on the complete newly
  processed dataset;
- preserve Notebook 02's `random_state=123` for the train/remainder split and
  `random_state=456` for the validation/test split;
- do not preserve the prior test set;
- save the resulting split manifests and effective seeds with the new run so
  the new split is reproducible for that exact input table.

## Phase 3 - Add Runnable Scripts

Goal: expose the reusable pipeline through OS-neutral commands.

Tentative commands/files:

```text
scripts/download_drive_images.py
scripts/preprocess.py
scripts/train_torch.py
scripts/evaluate_torch.py
scripts/predict_torch.py
scripts/validate_pipeline.py
```

Names remain open, but the confirmed direction is separate, clearly labeled
scripts. The scripts should import and call existing source functions rather
than invoke notebooks or duplicate notebook logic.

Tentative invocation shape:

```text
python scripts/download_drive_images.py ...
python scripts/preprocess.py ...
python scripts/train_torch.py ...
python scripts/evaluate_torch.py ...
python scripts/predict_torch.py ...
python scripts/validate_pipeline.py ...
```

`python -m greenspace predict ...` would instead be one package-level command
with subcommands. That is not the selected initial direction. A later packaging
layer could add it as a convenience without changing the underlying scripts.

Minimum command behavior:

### Preprocess

- accept explicit survey, Drive/cache, data-root, output, and split arguments;
- consume an already prepared local image cache; Drive authentication/download
  remains a separate script;
- support cached/offline operation when images are already available;
- fail clearly on missing columns, missing images, duplicate identifiers, or
  overlapping splits;
- publish related outputs together so partial runs are not mistaken for a
  completed dataset.
- default to the existing two-stage 60/20/20 split with seeds `123` and `456`.

### Train

- support `--smoke` and explicit full-run settings;
- print and save the complete effective configuration before training starts;
- estimate batches and provide a rough runtime projection after a short timing
  sample;
- save resumable state every epoch;
- handle interruption without discarding the latest completed epoch.

### Evaluate

- accept an explicit checkpoint/run bundle and split root;
- predict splits, tune validation thresholds, and save reports;
- never silently select a different model or threshold artifact.

### Predict

- accept checkpoint/run bundle, image directory, output path, device, batch
  size, workers, and optional limit;
- use `torch.inference_mode()` and deterministic image ordering;
- validate output schema, bounds, row count, and duplicate filenames;
- support an output/dataset tag so separate image folders do not overwrite one
  another;
- consider incremental/chunked output or a resume manifest for large jobs.

### Validate

- verify environment, data manifests, image availability, run bundle,
  thresholds, output permissions, and selected device before expensive work.

Note: the historical coding-conventions note discourages CLI argument parsing,
but that convention conflicts with the newer explicit goal of portable scripts.
The accessibility goal takes precedence; the conventions document should be
updated once the interface direction is approved.

## Phase 4 - Harden the 10,000-Row Survey Training Path

Goal: make a long run measurable, interruptible, and reasonably efficient
before starting it.

Planned work, subject to measurement:

1. Treat 10,000 as raw survey-response rows; report the actual post-filter row
   count and unique included-image count before estimating training duration.
2. Add epoch and data-loader timing to the saved history.
3. Separate smoke diagnostics from production metrics:
   - do not calculate full-model gradient finiteness and norm on every batch;
   - avoid repeated accelerator synchronization from per-batch Python scalar
     conversion;
   - retain periodic/debug diagnostics behind an explicit option.
4. Benchmark batch size and loader workers by device:
   - safe Windows default: zero workers until spawn behavior is verified;
   - macOS/MPS profile;
   - CUDA profile when available.
5. Evaluate mixed precision only after output/loss parity checks.
6. Avoid redundant resize work when cached inputs already match the required
   size, if the selected transform contract permits it.
7. Add a real resume contract containing model, optimizer, phase/epoch,
   scheduler/control state, best metrics, history, random state, and sampler
   state where needed.
8. Save a lightweight inference artifact without optimizer state.
9. Run a capped timing trial before approving the full schedule.

No performance optimization should be accepted solely because it is faster;
the smoke run must verify finite losses, matching schema, bounded score/veg
outputs, and equivalent inference behavior.

## Phase 5 - Build the Clean End-to-End Demonstration Notebook

Goal: provide one readable handoff entry point that illustrates the real
packaged pipeline without duplicating its implementation.

Confirmed notebook path:

```text
notebooks/CORE_pipeline_v1.ipynb
```

Proposed sections:

1. Project purpose and pipeline diagram.
2. Environment and path configuration.
3. Google Cloud Console OAuth credential prerequisites and Google Drive
   authentication using the existing PyDrive2 flow.
4. Google Drive image download using the separate download interface.
5. Raw survey cleaning and `include_tile == yes` filtering.
6. Per-image soft/hard label aggregation and input validation.
7. Existing two-stage 60/20/20 split with the recorded `123`/`456` seeds.
8. PyTorch/TorchGeo imports and selected backbone/preprocessing explanation.
9. GreenSpace multi-task heads, losses, oversampling, augmentation, and shade
   masking as implemented by the current source modules.
10. One warm-up epoch with the backbone frozen.
11. One fine-tuning epoch with the backbone unfrozen.
12. Saved smoke artifacts and a concise explanation of how full training,
    evaluation, threshold calibration, and later inference are launched.

Notebook rules:

- import and call the same reusable Python functions used by the production
  scripts;
- show the equivalent separate script command in nearby Markdown so a teammate
  can choose notebook or terminal execution;
- do not make notebook correctness depend on shell magics or subprocess calls
  when the underlying Python function is available;
- default to small, safe smoke inputs;
- select a deterministic 50-image demonstration subset from the newly prepared
  image-level dataset, using an explicit recorded seed;
- never begin the multi-hour full training run merely by running all cells;
- clearly mark authenticated/download steps and the distinction between the
  1+1 demonstration run and the later full run;
- use project-relative or configured paths only;
- work from both repository root and notebook directory;
- do not require the current developer's cached outputs to explain the flow;
- keep legacy TensorFlow material out of the primary narrative while linking it
  as historical reference where useful.
- reuse `TORCH_TRAINING_CONFIG` test mode, which already maps to one warm-up
  epoch plus one fine-tuning epoch, rather than implementing a second demo
  training loop.
- state clearly that metrics from 50 images and two total training epochs are
  functional smoke evidence only and must not be compared with full model runs.
- finish with the trained demo model's evaluation and threshold-calibration
  outputs; optionally point to the separate prediction command as the next
  production action.

## Phase 6 - Cross-Platform Readiness

Goal: make the documented workflow credible on Windows, macOS, and Linux.

Planned work:

1. Remove `/private/tmp` and other host-specific runtime paths from active
   notebooks/scripts.
2. Document activation and environment-variable examples for:
   - PowerShell on Windows;
   - zsh/bash on macOS/Linux.
3. Keep all runtime paths based on `pathlib` and explicit roots.
4. Verify Google OAuth behavior from a fresh Windows environment.
5. Verify default Windows data loading with `num_workers=0`; add multi-worker
   support only after spawn-safe testing.
6. Run a CPU inference smoke test on each operating-system family.
7. Add automated tests for source imports, preprocessing schemas, split
   validation, run-bundle validation, and small inference output.
8. Add CI jobs where practical; document hardware-specific training as a
   manual validation rather than claiming CI coverage.

## Phase 7 - Documentation and Release Cleanup

Goal: make `main` self-explanatory after the interfaces stabilize.

Planned updates:

- revise `README.md` around one canonical PyTorch quickstart;
- revise `STRUCTURE.md` to match the actual repository;
- clearly label TensorFlow notebooks/source as legacy reference;
- document data, credential, model-bundle, and output boundaries;
- resolve the stale unversioned `labels_soft.csv` / `labels_hard.csv` naming
  convention;
- document the production model bundle and checksum/version policy;
- document expected runtimes as measured ranges, not guarantees;
- add troubleshooting for Windows paths, OAuth, CPU-only execution, Apple
  Silicon, CUDA, and interrupted runs.

## Proposed Implementation Order

```text
0. post-merge baseline verification
1. explicit pipeline contracts and decisions
2. preprocessing extraction
3. separate Google Drive download script cleanup
4. prediction script (smallest complete user-facing path)
5. evaluation script
6. training script plus resume/performance hardening
7. preprocessing script and end-to-end validation command
8. clean demonstration notebook
9. Windows/macOS/Linux verification
10. documentation and release cleanup
```

Prediction is proposed first because the portable run-bundle and inference
helpers already exist, making it the shortest path to a complete external-user
workflow. Preprocessing extraction can proceed before or alongside it once the
input/output contract is approved.

## Acceptance Criteria for the Overall Effort

A new collaborator should be able to:

1. Clone `main` and create a supported Python environment.
2. Run a no-data/offline bundle smoke check.
3. Configure approved data and model locations without editing source.
4. Validate a prepared dataset before allocating a long training job.
5. Run inference on a small folder through one documented command.
6. Reproduce evaluation for a selected checkpoint and its thresholds.
7. Launch a smoke training run and resume a deliberately interrupted run.
8. Understand the same workflow through the demonstration notebook.
9. Follow equivalent setup instructions on Windows, macOS, or Linux.

## Decisions Needed From the User

These questions should be resolved before substantial implementation:

1. What filename should the demonstration notebook use for the new raw survey,
   or should it require the user to set `RAW_SURVEY_PATH` explicitly?
2. Should the deterministic 50-image demonstration subset use the existing
   PyTorch training seed `37`, the preprocessing seed `123`, or a separately
   named demonstration seed? Recommendation: reuse `37` and record it.
3. Where should external collaborators obtain approved data and the eventual
   new model bundle?

## Explicit Non-Goals Until Further Direction

- changing the Swin V2/NAIP architecture;
- changing labels, losses, thresholds, or reported model metrics;
- beginning the full training run derived from the 10,000-row survey;
- deleting TensorFlow notebooks or historical reports;
- committing data, credentials, model checkpoints, or prediction outputs;
- deploying to a cloud/cluster environment;
- merging branches on the user's behalf.
