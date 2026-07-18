# Process Update — Core Pipeline Packaging and Handoff

Date: 2026-07-18

## Objective

Make the current PyTorch pipeline reproducible and approachable for teammates who clone the repository on macOS or Windows. The implementation must remain grounded in the existing, tested notebook orchestration rather than redesigning the modeling workflow.

The PyTorch work has been merged into `main`; the merge baseline for this packaging work is commit `1f5c6b3`.

## Confirmed Decisions

- The upcoming full run begins with approximately 10,000 raw survey rows. The number of unique included images will be lower after cleaning, filtering, and aggregation.
- A new deterministic train/validation/test split will be created from the new dataset. The previous test set does not need to be preserved.
- Preprocessing keeps the existing seeds: master seed `123`, first split seed `123`, and validation/test split seed `456`. PyTorch training keeps seed `37`.
- The current June model remains a reference artifact. A new model trained and approved on the larger dataset will become the default later; merely training it does not automatically promote it.
- Google Drive download remains a separate command from preprocessing.
- The supported pipeline will be exposed through clearly named scripts rather than a single multi-command dispatcher. The intended entry points are download, preprocess, train, evaluate, predict, and validate scripts.
- TensorFlow is retained only as a historical reference. New handoff work is PyTorch-first.
- The current survey headers and Google Drive image layout remain the input contract.
- “Google Cloud” in the early discussion referred to the Google Cloud Console OAuth setup for Google Drive. Google Cloud Storage is not part of the current pipeline.

## Demonstration Notebook Contract

The handoff notebook is `notebooks/CORE_pipeline_v1.ipynb`. It is designed to demonstrate the full workflow while keeping costly or interactive actions opt-in:

1. Configure paths and deterministic seeds.
2. Authenticate to Google Drive through the existing PyDrive2 flow.
3. Download rated images through the planned separate download command.
4. Clean survey responses, filter included rows, aggregate rater labels, and prepare samples.
5. Select a deterministic 50-unique-image demonstration subset and create an approximately 30/10/10 train/validation/test split.
6. Construct the existing TorchGeo backbone and GreenSpace regression heads.
7. Run one frozen-backbone warm-up epoch and one unfrozen fine-tuning epoch.
8. Evaluate the demonstration model and calibrate thresholds.

Authentication, model construction, training, and evaluation switches default to off so a cloned repository can inspect and execute the inexpensive preparation path first. Demonstration metrics from 50 images and two epochs are a wiring check, not a performance claim.

## Implemented in This Step

- Added reusable preprocessing functions in `src/preprocessing.py`.
- Refactored `scripts/clean_survey.py` to call the reusable cleaning implementation.
- Added regression tests in `tests/test_preprocessing.py`.
- Added `notebooks/CORE_pipeline_v1.ipynb` as the clean illustrative entry point.
- Preserved the existing cleaning, inclusion filtering, label aggregation, and two-stage 60/20/20 split behavior.
- Marked rated-image download packaging explicitly as pending instead of inventing a replacement for the existing Drive orchestration.

## Verification Evidence

The reusable preprocessing implementation was compared with the current June artifacts:

- Clean survey output: 7,991 rows and 16 columns, exactly matching the saved cleaned table.
- Before inclusion filtering: 7,991 survey rows and 5,427 unique images.
- After inclusion filtering: 7,655 survey rows and 5,242 unique images.
- Aggregated soft and hard labels exactly match the saved June tables for all 5,242 included images.
- Split membership and order exactly match the current saved split files:
  - train: 3,145 images
  - validation: 1,048 images
  - test: 1,049 images
- The Python unit tests, supported-version check, offline checkpoint load/prediction check, dependency consistency check, notebook schema validation, and Python syntax checks pass.

## Scope Notes

- The older orchestration notebooks remain unchanged and continue to document the historical workflow.
- The local planning document under `.cursor/plans/` is intentionally ignored by Git. This tracked process update preserves the project decisions and handoff state that teammates need.
- The notebook does not silently authenticate, download images, construct a large model, or train when opened.
- The 10,000-row production run should not begin until the scripted training path has resume support, useful timing/progress output, and a successful small end-to-end smoke test.

## Next Steps

1. Extract the rated-image Google Drive listing, ID join, cache-aware download, and retry behavior from the current tested notebook into a reusable function and `scripts/download_drive_images.py`.
2. Replace the pending download cell in `CORE_pipeline_v1.ipynb` with a real call to that reusable downloader.
3. Add `scripts/preprocess.py` as a thin command-line adapter around `src.preprocessing`, with explicit input/output paths, run tags, seeds, and a concise data summary.
4. Package prediction and evaluation as thin adapters around their current tested functions, then package training with explicit warm-up/fine-tuning settings, checkpoint resume, progress/timing diagnostics, and safe output directories.
5. Run the notebook on the cached 50-image subset for one warm-up and one fine-tuning epoch, then verify evaluation and threshold calibration outputs.
6. Perform the Windows handoff pass: environment creation, OAuth/browser behavior, path handling, commands, and a clone-to-first-run walkthrough.
7. Only after those checks, prepare and launch the full approximately 10,000-row training run, evaluate it, and decide whether its run bundle should become the default.
