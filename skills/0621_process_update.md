# Process Update — 2026-06-21


## PyTorch Portability for External Review

**Goal:** a new user runs an approved model without this developer's laptop
paths, cache, or device type — no change to model behavior or training results.

**Changes (env vars; all optional, local default unchanged):**

| Env var | Selects |
| --- | --- |
| `GREENSPACE_DATA_ROOT` | directory of split manifests |
| `GREENSPACE_IMAGE_ROOT` | training/eval images, by filename |
| `GREENSPACE_INFERENCE_IMAGE_ROOT` | unseen-image folder for inference |
| `GREENSPACE_PREDICTION_OUTPUT_ROOT` | prediction output location |

- An explicit image folder must contain every requested file; a partial folder
  fails clearly instead of silently mixing in the laptop cache.
- Checkpoints rebuild **without re-downloading weights**; inference uses the
  label order saved in the checkpoint, not a live split.
- Device auto-selects (CUDA / Apple Silicon / CPU); no absolute checkpoint path
  remains in evaluation. README documents the env vars.

**Verified:**

- Smoke test — 10 images from `clusters/inference/` → CPU inference → 10 valid
  rows (19 cols, finite, score/veg within 1–5) at
  `clusters/outputs/predictions_portability_smoke.csv`.
- Offline checkpoint-load test passes (no network download); the real
  `PyTorch_20260614_220926` run loads end-to-end.
  Run: `.venv/bin/python scripts/check_offline_checkpoint_load.py`

## Run-Bundle Work

**Purpose:** let other users test our trained model on new photos
(photos → saved model → prediction CSV) offline — no absolute laptop paths, no download,
no retraining. A "bundle" = weights + settings + label names + thresholds.

| Piece | Done? |
| --- | --- |
| Load weights + settings + labels + thresholds as one validated unit | ✅ `load_run_bundle()` (`src_torch/run_bundle.py`) |
| Proof it works offline, labels from checkpoint not live split | ✅ `scripts/check_offline_checkpoint_load.py` + real-run load |
| Workflow-1 prediction on a photo folder | ✅ 10-image smoke test |

Also done: loader knobs `GREENSPACE_NUM_WORKERS` / `GREENSPACE_PIN_MEMORY`
(`config.py`, defaults unchanged); README reoriented to PyTorch-as-canonical.

**Files to reuse a model:** `models/runs/<tag>/best_mcmae_<tag>.pt`,
`models/runs/<tag>/model_config_<tag>.json`,
`monitoring_output/runs/<tag>/thresholds_best_mcmae.csv` — plus the repo code and
a venv (`pip install -r requirements.txt`).

**Open decisions:** transform contract (`[0,1]` vs `[0,255]`) not carried by the
bundle; `torch.load(weights_only=False)` policy undecided.

## Handoff Verification

Copied the repo code + the 3 model files to a separate location outside the
project (`GreenSpace_review_test/`) and confirmed the model works from the copy:

| Check | Result |
| --- | --- |
| 3 model files byte-for-byte identical to source | ✅ |
| `scripts/check_offline_checkpoint_load.py` from the copy, offline | ✅ PASS |
| Real `PyTorch_20260614_220926` model loads from the copy (weights + config + thresholds) | ✅ |
| Fresh venv + `pip install -r requirements.txt` + offline check (arm64) | ✅ PASS — `torch 2.10.0` / `torchgeo 0.8.1` |

### Reproduce the proof (run in a fresh checkout)

A reviewer can confirm the model is self-contained — clean environment,
dependencies, and offline model load — by running this from the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt          # pulls torch / torchgeo (multi-GB)
python scripts/check_offline_checkpoint_load.py   # expect: PASS
```

**Apple Silicon:** `python3.11` must be an **arm64** build. An x86_64 Python
(e.g. an Intel pyenv/Homebrew one) fails at install — `torch==2.10.0` has no
x86_64 macOS wheel (they stop at 2.2.2). Verify with
`python3.11 -c "import platform; print(platform.machine())"` (want `arm64`); see
the README arm64 troubleshooting if it prints `x86_64`. Linux/CUDA clusters are
unaffected.
