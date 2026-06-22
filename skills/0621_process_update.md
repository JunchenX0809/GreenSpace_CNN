# Process Update — 2026-06-21

> **Headline:**  made a PyTorch model run portable as one folder (checkpoint + settings + thresholds) and verified it offline; after activating the project environment, run `python scripts/check_offline_checkpoint_load.py`, then review: `README.md` → `src_torch/run_bundle.py` → that check script → `notebooks/05_pyTorch_prediction_demo.ipynb`.

## Full Inference — Swin V2 + NAIP

Full inference completed with `PyTorch_20260614_220926/best_mcmae` and its matching validation-tuned thresholds: **15,019** unseen images, **19** prediction columns, and no notebook errors.

Artifacts: [prediction CSV](../predictions/predictions_PyTorch_20260614_220926.csv), [inference summary](../predictions/summary_stats_PyTorch_20260614_220926.png), and [training-vs-inference diagnostic](../predictions/train_vs_inference_PyTorch_20260614_220926.png).

| Measure | Training reference | Inference | Difference |
| --- | ---: | ---: | ---: |
| Score mean | `3.09` | `3.74` | `+0.65` |
| Vegetation mean | `3.01` | `3.52` | `+0.51` |
| Score `>= 4.9` | `17.1%` | `29.0%` | `+11.9 pp` |
| Vegetation `>= 4.9` | `12.8%` | `23.2%` | `+10.3 pp` |

The prior exact-5 clipping artifact is resolved: score had one exact `5.0` prediction and vegetation had none. Upper-end saturation remains (`>= 4.99`: score `21.7%`, vegetation `16.9%`), so this is a calibration/domain-shift signal rather than the prior inference-pipeline defect. Binary predictions were below training-label prevalence for six of seven labels; `water_feature` was broadly similar.


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

**Files to reuse a model:** one `models/runs/<tag>/` folder containing
`best_mcmae_<tag>.pt`, `model_config_<tag>.json`, and
`thresholds_best_mcmae.csv` — plus the repo code and a venv
(`pip install -r requirements.txt`). Legacy monitoring copies remain for
historical comparisons.

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
| Three-file bundle copied to `clusters/model_bundle/` and compared on 10 images | ✅ same predictions as the prior smoke run |
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
