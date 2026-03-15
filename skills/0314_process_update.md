# Process Update — GreenSpace CNN Repo

---

## 1) Data & Storage Layer

- **Primary repository:** Google Drive folder(s) containing imagery.
- **Access method:** Google Drive API via OAuth2 (client secrets JSON + cached credentials).

**Verification checks:**

- First run: interactive consent → creates credentials cache.
- Subsequent runs: reuses cache without prompting; confirms ability to list and download sample files.
- **Stable identifiers:** Prefer Drive file IDs over filenames to avoid ambiguity.
- **Local caching:** download (how to improve?) rated images into a local cache for stable training I/O and faster epochs.

---

## 2) Label Table Construction (Survey → Training Rows)

**Clean Raw Survey Files:**

- Standardized filename and join to Drive file list to attach `drive_file_id`.
- Filter to only use images that are marked `include_tile == "yes"`.
- Produce split CSVs (train/val/test).

**Group by image filename and compute:**

- Binary soft labels (including shade): mean of 1/0s → columns like `*_p`.
- Score/veg soft distributions + means.
- Hard labels: binary with `>= 0.5`.
- `shade_class` with argmax of prob.
- Score/veg with argmax + 1.

**Argmax?**

- The index of the largest value. Good for multi-class labels & standard way to turn multi-class prob vector into a single hard label.

**Soft/hard?**

- **Soft labels:** Preserve disagreement among raters. Used directly for training binary heads. The model is learning “how likely” each feature is, based on rater consensus.
- **Hard labels:** One result per survey question. Used for classification head.

---

## 3) Dataset Build (tf.data)

- Load image from local cache.
- Decode images.
- 512×512 unchanged.
- Batch and prefetch: `batch(BATCH_SIZE)` + `prefetch(AUTOTUNE)`.

---

## 4) Model Architecture (Multi-Task)

- **Shared:** ImageNet CNN weights — EfficientNetB0 (`include_top=False`) with input `(512, 512, 3)`.

**Heads:**

- **Bin_head:** Sigmoid outputs for multi-label binaries.
- **Score_head, veg_head:** Categorical softmax for 1–5 levels; evaluated expected-value MAE.

---

## 5) Training Strategy

- **Binary loss:** BCE for past trainings.
- **Score/veg losses:** Sparse categorical cross-entropy.
- **Loss weights:** Defaults to 1s; different dictionaries when compiling.

**Augmentation (train only):**

We apply on the fly augmentation to the training stream only, producing one transformed image per sample per epoch (no increase in dataset size). Augmentation is implemented in the tf.data pipeline after decode and optional cache, and combines random geometric and discrete photometric transforms to improve robustness to viewpoint and lighting.

- Applied in tf.data after decode/cache; one image in → one image out (no sample inflation).

- **Geometry:** Random 90° rotation (k = 0,1,2,3), random flip left-right, random flip up-down.
- **Photometry (discrete):** Brightness add ±0.15 or ±0.05; contrast × 0.85–1.15; saturation × 0.95–1.05; hue ±0.03; then clip to [0, 1].

**Schedule:**

- **Warm up:** Freeze backbone, train heads only.
- **Fine tune:** Unfreeze backbone, train end-to-end.

**Optimization / callback control:**

- **Primary metric (binary head):** Validation **weighted macro PR-AUC** (`val_bin_head_weighted_pr_auc`). Two interchangeable implementations via `USE_BUILTIN_WEIGHTED_AUC` toggle in `03_model_training.ipynb`.

- **Weighting direction:** The built-in `AUC(multi_label=True, label_weights=[...])` accepts any weights — it does not enforce prevalence-based direction. By passing `w_i ∝ 1/sqrt(pos_i)` we explicitly invert the direction so **rare labels drive the metric** rather than common ones. Both modes use the same formula and the same `BINARY_MONITOR` key.

- **Mode A — built-in (conventional, `USE_BUILTIN_WEIGHTED_AUC=True`):** `tf.keras.metrics.AUC(curve='PR', multi_label=True, label_weights=INV_SQRT_LABEL_WEIGHTS)`. No custom callback. Approximates AUC via 200 fixed thresholds (~0.02 error on rare labels like `gardens`). Single-class labels return a degenerate value rather than being skipped.

- **Mode B — custom callback (exact, `USE_BUILTIN_WEIGHTED_AUC=False`, default):** `WeightedMacroPRAUC` runs `sklearn.average_precision_score` on the full val set each epoch. Exact AP at every unique threshold. Single-class labels are skipped and weights renormalized. Formula: `score = Σ w_i × AP_i` over valid labels.

- **Weighted macro PR-AUC formula:** `w_i ∝ 1/sqrt(max(pos_i, 1))`, normalized so `Σ w_i = 1`.

  | Label | pos_count | weight (1/sqrt) |
  |---|---|---|
  | gardens | 83 | 0.295 ← rarest, highest weight |
  | children_s_playground | 280 | 0.161 |
  | water_feature | 478 | 0.123 |
  | sports_field | 635 | 0.107 |
  | parking_lots | 701 | 0.102 |
  | built_structures | 962 | 0.087 |
  | walking_paths | 1750 | 0.064 |
  | multipurpose_open_area | 1853 | 0.062 ← most common, lowest weight |

  Rarest/most-common ratio: **4.7×** (sqrt). More aggressive: `1/n` → 22×. Less aggressive: `1/log(n)` → 1.7×. Tune with `1/n^alpha`, `alpha ∈ (0.5, 1.0)`.

- **`SingleClassValWarning` callback (always on):** Fires once at `on_train_begin`. Emits `CRITICAL` if any label is single-class in val; emits `WATCH` for labels under 50 val positives (`gardens` has 32 — currently WATCH).
- **ModelCheckpoint (PR-AUC):** Save best model by `val_bin_head_weighted_pr_auc` (max), e.g. `best_<run_tag>.keras`.
- **BestMcMaeCheckpoint:** Separate checkpoint that saves best by multi-task MAE (`mc_mae`) over score/shade/veg heads, e.g. `best_mcmae_<run_tag>.keras`. Keeps a regression/ordinal-focused snapshot alongside the PR-AUC one.
- **EarlyStopping:** Monitor `val_bin_head_weighted_pr_auc` (max), `patience=10`; restore best weights on stop.
- **ReduceLROnPlateau:** Monitor `val_bin_head_weighted_pr_auc` (max), `factor=0.5`, `patience=2`.

**Walking path + shade along paths combo:**

- If paths absent, shade is effectively not applicable. We should not teach the model to learn from shade losses when the previous walking-path question has a 0.

**Current method:**

- Keep predicting shade for all samples.
- Mask the shade loss during training using path presence: `shade_sample_weight = walking_paths_p` (soft weight in [0, 1]).
- If walking path’s soft rating is near 0, the sample does not contribute to shade loss.
- Same rule in train/val so `val_loss`, early stopping, and LR reduction reflect it.

---

## 6) Results

Refer to Update Report:  
https://docs.google.com/document/d/1TNlDL9u-TLdxdjNEPeRYFU3WLsinn_bZX5JMdwy--S8/edit?tab=t.3itocxr3i17o#heading=h.gg02jx8tcroq
