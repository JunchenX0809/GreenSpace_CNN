# Process Update — 2026-05-09

## Scope (planned — no implementation yet)

Notebook / pipeline changes will be designed **later**. This note only records **intent** for the next experiments.

---

## 1. Score & veg heads: softmax → continuous regression

**Goal:** Replace (or trial alongside) the **5-way softmax** heads for **structured score** and **vegetation** with **scalar regression** outputs on a **continuous** scale (e.g. **[1, 5]** with clipping / sensible loss), and compare performance to the current **softmax** (and **soft-target**) setup.

**Open items (for a later plan):** target (`score_mean` / `veg_mean` vs EV of soft bins), loss (MAE vs Huber vs MSE), metrics (\(R^2\), MAE vs means, optional bucketed accuracy), and how this interacts with **McMAE / checkpoint** selection.

---

## 2. Regularization (score & veg overfitting)

**Problem:** Score and veg heads show a ~3x train→val loss gap (e.g. score loss 0.69 train → 1.75 val). The model memorizes training patterns instead of learning generalizable features. This is the most likely reason these two heads underperform — changing the label format (sparse → soft → regression) won't help if the model overfits regardless.

**Goal:** Close the train/val gap so that improvements during training actually transfer to unseen data.

**Two concrete options to try:**

- **Dropout before score/veg heads:** Add a `Dropout(0.3)` layer immediately before each `Dense` head (score and veg only). This forces the network to spread its learning across more features instead of relying on a few memorized patterns. Implementation: in NB03's model-build cell, insert `x_score = layers.Dropout(0.3)(x)` before `score_out = layers.Dense(...)`, same for veg. Start with 0.3, tune up/down based on whether the train/val gap narrows.

- **Label smoothing:** When using soft CE loss, mix a small uniform component into the target distribution — e.g. `target = 0.9 * soft_probs + 0.1 * (1/5)`. Prevents the model from becoming overconfident on any single class. Implementation: a one-line change in `make_unbatched_ds` where `y_score` is built for soft mode.

---

## 3. Freeze strategy (backbone fine-tuning depth)

**Problem:** Currently the entire EfficientNetB0 backbone is unfrozen during fine-tuning. The binary heads generalize well (train/val ratio ~1.15x), but the ordinal heads don't — suggesting the full unfreeze causes the backbone to overfit to score/veg training signal while the binary heads are robust enough to tolerate it.

**Goal:** Unfreeze only the last N layers of the backbone (e.g. last 20–40 layers) instead of all ~240. This limits how much the backbone can shift toward memorizing ordinal patterns, while still allowing the top layers to adapt.

**Implementation:** In NB03's unfreeze cell, replace `base.trainable = True` with a loop that keeps early layers frozen: `for layer in base.layers[:-N]: layer.trainable = False`. Start with N=30 and compare the train/val gap.

---

## 4. Oversampling

**Phase A — binary head:** Start by **oversampling** the binary label with the **lowest positive support** (train stream only). Measure impact on **that label** and on **overall** multi-task stability.

**Phase B — ordinal heads (conditional):** If Phase A helps without hurting the rest of the model, extend oversampling ideas to **score** and **veg** (e.g. emphasis on **rare ordinal bins** or hard examples — exact rule **TBD**).

---

## Summary: two levers to improve score & veg

| Approach | What it targets | Expected effect |
|---|---|---|
| **Regularization** (dropout, label smoothing) | Model memorizing training data | Narrows the 3x train/val loss gap so learned features actually transfer |
| **Freeze strategy** (partial backbone unfreeze) | Backbone overfitting to ordinal signal | Preserves general visual features while still allowing top-layer adaptation |

Both attack the same root cause — overfitting on ~2800 training images — from different angles. They can be combined and tested independently.

---

## Links

- Prior ordinal / soft context: [0502_process_update.md](0502_process_update.md)
