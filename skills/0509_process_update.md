# Process Update — 2026-05-09

## Scope (planned — no implementation yet)
---


trajectory so far with score & veg:
- **Iteration 1**: Hard labels (`score_class`, sparse CE) → low accuracy on val
- **Iteration 2**: Soft labels (`score_p_1..5`, categorical CE) → same generalization gap
- **Iteration 3 (this plan)**: Continuous target (`score_mean`, MAE) → ?

Each step has been a different way to **encode the same rater information**. `score_mean` is literally the expected value of the `score_p_1..5` distribution --> which we already train on in soft mode.
---

## Next steps (priority order)

### Immdiate (priority)

**1. Continuous range (softmax → regression)**

**Goal:** Replace (or trial alongside) the **5-way softmax** heads for **score** and **vegetation** with **scalar regression** outputs on a **continuous** scale (e.g. **[1, 5]** with clipping / sensible loss), and compare performance to the current **softmax** (and **soft-target**) setup.

**Open items (for a later plan):** target (`score_mean` / `veg_mean` vs EV of soft bins), loss (MAE vs Huber vs MSE), metrics (\(R^2\), MAE vs means, optional bucketed accuracy), and how this interacts with **McMAE / checkpoint** selection.

**2. Oversampling**

**Phase A — binary head:** Start by **oversampling** the binary label with the **lowest positive support** (train stream only). Measure impact on **that label** and on **overall** multi-task stability.

**Phase B — ordinal heads (conditional):** If Phase A helps without hurting the rest of the model, extend oversampling ideas to **score** and **veg** (e.g. emphasis on **rare ordinal bins** or hard examples — exact rule **TBD**).

---

### Follow-up experiments (to address overfitting)

Both approaches below target the same root cause — the ~3x train→val loss gap on score/veg heads, which suggests the model memorizes training data instead of learning generalizable features. They can be combined and tested independently.

**3. Regularization (score & veg overfitting)**

**Goal:** Close the train/val gap so that improvements during training actually transfer to unseen data.

Two concrete options to try:

- **Dropout before score/veg heads:** Add a `Dropout(0.3)` layer immediately before each `Dense` head (score and veg only). This forces the network to spread its learning across more features instead of relying on a few memorized patterns. Implementation: in NB03's model-build cell, insert `x_score = layers.Dropout(0.3)(x)` before `score_out = layers.Dense(...)`, same for veg. Start with 0.3, tune up/down based on whether the train/val gap narrows.

- **Label smoothing:** When using soft CE loss, mix a small uniform component into the target distribution — e.g. `target = 0.9 * soft_probs + 0.1 * (1/5)`. Prevents the model from becoming overconfident on any single class. Implementation: a one-line change in `make_unbatched_ds` where `y_score` is built for soft mode.

**4. Freeze strategy (backbone fine-tuning depth)**

**Goal:** Unfreeze only the last N layers of the backbone (e.g. last 20–40 layers) instead of all ~240. The binary heads generalize well (train/val ratio ~1.15x), but the ordinal heads don't — suggesting the full unfreeze causes the backbone to overfit to score/veg training signal. Keeping early layers frozen preserves general visual features while still allowing top-layer adaptation.

**Implementation:** In NB03's unfreeze cell, replace `base.trainable = True` with a loop that keeps early layers frozen: `for layer in base.layers[:-N]: layer.trainable = False`. Start with N=30 and compare the train/val gap.

---

## Summary

| # | Approach | Source | What it targets |
|---|---|---|---|
| 1 | **Continuous range** (regression heads) | Immediate Next | Test if scalar targets improve score/veg over softmax |
| 2 | **Oversampling** | Immediate Next | Improve underrepresented labels |
| 3 | **Regularization** (dropout, label smoothing) | Suggested | Narrows the 3x train/val gap (memorization) |
| 4 | **Freeze strategy** (partial backbone unfreeze) | Suggested | Prevents backbone from overfitting to ordinal signal |

---

## Links

- Prior ordinal / soft context: [0502_process_update.md](0502_process_update.md)
