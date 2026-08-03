# Process Update — 2026-08-02

## Local Best-MCMAE Comparison

| Test metric | Previous local run<br>`PyTorch_20260614_220926` | Current local evaluation<br>`PyTorch_20260719_full_windows_local_eval_0802` | Delta |
| --- | ---: | ---: | ---: |
| Macro PR-AUC | 0.888 | 0.877 | -0.011 |
| Macro ROC-AUC | 0.939 | 0.935 | -0.004 |
| Tuned macro F1 | 0.824 | 0.806 | -0.018 |
| Shade accuracy, conditional | 0.668 | 0.693 | +0.025 |
| Score MAE vs. rater mean | 0.576 | 0.558 | -0.018 |
| Vegetation MAE vs. rater mean | 0.430 | 0.417 | -0.013 |

Binary ranking and tuned F1 declined slightly, while shade accuracy and both
continuous MAE outcomes improved. The runs use different regenerated test
splits, so the deltas are directional rather than a controlled same-split
comparison.

## Script Annotation Example

Before:

```python
predictions_by_split = {
    split: predict_split(model, split, device=device)
    for split in ("train", "val", "test")
}
```

After:

```python
# Run inference once on train, validation, and test.
predictions_by_split = {
    split: predict_split(model, split, device=device)
    for split in ("train", "val", "test")
}
```

All 15 Python entry-point scripts now use concise workflow-stage annotations;
no program behavior changed.

## July Training History

![July PyTorch training curves](../presentation_visuals_only/epoch_visual/0719_training_curves.png)

The run completed 5 warm-up and 24 fine-tuning epochs. The visual uses global
epoch numbering, so fine-tuning epoch 24 is global epoch 29; the epoch-24 star
marks the best validation PR-AUC, not the stopping point.

## Reviewer Inference Test

The packaged best-MCMAE bundle passed CPU validation and inference on all 50
review images. It generated `predictions_review_test_v1.csv` with 50 rows, 19
columns, no duplicate filenames, and no missing values.

## Remaining Review Handoff

| Item | Status |
| --- | --- |
| Reconcile and commit the local packaged workflow | Pending |
| Push the reviewed commit for fresh cloning | Pending |
| Zip checkpoint, matching configuration, and thresholds | Pending |
| Repeat inference from a clean clone | Pending |
| Provide labeled splits/images for evaluation | Optional; not needed for inference |
