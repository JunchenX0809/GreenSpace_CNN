# Process Update — 2026-05-24

## Bounded Regression

We implemented bounded regression for the current score/veg `regression` mode. The heads now use `Dense(1, sigmoid)` followed by `Rescaling(4.0, offset=1.0)`, preserving continuous 1–5 predictions while preventing invalid values above 5 or below 1.

The implementation keeps the existing orchestration, config keys, MAE losses, monitor keys, checkpointing, oversampling, and prediction schema unchanged. A smoke run (`20260524_144021`) confirmed the bounded architecture is active; full NB04/NB05 quality and inference-distribution validation remain the next manual checks.

## Backbone Research

We reviewed remote-sensing pretrained backbone options, mostly in the PyTorch ecosystem. Real options exist, including TorchGeo pretrained ResNet weights, SatlasPretrain aerial/satellite RGB models, SatMAE-style transformers, Prithvi, and RemoteCLIP, but they are not drop-in replacements for the current TensorFlow/Keras EfficientNet workflow.

## Low-Debt Sequence

1. Finish bounded regression validation.
2. Try partial unfreeze of current EfficientNetB0.
3. Test one remote-sensing pretrained backbone as a controlled PyTorch-side experiment.

### Partial Unfreeze

TensorFlow’s transfer learning guide says that in most convnets, lower layers learn generic features, higher layers become more specialized, and fine-tuning usually means unfreezing the top layers while keeping lower layers frozen.

- Because our weak labels, especially children_s_playground, may be limited by label support and object scale. If we unfreeze too much, the model may memorize local quirks without actually improving generalization. water_feature may benefit more from domain adaptation than playgrounds, but even there, late-block adaptation should be enough to test whether EfficientNet’s higher-level features need retuning for aerial water/park patterns.

## Experiment Order

1. Fix warm-up freeze and run the heads-only baseline.
2. Run full unfreeze with corrected warm-up.
3. Test partial unfreeze only after those two baselines.

## Heads-Only Baseline Result

Run `20260524_194825` confirmed the backbone freeze worked (`fine_tune_backbone=false`, about `14,091` trainable params), but performance was not competitive. On test, heads-only reached PR-AUC `0.472`, tuned F1 `0.560`, score MAE `1.113`, and veg MAE `0.943`, versus prior `20260511_200157` at PR-AUC `0.811`, tuned F1 `0.757`, score MAE `0.588`, and veg MAE `0.448`.

Interpretation: train/val/test were all weak and close together, indicating underfitting from frozen ImageNet features rather than overfitting. Do not use heads-only as the final strategy; proceed to corrected full unfreeze, then partial unfreeze if needed.

## Corrected Full-Unfreeze Result

Run `20260524_220623` confirmed the intended transition: warm-up had `14,091` trainable params, then fine-tune had `4,021,639`, so the EfficientNet backbone unfroze at epoch 6. Test split sizes differ from May 11 (`1022` vs `922`), so treat this as directional rather than perfectly controlled.


| Metric           | Full unfreeze `20260524_220623` | May 11 `20260511_200157` | Delta    |
| ---------------- | ------------------------------- | ------------------------ | -------- |
| Overall PR-AUC   | `0.817`                         | `0.811`                  | `+0.006` |
| Overall tuned F1 | `0.758`                         | `0.757`                  | `+0.001` |
| Score MAE mean   | `0.583`                         | `0.588`                  | `-0.006` |
| Veg MAE mean     | `0.425`                         | `0.448`                  | `-0.024` |


Per-label notes:

Most useful gains:

- `parking_lots`: PR-AUC `+0.035`
- `built_structures`: PR-AUC `+0.035`
- `multipurpose_open_area`: PR-AUC `+0.008`
- `children_s_playground`: PR-AUC `+0.005`, tuned F1 `+0.019`

Small regressions:

- `water_feature`: PR-AUC `-0.025`, tuned F1 `-0.019`
- `sports_field`: PR-AUC `-0.023`, tuned F1 `-0.023`
- `walking_paths`: PR-AUC `+0.007`, but tuned F1 `-0.013`

Interpretation: corrected full unfreeze restores performance to the May 11 level and slightly improves several test metrics, especially vegetation MAE. Use this as the current baseline before testing partial unfreeze.

## Partial-Unfreeze Result

Run `20260525_220942` tested partial unfreeze with `block6d`, `block7a`, and `top_` trainable while BatchNorm stayed frozen. The run executed as intended, but performance regressed sharply versus corrected full unfreeze.

| Metric | Partial `20260525_220942` | Full unfreeze `20260524_220623` | Delta |
| --- | ---: | ---: | ---: |
| Overall PR-AUC | `0.434` | `0.817` | `-0.383` |
| Overall ROC-AUC | `0.531` | `0.912` | `-0.380` |
| Overall tuned F1 | `0.545` | `0.758` | `-0.212` |
| Score MAE mean | `1.113` | `0.583` | `+0.530` |
| Veg MAE mean | `0.941` | `0.425` | `+0.516` |

Per-label pattern:

Largest PR-AUC drops:

- `sports_field`: `-0.593`
- `parking_lots`: `-0.499`
- `built_structures`: `-0.435`
- `water_feature`: `-0.391`
- `children_s_playground`: `-0.350`

High-prevalence labels also deteriorated:

- `walking_paths`: PR-AUC `-0.236`
- `multipurpose_open_area`: PR-AUC `-0.174`

Preliminary interpretation: this narrow partial-unfreeze cut is too constrained and behaves closer to the heads-only underfit run than to full unfreeze. Do not keep this exact partial strategy; use corrected full unfreeze as the current TensorFlow/EfficientNet baseline, or only test a wider partial cut if another ablation is needed.
