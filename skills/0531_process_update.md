# Process Update — 2026-05-31

## PyTorch Full-Run Comparison

Run `PyTorch_20260531_211840` tested the first full PyTorch/TorchGeo ResNet-50 pipeline against the current TensorFlow/EfficientNet full-unfreeze baseline, `20260524_220623`. This is the apples-to-apples comparison; `20260525_220942` was the narrow partial-unfreeze TensorFlow run and is not the primary comparator.

| Metric | PyTorch `20260531_211840` | TensorFlow full `20260524_220623` | Delta |
| --- | ---: | ---: | ---: |
| Overall PR-AUC | `0.797` | `0.817` | `-0.020` |
| Overall ROC-AUC | `0.894` | `0.912` | `-0.018` |
| Overall tuned F1 | `0.744` | `0.758` | `-0.014` |
| Score MAE mean | `0.631` | `0.583` | `+0.048` |
| Veg MAE mean | `0.509` | `0.425` | `+0.085` |
| Shade accuracy overall | `0.593` | `0.653` | `-0.060` |

Interpretation: the first full PyTorch/TorchGeo run is competitive but does not beat the corrected TensorFlow full-unfreeze baseline overall. TensorFlow remains stronger on aggregate binary metrics, score/veg MAE, and shade accuracy.

## Per-Label Pattern

Useful gain:

- `water_feature`: PR-AUC `+0.053`, tuned F1 `+0.027`

Near parity:

- `walking_paths`: PR-AUC `-0.006`, tuned F1 `+0.004`
- `multipurpose_open_area`: PR-AUC `-0.013`, tuned F1 `-0.009`

Small regressions:

- `built_structures`: PR-AUC `-0.020`, tuned F1 `-0.029`
- `sports_field`: PR-AUC `-0.025`, tuned F1 `-0.032`
- `parking_lots`: PR-AUC `-0.047`, tuned F1 `-0.023`

Largest regression:

- `children_s_playground`: PR-AUC `-0.083`, tuned F1 `-0.034`

## TorchGeo Read

TorchGeo gave us a useful signal for `water_feature`, but the current ResNet-50 + 224 preprocessing is probably not ideal for small amenity detection like `children_s_playground`. The next most sensible investigation is whether preserving more spatial detail, for example avoiding forced 224 resize or testing a higher-resolution-compatible backbone/setup, helps playgrounds without losing the water gain.
