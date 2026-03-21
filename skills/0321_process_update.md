# 0321 Process Update

## Augmentation: add identity values, reduce to 3 per station

**Problem**: The photometric pipeline always modified every image — no identity option. The model never trained on true pixel values.

**Fix**: Add identity (brightness=0, contrast=1, saturation=1) and drop the near-identity steps (±0.05) that were barely perceptible on 512×512 tiles.

| Station | Before | After |
|---|---|---|
| Brightness | [-0.15, -0.05, +0.05, +0.15] | [-0.15, **0.0**, +0.15] |
| Contrast | [0.85, 0.95, 1.05, 1.15] | [0.85, **1.0**, 1.15] |
| Saturation | [0.90, 0.95, 1.05, 1.10] | [0.90, **1.0**, 1.10] |
| Hue | uniform [-0.03, +0.03] | unchanged |

**Why 3 values**: Standard pipelines use binary apply/skip (`p=0.5`). Three choices (low / identity / high) is already richer. Each station has 1/3 identity chance; all three simultaneously: (1/3)^3 ≈ 3.7% of samples per epoch are photometrically unmodified. Combined with geometry (4×2×2=16 combos), total augmentation space is 432 distinct transforms — sufficient for ~2,300 training images over 100 epochs.

**Changed**: `src/augmentation.py` — single source of truth, no notebook changes needed.
