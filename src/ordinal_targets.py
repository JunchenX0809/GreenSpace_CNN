"""Shared helpers for soft ordinal targets (score / veg).

Single source of truth for the canonical column order so NB03 (training) and
NB04 (evaluation) stack `score_p_1..5` / `veg_p_1..5` the same way the model's
softmax dimension is interpreted (index 0 ↔ class 1, ..., index 4 ↔ class 5).
"""

import numpy as np

SCORE_PROB_COLS = [f'score_p_{i}' for i in range(1, 6)]
VEG_PROB_COLS = [f'veg_p_{i}' for i in range(1, 6)]


def soft_score_veg_arrays(df, eps=1e-8):
    """Return (score_probs, veg_probs) as float32 (N, 5), clipped + row-normalised."""
    missing = [c for c in SCORE_PROB_COLS + VEG_PROB_COLS if c not in df.columns]
    assert not missing, f"Missing soft prob columns: {missing}. Re-run NB02 to refresh splits."

    score = df[SCORE_PROB_COLS].fillna(0.0).astype(np.float32).values
    veg = df[VEG_PROB_COLS].fillna(0.0).astype(np.float32).values
    score = np.clip(score, 0.0, 1.0)
    veg = np.clip(veg, 0.0, 1.0)
    score = score / np.maximum(score.sum(axis=1, keepdims=True), eps)
    veg = veg / np.maximum(veg.sum(axis=1, keepdims=True), eps)
    return score.astype(np.float32), veg.astype(np.float32)
