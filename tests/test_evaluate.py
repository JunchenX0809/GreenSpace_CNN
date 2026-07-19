from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src_torch.evaluation import tune_val_thresholds


BIN_NAMES = ["a", "b"]
BINARY_COLS = ["a_p", "b_p"]
# Column 0 tracks label "a", column 1 tracks label "b"; both are separable so
# every label yields a finite tuned threshold.
BIN_HEAD_PROBS = np.array(
    [
        [0.90, 0.80],
        [0.80, 0.20],
        [0.70, 0.70],
        [0.30, 0.30],
        [0.20, 0.60],
        [0.10, 0.10],
    ]
)


class TuneValThresholdsTests(unittest.TestCase):
    def test_uses_hard_label_columns_when_present(self) -> None:
        val_df = pd.DataFrame(
            {
                "a": [1, 1, 1, 0, 0, 0],
                "b": [1, 0, 1, 0, 1, 0],
                "a_p": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
                "b_p": [0.8, 0.2, 0.7, 0.3, 0.6, 0.1],
            }
        )
        prediction = (val_df, {"bin_head": BIN_HEAD_PROBS}, {"loss": 0.0})

        thresholds_df, best_thresholds = tune_val_thresholds(
            prediction, BIN_NAMES, BINARY_COLS
        )

        self.assertEqual(thresholds_df["label"].tolist(), ["a", "b"])
        self.assertEqual(set(best_thresholds), {"a", "b"})
        for value in best_thresholds.values():
            self.assertTrue(np.isfinite(value))

    def test_falls_back_to_soft_columns_when_hard_labels_absent(self) -> None:
        val_df = pd.DataFrame(
            {
                "a_p": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
                "b_p": [0.6, 0.2, 0.7, 0.3, 0.8, 0.1],
            }
        )
        prediction = (val_df, {"bin_head": BIN_HEAD_PROBS}, {"loss": 0.0})

        thresholds_df, best_thresholds = tune_val_thresholds(
            prediction, BIN_NAMES, BINARY_COLS
        )

        self.assertEqual(thresholds_df["label"].tolist(), ["a", "b"])
        self.assertEqual(set(best_thresholds), {"a", "b"})

    def test_drops_labels_without_a_tuneable_threshold(self) -> None:
        # Label "b" is single-class here, so it must be excluded from the lookup.
        val_df = pd.DataFrame(
            {
                "a": [1, 1, 1, 0, 0, 0],
                "b": [0, 0, 0, 0, 0, 0],
            }
        )
        prediction = (val_df, {"bin_head": BIN_HEAD_PROBS}, {"loss": 0.0})

        thresholds_df, best_thresholds = tune_val_thresholds(
            prediction, BIN_NAMES, BINARY_COLS
        )

        self.assertEqual(thresholds_df["label"].tolist(), ["a", "b"])
        self.assertEqual(set(best_thresholds), {"a"})


if __name__ == "__main__":
    unittest.main()
