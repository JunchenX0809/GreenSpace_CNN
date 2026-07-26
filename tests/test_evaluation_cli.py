from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import evaluate_torch
from src_torch.evaluation import tune_validation_thresholds


class EvaluationHelpersTests(unittest.TestCase):
    def test_validation_threshold_wrapper_aligns_hard_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "sports_field": [0, 1, 1, 0],
                "sports_field_p": [0.0, 1.0, 1.0, 0.0],
            }
        )
        prediction = (
            frame,
            {
                "bin_head": np.array([[0.1], [0.8], [0.7], [0.2]]),
            },
            {},
        )

        thresholds, threshold_map = tune_validation_thresholds(
            prediction,
            ["sports_field_p"],
        )

        self.assertEqual(thresholds["label"].tolist(), ["sports_field"])
        self.assertIn("sports_field", threshold_map)
        self.assertAlmostEqual(
            threshold_map["sports_field"],
            float(thresholds.loc[0, "best_threshold"]),
        )


class EvaluationCliTests(unittest.TestCase):
    def test_cli_forwards_explicit_data_paths_and_saves_beside_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "splits"
            image_root = root / "images"
            run_dir = root / "models" / "runs" / "PyTorch_test"
            split_dir.mkdir()
            image_root.mkdir()
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_mcmae_PyTorch_test.pt"
            checkpoint.touch()
            split_frame = pd.DataFrame({"sports_field_p": [0.0]})
            for split in ("train", "val", "test"):
                split_frame.to_csv(split_dir / f"{split}.csv", index=False)

            prediction = (
                split_frame,
                {"bin_head": np.array([[0.1]])},
                {"loss": 0.0},
            )
            empty_table = pd.DataFrame()

            with (
                patch.object(evaluate_torch, "resolve_device", return_value="cpu"),
                patch.object(
                    evaluate_torch,
                    "load_torch_checkpoint_model",
                    return_value=(object(), {"binary_cols": ["sports_field_p"]}, {}),
                ),
                patch.object(
                    evaluate_torch,
                    "infer_run_tag_and_variant",
                    return_value=("PyTorch_test", "best_mcmae"),
                ),
                patch.object(evaluate_torch, "predict_split", return_value=prediction) as predict,
                patch.object(
                    evaluate_torch,
                    "tune_validation_thresholds",
                    return_value=(empty_table, {}),
                ),
                patch.object(
                    evaluate_torch,
                    "evaluate_loss_monitoring",
                    return_value=empty_table,
                ),
                patch.object(
                    evaluate_torch,
                    "evaluate_all_splits",
                    return_value=(empty_table, empty_table),
                ),
                patch.object(
                    evaluate_torch,
                    "save_evaluation_outputs",
                    return_value={},
                ) as save,
            ):
                result = evaluate_torch.main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--split-dir",
                        str(split_dir),
                        "--image-root",
                        str(image_root),
                        "--batch-size",
                        "2",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(predict.call_count, 3)
            for call in predict.call_args_list:
                self.assertEqual(call.kwargs["split_dir"], split_dir)
                self.assertEqual(call.kwargs["image_root"], image_root)
                self.assertEqual(call.kwargs["batch_size"], 2)
            self.assertEqual(save.call_args.kwargs["run_dir"], run_dir)


if __name__ == "__main__":
    unittest.main()
