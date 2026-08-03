from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import predict_torch, validate_pipeline
from src_torch.inference import (
    expected_prediction_columns,
    validate_prediction_dataframe,
)


class PredictionDataframeContractTests(unittest.TestCase):
    def test_validates_stable_schema_and_bounds(self) -> None:
        frame = pd.DataFrame(
            {
                "image_filename": ["one.jpg"],
                "sports_field_prob": [0.75],
                "sports_field_pred": [1],
                "shade_class": ["abundant"],
                "shade_confidence": [0.8],
                "score_ev": [3.5],
                "veg_ev": [4.0],
            }
        )

        self.assertEqual(
            frame.columns.tolist(),
            expected_prediction_columns(["sports_field"]),
        )
        validate_prediction_dataframe(
            frame,
            binary_labels=["sports_field"],
            expected_rows=1,
        )

    def test_rejects_out_of_bounds_probability(self) -> None:
        frame = pd.DataFrame(
            {
                "image_filename": ["one.jpg"],
                "sports_field_prob": [1.2],
                "sports_field_pred": [1],
                "shade_class": ["minimal"],
                "shade_confidence": [0.8],
                "score_ev": [3.5],
                "veg_ev": [4.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "probabilities"):
            validate_prediction_dataframe(frame, ["sports_field"], expected_rows=1)


class PredictionCliTests(unittest.TestCase):
    def test_cli_writes_valid_csv_and_forwards_loader_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best_mcmae_PyTorch_test.pt"
            checkpoint.touch()
            image_dir = root / "new_images"
            image_dir.mkdir()
            (image_dir / "b.jpg").touch()
            (image_dir / "a.jpg").touch()
            output = root / "exports" / "result.csv"

            bundle = SimpleNamespace(
                model=object(),
                checkpoint_path=checkpoint,
                threshold_path=root / "thresholds_best_mcmae.csv",
                run_tag="PyTorch_test",
                variant="best_mcmae",
                bin_names=["sports_field"],
                thresholds={"sports_field": 0.5},
                img_size=(512, 512),
            )
            predictions = {
                "bin_head": np.array([[0.25], [0.75]], dtype=np.float32),
                "shade_head": np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
                "score_head": np.array([[2.5], [3.5]], dtype=np.float32),
                "veg_head": np.array([[4.0], [4.5]], dtype=np.float32),
            }

            with (
                patch.object(predict_torch, "resolve_device", return_value="cpu"),
                patch.object(predict_torch, "load_run_bundle", return_value=bundle),
                patch.object(
                    predict_torch,
                    "predict_image_paths",
                    return_value=predictions,
                ) as predict,
            ):
                result = predict_torch.main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--image-dir",
                        str(image_dir),
                        "--output",
                        str(output),
                        "--batch-size",
                        "2",
                        "--num-workers",
                        "1",
                        "--pin-memory",
                    ]
                )

            self.assertEqual(result, 0)
            exported = pd.read_csv(output)
            self.assertEqual(exported["image_filename"].tolist(), ["a.jpg", "b.jpg"])
            self.assertEqual(
                exported.columns.tolist(),
                expected_prediction_columns(["sports_field"]),
            )
            self.assertEqual(predict.call_args.kwargs["batch_size"], 2)
            self.assertEqual(predict.call_args.kwargs["num_workers"], 1)
            self.assertTrue(predict.call_args.kwargs["pin_memory"])
            self.assertEqual(predict.call_args.kwargs["img_size"], (512, 512))

    def test_cli_protects_existing_output_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.csv"
            output.write_text("existing\n")
            with self.assertRaises(FileExistsError):
                predict_torch._write_csv_atomic(
                    pd.DataFrame({"value": [1]}),
                    output,
                    overwrite=False,
                )
            self.assertEqual(output.read_text(), "existing\n")


class ValidationCliTests(unittest.TestCase):
    def test_inference_only_validation_passes_without_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "best_mcmae_PyTorch_test.pt"
            checkpoint.touch()
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "one.jpg").touch()
            bundle = SimpleNamespace(
                run_tag="PyTorch_test",
                variant="best_mcmae",
                bin_names=["sports_field"],
                binary_cols=["sports_field_p"],
                threshold_path=root / "thresholds_best_mcmae.csv",
            )

            with (
                patch.object(validate_pipeline, "_check_environment", return_value="ok"),
                patch.object(validate_pipeline, "_check_device", return_value=("cpu", "selected cpu")),
                patch.object(validate_pipeline, "load_run_bundle", return_value=bundle),
                patch.object(validate_pipeline, "_check_output_dir", return_value="writable"),
            ):
                result = validate_pipeline.main(
                    [
                        "--checkpoint",
                        str(checkpoint),
                        "--skip-data",
                        "--inference-dir",
                        str(image_dir),
                    ]
                )

            self.assertEqual(result, 0)

    def test_data_validation_reports_missing_images(self) -> None:
        binary_cols = [
            "sports_field_p",
            "multipurpose_open_area_p",
            "children_s_playground_p",
            "water_feature_p",
            "walking_paths_p",
            "built_structures_p",
            "parking_lots_p",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "splits"
            split_dir.mkdir()
            image_root = root / "images"
            image_root.mkdir()
            for index, split in enumerate(("train", "val", "test")):
                filename = f"{split}_{index}.jpg"
                row = {
                    "image_path": str(image_root / filename),
                    "image_filename": filename,
                    "shade_class": 0,
                    "score_mean": 3.0,
                    "veg_mean": 3.0,
                }
                row.update({column: 0.0 for column in binary_cols})
                pd.DataFrame([row]).to_csv(split_dir / f"{split}.csv", index=False)

            with self.assertRaisesRegex(FileNotFoundError, "missing 1 images"):
                validate_pipeline._check_data(split_dir, image_root, binary_cols)


if __name__ == "__main__":
    unittest.main()
