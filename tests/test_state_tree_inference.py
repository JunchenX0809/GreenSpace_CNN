from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from PIL import Image

from scripts import predict_state_tree


def _write_rgb(path: Path, color: tuple[int, int, int] = (20, 40, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _state_root(image_root: Path, code: str = "AL", suffix: str = "one") -> Path:
    return image_root / f"USA_{code}_2023_Full-{suffix}" / f"USA_{code}_2023_Full"


class StateTreeInventoryTests(unittest.TestCase):
    def test_discovers_nested_jpg_directories_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp) / "GEE Derived"
            first = _state_root(image_root) / "04033-1402" / "jpg"
            second = _state_root(image_root) / "04033-1403" / "jpg"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "b.jpg").touch()
            (first / "a.JPG").touch()
            (first / "notes.txt").touch()
            (second / "a.jpg").touch()
            ignored = _state_root(image_root) / "04033-1403" / "preview"
            ignored.mkdir()
            (ignored / "outside.jpg").touch()
            (image_root / "inference_outputs").mkdir()

            states = predict_state_tree.discover_state_directories(image_root, ["AL"])
            inventory = predict_state_tree.discover_state_inventory(
                image_root,
                "AL",
                states["AL"],
            )

            self.assertEqual(len(inventory.records), 3)
            self.assertEqual(inventory.jpg_directories, 2)
            self.assertEqual(inventory.ignored_files, 1)
            self.assertEqual(inventory.duplicate_basename_groups, 1)
            self.assertEqual(inventory.layout_warnings, ())
            self.assertEqual(
                [record.path.name for record in inventory.records],
                ["a.JPG", "b.jpg", "a.jpg"],
            )
            self.assertEqual(
                [len(part) for part in predict_state_tree._partition_records(inventory.records, 2)],
                [2, 1],
            )

    def test_duplicate_state_directories_fail_before_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp)
            (image_root / "USA_AL_first").mkdir()
            (image_root / "USA_AL_second").mkdir()
            with self.assertRaisesRegex(ValueError, "Multiple top-level folders"):
                predict_state_tree.discover_state_directories(image_root, ["AL"])

    def test_exact_image_must_be_under_a_state_jpg_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp) / "GEE Derived"
            image_path = _state_root(image_root) / "04033-1402" / "jpg" / "one.jpg"
            _write_rgb(image_path)

            record = predict_state_tree._record_for_exact_path(image_root, str(image_path))

            self.assertEqual(record.state_code, "AL")
            self.assertEqual(record.park_code, "04033-1402")
            self.assertTrue(record.relative_path.endswith("/04033-1402/jpg/one.jpg"))

    def test_inventory_only_allows_sibling_output_under_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp) / "GEE Derived"
            image_path = _state_root(image_root) / "04033-1402" / "jpg" / "one.jpg"
            _write_rgb(image_path)
            output_root = image_root / "inference_outputs"

            result = predict_state_tree.main(
                [
                    "--image-root",
                    str(image_root),
                    "--output-dir",
                    str(output_root),
                    "--run-id",
                    "al_inventory",
                    "--states",
                    "AL",
                    "--inventory-only",
                ]
            )

            self.assertEqual(result, 0)
            inventory_path = (
                output_root / "al_inventory" / "inventory" / "USA_AL_inventory.csv"
            )
            self.assertTrue(inventory_path.is_file())
            inventory = pd.read_csv(inventory_path)
            self.assertEqual(len(inventory), 1)
            summary = json.loads(
                (output_root / "al_inventory" / "run_summary.json").read_text()
            )
            self.assertEqual(summary["status"], "inventory_complete")
            self.assertEqual(summary["images_discovered"], 1)
            self.assertEqual(summary["images_selected"], 1)

    def test_smoke_limit_reports_found_and_selected_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp) / "GEE Derived"
            jpg_root = _state_root(image_root) / "04033-1402" / "jpg"
            _write_rgb(jpg_root / "one.jpg")
            _write_rgb(jpg_root / "two.jpg")
            output_root = image_root / "inference_outputs"

            result = predict_state_tree.main(
                [
                    "--image-root",
                    str(image_root),
                    "--output-dir",
                    str(output_root),
                    "--run-id",
                    "al_inventory_limit",
                    "--states",
                    "AL",
                    "--max-images",
                    "1",
                    "--inventory-only",
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(
                (output_root / "al_inventory_limit" / "run_summary.json").read_text()
            )
            self.assertEqual(summary["images_discovered"], 2)
            self.assertEqual(summary["images_selected"], 1)


class TinyInferenceModel:
    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.forward_calls = 0

    def eval(self):
        return self

    def __call__(self, images):
        self.forward_calls += 1
        count = images.shape[0]
        torch = self.torch
        return {
            "bin_head": torch.zeros((count, 1), device=images.device),
            "shade_head": torch.zeros((count, 2), device=images.device),
            "score_head": torch.full((count, 1), 3.0, device=images.device),
            "veg_head": torch.full((count, 1), 4.0, device=images.device),
        }


class StateTreePredictionTests(unittest.TestCase):
    def test_model_setup_failure_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "GEE Derived"
            image_path = _state_root(image_root) / "04033-1402" / "jpg" / "one.jpg"
            _write_rgb(image_path)
            checkpoint = root / "best_mcmae_PyTorch_test.pt"
            checkpoint.touch()
            output_root = image_root / "inference_outputs"
            arguments = [
                "--checkpoint",
                str(checkpoint),
                "--image-root",
                str(image_root),
                "--output-dir",
                str(output_root),
                "--run-id",
                "setup_failure",
                "--states",
                "AL",
                "--device",
                "cuda",
            ]

            with patch.object(
                predict_state_tree,
                "_resolve_prediction_runtime",
                side_effect=RuntimeError("CUDA unavailable for test"),
            ):
                result = predict_state_tree.main(arguments)

            self.assertEqual(result, 1)
            summary = json.loads(
                (output_root / "setup_failure" / "run_summary.json").read_text()
            )
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["error_type"], "RuntimeError")

    def test_prediction_records_one_corrupt_image_and_resume_skips_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "GEE Derived"
            first = _state_root(image_root) / "04033-1402" / "jpg"
            second = _state_root(image_root) / "04033-1403" / "jpg"
            _write_rgb(first / "same.jpg", color=(1, 2, 3))
            _write_rgb(second / "same.jpg", color=(4, 5, 6))
            corrupt = second / "bad.jpg"
            corrupt.write_text("not an image")

            checkpoint = root / "best_mcmae_PyTorch_test.pt"
            checkpoint.touch()
            output_root = image_root / "inference_outputs"
            model = TinyInferenceModel()
            bundle = SimpleNamespace(
                model=model,
                checkpoint_path=checkpoint,
                threshold_path=root / "thresholds_best_mcmae.csv",
                run_tag="PyTorch_test",
                variant="best_mcmae",
                bin_names=["sports_field"],
                thresholds={"sports_field": 0.5},
                img_size=(16, 16),
                model_config={
                    "torch_model_config": {
                        "torchgeo_model_name": "swin_v2_b",
                        "torchgeo_weight": "Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS",
                    }
                },
            )
            arguments = [
                "--checkpoint",
                str(checkpoint),
                "--image-root",
                str(image_root),
                "--output-dir",
                str(output_root),
                "--run-id",
                "al_smoke",
                "--states",
                "AL",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--num-workers",
                "0",
                "--no-pin-memory",
            ]

            with patch.object(
                predict_state_tree,
                "_resolve_prediction_runtime",
                return_value=("cpu", bundle),
            ):
                result = predict_state_tree.main(arguments)

            self.assertEqual(result, 0)
            state_root = output_root / "al_smoke" / "states" / "USA_AL"
            predictions = pd.read_csv(
                state_root / "predictions_USA_AL_part_00001.csv"
            )
            failures = pd.read_csv(state_root / "failures_USA_AL_part_00001.csv")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(len(failures), 1)
            self.assertEqual(
                predictions.columns[-3:].tolist(),
                ["state_code", "park_code", "image_relative_path"],
            )
            self.assertEqual(predictions["image_filename"].tolist(), ["same.jpg", "same.jpg"])
            self.assertTrue(predictions["image_relative_path"].is_unique)
            self.assertEqual(failures.loc[0, "image_filename"], "bad.jpg")
            summary = json.loads((state_root / "state_summary.json").read_text())
            self.assertEqual(summary["images_attempted"], 3)
            self.assertEqual(summary["images_succeeded"], 2)
            self.assertEqual(summary["images_failed"], 1)
            calls_after_first_run = model.forward_calls

            with patch.object(
                predict_state_tree,
                "_resolve_prediction_runtime",
                return_value=("cpu", bundle),
            ):
                resumed = predict_state_tree.main([*arguments, "--resume"])

            self.assertEqual(resumed, 0)
            self.assertEqual(model.forward_calls, calls_after_first_run)


if __name__ == "__main__":
    unittest.main()
