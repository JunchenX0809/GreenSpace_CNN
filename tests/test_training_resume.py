from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src_torch.training import (
    MaeGuardrailControl,
    PlateauTrainingControl,
    capture_training_rng_state,
    load_training_resume_checkpoint,
    restore_training_rng_state,
    run_persistent_warmup_finetune,
    set_torch_seed,
    split_frame_signature,
)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)

    def forward(self, value):  # pragma: no cover - epoch functions are mocked.
        return self.head(self.backbone(value))

    def metadata(self) -> dict[str, str]:
        return {"model_name": "tiny-test-model"}


class DummyLoader:
    def __init__(self) -> None:
        self.sampler = SimpleNamespace(generator=torch.Generator().manual_seed(37))


class ResumeStateTests(unittest.TestCase):
    def test_split_signature_ignores_host_image_paths(self) -> None:
        first = pd.DataFrame(
            {"image_path": ["/mac/a.jpg"], "image_filename": ["a.jpg"], "label": [1]}
        )
        second = pd.DataFrame(
            {"image_path": ["C:/win/a.jpg"], "image_filename": ["a.jpg"], "label": [1]}
        )
        changed = second.assign(label=0)
        self.assertEqual(split_frame_signature(first), split_frame_signature(second))
        self.assertNotEqual(split_frame_signature(first), split_frame_signature(changed))

    def test_rng_state_round_trip(self) -> None:
        set_torch_seed(37)
        loader = DummyLoader()
        state = capture_training_rng_state(loader)
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1)),
            torch.rand(1, generator=loader.sampler.generator).item(),
        )
        restore_training_rng_state(state, loader)
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1)),
            torch.rand(1, generator=loader.sampler.generator).item(),
        )
        self.assertEqual(expected, actual)

    def test_training_control_state_round_trip(self) -> None:
        model = TinyTrainingModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        control = PlateauTrainingControl("training_combo")
        control.update(model, optimizer, 1, {"training_combo": 0.6})
        restored = PlateauTrainingControl("training_combo")
        restored.load_state_dict(control.state_dict())
        self.assertEqual(restored.best, 0.6)
        self.assertEqual(restored.best_epoch, 1)
        self.assertIsNotNone(restored.best_state_dict)

        guardrail = MaeGuardrailControl()
        guardrail.update(1, {"metric_score_mae": 1.0, "metric_veg_mae": 1.2})
        restored_guardrail = MaeGuardrailControl()
        restored_guardrail.load_state_dict(guardrail.state_dict())
        self.assertEqual(restored_guardrail.best_mc_mae, 1.1)

    def test_runner_resumes_after_last_completed_warmup_epoch(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": ["ignored.jpg"],
                "image_filename": ["image.jpg"],
                "label_p": [1.0],
            }
        )
        schema = SimpleNamespace(binary_cols=["label_p"], bin_names=["label"])
        train_metrics = {
            "loss_total": 1.0,
            "metric_score_mae": 1.0,
            "metric_veg_mae": 1.0,
        }
        val_metrics = {
            "loss_total": 0.9,
            "metric_score_mae": 0.9,
            "metric_veg_mae": 1.1,
            "metric_score_veg_mae_mean": 1.0,
            "metric_bin_head_weighted_pr_auc": 0.5,
        }
        data_config = {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "use_oversampling": False,
            "use_augmentation": False,
        }
        training_config = {
            "test_run_mode": True,
            "test_warmup_epochs": 3,
            "early_stopping_patience": 1,
            "device": "cpu",
        }
        loader = DummyLoader()

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src_torch.training.load_split_df", return_value=frame
        ), patch(
            "src_torch.training.resolve_split_schema", return_value=schema
        ), patch(
            "src_torch.training.make_train_dataloader", return_value=(loader, None)
        ), patch(
            "src_torch.training.make_eval_dataloader", return_value=object()
        ), patch(
            "src_torch.training.build_torchgeo_model",
            side_effect=lambda **_kwargs: TinyTrainingModel(),
        ), patch(
            "src_torch.training.save_training_curves", return_value=None
        ), patch(
            "src_torch.training.train_one_epoch"
        ) as train_epoch, patch(
            "src_torch.training.evaluate_one_epoch", return_value=val_metrics
        ):
            train_epoch.side_effect = [
                train_metrics,
                train_metrics,
                RuntimeError("interrupted fine-tune"),
            ]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_persistent_warmup_finetune(
                    run_tag="PyTorch_resume_unit",
                    training_config=training_config,
                    data_config=data_config,
                    runs_root=Path(temp_dir),
                )

            last_path = (
                Path(temp_dir)
                / "PyTorch_resume_unit"
                / "last_PyTorch_resume_unit.pt"
            )
            interrupted = load_training_resume_checkpoint(last_path)
            interrupted_state = interrupted["training_state"]
            self.assertEqual(interrupted_state["phase"], "warmup")
            self.assertEqual(interrupted_state["phase_epoch"], 2)
            self.assertEqual(interrupted_state["global_epoch"], 2)
            self.assertTrue(interrupted_state["phase_complete"])

            train_epoch.side_effect = None
            train_epoch.return_value = train_metrics
            result = run_persistent_warmup_finetune(
                training_config=training_config,
                data_config=data_config,
                resume_from=last_path,
            )

            self.assertTrue(result["resumed"])
            self.assertEqual(
                result["history"]["phase"],
                ["warmup", "warmup", "finetune"],
            )
            completed = load_training_resume_checkpoint(last_path)
            self.assertTrue(completed["training_state"]["run_complete"])
            self.assertFalse(last_path.with_name(f"{last_path.name}.tmp").exists())

    def test_runner_resumes_inside_finetuning_phase(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": ["ignored.jpg"],
                "image_filename": ["image.jpg"],
                "label_p": [1.0],
            }
        )
        schema = SimpleNamespace(binary_cols=["label_p"], bin_names=["label"])
        train_metrics = {
            "loss_total": 1.0,
            "metric_score_mae": 1.0,
            "metric_veg_mae": 1.0,
        }
        val_metrics = {
            "loss_total": 0.9,
            "metric_score_mae": 0.9,
            "metric_veg_mae": 1.1,
            "metric_score_veg_mae_mean": 1.0,
            "metric_bin_head_weighted_pr_auc": 0.5,
        }
        data_config = {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "use_oversampling": False,
            "use_augmentation": False,
        }
        training_config = {
            "test_run_mode": True,
            "test_finetune_epochs": 2,
            "device": "cpu",
        }
        loader = DummyLoader()

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src_torch.training.load_split_df", return_value=frame
        ), patch(
            "src_torch.training.resolve_split_schema", return_value=schema
        ), patch(
            "src_torch.training.make_train_dataloader", return_value=(loader, None)
        ), patch(
            "src_torch.training.make_eval_dataloader", return_value=object()
        ), patch(
            "src_torch.training.build_torchgeo_model",
            side_effect=lambda **_kwargs: TinyTrainingModel(),
        ), patch(
            "src_torch.training.save_training_curves", return_value=None
        ), patch(
            "src_torch.training.train_one_epoch"
        ) as train_epoch, patch(
            "src_torch.training.evaluate_one_epoch", return_value=val_metrics
        ):
            train_epoch.side_effect = [
                train_metrics,
                train_metrics,
                RuntimeError("interrupted second fine-tune epoch"),
            ]
            with self.assertRaisesRegex(RuntimeError, "second fine-tune"):
                run_persistent_warmup_finetune(
                    run_tag="PyTorch_finetune_resume_unit",
                    training_config=training_config,
                    data_config=data_config,
                    runs_root=Path(temp_dir),
                )

            last_path = (
                Path(temp_dir)
                / "PyTorch_finetune_resume_unit"
                / "last_PyTorch_finetune_resume_unit.pt"
            )
            interrupted = load_training_resume_checkpoint(last_path)
            interrupted_state = interrupted["training_state"]
            self.assertEqual(interrupted_state["phase"], "finetune")
            self.assertEqual(interrupted_state["phase_epoch"], 1)
            self.assertEqual(interrupted_state["global_epoch"], 2)
            self.assertFalse(interrupted_state["phase_complete"])
            self.assertIsInstance(interrupted_state["guardrail_state"], dict)

            train_epoch.side_effect = None
            train_epoch.return_value = train_metrics
            result = run_persistent_warmup_finetune(
                training_config=training_config,
                data_config=data_config,
                resume_from=last_path,
            )

            self.assertEqual(
                result["history"]["phase"],
                ["warmup", "finetune", "finetune"],
            )
            self.assertEqual(result["history"]["epoch"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
