from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src_torch.artifacts import save_training_curves, save_training_metric_curves


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = {
            "epoch": [1, 2, 3],
            "loss_total": [1.0, 0.8, 0.7],
            "val_loss_total": [1.1, 0.9, 0.8],
            "val_metric_bin_head_weighted_pr_auc": [0.5, 0.7, 0.6],
            "metric_score_mae": [0.9, 0.7, 0.6],
            "val_metric_score_mae": [1.0, 0.8, 0.7],
            "metric_veg_mae": [0.8, 0.6, 0.5],
            "val_metric_veg_mae": [0.9, 0.7, 0.6],
        }

    def test_metric_plot_does_not_require_train_pr_auc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metric_curves.png"
            actual = save_training_metric_curves(
                self.history,
                output_path,
                warmup_epochs=1,
            )
            self.assertEqual(actual, output_path)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_standard_plot_also_saves_metric_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            actual = save_training_curves(self.history, run_dir, warmup_epochs=1)
            self.assertEqual(actual, run_dir / "training_curves.png")
            self.assertTrue((run_dir / "training_curves.png").is_file())
            self.assertTrue((run_dir / "training_metric_curves.png").is_file())


if __name__ == "__main__":
    unittest.main()
