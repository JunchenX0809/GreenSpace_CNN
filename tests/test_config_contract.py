from __future__ import annotations

import unittest

from src.label_schema import MODEL_TASK_CONFIG
from src_torch.config import TORCH_TRAINING_CONFIG


class ConfigurationContractTests(unittest.TestCase):
    def test_active_training_controls_live_only_in_torch_config(self) -> None:
        legacy_training_keys = {
            "test_run_mode",
            "test_warmup_epochs",
            "test_finetune_epochs",
            "epochs_warmup",
            "epochs_finetune",
            "fine_tune_backbone",
        }

        self.assertTrue(legacy_training_keys.isdisjoint(MODEL_TASK_CONFIG))
        self.assertEqual(TORCH_TRAINING_CONFIG["warmup_epochs"], 5)
        self.assertEqual(TORCH_TRAINING_CONFIG["finetune_epochs"], 100)


if __name__ == "__main__":
    unittest.main()
