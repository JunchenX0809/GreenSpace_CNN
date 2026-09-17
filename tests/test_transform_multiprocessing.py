from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src_torch.data import (
    ResilientImagePathTorchDataset,
    _collate_resilient_image_batch,
)
from src_torch.transforms import build_image_transform


class TransformMultiprocessingTests(unittest.TestCase):
    def test_inference_transform_survives_pickle_round_trip(self) -> None:
        import torch

        image = Image.new("RGB", (3, 2), color=(10, 20, 30))
        for mode, expected in (
            ("rgb_255", torch.tensor([10.0, 20.0, 30.0])),
            ("tf_parity", torch.tensor([10.0, 20.0, 30.0]) / 255.0),
        ):
            with self.subTest(mode=mode):
                transform = build_image_transform(
                    img_size=(3, 2),
                    mode=mode,
                    augment=False,
                )
                restored = pickle.loads(pickle.dumps(transform))

                actual = restored(image)

                self.assertEqual(tuple(actual.shape), (3, 2, 3))
                torch.testing.assert_close(actual[:, 0, 0], expected)

    def test_resilient_loader_works_with_spawned_workers(self) -> None:
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_paths = [root / "one.jpg", root / "two.jpg"]
            for index, path in enumerate(image_paths, start=1):
                Image.new("RGB", (4, 4), color=(index, index, index)).save(path)

            dataset = ResilientImagePathTorchDataset(
                image_paths,
                img_size=(4, 4),
                image_transform="rgb_255",
            )
            loader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                collate_fn=_collate_resilient_image_batch,
                multiprocessing_context="spawn",
            )

            batches = list(loader)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["failures"], [])
        self.assertEqual(tuple(batches[0]["images"].shape), (2, 3, 4, 4))


if __name__ == "__main__":
    unittest.main()
