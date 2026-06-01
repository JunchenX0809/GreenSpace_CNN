"""PyTorch data-loading layer for existing NB02 split manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src_torch.config import EXPERIMENT_CONFIG, SPLIT_DIR, TORCH_DATA_CONFIG, resolve_label_cols
from src_torch.sampling import make_weighted_sampler
from src_torch.transforms import build_image_transform

REQUIRED_COLUMNS = (
    "image_path",
    "shade_class",
    "score_mean",
    "veg_mean",
)


@dataclass(frozen=True)
class SplitSchema:
    """Resolved label columns for one split manifest."""

    binary_cols: list[str]
    bin_names: list[str]


def torch_available() -> bool:
    """Return whether torch can be imported in the current environment."""

    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "constructing GreenSpaceTorchDataset or DataLoader objects."
        ) from exc
    return torch


def load_split_df(split: str, split_dir: Path = SPLIT_DIR) -> pd.DataFrame:
    """Load one current NB02 split manifest."""

    path = split_dir / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    return pd.read_csv(path)


def load_split_dfs(split_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Load train/val/test manifests from the current processed split folder."""

    return {split: load_split_df(split, split_dir=split_dir) for split in ("train", "val", "test")}


def resolve_split_schema(df: pd.DataFrame) -> SplitSchema:
    """Resolve binary labels using the same exclusion config as NB03."""

    schema = resolve_label_cols(df)
    excluded = set(EXPERIMENT_CONFIG.get("exclude_binary", []))
    binary_cols = [c for c in schema["binary_cols"] if c not in excluded]
    bin_names = [c[:-2] for c in binary_cols]
    return SplitSchema(binary_cols=binary_cols, bin_names=bin_names)


def validate_split_df(df: pd.DataFrame, schema: SplitSchema) -> None:
    """Validate the columns needed for the first PyTorch smoke test."""

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing.extend(c for c in schema.binary_cols if c not in df.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {missing}")
    if len(schema.binary_cols) != 7:
        raise ValueError(
            f"Expected 7 binary labels after exclusions, found {len(schema.binary_cols)}: "
            f"{schema.binary_cols}"
        )


def missing_image_paths(df: pd.DataFrame, limit: int | None = 10) -> list[str]:
    """Return missing image paths for quick data-health checks."""

    missing = [str(p) for p in df["image_path"].astype(str) if not Path(p).exists()]
    return missing if limit is None else missing[:limit]


def _load_rgb_tensor(path: str, transform):
    from PIL import Image

    with Image.open(path) as img:
        return transform(img)


class GreenSpaceTorchDataset:
    """Minimal Dataset-compatible object for one split manifest."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: SplitSchema | None = None,
        img_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
        image_transform: str = TORCH_DATA_CONFIG["image_transform"],
        augment: bool = False,
    ) -> None:
        self.torch = _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.schema = schema or resolve_split_schema(self.df)
        self.img_size = img_size
        self.image_transform = build_image_transform(img_size=img_size, mode=image_transform, augment=augment)
        self.augment = augment
        validate_split_df(self.df, self.schema)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        row = self.df.iloc[idx]
        image = _load_rgb_tensor(str(row["image_path"]), self.image_transform)

        bin_target = row[self.schema.binary_cols].fillna(0.0).astype(np.float32).to_numpy(copy=True)
        shade_target = int(np.clip(row["shade_class"], 0, 1))
        score_target = np.array([np.clip(float(row["score_mean"]), 1.0, 5.0)], dtype=np.float32)
        veg_target = np.array([np.clip(float(row["veg_mean"]), 1.0, 5.0)], dtype=np.float32)

        targets = {
            "bin_head": self.torch.from_numpy(bin_target),
            "shade_head": self.torch.tensor(shade_target, dtype=self.torch.long),
            "score_head": self.torch.from_numpy(score_target),
            "veg_head": self.torch.from_numpy(veg_target),
        }
        return image, targets


def make_dataset(
    split: str,
    split_dir: Path = SPLIT_DIR,
    image_transform: str = TORCH_DATA_CONFIG["image_transform"],
    augment: bool = False,
) -> GreenSpaceTorchDataset:
    """Construct the smoke-test dataset for one split."""

    df = load_split_df(split, split_dir=split_dir)
    schema = resolve_split_schema(df)
    return GreenSpaceTorchDataset(df, schema=schema, image_transform=image_transform, augment=augment)


def make_dataloader(
    split: str,
    batch_size: int = TORCH_DATA_CONFIG["batch_size"],
    shuffle: bool = False,
    split_dir: Path = SPLIT_DIR,
    image_transform: str = TORCH_DATA_CONFIG["image_transform"],
    augment: bool = False,
):
    """Construct a DataLoader for one split."""

    _require_torch()
    from torch.utils.data import DataLoader

    dataset = make_dataset(split, split_dir=split_dir, image_transform=image_transform, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=TORCH_DATA_CONFIG["num_workers"],
        pin_memory=TORCH_DATA_CONFIG["pin_memory"],
    )


def make_train_dataloader(
    batch_size: int = TORCH_DATA_CONFIG["batch_size"],
    split_dir: Path = SPLIT_DIR,
    image_transform: str = "rgb_255",
    use_oversampling: bool = TORCH_DATA_CONFIG["use_oversampling"],
    use_augmentation: bool = TORCH_DATA_CONFIG["use_augmentation"],
    seed: int | None = None,
    return_plan: bool = False,
):
    """Construct the train DataLoader with optional NB03-style parity behavior."""

    _require_torch()
    from torch.utils.data import DataLoader

    df = load_split_df("train", split_dir=split_dir)
    schema = resolve_split_schema(df)
    dataset = GreenSpaceTorchDataset(
        df,
        schema=schema,
        image_transform=image_transform,
        augment=use_augmentation,
    )
    sampler = None
    plan = None
    shuffle = True
    if use_oversampling:
        sampler, plan = make_weighted_sampler(df, seed=37 if seed is None else seed)
        shuffle = sampler is None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=TORCH_DATA_CONFIG["num_workers"],
        pin_memory=TORCH_DATA_CONFIG["pin_memory"],
    )
    if return_plan:
        return loader, plan
    return loader


def make_eval_dataloader(
    split: str,
    batch_size: int = TORCH_DATA_CONFIG["batch_size"],
    split_dir: Path = SPLIT_DIR,
    image_transform: str = "rgb_255",
):
    """Construct a val/test DataLoader with no oversampling or augmentation."""

    if split == "train":
        raise ValueError("Use make_train_dataloader for train split behavior.")
    return make_dataloader(
        split,
        batch_size=batch_size,
        shuffle=False,
        split_dir=split_dir,
        image_transform=image_transform,
        augment=False,
    )


def tensor_shapes(image: Any, targets: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return shape tuples for a sample or batch."""

    shapes = {"image": tuple(image.shape)}
    shapes.update({name: tuple(value.shape) for name, value in targets.items()})
    return shapes
