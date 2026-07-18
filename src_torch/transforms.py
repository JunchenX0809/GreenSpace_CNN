"""PyTorch image transforms that mirror current TensorFlow input behavior."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from PIL import Image

from src.augmentation import AUG_PARAMS
from src_torch.config import TORCH_DATA_CONFIG


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "constructing image transforms."
        ) from exc
    return torch


def _require_torchvision_functional() -> Any:
    try:
        import torchvision.transforms.functional as functional
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is not installed. Install the repo requirements.txt before "
            "using PyTorch image augmentations."
        ) from exc
    return functional


def _pil_to_unit_tensor(img: Image.Image, img_size: tuple[int, int]) -> Any:
    torch = _require_torch()
    rgb = img.convert("RGB").resize(img_size, Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)


def _augment_unit_tensor(tensor: Any) -> Any:
    """Apply NB03-style train augmentation to a CHW tensor in [0, 1]."""

    torch = _require_torch()
    functional = _require_torchvision_functional()

    k = int(torch.randint(0, 4, ()).item())
    tensor = torch.rot90(tensor, k, dims=(-2, -1))
    if bool((torch.rand(()) < 0.5).item()):
        tensor = torch.flip(tensor, dims=(-1,))
    if bool((torch.rand(()) < 0.5).item()):
        tensor = torch.flip(tensor, dims=(-2,))

    brightness_deltas = torch.tensor(AUG_PARAMS["brightness_deltas"], dtype=tensor.dtype, device=tensor.device)
    contrast_factors = torch.tensor(AUG_PARAMS["contrast_factors"], dtype=tensor.dtype, device=tensor.device)
    saturation_factors = torch.tensor(AUG_PARAMS["saturation_factors"], dtype=tensor.dtype, device=tensor.device)

    tensor = tensor + brightness_deltas[torch.randint(0, len(brightness_deltas), ())]
    tensor = functional.adjust_contrast(tensor, float(contrast_factors[torch.randint(0, len(contrast_factors), ())]))
    tensor = functional.adjust_saturation(tensor, float(saturation_factors[torch.randint(0, len(saturation_factors), ())]))
    hue_max = float(AUG_PARAMS["hue_max_delta"])
    hue_delta = float(torch.empty(()).uniform_(-hue_max, hue_max).item())
    tensor = functional.adjust_hue(tensor.clamp(0.0, 1.0), hue_delta)
    return tensor.clamp(0.0, 1.0)


def tf_parity_image_transform(
    img_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
    augment: bool = False,
) -> Callable[[Image.Image], Any]:
    """Return the current NB03-equivalent image transform.

    TensorFlow NB03 decodes RGB images, casts to float32, and scales pixels to
    [0, 1]. Cached images are already 512x512 RGB, but resize is kept explicit
    so the transform contract is stable.
    """

    def transform(img: Image.Image) -> Any:
        tensor = _pil_to_unit_tensor(img, img_size)
        return _augment_unit_tensor(tensor) if augment else tensor

    return transform


def rgb_255_image_transform(
    img_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
    augment: bool = False,
) -> Callable[[Image.Image], Any]:
    """Return RGB tensor in 0..255 scale for TorchGeo weight transforms."""

    def transform(img: Image.Image) -> Any:
        tensor = _pil_to_unit_tensor(img, img_size)
        if augment:
            tensor = _augment_unit_tensor(tensor)
        return tensor * 255.0

    return transform


def build_image_transform(
    img_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
    mode: str = TORCH_DATA_CONFIG["image_transform"],
    backbone_preprocess: Callable[[Any], Any] | None = TORCH_DATA_CONFIG["backbone_preprocess"],
    augment: bool = False,
) -> Callable[[Image.Image], Any]:
    """Build an image transform for the current PyTorch smoke stage."""

    if mode == "tf_parity":
        base_transform = tf_parity_image_transform(img_size=img_size, augment=augment)
    elif mode == "rgb_255":
        base_transform = rgb_255_image_transform(img_size=img_size, augment=augment)
    else:
        raise ValueError(f"Unsupported image transform mode: {mode!r}")

    if backbone_preprocess is None:
        return base_transform

    def transform(img: Image.Image) -> Any:
        return backbone_preprocess(base_transform(img))

    return transform


def tensor_summary(tensor: Any) -> dict[str, Any]:
    """Return lightweight tensor diagnostics for notebook smoke checks."""

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }
