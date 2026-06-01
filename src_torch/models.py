"""Minimal PyTorch model builders for forward-pass smoke tests."""

from __future__ import annotations

from typing import Any

from src_torch.config import TORCH_MODEL_CONFIG


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "constructing Torch model objects."
        ) from exc
    return torch


def _require_torchgeo_resnet50() -> tuple[Any, Any]:
    try:
        from torchgeo.models import ResNet50_Weights, resnet50
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TorchGeo is not installed. Install TorchGeo before running the "
            "TorchGeo forward-pass smoke test."
        ) from exc
    return resnet50, ResNet50_Weights


def resolve_resnet50_weight(weight_name: str = TORCH_MODEL_CONFIG["torchgeo_weight"]) -> Any:
    """Resolve the configured TorchGeo ResNet-50 weight enum."""

    _, weights_enum = _require_torchgeo_resnet50()
    enum_name = weight_name.split(".")[-1]
    try:
        return getattr(weights_enum, enum_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown ResNet50 weight: {weight_name!r}") from exc


class GreenSpaceTorchGeoResNet50(_require_torch().nn.Module):
    """TorchGeo ResNet-50 backbone with GreenSpace multi-task heads."""

    def __init__(
        self,
        weight_name: str = TORCH_MODEL_CONFIG["torchgeo_weight"],
        load_pretrained_weights: bool = TORCH_MODEL_CONFIG["load_pretrained_weights"],
        num_binary: int = TORCH_MODEL_CONFIG["num_binary"],
        num_shade: int = TORCH_MODEL_CONFIG["num_shade"],
        score_output_range: tuple[float, float] = TORCH_MODEL_CONFIG["score_output_range"],
        veg_output_range: tuple[float, float] = TORCH_MODEL_CONFIG["veg_output_range"],
    ) -> None:
        super().__init__()
        torch = _require_torch()
        resnet50, _ = _require_torchgeo_resnet50()
        self.weight = resolve_resnet50_weight(weight_name)
        self.weight_name = weight_name
        self.load_pretrained_weights = load_pretrained_weights
        self.backbone_preprocess = self.weight.transforms
        self.backbone = resnet50(
            weights=self.weight if load_pretrained_weights else None,
            num_classes=0,
            in_chans=int(self.weight.meta["in_chans"]),
        )
        feature_dim = int(getattr(self.backbone, "num_features", 2048))
        self.bin_head = torch.nn.Linear(feature_dim, num_binary)
        self.shade_head = torch.nn.Linear(feature_dim, num_shade)
        self.score_head_raw = torch.nn.Linear(feature_dim, 1)
        self.veg_head_raw = torch.nn.Linear(feature_dim, 1)
        self.score_output_range = score_output_range
        self.veg_output_range = veg_output_range

    @staticmethod
    def _bounded_sigmoid(raw: Any, output_range: tuple[float, float]) -> Any:
        torch = _require_torch()
        low, high = output_range
        return low + (high - low) * torch.sigmoid(raw)

    def preprocess_images(self, images: Any) -> Any:
        """Apply the selected TorchGeo weight transform to 0..255 RGB tensors."""

        return self.backbone_preprocess(images.clone())

    def forward(self, images: Any) -> dict[str, Any]:
        x = self.preprocess_images(images)
        features = self.backbone(x)
        return {
            "bin_head": self.bin_head(features),
            "shade_head": self.shade_head(features),
            "score_head": self._bounded_sigmoid(self.score_head_raw(features), self.score_output_range),
            "veg_head": self._bounded_sigmoid(self.veg_head_raw(features), self.veg_output_range),
        }

    def metadata(self) -> dict[str, Any]:
        """Return the selected TorchGeo backbone metadata for logging."""

        return {
            "model_name": TORCH_MODEL_CONFIG["torchgeo_model_name"],
            "weight_name": self.weight_name,
            "load_pretrained_weights": self.load_pretrained_weights,
            "weight_meta": dict(self.weight.meta),
            "weight_transforms": repr(self.weight.transforms),
        }


def build_torchgeo_resnet50_forward_model(
    load_pretrained_weights: bool = TORCH_MODEL_CONFIG["load_pretrained_weights"],
) -> GreenSpaceTorchGeoResNet50:
    """Build the configured TorchGeo ResNet-50 model for one-batch smoke tests."""

    return GreenSpaceTorchGeoResNet50(load_pretrained_weights=load_pretrained_weights)


def output_summary(outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return lightweight diagnostics for model outputs."""

    return {
        name: {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.detach().min()),
            "max": float(value.detach().max()),
        }
        for name, value in outputs.items()
    }
