"""TorchGeo model builders for GreenSpace multi-task training."""

from __future__ import annotations

from typing import Any

from src_torch.config import TORCH_DATA_CONFIG, TORCH_MODEL_CONFIG


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "constructing Torch model objects."
        ) from exc
    return torch


def _require_torchgeo_model(model_name: str) -> tuple[Any, Any]:
    try:
        from torchgeo.models import (
            ResNet50_Weights,
            Swin_V2_B_Weights,
            resnet50,
            swin_v2_b,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TorchGeo is not installed. Install the repo requirements before "
            "constructing TorchGeo models."
        ) from exc

    supported = {
        "resnet50": (resnet50, ResNet50_Weights),
        "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights),
    }
    try:
        return supported[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported TorchGeo model {model_name!r}. "
            f"Supported models: {sorted(supported)}"
        ) from exc


def resolve_torchgeo_weight(model_name: str, weight_name: str) -> Any:
    """Resolve a configured TorchGeo weight for its matching model."""

    _, weights_enum = _require_torchgeo_model(model_name)
    enum_class_name, separator, enum_name = weight_name.partition(".")
    if not separator or enum_class_name != weights_enum.__name__:
        raise ValueError(
            f"Weight {weight_name!r} does not match model {model_name!r}; "
            f"expected {weights_enum.__name__}.<member>."
        )
    try:
        return getattr(weights_enum, enum_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown TorchGeo weight: {weight_name!r}") from exc


def _build_backbone_preprocess(weight: Any, preserve_input_resolution: bool) -> Any:
    """Return official preprocessing, optionally without its spatial transform."""

    torch = _require_torch()
    official_preprocess = weight.transforms
    if not preserve_input_resolution:
        return official_preprocess

    from torchvision.transforms.v2 import CenterCrop, Resize

    transforms = list(official_preprocess.children())
    spatial_types = (Resize, CenterCrop)
    if not transforms or not isinstance(transforms[0], spatial_types):
        raise ValueError(
            "Cannot preserve input resolution safely: the configured TorchGeo "
            "weight transform does not begin with a recognized Resize or "
            "CenterCrop operation."
        )
    return torch.nn.Sequential(*transforms[1:])


class GreenSpaceTorchGeoModel(_require_torch().nn.Module):
    """Configured TorchGeo backbone with GreenSpace multi-task heads."""

    def __init__(
        self,
        model_name: str = TORCH_MODEL_CONFIG["torchgeo_model_name"],
        weight_name: str = TORCH_MODEL_CONFIG["torchgeo_weight"],
        load_pretrained_weights: bool = TORCH_MODEL_CONFIG["load_pretrained_weights"],
        preserve_input_resolution: bool = TORCH_MODEL_CONFIG["preserve_input_resolution"],
        input_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
        num_binary: int = TORCH_MODEL_CONFIG["num_binary"],
        num_shade: int = TORCH_MODEL_CONFIG["num_shade"],
        score_output_range: tuple[float, float] = TORCH_MODEL_CONFIG["score_output_range"],
        veg_output_range: tuple[float, float] = TORCH_MODEL_CONFIG["veg_output_range"],
    ) -> None:
        super().__init__()
        torch = _require_torch()
        model_builder, _ = _require_torchgeo_model(model_name)
        self.model_name = model_name
        self.weight = resolve_torchgeo_weight(model_name, weight_name)
        self.weight_name = weight_name
        self.load_pretrained_weights = load_pretrained_weights
        self.preserve_input_resolution = preserve_input_resolution
        self.input_size = input_size
        self.official_backbone_preprocess = self.weight.transforms
        self.backbone_preprocess = _build_backbone_preprocess(
            self.weight,
            preserve_input_resolution=preserve_input_resolution,
        )
        selected_weight = self.weight if load_pretrained_weights else None
        if model_name == "resnet50":
            self.backbone = model_builder(
                weights=selected_weight,
                num_classes=0,
                in_chans=int(self.weight.meta["in_chans"]),
            )
            feature_dim = int(getattr(self.backbone, "num_features", 2048))
        elif model_name == "swin_v2_b":
            self.backbone = model_builder(weights=selected_weight)
            feature_dim = int(self.backbone.head.in_features)
            self.backbone.head = torch.nn.Identity()
        else:  # Guarded by _require_torchgeo_model.
            raise ValueError(f"Unsupported TorchGeo model: {model_name!r}")

        self.feature_dim = feature_dim
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
            "model_name": self.model_name,
            "weight_name": self.weight_name,
            "load_pretrained_weights": self.load_pretrained_weights,
            "preserve_input_resolution": self.preserve_input_resolution,
            "requested_input_size": self.input_size,
            "feature_dim": self.feature_dim,
            "weight_meta": dict(self.weight.meta),
            "weight_transforms": repr(self.official_backbone_preprocess),
            "official_weight_transforms": repr(self.official_backbone_preprocess),
            "effective_weight_transforms": repr(self.backbone_preprocess),
        }


def build_torchgeo_model(
    model_name: str = TORCH_MODEL_CONFIG["torchgeo_model_name"],
    weight_name: str = TORCH_MODEL_CONFIG["torchgeo_weight"],
    load_pretrained_weights: bool = TORCH_MODEL_CONFIG["load_pretrained_weights"],
    preserve_input_resolution: bool = TORCH_MODEL_CONFIG["preserve_input_resolution"],
    input_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
) -> GreenSpaceTorchGeoModel:
    """Build the centrally configured TorchGeo multi-task model."""

    return GreenSpaceTorchGeoModel(
        model_name=model_name,
        weight_name=weight_name,
        load_pretrained_weights=load_pretrained_weights,
        preserve_input_resolution=preserve_input_resolution,
        input_size=input_size,
    )


def build_torchgeo_resnet50_forward_model(
    load_pretrained_weights: bool = TORCH_MODEL_CONFIG["load_pretrained_weights"],
    preserve_input_resolution: bool = TORCH_MODEL_CONFIG["preserve_input_resolution"],
    input_size: tuple[int, int] = TORCH_DATA_CONFIG["img_size"],
) -> GreenSpaceTorchGeoModel:
    """Build ResNet-50 for compatibility with the original smoke notebook."""

    return build_torchgeo_model(
        model_name="resnet50",
        weight_name="ResNet50_Weights.FMOW_RGB_GASSL",
        load_pretrained_weights=load_pretrained_weights,
        preserve_input_resolution=preserve_input_resolution,
        input_size=input_size,
    )


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
