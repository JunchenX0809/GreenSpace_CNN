"""TorchGeo API discovery helpers for the PyTorch experiment branch.

This module intentionally avoids model construction and weight downloads. It
only inspects local package availability and, when TorchGeo is installed,
available model/weight metadata.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


TORCHGEO_MODELS_DOC = "https://docs.torchgeo.org/en/stable/api/models.html"

# Doc-based candidates from the TorchGeo 0.9.0 models page. These are not a
# final selection; the installed TorchGeo metadata still needs to be inspected.
DOC_RGB_CANDIDATES = [
    {
        "weight": "ResNet50_Weights.FMOW_RGB_GASSL",
        "family": "sensor-agnostic",
        "channels": 3,
        "why_check": "RGB remote-sensing weight listed in the sensor-agnostic section.",
    },
    {
        "weight": "ResNet18_Weights.SENTINEL2_RGB_MOCO",
        "family": "Sentinel-2",
        "channels": 3,
        "why_check": "3-channel Sentinel-2 RGB ResNet candidate.",
    },
    {
        "weight": "ResNet18_Weights.SENTINEL2_RGB_SECO",
        "family": "Sentinel-2",
        "channels": 3,
        "why_check": "3-channel Sentinel-2 RGB ResNet candidate.",
    },
    {
        "weight": "ResNet50_Weights.SENTINEL2_MI_RGB_SATLAS",
        "family": "Sentinel-2 / Satlas",
        "channels": 3,
        "why_check": "3-channel Satlas RGB ResNet candidate.",
    },
    {
        "weight": "ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS",
        "family": "Sentinel-2 / Satlas",
        "channels": 3,
        "why_check": "3-channel Satlas RGB ResNet candidate.",
    },
    {
        "weight": "ResNet50_Weights.SENTINEL2_RGB_MOCO",
        "family": "Sentinel-2",
        "channels": 3,
        "why_check": "3-channel Sentinel-2 RGB ResNet candidate.",
    },
    {
        "weight": "ResNet50_Weights.SENTINEL2_RGB_SECO",
        "family": "Sentinel-2",
        "channels": 3,
        "why_check": "3-channel Sentinel-2 RGB ResNet candidate.",
    },
    {
        "weight": "EarthLoc_Weights.SENTINEL2_RESNET50",
        "family": "Sentinel-2",
        "channels": 3,
        "why_check": "3-channel Sentinel-2 ResNet50-style entry in the docs.",
    },
]


def package_status(package_names: list[str] | None = None) -> list[dict[str, str]]:
    """Return import/version status for packages needed by TorchGeo probing."""

    package_names = package_names or ["torch", "torchvision", "torchgeo", "timm"]
    rows = []
    for name in package_names:
        try:
            import_module(name)
        except Exception as exc:
            rows.append(
                {
                    "package": name,
                    "available": "no",
                    "version": "",
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            continue

        try:
            pkg_version = version(name)
        except PackageNotFoundError:
            pkg_version = "unknown"

        rows.append({"package": name, "available": "yes", "version": pkg_version, "error": ""})
    return rows


def torchgeo_available() -> bool:
    """Return whether TorchGeo can be imported locally."""

    return any(row["package"] == "torchgeo" and row["available"] == "yes" for row in package_status(["torchgeo"]))


def list_local_torchgeo_models() -> dict[str, Any]:
    """List locally registered TorchGeo model names if TorchGeo is installed."""

    try:
        models = import_module("torchgeo.models")
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "models": []}

    try:
        names = models.list_models()
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}", "models": []}

    return {"available": True, "error": "", "models": list(names)}


def inspect_local_weight_enum(model_name: str) -> dict[str, Any]:
    """Inspect a TorchGeo weight enum without instantiating a model."""

    try:
        models = import_module("torchgeo.models")
        weights_enum = models.get_model_weights(model_name)
    except Exception as exc:
        return {"model_name": model_name, "available": False, "error": f"{type(exc).__name__}: {exc}", "weights": []}

    rows = []
    for weight in weights_enum:
        meta = getattr(weight, "meta", {}) or {}
        transforms = getattr(weight, "transforms", None)
        rows.append(
            {
                "name": str(weight),
                "in_chans": meta.get("in_chans"),
                "bands": meta.get("bands"),
                "categories": meta.get("categories"),
                "transforms": repr(transforms),
            }
        )
    return {"model_name": model_name, "available": True, "error": "", "weights": rows}


def doc_rgb_candidates() -> list[dict[str, Any]]:
    """Return official-doc RGB candidates to check after installing TorchGeo."""

    return [dict(row, source=TORCHGEO_MODELS_DOC) for row in DOC_RGB_CANDIDATES]
