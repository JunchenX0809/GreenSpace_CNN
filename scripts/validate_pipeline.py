#!/usr/bin/env python3
"""Validate GreenSpace data, images, model bundle, device, and output readiness."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--checkpoint", help="Explicit checkpoint that anchors the run bundle")
    selection.add_argument(
        "--run-dir",
        help="Explicit run directory containing one checkpoint for the preferred variant",
    )
    parser.add_argument(
        "--preferred-variant",
        choices=("best_mcmae", "best_prauc", "final"),
        default="best_mcmae",
        help="Checkpoint variant used with --run-dir (default: best_mcmae)",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Root containing processed/splits and cache/images",
    )
    parser.add_argument("--split-dir", help="Override the train/val/test manifest directory")
    parser.add_argument("--image-root", help="Override the labeled image directory")
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Validate inference readiness without requiring labeled split manifests",
    )
    parser.add_argument(
        "--inference-dir",
        help="Optional unlabeled image directory to validate for prediction",
    )
    parser.add_argument(
        "--output-dir",
        help="Prediction output directory to check (default: predictions/)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser


def _checkpoint_from_run_dir(
    parser: argparse.ArgumentParser,
    run_dir: Path,
    variant: str,
) -> Path:
    if not run_dir.is_dir():
        parser.error(f"missing run directory: {run_dir}")
    candidates = sorted(run_dir.glob(f"{variant}_*.pt"))
    if not candidates:
        parser.error(f"no {variant} checkpoint found in run directory: {run_dir}")
    if len(candidates) > 1:
        parser.error(
            f"multiple {variant} checkpoints found in {run_dir}; use --checkpoint explicitly"
        )
    return candidates[0]


def _run_check(
    results: list[CheckResult],
    name: str,
    operation: Callable[[], str],
) -> None:
    try:
        detail = operation()
    except Exception as exc:
        results.append(CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"))
    else:
        results.append(CheckResult(name=name, ok=True, detail=detail))


def resolve_prediction_output_root(output_root: str | Path | None = None) -> Path:
    from src_torch.config import resolve_prediction_output_root as resolve

    return resolve(output_root)


def resolve_device(device: str):
    from src_torch.training import resolve_device as resolve

    return resolve(device)


def load_run_bundle(checkpoint: Path, device=None):
    from src_torch.run_bundle import load_run_bundle as load

    return load(checkpoint, device=device)


def list_inference_image_paths(image_dir: Path):
    from src_torch.inference import list_inference_image_paths as list_paths

    return list_paths(image_dir)


def _check_environment() -> str:
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        raise RuntimeError(
            f"Python {platform.python_version()} is unsupported; use Python 3.11 or 3.12."
        )
    required_modules = ("numpy", "pandas", "PIL", "torch", "torchvision", "torchgeo")
    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name} ({exc})")
    if missing:
        raise RuntimeError(f"required imports failed: {missing}")
    return f"Python {platform.python_version()}; required imports available"


def _check_device(requested: str) -> tuple[object, str]:
    import torch

    normalized = requested.split(":", maxsplit=1)[0].lower()
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available.")
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, cuda:<index>, or mps")

    device = resolve_device(requested)
    probe = torch.zeros(1, device=device)
    del probe
    return device, f"selected {device}"


def _check_output_dir(output_dir: Path) -> str:
    candidate = output_dir.expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise NotADirectoryError(f"output path is not a directory: {candidate}")
        writable_target = candidate
    else:
        writable_target = candidate
        while not writable_target.exists() and writable_target != writable_target.parent:
            writable_target = writable_target.parent
        if not writable_target.is_dir():
            raise NotADirectoryError(
                f"no existing parent directory is available for: {candidate}"
            )
    if not os.access(writable_target, os.W_OK | os.X_OK):
        raise PermissionError(f"output location is not writable: {writable_target}")
    status = "exists" if candidate.exists() else f"creatable under {writable_target}"
    return f"{candidate} ({status})"


def _check_data(
    split_dir: Path,
    image_root: Path,
    expected_binary_cols: list[str],
) -> str:
    from src_torch.data import (
        load_split_df,
        missing_image_paths,
        resolve_split_schema,
        validate_split_df,
    )

    filenames_by_split: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        frame = load_split_df(split, split_dir=split_dir)
        schema = resolve_split_schema(frame)
        validate_split_df(frame, schema)
        if schema.binary_cols != expected_binary_cols:
            raise ValueError(
                f"{split} binary schema differs from checkpoint: "
                f"manifest={schema.binary_cols}, checkpoint={expected_binary_cols}"
            )

        filenames = frame["image_filename"].astype(str)
        normalized = filenames.str.casefold()
        if normalized.duplicated().any():
            duplicates = filenames[normalized.duplicated(keep=False)].head(5).tolist()
            raise ValueError(f"{split} contains duplicate image filenames: {duplicates}")
        filenames_by_split[split] = set(normalized)
        row_counts[split] = len(frame)

        missing = missing_image_paths(frame, limit=None, image_root=image_root)
        if missing:
            raise FileNotFoundError(
                f"{split} is missing {len(missing)} images under {image_root}; "
                f"first examples: {missing[:5]}"
            )

    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = filenames_by_split[left] & filenames_by_split[right]
        if shared:
            overlaps.append(f"{left}/{right}={len(shared)}")
    if overlaps:
        raise ValueError(f"split image overlap detected: {', '.join(overlaps)}")
    return (
        f"train={row_counts['train']}, val={row_counts['val']}, "
        f"test={row_counts['test']}; all images available"
    )


def main(argv: list[str] | None = None) -> int:
    # Resolve one explicit checkpoint bundle and the workflow paths to inspect.
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint = (
        Path(args.checkpoint).expanduser()
        if args.checkpoint
        else _checkpoint_from_run_dir(
            parser,
            Path(args.run_dir).expanduser(),
            args.preferred_variant,
        )
    )
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")

    data_root = Path(args.data_root).expanduser()
    split_dir = (
        Path(args.split_dir).expanduser()
        if args.split_dir
        else data_root / "processed" / "splits"
    )
    image_root = (
        Path(args.image_root).expanduser()
        if args.image_root
        else data_root / "cache" / "images"
    )
    output_dir = resolve_prediction_output_root(args.output_dir)

    # Collect independent readiness checks instead of stopping at the first failure.
    results: list[CheckResult] = []
    _run_check(results, "Environment", _check_environment)

    selected_device: object | None = None

    def device_operation() -> str:
        nonlocal selected_device
        selected_device, detail = _check_device(args.device)
        return detail

    _run_check(results, "Device", device_operation)

    bundle = None

    def bundle_operation() -> str:
        nonlocal bundle
        bundle = load_run_bundle(checkpoint, device=selected_device)
        return (
            f"{bundle.run_tag} ({bundle.variant}); {len(bundle.bin_names)} binary labels; "
            f"thresholds={bundle.threshold_path}"
        )

    _run_check(results, "Run bundle", bundle_operation)
    _run_check(results, "Output", lambda: _check_output_dir(output_dir))

    # Validate labeled manifests only when the selected workflow requires them.
    if not args.skip_data:
        if bundle is None:
            results.append(
                CheckResult(
                    name="Prepared data",
                    ok=False,
                    detail="run bundle must load before manifest schema can be compared",
                )
            )
        else:
            _run_check(
                results,
                "Prepared data",
                lambda: _check_data(split_dir, image_root, bundle.binary_cols),
            )

    # Optionally inspect the unlabeled image folder used for prediction.
    if args.inference_dir:
        inference_dir = Path(args.inference_dir).expanduser()
        _run_check(
            results,
            "Inference images",
            lambda: (
                f"{len(list_inference_image_paths(inference_dir))} supported images in "
                f"{inference_dir}"
            ),
        )

    # Print one concise pass/fail report and return a CI-friendly exit code.
    print("GreenSpace pipeline validation")
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")

    failures = [result for result in results if not result.ok]
    if failures:
        print(f"Validation failed: {len(failures)} check(s) need attention.")
        return 1
    print("Validation passed: the selected workflow is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
