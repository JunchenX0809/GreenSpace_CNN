#!/usr/bin/env python3
"""Regenerate the presentation-style epoch visual from a saved PyTorch run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_torch.artifacts import save_training_metric_curves  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing one training history and one model configuration",
    )
    parser.add_argument(
        "--output",
        help="Output PNG (default: <run-dir>/training_metric_curves.png)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser


def _one_matching_file(
    parser: argparse.ArgumentParser,
    run_dir: Path,
    pattern: str,
    description: str,
) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        parser.error(f"no {description} found in run directory: {run_dir}")
    if len(matches) > 1:
        parser.error(f"multiple {description} files found in {run_dir}")
    return matches[0]


def _load_json(parser: argparse.ArgumentParser, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"could not read {path}: {exc}")
    if not isinstance(payload, dict):
        parser.error(f"expected a JSON object in {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    # Resolve the saved history, configuration, and requested output location.
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"missing run directory: {run_dir}")
    history_path = _one_matching_file(
        parser,
        run_dir,
        "training_history_*.json",
        "training history",
    )
    config_path = _one_matching_file(
        parser,
        run_dir,
        "model_config_*.json",
        "model configuration",
    )
    output_path = (
        Path(args.output) if args.output else run_dir / "training_metric_curves.png"
    )
    if output_path.exists() and not args.overwrite:
        parser.error(f"output already exists; pass --overwrite to replace it: {output_path}")

    # Use the run's saved warm-up boundary rather than a current default.
    history = _load_json(parser, history_path)
    config = _load_json(parser, config_path)
    warmup_epochs = config.get("warmup_epochs")
    if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
        parser.error(f"invalid warmup_epochs in {config_path}")

    # Render only metrics that were actually recorded in the history.
    rendered_path = save_training_metric_curves(history, output_path, warmup_epochs)
    if rendered_path is None:
        parser.error("plot was not created; check that epochs exist and matplotlib is installed")
    print(f"Saved training metric curves: {rendered_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
