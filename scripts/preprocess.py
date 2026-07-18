#!/usr/bin/env python3
"""Clean, aggregate, and split the rated GreenSpace survey data."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import run_preprocessing_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-csv", required=True, help="Raw or cleaned survey CSV")
    parser.add_argument(
        "--filelist-csv",
        help="Optional filelist CSV containing image_filename and drive_file_id",
    )
    parser.add_argument(
        "--image-dir",
        default=str(PROJECT_ROOT / "data" / "cache" / "images"),
        help="Cached rated-image directory",
    )
    parser.add_argument(
        "--run-tag",
        default=datetime.now().strftime("%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--interim-dir",
        default=str(PROJECT_ROOT / "data" / "interim"),
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional deterministic cached-image subset for a smoke run",
    )
    parser.add_argument(
        "--fail-on-missing-images",
        action="store_true",
        help="Stop instead of filtering uncached labeled images from the splits",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_preprocessing_pipeline(
        survey_path=args.survey_csv,
        image_dir=args.image_dir,
        run_tag=args.run_tag,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        filelist_path=args.filelist_csv,
        sample_size=args.sample_size,
        require_all_images=args.fail_on_missing_images,
    )

    print(f"Preprocessing complete: {result['run_tag']}")
    print("Inclusion:", result["inclusion_summary"])
    print("Selection:", result["selection_summary"])
    print("Splits:", result["split_summary"])
    print("Artifacts:")
    for name, path in result["paths"].items():
        if path is not None:
            print(f"  {name}: {Path(path).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
