#!/usr/bin/env python3
"""Build rated-image Drive manifests and cache the matching images locally."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drive_download import (  # noqa: E402
    attach_drive_ids,
    build_drive_manifest,
    build_rated_download_manifest,
    download_images,
)
from src.drive_utils import (  # noqa: E402
    get_drive,
    list_files_in_folder,
    load_drive_environment,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _run_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "run tag must contain only letters, numbers, dots, underscores, or hyphens"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List the existing flat Google Drive image folder, attach Drive IDs "
            "to the canonical file list, select included survey images, and "
            "cache one file per unique rated image."
        )
    )
    parser.add_argument("--survey-csv", required=True, help="Raw or cleaned survey CSV")
    parser.add_argument("--filelist-csv", required=True, help="Canonical filelist CSV")
    parser.add_argument(
        "--folder-id",
        help="Drive folder ID; defaults to GOOGLE_DRIVE_FOLDER_ID from .env",
    )
    parser.add_argument("--cache-dir", default="data/cache/images")
    parser.add_argument("--output-dir", default="data/interim")
    parser.add_argument(
        "--run-tag",
        type=_run_tag,
        default=datetime.now().strftime("%m%d_%H%M%S"),
    )
    parser.add_argument("--survey-image-col", default="Image Name")
    parser.add_argument("--filelist-image-col", default="ImageName")
    parser.add_argument(
        "--mime-prefix",
        help="Drive MIME prefix; defaults to GOOGLE_DRIVE_MIME_PREFIX or image/",
    )
    parser.add_argument(
        "--include-shared-drives",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--client-secrets", help="Override OAuth client secrets path")
    parser.add_argument("--credentials-cache", help="Override OAuth token cache path")
    parser.add_argument(
        "--command-line-auth",
        action="store_true",
        help="Use PyDrive2 command-line OAuth instead of the default browser flow",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("first", "error"),
        default="first",
        help="How to handle duplicate normalized filenames in Drive",
    )
    parser.add_argument("--limit", type=int, help="Download only the first N unique images")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Build and save manifests without downloading image bytes",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit before download if any included image lacks a Drive ID",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds cannot be negative")

    load_drive_environment()
    folder_id = args.folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    if not folder_id:
        parser.error("set --folder-id or GOOGLE_DRIVE_FOLDER_ID in the project .env")
    mime_prefix = (
        args.mime_prefix
        if args.mime_prefix is not None
        else os.getenv("GOOGLE_DRIVE_MIME_PREFIX", "image/")
    )
    include_shared = (
        args.include_shared_drives
        if args.include_shared_drives is not None
        else _env_bool("GOOGLE_DRIVE_INCLUDE_SHARED", True)
    )

    survey_path = Path(args.survey_csv)
    filelist_path = Path(args.filelist_csv)
    for label, path in (("survey", survey_path), ("filelist", filelist_path)):
        if not path.is_file():
            parser.error(f"{label} CSV does not exist: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "drive_manifest": output_dir / f"drive_manifest_{args.run_tag}.csv",
        "filelist_with_ids": output_dir
        / f"filelist_{args.run_tag}_with_drive_fileid.csv",
        "survey_with_ids": output_dir / f"survey_with_fileid_{args.run_tag}.csv",
        "summary": output_dir / f"drive_download_summary_{args.run_tag}.json",
    }

    print("Authenticating to Google Drive...")
    drive = get_drive(
        use_local_server=not args.command_line_auth,
        client_secrets_path=args.client_secrets,
        credentials_cache_path=args.credentials_cache,
    )
    print("Listing direct image children of the Drive folder...")
    files = list_files_in_folder(
        drive,
        folder_id,
        mime_prefix=mime_prefix,
        include_shared_drives=include_shared,
    )
    drive_manifest, drive_summary = build_drive_manifest(
        files,
        duplicate_policy=args.duplicate_policy,
    )
    filelist_with_ids, filelist_summary = attach_drive_ids(
        pd.read_csv(filelist_path),
        drive_manifest,
        filelist_image_col=args.filelist_image_col,
    )
    row_manifest, download_manifest, survey_summary = build_rated_download_manifest(
        pd.read_csv(survey_path),
        filelist_with_ids,
        survey_image_col=args.survey_image_col,
    )

    drive_manifest.to_csv(paths["drive_manifest"], index=False)
    filelist_with_ids.to_csv(paths["filelist_with_ids"], index=False)
    row_manifest.to_csv(paths["survey_with_ids"], index=False)

    print("Drive listing:", drive_summary)
    print("Filelist join:", filelist_summary)
    print("Included survey join:", survey_summary)
    if drive_summary["duplicate_names"]:
        print(
            "WARNING: duplicate Drive filenames were resolved by keeping the first. "
            "Use --duplicate-policy error for strict handling."
        )
    stop_for_missing = False
    if survey_summary["missing_unique_images"]:
        print(
            "WARNING: some included survey images do not have a Drive ID. "
            "Inspect the saved survey manifest."
        )
        if args.fail_on_missing:
            print("Stopping because --fail-on-missing was supplied.")
            stop_for_missing = True

    download_summary: dict[str, object]
    if stop_for_missing:
        requested = int(
            min(len(download_manifest), args.limit)
            if args.limit is not None
            else len(download_manifest)
        )
        download_summary = {
            "requested": requested,
            "downloaded": 0,
            "skipped_existing": 0,
            "failed": 0,
            "failures": [],
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "manifest_only": bool(args.manifest_only),
            "aborted_missing_drive_ids": True,
        }
    elif args.manifest_only:
        download_summary = {
            "requested": int(
                min(len(download_manifest), args.limit)
                if args.limit is not None
                else len(download_manifest)
            ),
            "downloaded": 0,
            "skipped_existing": 0,
            "failed": 0,
            "failures": [],
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "manifest_only": True,
        }
        print("Manifest-only mode: image download skipped.")
    else:
        download_summary = download_images(
            drive,
            download_manifest,
            args.cache_dir,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
            limit=args.limit,
        )
        download_summary["manifest_only"] = False
        print("Download:", {k: v for k, v in download_summary.items() if k != "failures"})

    summary = {
        "run_tag": args.run_tag,
        "inputs": {
            "survey_csv": str(survey_path.resolve()),
            "filelist_csv": str(filelist_path.resolve()),
            "folder_id": folder_id,
            "mime_prefix": mime_prefix,
            "include_shared_drives": include_shared,
        },
        "drive": drive_summary,
        "filelist": filelist_summary,
        "survey": survey_summary,
        "download": download_summary,
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Saved manifests and summary:")
    for path in paths.values():
        print(f"  {path.resolve()}")
    if stop_for_missing:
        return 2
    return 1 if int(download_summary["failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
