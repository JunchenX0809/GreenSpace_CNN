"""Reusable Google Drive orchestration for the rated survey images.

This module extracts the tested flow from ``02_data_preprocessing.ipynb``:

1. normalize a direct-folder Google Drive listing;
2. attach Drive file IDs to the canonical project file list;
3. retain survey rows where ``include_tile == 'yes'``;
4. download each unique rated image into a local cache.

Authentication and low-level Drive access remain in :mod:`src.drive_utils`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from src.drive_utils import download_file_bytes
from src.preprocessing import (
    clean_survey_dataframe,
    filter_included_rows,
    normalize_name,
)


DownloadBytes = Callable[[Any, str], bytes]


def build_drive_manifest(
    files: Sequence[Mapping[str, Any]],
    *,
    duplicate_policy: str = "first",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize a Drive listing and resolve duplicate normalized filenames.

    ``duplicate_policy='first'`` preserves the historical notebook behavior and
    reports duplicate counts. Use ``'error'`` when ambiguous Drive names must
    stop the run.
    """

    if duplicate_policy not in {"first", "error"}:
        raise ValueError("duplicate_policy must be 'first' or 'error'")

    manifest = pd.DataFrame(files)
    required = {"file_id", "name"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        if manifest.empty:
            raise ValueError(
                "Drive manifest is empty. Check the folder ID, permissions, "
                "and MIME filter."
            )
        raise ValueError(f"Drive listing missing required fields: {missing}")

    manifest = manifest.copy()
    invalid = manifest["file_id"].isna() | manifest["name"].isna()
    if invalid.any():
        raise ValueError(
            f"Drive listing contains {int(invalid.sum())} rows without a name "
            "or file ID."
        )
    manifest["image_filename"] = (
        manifest["name"].astype(str).map(normalize_name)
    )

    duplicate_mask = manifest["image_filename"].duplicated(keep=False)
    duplicate_names = int(
        manifest.loc[duplicate_mask, "image_filename"].nunique()
    )
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_names and duplicate_policy == "error":
        examples = (
            manifest.loc[duplicate_mask, "image_filename"]
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Duplicate normalized filenames found in Drive. "
            f"Examples: {examples}"
        )
    if duplicate_names:
        manifest = manifest.drop_duplicates("image_filename", keep="first")

    summary = {
        "listed_files": int(len(files)),
        "manifest_images": int(len(manifest)),
        "duplicate_names": duplicate_names,
        "duplicate_rows": duplicate_rows,
    }
    return manifest.reset_index(drop=True), summary


def attach_drive_ids(
    filelist: pd.DataFrame,
    drive_manifest: pd.DataFrame,
    *,
    filelist_image_col: str = "ImageName",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach Drive IDs to the existing canonical ``filelist_*.csv`` schema."""

    if filelist_image_col not in filelist.columns:
        raise ValueError(
            f"Filelist missing image column {filelist_image_col!r}. "
            f"Available columns: {list(filelist.columns)}"
        )
    required = {"image_filename", "file_id"}
    missing = sorted(required.difference(drive_manifest.columns))
    if missing:
        raise ValueError(f"Drive manifest missing required columns: {missing}")
    if drive_manifest["image_filename"].duplicated().any():
        raise ValueError("Drive manifest must contain one row per image_filename")

    joined = filelist.copy()
    joined["image_filename"] = (
        joined[filelist_image_col].astype(str).map(normalize_name)
    )
    duplicate_filelist = joined["image_filename"].duplicated(keep=False)
    if duplicate_filelist.any():
        examples = (
            joined.loc[duplicate_filelist, "image_filename"]
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Filelist contains duplicate normalized image filenames. "
            f"Examples: {examples}"
        )

    joined = joined.merge(
        drive_manifest[["image_filename", "file_id"]],
        on="image_filename",
        how="left",
        validate="one_to_one",
    ).rename(columns={"file_id": "drive_file_id"})
    missing_ids = joined["drive_file_id"].isna()
    summary = {
        "filelist_rows": int(len(joined)),
        "filelist_matched": int((~missing_ids).sum()),
        "filelist_missing": int(missing_ids.sum()),
    }
    return joined, summary


def build_rated_download_manifest(
    survey: pd.DataFrame,
    filelist_with_ids: pd.DataFrame,
    *,
    survey_image_col: str = "Image Name",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build row-level and unique-image manifests for included survey images."""

    cleaned = survey.copy()
    if "image_filename" not in cleaned.columns:
        cleaned = clean_survey_dataframe(
            cleaned,
            image_col=survey_image_col,
        )
    included, inclusion_summary = filter_included_rows(cleaned)

    required = {"image_filename", "drive_file_id"}
    missing = sorted(required.difference(filelist_with_ids.columns))
    if missing:
        raise ValueError(f"Filelist with IDs missing required columns: {missing}")
    lookup_columns = ["image_filename", "drive_file_id"]
    if "ReviewFlag" in filelist_with_ids.columns:
        lookup_columns.append("ReviewFlag")
    lookup = filelist_with_ids[lookup_columns].copy()
    if lookup["image_filename"].duplicated().any():
        raise ValueError("Filelist with IDs must contain one row per image_filename")

    row_manifest = included.merge(
        lookup,
        on="image_filename",
        how="left",
        validate="many_to_one",
    )
    unique_manifest = row_manifest[
        ["image_filename", "drive_file_id"]
    ].drop_duplicates("image_filename", keep="first")

    missing_rows = row_manifest["drive_file_id"].isna()
    missing_images = unique_manifest["drive_file_id"].isna()
    downloadable = unique_manifest.loc[~missing_images].reset_index(drop=True)
    summary = {
        **inclusion_summary,
        "matched_survey_rows": int((~missing_rows).sum()),
        "missing_survey_rows": int(missing_rows.sum()),
        "missing_unique_images": int(missing_images.sum()),
        "downloadable_unique_images": int(len(downloadable)),
    }
    return row_manifest, downloadable, summary


def _safe_cache_filename(filename: str) -> str:
    value = str(filename).strip()
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"Unsafe image filename: {filename!r}")
    return value


def download_images(
    drive: Any,
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    *,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
    limit: int | None = None,
    fetch_bytes: DownloadBytes = download_file_bytes,
    sleep: Callable[[float], None] = time.sleep,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Cache unique manifest images with retries and atomic final writes."""

    required = {"image_filename", "drive_file_id"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Download manifest missing required columns: {missing}")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when supplied")

    selected = (
        manifest[["image_filename", "drive_file_id"]]
        .dropna()
        .drop_duplicates("image_filename", keep="first")
        .copy()
    )
    if limit is not None:
        selected = selected.head(limit)

    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: Iterable[tuple[Any, pd.Series]] = selected.iterrows()
    if show_progress:
        from tqdm import tqdm

        rows = tqdm(rows, total=len(selected), desc="Rated images")

    downloaded = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for _, row in rows:
        filename = str(row["image_filename"])
        file_id = str(row["drive_file_id"])
        try:
            filename = _safe_cache_filename(filename)
        except ValueError as error:
            failures.append(
                {"image_filename": filename, "drive_file_id": file_id, "error": str(error)}
            )
            continue

        output_path = destination / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue

        partial_path = output_path.with_name(f"{output_path.name}.part")
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = fetch_bytes(drive, file_id)
                if not raw:
                    raise ValueError("Drive returned an empty file")
                partial_path.write_bytes(raw)
                os.replace(partial_path, output_path)
                downloaded += 1
                last_error = None
                break
            except Exception as error:  # Drive client exposes several error types.
                last_error = error
                partial_path.unlink(missing_ok=True)
                if attempt < max_retries:
                    sleep(retry_delay_seconds * attempt)
        if last_error is not None:
            failures.append(
                {
                    "image_filename": filename,
                    "drive_file_id": file_id,
                    "error": str(last_error),
                }
            )

    return {
        "requested": int(len(selected)),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": int(len(failures)),
        "failures": failures,
        "cache_dir": str(destination.resolve()),
    }


__all__ = [
    "attach_drive_ids",
    "build_drive_manifest",
    "build_rated_download_manifest",
    "download_images",
]
