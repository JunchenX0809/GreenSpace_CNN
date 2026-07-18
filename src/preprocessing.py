"""Reusable survey preprocessing extracted from Notebook 02.

The functions in this module preserve the current notebook contract:

- normalize raw survey headers and image filenames;
- keep only rows where ``include_tile`` is ``yes``;
- aggregate multiple rater rows into soft and hard per-image labels;
- create the same two-stage 60/20/20 train/validation/test split.

Google Drive authentication and image download remain separate concerns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


BINARY_LABELS = (
    "sports_field",
    "multipurpose_open_area",
    "children_s_playground",
    "water_feature",
    "gardens",
    "walking_paths",
    "built_structures",
    "parking_lots",
)

VEGETATION_RATINGS = {
    "only low vegetation": 1,
    "low vegetation with some trees": 2,
    "mixed": 3,
    "trees with some low vegetation": 4,
    "only trees": 5,
}


def normalize_name(name: str) -> str:
    """Trim a filename and lowercase its extension."""

    if not isinstance(name, str):
        return name
    value = name.strip()
    return re.sub(
        r"\.([A-Za-z0-9]+)$",
        lambda match: f".{match.group(1).lower()}",
        value,
    )


def normalize_column_name(name: str) -> str:
    """Convert one raw survey header to the existing snake-case convention."""

    value = name.strip().lower()
    value = re.sub(r"[^0-9a-z]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_dataframe_columns(
    df: pd.DataFrame,
    *,
    dedupe: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize headers, preserving the cleaner's collision behavior."""

    raw_cols = list(df.columns)
    norm_cols = [normalize_column_name(column) for column in raw_cols]

    seen: dict[str, str] = {}
    collisions: dict[str, set[str]] = {}
    for raw, normalized in zip(raw_cols, norm_cols):
        if normalized in seen and seen[normalized] != raw:
            collisions.setdefault(normalized, set()).update(
                {seen[normalized], raw}
            )
        else:
            seen[normalized] = raw

    if collisions and not dedupe:
        lines = ["Column-name collision(s) after normalization:"]
        for normalized, raw_names in sorted(collisions.items()):
            names = ", ".join(sorted(map(str, raw_names)))
            lines.append(f"  - {normalized}: {names}")
        lines.append(
            "Fix the input headers, or rerun with --dedupe-columns to "
            "auto-suffix duplicates."
        )
        raise ValueError("\n".join(lines))

    column_map: dict[str, str] = {}
    used: set[str] = set()
    for raw, base in zip(raw_cols, norm_cols):
        new_name = base
        if dedupe:
            suffix = 2
            while new_name in used:
                new_name = f"{base}_{suffix}"
                suffix += 1
        used.add(new_name)
        column_map[raw] = new_name

    return df.rename(columns=column_map), column_map


def clean_survey_dataframe(
    df: pd.DataFrame,
    *,
    image_col: str = "Image Name",
    dedupe_columns: bool = False,
) -> pd.DataFrame:
    """Return the Notebook 02-compatible cleaned survey table."""

    if image_col not in df.columns:
        raise ValueError(f"Missing image column: {image_col}")
    cleaned, _ = normalize_dataframe_columns(df, dedupe=dedupe_columns)
    normalized_image_col = normalize_column_name(image_col)
    if normalized_image_col not in cleaned.columns:
        raise ValueError(f"Missing normalized image column: {normalized_image_col}")
    cleaned = cleaned.copy()
    cleaned["image_filename"] = (
        cleaned[normalized_image_col].astype(str).map(normalize_name)
    )
    return cleaned


def clean_survey_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    image_col: str = "Image Name",
    dedupe_columns: bool = False,
) -> pd.DataFrame:
    """Clean one raw survey CSV and write the current cleaner output schema."""

    source = Path(input_path)
    destination = Path(output_path)
    cleaned = clean_survey_dataframe(
        pd.read_csv(source),
        image_col=image_col,
        dedupe_columns=dedupe_columns,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)
    return cleaned


def filter_included_rows(
    df: pd.DataFrame,
    *,
    image_key: str = "image_filename",
    include_col: str = "include_tile",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the rows Notebook 02 includes before joins and aggregation."""

    missing = [column for column in (image_key, include_col) if column not in df]
    if missing:
        raise ValueError(f"Survey table missing required columns: {missing}")

    before_rows = len(df)
    before_images = df[image_key].astype(str).nunique(dropna=False)
    included = df[
        df[include_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    ].copy()
    summary = {
        "before_rows": int(before_rows),
        "before_unique_images": int(before_images),
        "included_rows": int(len(included)),
        "included_unique_images": int(
            included[image_key].astype(str).nunique(dropna=False)
        ),
    }
    return included, summary


def validate_aggregation_columns(
    df: pd.DataFrame,
    *,
    image_key: str = "image_filename",
    binary_labels: Sequence[str] = BINARY_LABELS,
) -> None:
    """Validate the raw columns used by the current label aggregation."""

    required = [
        image_key,
        "include_tile",
        "structured_unstructured",
        "vegetation_cover_distribution",
        "shade_along_paths",
        *binary_labels,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in cleaned survey: {missing}")


def _parse_survey_targets(
    df: pd.DataFrame,
    *,
    binary_labels: Sequence[str],
) -> pd.DataFrame:
    """Parse the current binary, shade, score, and vegetation responses."""

    parsed = df.copy()
    yes_no = {"yes": 1, "no": 0}
    for column in binary_labels:
        parsed[column] = (
            parsed[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(yes_no)
            .astype(float)
        )

    shade_first = (
        parsed["shade_along_paths"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.extract(r"^(minimal|abundant)")[0]
    )
    parsed["shade_i"] = shade_first.map(
        {"minimal": 0, "abundant": 1}
    ).astype(float)
    parsed["score_i"] = pd.to_numeric(
        parsed["structured_unstructured"]
        .astype(str)
        .str.extract(r"^(\d)")[0],
        errors="coerce",
    )
    vegetation = (
        parsed["vegetation_cover_distribution"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s*\(.*\)$", "", regex=True)
    )
    parsed["veg_i"] = vegetation.map(VEGETATION_RATINGS).astype(float)
    return parsed


def aggregate_rater_labels(
    cleaned_survey: pd.DataFrame,
    *,
    image_key: str = "image_filename",
    binary_labels: Sequence[str] = BINARY_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Aggregate rater rows into the existing soft and hard label schemas."""

    validate_aggregation_columns(
        cleaned_survey,
        image_key=image_key,
        binary_labels=binary_labels,
    )
    included, summary = filter_included_rows(
        cleaned_survey,
        image_key=image_key,
    )
    parsed = _parse_survey_targets(included, binary_labels=binary_labels)
    grouped = parsed.groupby(image_key, dropna=False)
    n_ratings = grouped.size().rename("n_ratings")

    binary_soft = grouped[list(binary_labels)].mean().astype(float)
    binary_soft.columns = [f"{column}_p" for column in binary_labels]

    shade_probs = grouped["shade_i"].apply(
        lambda values: values.value_counts(normalize=True)
    ).unstack(fill_value=0.0)
    shade_probs = shade_probs.reindex(
        columns=[0.0, 1.0], fill_value=0.0
    ).astype(float)
    shade_probs.columns = ["shade_p_minimal", "shade_p_abundant"]

    score_probs = grouped["score_i"].apply(
        lambda values: values.value_counts(normalize=True)
    ).unstack(fill_value=0.0)
    score_probs = score_probs.reindex(
        columns=[1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0.0
    ).astype(float)
    score_probs.columns = [f"score_p_{index}" for index in range(1, 6)]
    score_mean = grouped["score_i"].mean().rename("score_mean").astype(float)

    vegetation_probs = grouped["veg_i"].apply(
        lambda values: values.value_counts(normalize=True)
    ).unstack(fill_value=0.0)
    vegetation_probs = vegetation_probs.reindex(
        columns=[1.0, 2.0, 3.0, 4.0, 5.0], fill_value=0.0
    ).astype(float)
    vegetation_probs.columns = [f"veg_p_{index}" for index in range(1, 6)]
    vegetation_mean = grouped["veg_i"].mean().rename("veg_mean").astype(float)

    soft = pd.concat(
        [
            n_ratings,
            binary_soft,
            shade_probs,
            score_probs,
            score_mean,
            vegetation_probs,
            vegetation_mean,
        ],
        axis=1,
    ).reset_index()

    hard = soft[[image_key, "n_ratings"]].copy()
    for column in binary_labels:
        hard[column] = soft[f"{column}_p"].fillna(0.0).ge(0.5).astype(int)

    shade_columns = ["shade_p_minimal", "shade_p_abundant"]
    hard["shade_class"] = (
        soft[shade_columns].fillna(0.0).to_numpy(dtype=float).argmax(axis=1)
    )
    score_columns = [f"score_p_{index}" for index in range(1, 6)]
    hard["score_class"] = (
        soft[score_columns].fillna(0.0).to_numpy(dtype=float).argmax(axis=1) + 1
    )
    vegetation_columns = [f"veg_p_{index}" for index in range(1, 6)]
    hard["veg_class"] = (
        soft[vegetation_columns]
        .fillna(0.0)
        .to_numpy(dtype=float)
        .argmax(axis=1)
        + 1
    )
    return soft, hard, summary


def write_aggregated_labels(
    cleaned_survey_path: str | Path,
    processed_dir: str | Path,
    run_tag: str,
) -> tuple[Path, Path, dict[str, int]]:
    """Aggregate a cleaned survey CSV and write run-tagged label tables."""

    soft, hard, summary = aggregate_rater_labels(
        pd.read_csv(cleaned_survey_path)
    )
    output_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    soft_path = output_dir / f"labels_soft_{run_tag}.csv"
    hard_path = output_dir / f"labels_hard_{run_tag}.csv"
    soft.to_csv(soft_path, index=False)
    hard.to_csv(hard_path, index=False)
    return soft_path, hard_path, summary


def build_split_frames(
    soft: pd.DataFrame,
    hard: pd.DataFrame,
    image_dir: str | Path,
    *,
    filelist: pd.DataFrame | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    split_seed: int = 123,
    split_seed_2: int = 456,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Build the same two-stage split manifests as Notebook 02."""

    from sklearn.model_selection import train_test_split

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    required = {"image_filename", "n_ratings"}
    for name, frame in (("soft", soft), ("hard", hard)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} labels missing required columns: {missing}")

    label_merged = hard.merge(
        soft,
        on=["image_filename", "n_ratings"],
        how="inner",
    )
    merged = label_merged
    if filelist is not None:
        filelist_required = {"image_filename", "drive_file_id"}
        missing = sorted(filelist_required.difference(filelist.columns))
        if missing:
            raise ValueError(f"Filelist missing required columns: {missing}")
        if filelist["image_filename"].duplicated().any():
            raise ValueError(
                "Filelist must contain one row per image_filename before splitting."
            )
        merged = merged.merge(
            filelist[["image_filename", "drive_file_id"]],
            on="image_filename",
            how="left",
        )

    cache_root = Path(image_dir)
    merged["image_path"] = merged["image_filename"].apply(
        lambda filename: str((cache_root / filename).resolve())
    )
    exists = merged["image_path"].apply(lambda path: Path(path).exists())
    missing_images = int((~exists).sum())
    merged = merged.loc[exists].copy()
    if len(merged) < 3:
        raise ValueError(
            "At least three cached, labeled images are required to create splits."
        )

    probability_columns = [
        column
        for column in merged.columns
        if column.endswith("_p")
        or column.startswith(("shade_p_", "score_p_", "veg_p_"))
    ]
    mean_columns = [
        column for column in ("score_mean", "veg_mean") if column in merged
    ]
    hard_binary_columns = [
        column for column in BINARY_LABELS if column in merged.columns
    ]
    other_hard_columns = [
        column
        for column in ("shade_class", "score_class", "veg_class")
        if column in merged.columns
    ]
    keep_columns = ["image_path", "image_filename", "n_ratings"]
    if "drive_file_id" in merged.columns:
        keep_columns.append("drive_file_id")
    keep_columns.extend(
        probability_columns
        + mean_columns
        + hard_binary_columns
        + other_hard_columns
    )

    total = len(merged)
    train_size = max(1, round(train_ratio * total))
    val_size = max(1, round(val_ratio * total))
    test_size = max(1, total - train_size - val_size)
    if train_size + val_size + test_size != total:
        test_size = total - train_size - val_size

    train_df, remainder_df = train_test_split(
        merged,
        train_size=train_size,
        random_state=split_seed,
        shuffle=True,
    )
    val_proportion = val_size / max(1, len(remainder_df))
    val_df, test_df = train_test_split(
        remainder_df,
        test_size=(1 - val_proportion),
        random_state=split_seed_2,
        shuffle=True,
    )
    splits = {
        "train": train_df[keep_columns].copy(),
        "val": val_df[keep_columns].copy(),
        "test": test_df[keep_columns].copy(),
    }
    summary = {
        "labeled_images": int(len(label_merged)),
        "missing_images": missing_images,
        "split_images": total,
        "train_images": int(len(splits["train"])),
        "val_images": int(len(splits["val"])),
        "test_images": int(len(splits["test"])),
        "split_seed": int(split_seed),
        "split_seed_2": int(split_seed_2),
    }
    return splits, summary


def write_split_frames(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write train, validation, and test manifests using current filenames."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        if split not in splits:
            raise ValueError(f"Missing split frame: {split}")
        path = destination / f"{split}.csv"
        splits[split].to_csv(path, index=False)
        paths[split] = path
    return paths


def _validate_run_tag(run_tag: str) -> str:
    """Validate a run tag before using it in artifact filenames."""

    value = str(run_tag).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "run_tag must contain only letters, numbers, dots, underscores, "
            "or hyphens"
        )
    return value


def run_preprocessing_pipeline(
    survey_path: str | Path,
    image_dir: str | Path,
    *,
    run_tag: str,
    interim_dir: str | Path,
    processed_dir: str | Path,
    filelist_path: str | Path | None = None,
    split_dir: str | Path | None = None,
    image_col: str = "Image Name",
    split_seed: int = 123,
    split_seed_2: int = 456,
    sample_size: int | None = None,
    sample_seed: int = 37,
    require_all_images: bool = False,
) -> dict[str, object]:
    """Run the established survey-to-split preprocessing orchestration.

    Labels are always aggregated for the full included survey. ``sample_size``
    only limits the cached, labeled images used to create split manifests; this
    is how the core notebook creates its deterministic 50-image demonstration.
    """

    source = Path(survey_path)
    cache_root = Path(image_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Missing survey CSV: {source}")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Missing image cache directory: {cache_root}")
    tag = _validate_run_tag(run_tag)
    if sample_size is not None and sample_size < 3:
        raise ValueError("sample_size must be at least 3 when supplied")

    survey = pd.read_csv(source)
    if "image_filename" in survey.columns:
        cleaned = survey.copy()
    else:
        cleaned = clean_survey_dataframe(survey, image_col=image_col)
    soft, hard, inclusion_summary = aggregate_rater_labels(cleaned)

    filelist: pd.DataFrame | None = None
    resolved_filelist_path: Path | None = None
    if filelist_path is not None:
        resolved_filelist_path = Path(filelist_path)
        if not resolved_filelist_path.is_file():
            raise FileNotFoundError(
                f"Missing filelist-with-Drive-IDs CSV: {resolved_filelist_path}"
            )
        filelist = pd.read_csv(resolved_filelist_path)

    cached_mask = soft["image_filename"].apply(
        lambda filename: (cache_root / str(filename)).is_file()
    )
    cached_soft = soft.loc[cached_mask].copy()
    if sample_size is not None:
        if len(cached_soft) < sample_size:
            raise ValueError(
                f"Requested sample_size={sample_size}, but only "
                f"{len(cached_soft)} labeled images are cached in {cache_root}."
            )
        selected_names = cached_soft["image_filename"].sample(
            n=sample_size,
            random_state=sample_seed,
        ).tolist()
        split_soft = soft.set_index("image_filename").loc[selected_names].reset_index()
        split_hard = hard.set_index("image_filename").loc[selected_names].reset_index()
    else:
        split_soft = soft
        split_hard = hard

    splits, split_summary = build_split_frames(
        split_soft,
        split_hard,
        cache_root,
        filelist=filelist,
        split_seed=split_seed,
        split_seed_2=split_seed_2,
    )
    if require_all_images and split_summary["missing_images"]:
        raise FileNotFoundError(
            f"{split_summary['missing_images']} labeled images are missing from "
            f"{cache_root}. Download them or omit require_all_images."
        )

    interim_root = Path(interim_dir)
    processed_root = Path(processed_dir)
    resolved_split_dir = (
        Path(split_dir) if split_dir is not None else processed_root / "splits"
    )
    cleaned_path = interim_root / f"survey_response_clean_{tag}.csv"
    soft_path = processed_root / f"labels_soft_{tag}.csv"
    hard_path = processed_root / f"labels_hard_{tag}.csv"

    # Compute and validate every artifact before creating output directories.
    interim_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    soft.to_csv(soft_path, index=False)
    hard.to_csv(hard_path, index=False)
    split_paths = write_split_frames(splits, resolved_split_dir)

    selection_summary = {
        "aggregated_images": int(len(soft)),
        "cached_labeled_images": int(cached_mask.sum()),
        "uncached_labeled_images": int((~cached_mask).sum()),
        "selected_split_images": int(split_summary["split_images"]),
        "sample_size": sample_size,
        "sample_seed": int(sample_seed) if sample_size is not None else None,
    }
    return {
        "run_tag": tag,
        "paths": {
            "cleaned_survey": cleaned_path,
            "labels_soft": soft_path,
            "labels_hard": hard_path,
            "train": split_paths["train"],
            "val": split_paths["val"],
            "test": split_paths["test"],
            "filelist": resolved_filelist_path,
        },
        "inclusion_summary": inclusion_summary,
        "selection_summary": selection_summary,
        "split_summary": split_summary,
    }


__all__ = [
    "BINARY_LABELS",
    "VEGETATION_RATINGS",
    "aggregate_rater_labels",
    "build_split_frames",
    "clean_survey_csv",
    "clean_survey_dataframe",
    "filter_included_rows",
    "normalize_column_name",
    "normalize_dataframe_columns",
    "normalize_name",
    "run_preprocessing_pipeline",
    "validate_aggregation_columns",
    "write_aggregated_labels",
    "write_split_frames",
]
