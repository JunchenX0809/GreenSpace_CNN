#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    # Normalize extension case (e.g., .JPG/.JPEG/.Png -> lowercase)
    s = re.sub(r'\.([A-Za-z0-9]+)$', lambda m: f".{m.group(1).lower()}", s)
    return s
def normalize_column_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r'[^0-9a-z]+', '_', s)      # spaces/punct → _
    s = re.sub(r'_+', '_', s).strip('_')   # collapse/trim _
    return s
def normalize_dataframe_columns(df: pd.DataFrame, *, dedupe: bool = False):
    """
    Normalize column names (lowercase, spaces/punct → underscores).

    Collisions can happen *after* normalization (e.g., "Image Name" and "Image-Name"
    both become "image_name"). By default we fail fast with a clear error; if
    dedupe=True, we append _2, _3, ... to make names unique.
    """
    raw_cols = list(df.columns)
    norm_cols = [normalize_column_name(c) for c in raw_cols]

    # Detect collisions (same normalized name produced from multiple raw names)
    seen = {}
    collisions = {}
    for raw, norm in zip(raw_cols, norm_cols):
        if norm in seen and seen[norm] != raw:
            collisions.setdefault(norm, set()).update({seen[norm], raw})
        else:
            seen[norm] = raw

    if collisions and not dedupe:
        lines = ["Column-name collision(s) after normalization:"]
        for norm, raws in sorted(collisions.items()):
            raws_list = ", ".join(sorted(map(str, raws)))
            lines.append(f"  - {norm}: {raws_list}")
        lines.append("Fix the input headers, or rerun with --dedupe-columns to auto-suffix duplicates.")
        raise ValueError("\n".join(lines))

    # Apply mapping (with optional de-duplication suffixing)
    col_map = {}
    used = set()
    for raw, base in zip(raw_cols, norm_cols):
        new = base
        if dedupe:
            i = 2
            while new in used:
                new = f"{base}_{i}"
                i += 1
        used.add(new)
        col_map[raw] = new

    return df.rename(columns=col_map), col_map
def main() -> None:
    p = argparse.ArgumentParser(description='Clean survey CSV: normalize image filenames and (optionally) align with filelist.csv')
    p.add_argument('--in-csv', required=True, help='Path to raw survey_responses.csv')
    p.add_argument('--out-csv', default='data/raw/survey_responses_clean.csv', help='Output path for cleaned CSV')
    p.add_argument('--image-col', default='Image Name', help='Column name containing image names')
    p.add_argument('--filelist-csv', default='data/filelist.csv', help='Optional path to filelist.csv (ImageName/FileId columns expected)')
    p.add_argument('--filter-to-filelist', action='store_true', help='If set, keep only rows with image present in filelist')
    p.add_argument('--dedupe-columns', action='store_true', help='If set, append _2, _3, ... when normalization causes duplicate column names')
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)
    assert args.image_col in df.columns, f"Missing column: {args.image_col}"
    # Normalize column names
    df, col_map = normalize_dataframe_columns(df, dedupe=args.dedupe_columns)
    
    # Use normalized image column name
    normalized_image_col = normalize_column_name(args.image_col)
    assert normalized_image_col in df.columns, f"Missing column: {normalized_image_col}"

    df["image_filename"] = df[normalized_image_col].astype(str).map(normalize_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} (rows={len(df)}, cols={len(df.columns)})')


if __name__ == '__main__':
    main()
