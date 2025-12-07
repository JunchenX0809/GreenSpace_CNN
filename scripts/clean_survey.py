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
def normalize_dataframe_columns(df: pd.DataFrame):
    col_map = {}
    used = set()
    for c in df.columns:
        base = normalize_column_name(c)
        new = base
        i = 2
        while new in used:
            new = f"{base}_{i}"            # de-duplicate collisions
            i += 1
        used.add(new)
        col_map[c] = new
    return df.rename(columns=col_map), col_map
def main() -> None:
    p = argparse.ArgumentParser(description='Clean survey CSV: normalize image filenames and (optionally) align with filelist.csv')
    p.add_argument('--in-csv', required=True, help='Path to raw survey_responses.csv')
    p.add_argument('--out-csv', default='data/raw/survey_responses_clean.csv', help='Output path for cleaned CSV')
    p.add_argument('--image-col', default='Image Name', help='Column name containing image names')
    p.add_argument('--filelist-csv', default='data/filelist.csv', help='Optional path to filelist.csv (ImageName/FileId columns expected)')
    p.add_argument('--filter-to-filelist', action='store_true', help='If set, keep only rows with image present in filelist')
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)
    assert args.image_col in df.columns, f"Missing column: {args.image_col}"
    # Normalize column names
    df, col_map = normalize_dataframe_columns(df)
    
    # Use normalized image column name
    normalized_image_col = normalize_column_name(args.image_col)
    assert normalized_image_col in df.columns, f"Missing column: {normalized_image_col}"

    df["image_filename"] = df[normalized_image_col].astype(str).map(normalize_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} (rows={len(df)}, cols={len(df.columns)})')


if __name__ == '__main__':
    main()
