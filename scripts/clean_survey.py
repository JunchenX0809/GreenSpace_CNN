#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

PREFIX_PATTERNS = [r'^copy of\s+', r'^copia de\s+', r'^copy_of_', r'^copy\s+']


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    s = name.strip()
    for pat in PREFIX_PATTERNS:
        s2 = re.sub(pat, '', s, flags=re.IGNORECASE)
        if s2 != s:
            s = s2
            break
    # Normalize extension: .JPG -> .jpg
    s = re.sub(r'\.(JPG|JPEG)$', lambda m: f".{m.group(1).lower()}", s, flags=re.IGNORECASE)
    return s


def main() -> None:
    p = argparse.ArgumentParser(description='Clean survey CSV and produce survey_responses_clean.csv')
    p.add_argument('--in-csv', required=True, help='Path to raw survey_responses.csv')
    p.add_argument('--out-csv', default='data/raw/survey_responses_clean.csv', help='Output path for cleaned CSV')
    p.add_argument('--image-col', default='Image Name', help='Column name containing image names')
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)

    df = pd.read_csv(in_path)
    assert args.image_col in df.columns, f"Missing column: {args.image_col}"

    df['image_filename'] = df[args.image_col].astype(str).map(normalize_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} (rows={len(df)}, cols={len(df.columns)})')


if __name__ == '__main__':
    main()
