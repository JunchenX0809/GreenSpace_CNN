#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    clean_survey_dataframe,
    normalize_column_name,
    normalize_dataframe_columns,
    normalize_name,
)


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

    df = clean_survey_dataframe(
        pd.read_csv(in_path),
        image_col=args.image_col,
        dedupe_columns=args.dedupe_columns,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'Wrote {out_path} (rows={len(df)}, cols={len(df.columns)})')


if __name__ == '__main__':
    main()
