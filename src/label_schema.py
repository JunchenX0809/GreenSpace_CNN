# src/label_schema.py
"""Resolves label column schema from a split manifest DataFrame.

Call once after loading train_df; reuse the returned dict throughout
the notebook to avoid re-deriving the same lists in every cell.
"""


def resolve_label_cols(df):
    """Return label column variants derived from a split manifest DataFrame.

    Args:
        df: pandas DataFrame loaded from a split CSV (train, val, or test).

    Returns:
        dict with:
            binary_cols   : list of '*_p' soft probability columns
            bin_names     : label names without the '_p' suffix
            hard_bin_cols : subset of bin_names that also exist as hard (0/1)
                            columns directly in df
    """
    binary_cols = [c for c in df.columns if c.endswith('_p')]
    bin_names = [c[:-2] for c in binary_cols]
    hard_bin_cols = [n for n in bin_names if n in df.columns]
    return {
        'binary_cols': binary_cols,
        'bin_names': bin_names,
        'hard_bin_cols': hard_bin_cols,
    }
