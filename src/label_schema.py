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


# ---- Experiment-wide toggles (single source of truth) ----
# Change values in EXPERIMENT_CONFIG, then re-run notebooks that import CFG.

EXPERIMENT_CONFIG = {
    # Head modes — one of: 'sparse', 'soft', 'regression'
    # 'sparse': hard int labels (score_class), sparse CE, Dense(5, softmax)
    # 'soft':   prob vectors (score_p_1..5), CE, Dense(5, softmax)
    # 'regression': score_mean / veg_mean, MAE, Dense(1, sigmoid) scaled to [1, 5]
    'score_head_mode': 'regression',
    'veg_head_mode': 'regression',
    # Binary label exclusion — column names to drop from binary head. [] to include all.
    'exclude_binary': ['gardens_p'],
    # Oversampling — list of dicts with keys label, target_rate, pos_threshold. [] to disable.
    # Rows are assigned to streams in list order (first match wins). Sum of target_rates < 1.0.
    'oversample': [
    {'label': 'children_s_playground_p', 'target_rate': 0.20, 'pos_threshold': 0.50},
    {'label': 'water_feature_p',         'target_rate': 0.25, 'pos_threshold': 0.50},
],
    # Training control — keep smoke/full-run and backbone fine-tuning decisions centralized.
    'test_run_mode': False,
    'test_warmup_epochs': 1,
    'test_finetune_epochs': 1,
    'epochs_warmup': 5,
    'epochs_finetune': 100,
    'fine_tune_backbone': True,
}


HEAD_PRESETS = {
    'sparse': {
        'units': 5,
        'activation': 'softmax',
        'loss': 'sparse_categorical_crossentropy',
        'accuracy': 'sparse_categorical_accuracy',
        'use_ev_mae': True,
        'ev_mae_fn': 'sparse',
    },
    'soft': {
        'units': 5,
        'activation': 'softmax',
        'loss': 'categorical_crossentropy',
        'accuracy': 'accuracy',
        'use_ev_mae': True,
        'ev_mae_fn': 'soft',
    },
    'regression': {
        'units': 1,
        'activation': 'sigmoid',
        'output_range': (1.0, 5.0),
        'loss': 'mae',
        'accuracy': None,
        'use_ev_mae': False,
        'ev_mae_fn': None,
    },
}
