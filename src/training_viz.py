"""Training history merge and epoch-level metric visualization.

Usage from notebook:
    from src.training_viz import merge_histories, plot_training_curves
    merged = merge_histories(history_warmup.history, history_finetune.history)
    plot_training_curves(merged, warmup_epochs=5, save_dir=run_dir)

Or from saved JSON:
    import json
    history = json.load(open('training_history_<RUN_TAG>.json'))
    plot_training_curves(history, warmup_epochs=5, save_dir=run_dir)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Metric key mapping (logical name → Keras history key) ----
METRIC_KEYS = {
    'val_pr_auc': 'val_bin_head_weighted_pr_auc',
    'train_pr_auc': 'bin_head_weighted_pr_auc',
    'val_score_mae': 'val_score_head_score_ev_mae',
    'train_score_mae': 'score_head_score_ev_mae',
    'val_veg_mae': 'val_veg_head_veg_ev_mae',
    'train_veg_mae': 'veg_head_veg_ev_mae',
}


def merge_histories(hist_warmup, hist_finetune):
    """Concatenate two Keras History.history dicts into one continuous timeline.

    Keys present in only one phase are padded with NaN for the other phase
    so that epoch indices stay aligned.
    """
    all_keys = set(hist_warmup.keys()) | set(hist_finetune.keys())
    n_warmup = len(next(iter(hist_warmup.values()))) if hist_warmup else 0
    n_finetune = len(next(iter(hist_finetune.values()))) if hist_finetune else 0

    merged = {}
    for k in sorted(all_keys):
        w = hist_warmup.get(k, [float('nan')] * n_warmup)
        f = hist_finetune.get(k, [float('nan')] * n_finetune)
        merged[k] = list(w) + list(f)
    return merged


def _find_best_epoch(values, mode='max'):
    """Return 1-indexed epoch of best value, skipping NaN. Returns None if all NaN."""
    best_idx = None
    best_val = None
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        v = float(v)
        if best_val is None:
            best_val = v
            best_idx = i
        elif mode == 'max' and v > best_val:
            best_val = v
            best_idx = i
        elif mode == 'min' and v < best_val:
            best_val = v
            best_idx = i
    return (best_idx + 1) if best_idx is not None else None


def plot_training_curves(
    history,
    warmup_epochs,
    save_dir=None,
    score_val_key=None,
    score_train_key=None,
    veg_val_key=None,
    veg_train_key=None,
):
    """Plot PR-AUC and MAE training curves with best-checkpoint markers.

    Args:
        history: merged history dict (from merge_histories or loaded JSON).
        warmup_epochs: number of warmup epochs (for phase boundary line).
        save_dir: if provided, saves PNGs here.
        score_val_key, score_train_key, veg_val_key, veg_train_key: optional overrides
            for Keras history keys (e.g. val_score_head_mae when using regression heads).
    """
    n_epochs = len(next(iter(history.values())))
    epochs = list(range(1, n_epochs + 1))

    keys = dict(METRIC_KEYS)
    if score_val_key is not None:
        keys['val_score_mae'] = score_val_key
    if score_train_key is not None:
        keys['train_score_mae'] = score_train_key
    if veg_val_key is not None:
        keys['val_veg_mae'] = veg_val_key
    if veg_train_key is not None:
        keys['train_veg_mae'] = veg_train_key

    def _safe_get(key):
        """Get values from history, converting None to NaN for plotting."""
        vals = history.get(key)
        if vals is None:
            return None
        return [float('nan') if v is None else float(v) for v in vals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- Panel 1: PR-AUC ----
    val_pr = _safe_get(keys['val_pr_auc'])
    train_pr = _safe_get(keys['train_pr_auc'])

    if val_pr is not None:
        ax1.plot(epochs, val_pr, label='Val PR-AUC', color='tab:blue', linewidth=1.5)
        best_ep = _find_best_epoch(val_pr, mode='max')
        if best_ep is not None:
            ax1.plot(best_ep, val_pr[best_ep - 1], '*', color='tab:blue',
                     markersize=14, label=f'Best val (epoch {best_ep})')

    if train_pr is not None:
        ax1.plot(epochs, train_pr, label='Train PR-AUC', color='tab:blue',
                 linestyle='--', alpha=0.6)

    ax1.axvline(x=warmup_epochs + 0.5, color='gray', linestyle=':', alpha=0.7,
                label='Unfreeze backbone')
    ax1.set_ylabel('Weighted PR-AUC')
    ax1.set_title('Binary Head — Weighted PR-AUC')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: MAE (score + veg) ----
    val_score = _safe_get(keys['val_score_mae'])
    val_veg = _safe_get(keys['val_veg_mae'])
    train_score = _safe_get(keys['train_score_mae'])
    train_veg = _safe_get(keys['train_veg_mae'])

    if val_score is not None:
        ax2.plot(epochs, val_score, label='Val Score MAE', color='tab:orange', linewidth=1.5)
    if val_veg is not None:
        ax2.plot(epochs, val_veg, label='Val Veg MAE', color='tab:green', linewidth=1.5)
    if train_score is not None:
        ax2.plot(epochs, train_score, label='Train Score MAE', color='tab:orange',
                 linestyle='--', alpha=0.6)
    if train_veg is not None:
        ax2.plot(epochs, train_veg, label='Train Veg MAE', color='tab:green',
                 linestyle='--', alpha=0.6)

    # Best mc_mae checkpoint (average of val score + val veg MAE)
    if val_score is not None and val_veg is not None:
        mc_mae = [(s + v) / 2 if not (math.isnan(s) or math.isnan(v)) else float('nan')
                  for s, v in zip(val_score, val_veg)]
        best_ep = _find_best_epoch(mc_mae, mode='min')
        if best_ep is not None:
            ax2.plot(best_ep, val_score[best_ep - 1], '*', color='tab:orange', markersize=14)
            ax2.plot(best_ep, val_veg[best_ep - 1], '*', color='tab:green', markersize=14,
                     label=f'Best mc_mae (epoch {best_ep})')

    ax2.axvline(x=warmup_epochs + 0.5, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Expected-Value MAE')
    ax2.set_title('Multi-Class Heads — Score & Vegetation MAE')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        fig.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        print(f'Saved training_curves.png to {save_dir}')

    plt.show()
