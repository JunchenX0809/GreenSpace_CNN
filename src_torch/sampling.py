"""PyTorch sampling helpers that mirror TensorFlow NB03 oversampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src_torch.config import EXPERIMENT_CONFIG, TORCH_TRAINING_SMOKE_CONFIG


@dataclass(frozen=True)
class OversamplingStream:
    """One exclusive oversampling stream."""

    label: str
    target_rate: float
    pos_threshold: float
    row_count: int
    base_soft_mean: float
    threshold_rate: float


@dataclass(frozen=True)
class OversamplingPlan:
    """Resolved stream weights and per-row sampler weights."""

    streams: list[OversamplingStream]
    remainder_count: int
    remainder_weight: float
    row_weights: np.ndarray

    @property
    def active(self) -> bool:
        return bool(self.streams) and self.remainder_count > 0

    @property
    def weights(self) -> list[float]:
        return [stream.target_rate for stream in self.streams] + [self.remainder_weight]

    def summary(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "streams": [stream.__dict__ for stream in self.streams],
            "remainder_count": self.remainder_count,
            "remainder_weight": self.remainder_weight,
            "weights": self.weights,
        }


def build_oversampling_plan(
    df: pd.DataFrame,
    oversample_cfg: list[dict[str, Any]] | None = None,
) -> OversamplingPlan:
    """Resolve first-match-wins stream assignments from `EXPERIMENT_CONFIG`.

    TensorFlow NB03 builds repeated streams and samples them by stream weight.
    PyTorch uses equivalent per-row probabilities for `WeightedRandomSampler`.
    """

    cfgs = [
        dict(cfg)
        for cfg in (oversample_cfg if oversample_cfg is not None else EXPERIMENT_CONFIG.get("oversample", []))
        if cfg.get("label") in df.columns
    ]
    row_weights = np.ones(len(df), dtype=np.float64) / max(len(df), 1)
    if not cfgs:
        return OversamplingPlan(streams=[], remainder_count=len(df), remainder_weight=1.0, row_weights=row_weights)

    total_rate = sum(float(cfg["target_rate"]) for cfg in cfgs)
    if total_rate >= 1.0:
        raise ValueError(
            f"Sum of oversample target_rates is {total_rate:.2f} >= 1.0. "
            "Reduce rates so the remainder stream has positive weight."
        )

    assigned = np.zeros(len(df), dtype=bool)
    row_weights = np.zeros(len(df), dtype=np.float64)
    streams: list[OversamplingStream] = []

    for cfg in cfgs:
        label = str(cfg["label"])
        threshold = float(cfg["pos_threshold"])
        rate = float(cfg["target_rate"])
        values = df[label].fillna(0.0).to_numpy(dtype=np.float64)
        mask = (values >= threshold) & ~assigned
        row_count = int(mask.sum())
        if row_count == 0:
            continue
        row_weights[mask] = rate / row_count
        assigned |= mask
        streams.append(
            OversamplingStream(
                label=label,
                target_rate=rate,
                pos_threshold=threshold,
                row_count=row_count,
                base_soft_mean=float(values.mean()),
                threshold_rate=float((values >= threshold).mean()),
            )
        )

    remainder_mask = ~assigned
    remainder_count = int(remainder_mask.sum())
    remainder_weight = 1.0 - sum(stream.target_rate for stream in streams)
    if streams and remainder_count > 0:
        row_weights[remainder_mask] = remainder_weight / remainder_count
    else:
        row_weights = np.ones(len(df), dtype=np.float64) / max(len(df), 1)
        remainder_weight = 1.0

    return OversamplingPlan(
        streams=streams,
        remainder_count=remainder_count,
        remainder_weight=remainder_weight,
        row_weights=row_weights,
    )


def make_weighted_sampler(
    df: pd.DataFrame,
    oversample_cfg: list[dict[str, Any]] | None = None,
    seed: int = TORCH_TRAINING_SMOKE_CONFIG["seed"],
) -> tuple[Any | None, OversamplingPlan]:
    """Build a `WeightedRandomSampler` and its resolved plan."""

    try:
        import torch
        from torch.utils.data import WeightedRandomSampler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is not installed. Install the repo requirements.txt before "
            "constructing oversampling samplers."
        ) from exc

    plan = build_oversampling_plan(df, oversample_cfg=oversample_cfg)
    if not plan.active:
        return None, plan

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(plan.row_weights, dtype=torch.double),
        num_samples=len(df),
        replacement=True,
        generator=generator,
    )
    return sampler, plan


def oversampling_sanity_check(
    loader: Any,
    binary_cols: list[str],
    oversample_cfg: list[dict[str, Any]] | None = None,
    max_batches: int = 80,
) -> dict[str, Any]:
    """Measure realized positive rates in a sampled train loader."""

    cfgs = [
        dict(cfg)
        for cfg in (oversample_cfg if oversample_cfg is not None else EXPERIMENT_CONFIG.get("oversample", []))
        if cfg.get("label") in binary_cols
    ]
    label_counts = {cfg["label"]: 0 for cfg in cfgs}
    total = 0

    for batch_idx, (_, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        y_bin = targets["bin_head"]
        total += int(y_bin.shape[0])
        for cfg in cfgs:
            idx = binary_cols.index(cfg["label"])
            label_counts[cfg["label"]] += int((y_bin[:, idx] >= float(cfg["pos_threshold"])).sum().item())

    return {
        "samples": total,
        "rates": {
            cfg["label"]: {
                "realized": label_counts[cfg["label"]] / max(total, 1),
                "target": float(cfg["target_rate"]),
            }
            for cfg in cfgs
        },
    }
