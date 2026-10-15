"""Train / calibration / test split and K-fold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def _price_bins(y: np.ndarray, n_bins: int = 10) -> np.ndarray:
    return pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop").to_numpy()


def three_way_split(
    y: np.ndarray, seed: int = 42, calib_size: float = 0.2, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random split into train / calibration / test, stratified on price deciles.

    The calibration part is used only to calibrate prediction intervals; the test part only for the
    final report."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    bins = _price_bins(y)
    rest, test = train_test_split(idx, test_size=test_size, stratify=bins, random_state=seed)
    rest_bins = bins[rest]
    train, calib = train_test_split(
        rest, test_size=calib_size / (1 - test_size), stratify=rest_bins, random_state=seed + 1
    )
    return np.sort(train), np.sort(calib), np.sort(test)


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold as (train_positions, validation_positions)."""
    return [(tr, te) for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n))]
