"""Split-conformal prediction intervals (absolute-residual and conformalised quantile regression)."""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score; infinite when n is too small for the level."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


def split_conformal(
    pred_cal: np.ndarray, y_cal: np.ndarray, pred_test: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric intervals pred +/- q with q from absolute calibration residuals."""
    q = conformal_quantile(np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float)), alpha)
    pred_test = np.asarray(pred_test, float)
    return pred_test - q, pred_test + q


def split_conformal_price(
    pred_cal: np.ndarray, y_cal: np.ndarray, pred_test: np.ndarray, alpha: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Split conformal on log prices: multiplicative intervals in dollars, always positive."""
    lo, hi = split_conformal(np.log(pred_cal), np.log(y_cal), np.log(pred_test), alpha)
    return np.exp(lo), np.exp(hi)


def cqr(
    lo_cal: np.ndarray,
    hi_cal: np.ndarray,
    y_cal: np.ndarray,
    lo_test: np.ndarray,
    hi_test: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Conformalised quantile regression (Romano, Patterson and Candes, 2019).

    The score is max(lo - y, y - hi); the correction may be negative, so an over-wide band shrinks."""
    scores = np.maximum(
        np.asarray(lo_cal, float) - np.asarray(y_cal, float),
        np.asarray(y_cal, float) - np.asarray(hi_cal, float),
    )
    q = conformal_quantile(scores, alpha)
    return np.asarray(lo_test, float) - q, np.asarray(hi_test, float) + q
