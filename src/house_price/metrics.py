"""Regression and interval metrics. Prices are in dollars unless a name says otherwise."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def bias(y: np.ndarray, p: np.ndarray) -> float:
    """Mean signed error (prediction minus actual); positive means over-prediction."""
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def rmsle(y: np.ndarray, p: np.ndarray) -> float:
    """RMSE of log prices (the usual 'RMSLE' with natural log, no +1 since prices are large)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((np.log(p) - np.log(y)) ** 2)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - y) / y))


def median_ape(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.median(np.abs(p - y) / y))


METRIC_FUNCS: dict[str, MetricFn] = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "rmsle": rmsle,
    "mape": mape,
    "median_ape": median_ape,
    "bias": bias,
}


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {k: fn(y, p) for k, fn in METRIC_FUNCS.items()}


def segment_report(y: np.ndarray, p: np.ndarray, n_segments: int = 5) -> pd.DataFrame:
    """Error by segment of the *actual* price (equal-count bins).

    Conditioning on the actual price makes every model look like it over-predicts the cheapest homes
    and under-predicts the dearest (regression to the mean); read it as a diagnostic, not a defect."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    seg = pd.qcut(pd.Series(y).rank(method="first"), q=n_segments, labels=False).to_numpy()
    rows = []
    for s in range(n_segments):
        m = seg == s
        rows.append(
            {
                "segment": s + 1,
                "n": int(m.sum()),
                "price_min": float(y[m].min()),
                "price_max": float(y[m].max()),
                "mae": mae(y[m], p[m]),
                "rmse": rmse(y[m], p[m]),
                "mape": mape(y[m], p[m]),
                "bias": bias(y[m], p[m]),
                "bias_pct": float(np.mean((p[m] - y[m]) / y[m])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def ci_from_indices(
    y: np.ndarray, p: np.ndarray, fn: MetricFn, idx: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    y, p = np.asarray(y), np.asarray(p)
    vals = np.array([fn(y[i], p[i]) for i in idx])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, fn: MetricFn, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI over test rows."""
    return ci_from_indices(y, p, fn, bootstrap_indices(len(y), n_boot, seed), alpha)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = np.asarray(y), np.asarray(lo), np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def median_relative_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.median((np.asarray(hi) - np.asarray(lo)) / np.asarray(y)))
