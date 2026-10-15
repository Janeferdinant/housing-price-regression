from __future__ import annotations

import numpy as np
import pytest

from house_price import metrics


def test_basic_metrics_known_values():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 180.0, 330.0])
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert metrics.mae(y, p) == pytest.approx(20.0)
    assert metrics.bias(y, p) == pytest.approx((10 - 20 + 30) / 3)
    assert metrics.mape(y, p) == pytest.approx(0.10)
    assert metrics.median_ape(y, p) == pytest.approx(0.10)
    assert metrics.r2(y, p) == pytest.approx(1 - 1400.0 / 20000.0)


def test_rmsle_is_rmse_in_log_space():
    y = np.array([100.0, 1000.0])
    p = np.array([200.0, 500.0])
    assert metrics.rmsle(y, p) == pytest.approx(np.log(2.0))


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 5.0])
    m = metrics.all_metrics(y, y)
    assert m["rmse"] == 0 and m["r2"] == 1 and m["mape"] == 0


def test_segment_report_partitions_rows_by_actual_price():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(12, 0.5, 500))
    p = y * np.exp(rng.normal(0, 0.1, 500))
    rep = metrics.segment_report(y, p, n_segments=5)
    assert rep["n"].sum() == 500 and len(rep) == 5
    assert (rep["price_min"].iloc[1:].to_numpy() >= rep["price_max"].iloc[:-1].to_numpy() - 1e-9).all()
    assert rep["n"].between(99, 101).all()


def test_segment_report_exposes_regression_to_the_mean():
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(12, 0.6, 3000))
    p = np.exp(0.7 * np.log(y) + 0.3 * 12 + rng.normal(0, 0.1, 3000))  # shrunk predictions
    rep = metrics.segment_report(y, p, n_segments=3)
    assert rep["bias"].iloc[0] > 0 > rep["bias"].iloc[-1]  # over-predict cheap homes, under-predict dear ones


def test_bootstrap_ci_brackets_estimate_and_is_deterministic():
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(12, 0.4, 400))
    p = y * np.exp(rng.normal(0, 0.15, 400))
    lo, hi = metrics.bootstrap_ci(y, p, metrics.rmse, n_boot=200, seed=0)
    assert lo < metrics.rmse(y, p) < hi
    assert (lo, hi) == metrics.bootstrap_ci(y, p, metrics.rmse, n_boot=200, seed=0)


def test_interval_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.5, 2.5, 2.0, 3.0])
    hi = np.array([1.5, 3.5, 4.0, 5.0])
    assert metrics.coverage(y, lo, hi) == 0.75
    assert metrics.mean_width(lo, hi) == pytest.approx((1 + 1 + 2 + 2) / 4)
    assert metrics.median_relative_width(y, lo, hi) == pytest.approx(np.median([1.0, 0.5, 2 / 3, 0.5]))


def test_coverage_is_inclusive_at_edges():
    assert metrics.coverage(np.array([1.0]), np.array([1.0]), np.array([1.0])) == 1.0
