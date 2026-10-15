from __future__ import annotations

import numpy as np
import pytest

from house_price import conformal, metrics


def test_conformal_quantile_uses_finite_sample_correction():
    scores = np.arange(1.0, 100.0)  # n = 99
    # ceil((n + 1) * (1 - alpha)) = ceil(100 * 0.9) = 90 -> the 90th smallest score
    assert conformal.conformal_quantile(scores, alpha=0.1) == 90.0


def test_conformal_quantile_is_infinite_when_n_too_small():
    assert conformal.conformal_quantile(np.array([1.0, 2.0, 3.0]), alpha=0.05) == np.inf


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(20):
        xc, xt = rng.uniform(0, 5, 400), rng.uniform(0, 5, 2000)
        yc = 2 * xc + rng.normal(0, 1, 400)
        yt = 2 * xt + rng.normal(0, 1, 2000)
        lo, hi = conformal.split_conformal(2 * xc, yc, 2 * xt, alpha=0.1)
        covs.append(metrics.coverage(yt, lo, hi))
    assert 0.88 <= np.mean(covs) <= 0.93  # guarantee is >= 0.90 on average, and tight at n = 400


def test_split_conformal_intervals_have_constant_width():
    lo, hi = conformal.split_conformal(np.zeros(50), np.linspace(-1, 1, 50), np.zeros(5), alpha=0.2)
    assert np.allclose(hi - lo, (hi - lo)[0])


def test_log_scale_intervals_are_multiplicative_and_positive():
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(12, 0.4, 300))
    p = y * np.exp(rng.normal(0, 0.1, 300))
    lo, hi = conformal.split_conformal_price(p[:200], y[:200], p[200:], alpha=0.1)
    assert (lo > 0).all() and (hi > lo).all()
    ratio = hi / p[200:]
    assert np.allclose(ratio, ratio[0])  # same multiplicative factor for every home


def test_cqr_widens_a_too_narrow_band_to_reach_target_coverage():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.uniform(0, 1, n)
    y = x + rng.normal(0, 1, n)
    lo_raw, hi_raw = x - 0.3, x + 0.3  # far too narrow for a 90% band
    cal, test = slice(0, 1500), slice(1500, n)
    assert metrics.coverage(y[test], lo_raw[test], hi_raw[test]) < 0.5
    lo, hi = conformal.cqr(lo_raw[cal], hi_raw[cal], y[cal], lo_raw[test], hi_raw[test], alpha=0.1)
    assert metrics.coverage(y[test], lo, hi) == pytest.approx(0.9, abs=0.03)


def test_cqr_can_shrink_an_overly_wide_band():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 2000)
    lo_raw, hi_raw = np.full(2000, -5.0), np.full(2000, 5.0)
    lo, hi = conformal.cqr(lo_raw[:1000], hi_raw[:1000], y[:1000], lo_raw[1000:], hi_raw[1000:], alpha=0.1)
    assert (hi - lo)[0] < 10  # negative correction is allowed
