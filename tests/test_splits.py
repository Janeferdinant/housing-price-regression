from __future__ import annotations

import numpy as np

from house_price.splits import kfold_indices, three_way_split


def _y(n=1000, seed=0):
    return np.exp(np.random.default_rng(seed).normal(12, 0.5, n))


def test_three_way_split_is_disjoint_complete_and_sized():
    y = _y()
    tr, ca, te = three_way_split(y, seed=1, calib_size=0.2, test_size=0.2)
    assert set(tr).isdisjoint(ca) and set(tr).isdisjoint(te) and set(ca).isdisjoint(te)
    assert len(tr) + len(ca) + len(te) == len(y)
    assert len(te) == 200 and len(ca) == 200 and len(tr) == 600


def test_three_way_split_is_stratified_on_price():
    y = _y(2000)
    tr, ca, te = three_way_split(y, seed=2)
    for part in (ca, te):
        assert abs(np.median(y[part]) / np.median(y[tr]) - 1) < 0.05
        assert abs(y[part].mean() / y[tr].mean() - 1) < 0.06


def test_three_way_split_deterministic():
    y = _y()
    a = three_way_split(y, seed=5)
    b = three_way_split(y, seed=5)
    c = three_way_split(y, seed=6)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    assert not np.array_equal(a[2], c[2])


def test_kfold_indices_partition():
    folds = kfold_indices(103, 5, seed=0)
    assert len(folds) == 5
    te = np.concatenate([f[1] for f in folds])
    assert sorted(te) == list(range(103))
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
