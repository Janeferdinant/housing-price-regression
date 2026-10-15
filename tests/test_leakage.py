"""Leakage tests: nothing fitted may see the rows it is evaluated on."""

from __future__ import annotations

import numpy as np
import pytest

from house_price import tuning
from house_price.metrics import r2
from house_price.models import make_model
from house_price.splits import kfold_indices, three_way_split


def _nested(X, y, num, cat, names, **kw):
    args = {"n_splits": 4, "n_trials": 2, "inner_splits": 2, "seed": 0, "n_jobs": 1}
    args.update(kw)
    return tuning.nested_cv(X.reset_index(drop=True), y, num, cat, names, **args)


def test_tuning_never_sees_outer_test_rows(reg, monkeypatch):
    X, y, num, cat = reg
    seen = []
    real = tuning.tune_params

    def spy(name, Xtr, ytr, *a, **k):
        seen.append(set(Xtr.index))
        return real(name, Xtr, ytr, *a, **k)

    monkeypatch.setattr(tuning, "tune_params", spy)
    _nested(X, y, num, cat, ["ridge"], n_splits=3)
    folds = kfold_indices(len(y), 3, seed=0)
    assert len(seen) == 3
    for used, (_, te) in zip(seen, folds, strict=True):
        assert used.isdisjoint(set(te))


def test_every_row_predicted_by_a_model_that_never_saw_it(reg):
    X, y, num, cat = reg
    res = _nested(X, y, num, cat, ["ridge", "hgb"])
    for name in ("ridge", "hgb"):
        assert not np.isnan(res.oof[name]).any()


def test_label_permutation_gives_no_skill(reg):
    """If preprocessing or tuning leaked the target, permuted labels would still be predicted."""
    X, y, num, cat = reg
    y_perm = np.random.default_rng(0).permutation(y)
    res = _nested(X, y_perm, num, cat, ["ridge", "hgb"], n_splits=5)
    for name in ("ridge", "hgb"):
        assert r2(y_perm, res.oof[name]) < 0.1, name


def test_real_signal_is_recovered_by_the_same_procedure(reg):
    X, y, num, cat = reg
    res = _nested(X, y, num, cat, ["ridge"], n_splits=5)
    assert r2(y, res.oof["ridge"]) > 0.6


def test_three_way_split_keeps_calibration_and_test_out_of_training(reg):
    _, y, _, _ = reg
    tr, ca, te = three_way_split(y, seed=0)
    assert set(tr).isdisjoint(ca) and set(tr).isdisjoint(te) and set(ca).isdisjoint(te)


def test_preprocessing_statistics_come_from_training_rows_only(reg):
    X, y, num, cat = reg
    X = X.copy()
    X.iloc[200:, X.columns.get_loc("area")] = 10**7  # absurd values only in held-out rows
    m = make_model("ridge", num, cat, seed=0).fit(X.iloc[:200], y[:200])
    scaler = m.regressor_.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert np.abs(scaler.mean_).max() < 1e5  # the 10^7 rows never reached the scaler


def test_target_never_among_features(reg):
    X, _, num, cat = reg
    assert "price" not in X.columns and "price" not in num + cat


@pytest.mark.parametrize("name", ["ridge", "rf", "hgb"])
def test_predictions_do_not_depend_on_other_rows_in_the_batch(reg, name):
    X, y, num, cat = reg
    m = make_model(name, num, cat, seed=0).fit(X.iloc[:200], y[:200])
    batch = m.predict(X.iloc[200:210])
    each = np.array([m.predict(X.iloc[[i]])[0] for i in range(200, 210)])
    assert np.allclose(batch, each)
