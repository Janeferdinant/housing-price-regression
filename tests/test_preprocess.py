from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price.preprocess import SkewedLog, build_preprocessor


def test_skewed_log_chooses_columns_from_training_data_only():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": np.exp(rng.normal(0, 1.2, 500)), "b": rng.normal(10, 1, 500)})
    t = SkewedLog(threshold=0.75).fit(train)
    assert list(t.columns_) == [0]
    # a test frame in which b looks hugely skewed must not change the decision
    test = pd.DataFrame({"a": rng.normal(10, 1, 50), "b": np.exp(rng.normal(0, 3, 50))})
    out = t.transform(test)
    assert np.allclose(out[:, 1], test["b"])  # b untouched
    assert np.allclose(out[:, 0], np.log1p(test["a"]))


def test_skewed_log_skips_columns_with_negative_values():
    rng = np.random.default_rng(1)
    vals = np.concatenate([-np.exp(rng.normal(0, 1.5, 200)), np.exp(rng.normal(0, 1.5, 200))])
    X = pd.DataFrame({"neg": vals})
    assert len(SkewedLog().fit(X).columns_) == 0


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessors_handle_missing_and_unseen(reg, kind):
    X, _, num, cat = reg
    train, test = X.iloc[:200].copy(), X.iloc[200:].copy()
    test.iloc[0, test.columns.get_loc("area")] = np.nan
    test["style"] = test["style"].astype(object)
    test.iloc[1, test.columns.get_loc("style")] = "unseen"
    pre = build_preprocessor(kind, num, cat).fit(train)
    out = np.asarray(pre.transform(test), dtype=float)
    assert not np.isnan(out).any()


def test_linear_imputation_and_scaling_use_training_statistics(reg):
    X, _, num, cat = reg
    train, test = X.iloc[:200].copy(), X.iloc[200:].copy()
    train["age"] = train["age"].astype(float)
    train.iloc[0, train.columns.get_loc("age")] = np.nan
    test["age"] = 10_000.0
    pre = build_preprocessor("linear", num, cat).fit(train)
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    j = num.index("age")
    assert scaler.mean_[j] == pytest.approx(train["age"].fillna(train["age"].median()).mean())
    assert scaler.mean_[j] < 100


def test_missing_indicator_added_only_for_columns_missing_in_training(reg):
    X, _, num, cat = reg
    train = X.iloc[:200].copy()
    train["rooms"] = train["rooms"].astype(float)
    train.iloc[0, train.columns.get_loc("rooms")] = np.nan
    pre = build_preprocessor("linear", num, cat).fit(train)
    names = list(pre.get_feature_names_out())
    assert sum("missingindicator" in n for n in names) == 1


def test_tree_preprocessor_keeps_one_column_per_feature(reg):
    X, _, num, cat = reg
    pre = build_preprocessor("tree", num, cat).fit(X)
    assert pre.transform(X).shape[1] == len(num) + len(cat)


def test_rare_levels_are_grouped_in_linear_encoding(reg):
    X, _, num, cat = reg
    X = X.copy()
    X["neighborhood"] = X["neighborhood"].astype(object)
    X.iloc[0, X.columns.get_loc("neighborhood")] = "rare"
    pre = build_preprocessor("linear", num, cat).fit(X)
    names = " ".join(pre.get_feature_names_out())
    assert "_rare" not in names
