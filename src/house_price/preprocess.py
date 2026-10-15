"""Preprocessing pipelines. All statistics are learned inside ``fit`` on training rows only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class SkewedLog(BaseEstimator, TransformerMixin):
    """log1p on the non-negative columns whose *training* skewness exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        df = pd.DataFrame(np.asarray(X, dtype=float))
        skew = df.skew()
        nonneg = (df.min() >= 0).to_numpy()
        self.columns_ = np.where((skew.to_numpy() > self.threshold) & nonneg)[0]
        return self

    def transform(self, X):
        out = np.array(X, dtype=float, copy=True)
        if len(self.columns_):
            out[:, self.columns_] = np.log1p(np.clip(out[:, self.columns_], 0, None))
        return out

    def get_feature_names_out(self, names=None):
        return np.asarray(names)


def build_preprocessor(kind: str, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Unfitted ColumnTransformer.

    ``linear``: median imputation with missing-indicators, log1p of skewed columns, standardisation,
    one-hot categories (rare levels grouped, unseen levels ignored).
    ``tree``: median imputation, ordinal category codes (unseen levels become -1), one column per
    original feature so SHAP values map straight back to features.
    """
    transformers = []
    if kind == "linear":
        num = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("skew", _SkewOnNumeric(len(numeric))),
                ("scale", StandardScaler()),
            ]
        )
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
                ),
            ]
        )
    elif kind == "tree":
        num = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    else:
        raise ValueError(f"unknown preprocessor kind {kind!r}")
    if numeric:
        transformers.append(("num", num, list(numeric)))
    if categorical:
        transformers.append(("cat", cat, list(categorical)))
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


class _SkewOnNumeric(SkewedLog):
    """SkewedLog applied only to the first ``n_numeric`` columns (not to missing indicators)."""

    def __init__(self, n_numeric: int = 0, threshold: float = 0.75):
        super().__init__(threshold)
        self.n_numeric = n_numeric

    def fit(self, X, y=None):
        super().fit(np.asarray(X, dtype=float)[:, : self.n_numeric])
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if len(self.columns_):
            X[:, self.columns_] = np.log1p(np.clip(X[:, self.columns_], 0, None))
        return X
