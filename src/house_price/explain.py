"""SHAP attributions for the gradient-boosted model (log-price scale)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_values(fitted_xgb_model, X: pd.DataFrame) -> dict[str, Any]:
    """SHAP values of the fitted XGBoost pipeline inside a log-target wrapper.

    Values are in log-price units, so exp(value) is the multiplicative effect on the predicted price.
    The tree preprocessing keeps one column per original feature, so no aggregation is needed."""
    import shap

    pipe = fitted_xgb_model.regressor_
    prep, reg = pipe.named_steps["prep"], pipe.named_steps["reg"]
    Xt = np.asarray(prep.transform(X), dtype=float)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    explainer = shap.TreeExplainer(reg)
    values = pd.DataFrame(np.asarray(explainer.shap_values(Xt)), columns=names, index=X.index)
    return {"values": values, "base_value": float(np.ravel(explainer.expected_value)[0])}


def mean_abs_importance(values: pd.DataFrame) -> pd.Series:
    return values.abs().mean().sort_values(ascending=False)
