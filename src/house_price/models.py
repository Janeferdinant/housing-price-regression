"""Model factory: baselines, regularised linear models, forests, boosting and a stacked ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from house_price.preprocess import build_preprocessor

MODEL_NAMES = ("mean", "ols", "ridge", "lasso", "enet", "rf", "hgb", "xgb", "stack")
STACK_BASES = ("ridge", "rf", "hgb", "xgb")
MODEL_LABELS = {
    "mean": "Mean price (baseline)",
    "ols": "Linear regression (OLS)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "rf": "Random forest",
    "hgb": "HistGradientBoosting",
    "xgb": "XGBoost",
    "stack": "Stacked ensemble",
}
TUNABLE = ("ridge", "lasso", "enet", "rf", "hgb", "xgb")


def default_params(name: str) -> dict[str, Any]:
    return {
        "mean": {},
        "ols": {},
        "stack": {},
        "ridge": {"alpha": 10.0},
        "lasso": {"alpha": 0.001},
        "enet": {"alpha": 0.001, "l1_ratio": 0.5},
        "rf": {"n_estimators": 200, "max_features": 0.5, "min_samples_leaf": 2},
        "hgb": {
            "learning_rate": 0.06,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "max_iter": 300,
        },
        "xgb": {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 1.0,
        },
    }[name]


def suggest_params(name: str, trial) -> dict[str, Any]:
    """Optuna search spaces (documented in the README)."""
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}
    if name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True)}
    if name == "enet":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
        }
    if name == "rf":
        return {
            "n_estimators": 150,
            "max_features": trial.suggest_float("max_features", 0.2, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
        }
    if name == "hgb":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500),
        }
    if name == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    raise ValueError(f"no search space for model {name!r}")


def _core(
    name: str,
    numeric: list[str],
    categorical: list[str],
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> Pipeline:
    """Preprocessing + estimator, without any target transform."""
    if name == "mean":
        return Pipeline(
            [("prep", build_preprocessor("tree", numeric, categorical)), ("reg", DummyRegressor())]
        )
    linear = {
        "ols": lambda p: LinearRegression(**p),
        "ridge": lambda p: Ridge(random_state=seed, **p),
        "lasso": lambda p: Lasso(max_iter=10_000, tol=1e-3, random_state=seed, **p),
        "enet": lambda p: ElasticNet(max_iter=10_000, tol=1e-3, random_state=seed, **p),
    }
    if name in linear:
        return Pipeline(
            [("prep", build_preprocessor("linear", numeric, categorical)), ("reg", linear[name](params))]
        )
    if name == "rf":
        reg = RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    elif name == "hgb":
        reg = HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **params)
    elif name == "xgb":
        reg = XGBRegressor(tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0, **params)
    else:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return Pipeline([("prep", build_preprocessor("tree", numeric, categorical)), ("reg", reg)])


def make_model(
    name: str,
    numeric: list[str],
    categorical: list[str],
    params: dict[str, Any] | None = None,
    seed: int = 0,
    log_target: bool = True,
    stack_params: dict[str, dict[str, Any]] | None = None,
    n_jobs: int = 1,
):
    """Unfitted estimator mapping raw feature rows to prices in dollars.

    With ``log_target`` the model is trained on log(price) and predictions are exponentiated, so
    errors are multiplicative and predictions are always positive. Preprocessing is inside the
    pipeline (and inside the stack's internal folds), so it is fitted on training rows only.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    if name == "stack":
        sp = stack_params or {}
        bases = [
            (
                b,
                _core(b, numeric, categorical, {**default_params(b), **sp.get(b, {})}, seed, n_jobs),
            )
            for b in STACK_BASES
        ]
        core: Any = StackingRegressor(
            estimators=bases,
            final_estimator=LinearRegression(positive=True),
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            n_jobs=1,
        )
    else:
        core = _core(name, numeric, categorical, {**default_params(name), **(params or {})}, seed, n_jobs)
    if not log_target:
        return core
    return TransformedTargetRegressor(regressor=core, func=np.log, inverse_func=np.exp, check_inverse=False)


def make_quantile_model(
    quantile: float,
    numeric: list[str],
    categorical: list[str],
    seed: int = 0,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Gradient-boosted quantile regressor; fit it on log(price) and exponentiate its output."""
    p = {**default_params("hgb"), **(params or {})}
    reg = HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, early_stopping=False, random_state=seed, **p
    )
    return Pipeline([("prep", build_preprocessor("tree", numeric, categorical)), ("reg", reg)])
