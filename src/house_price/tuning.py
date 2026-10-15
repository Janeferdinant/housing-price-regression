"""Optuna tuning (inner loop) and nested cross-validation (outer loop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from house_price import metrics
from house_price.models import TUNABLE, make_model, suggest_params
from house_price.splits import kfold_indices

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuneResult:
    params: dict[str, Any]
    value: float  # best mean inner-CV RMSLE (RMSE of log price)
    n_trials: int


def tune_params(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_trials: int,
    inner_splits: int,
) -> TuneResult:
    """Minimise mean inner-CV RMSLE with a seeded TPE sampler, using only the (X, y) it is given."""
    if name not in TUNABLE:
        return TuneResult({}, float("nan"), 0)
    y = np.asarray(y, float)
    folds = kfold_indices(len(y), inner_splits, seed)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        losses = []
        for tr, va in folds:
            m = make_model(name, numeric, categorical, params=params, seed=seed, log_target=True)
            m.fit(X.iloc[tr], y[tr])
            losses.append(metrics.rmsle(y[va], m.predict(X.iloc[va])))
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuneResult(dict(study.best_params), float(study.best_value), n_trials)


@dataclass
class NestedCVResult:
    oof: dict[str, np.ndarray]  # model -> (n_rows,) out-of-fold predictions in dollars
    folds: list[dict[str, Any]] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)


def _outer_fold(
    fold: int,
    tr: np.ndarray,
    te: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    model_names: list[str],
    n_trials: int,
    inner_splits: int,
    seed: int,
) -> dict[str, Any]:
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
    fold_seed = seed + 1000 + fold
    tuned: dict[str, dict[str, Any]] = {}
    preds: dict[str, np.ndarray] = {}
    records, param_records = [], []
    ordered = [n for n in model_names if n != "stack"] + (["stack"] if "stack" in model_names else [])
    for name in ordered:
        if name == "stack":
            model = make_model("stack", numeric, categorical, seed=fold_seed, stack_params=tuned)
        else:
            res = tune_params(
                name, Xtr, ytr, numeric, categorical,
                seed=fold_seed, n_trials=n_trials, inner_splits=inner_splits,
            )  # fmt: skip
            tuned[name] = res.params
            param_records.append(
                {"fold": fold, "model": name, "params": res.params, "inner_rmsle": res.value}
            )
            model = make_model(
                name, numeric, categorical, params=res.params, seed=fold_seed,
                log_target=name != "mean",
            )  # fmt: skip
        p = model.fit(Xtr, ytr).predict(Xte)
        preds[name] = p
        records.append({"fold": fold, "model": name, **metrics.all_metrics(yte, p)})
    return {"te": te, "preds": preds, "records": records, "params": param_records}


def _outer_fold_limited(*args: Any) -> dict[str, Any]:
    """One outer fold with BLAS/OpenMP limited to one thread, so parallel folds do not oversubscribe."""
    with threadpool_limits(limits=1):
        return _outer_fold(*args)


def nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    model_names: list[str],
    n_splits: int = 5,
    n_trials: int = 15,
    inner_splits: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> NestedCVResult:
    """K-fold outer CV; every tuning step sees only the outer training part.

    The stacked ensemble reuses the parameters tuned for its base models in the same fold. Models
    are trained on log(price) (the mean baseline on raw price) and scored on dollar predictions."""
    y = np.asarray(y, float)
    folds = kfold_indices(len(y), n_splits, seed)
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_outer_fold_limited)(
            k, tr, te, X, y, numeric, categorical, model_names, n_trials, inner_splits, seed
        )
        for k, (tr, te) in enumerate(folds)
    )
    out = NestedCVResult(oof={})
    for res in results:
        for name, p in res["preds"].items():
            out.oof.setdefault(name, np.full(len(y), np.nan))[res["te"]] = p
        out.folds.extend(res["records"])
        out.params.extend(res["params"])
    return out
