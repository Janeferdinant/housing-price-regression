from __future__ import annotations

import numpy as np
import optuna
import pytest

from house_price.metrics import r2
from house_price.models import MODEL_NAMES, make_model, make_quantile_model, suggest_params


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts_positive_prices(reg, name):
    X, y, num, cat = reg
    m = make_model(name, num, cat, seed=0).fit(X.iloc[:200], y[:200])
    p = m.predict(X.iloc[200:])
    assert p.shape == (100,) and (p > 0).all()


def test_mean_baseline_predicts_training_mean(reg):
    X, y, num, cat = reg
    m = make_model("mean", num, cat, log_target=False).fit(X, y)
    assert np.allclose(m.predict(X), y.mean())


def test_log_target_is_inverted_to_price_scale(reg):
    X, y, num, cat = reg
    pred = make_model("ridge", num, cat, log_target=True, seed=0).fit(X, y).predict(X)
    assert 0.5 * y.mean() < pred.mean() < 2 * y.mean()  # prices, not log prices


def test_models_beat_the_mean_baseline(reg):
    X, y, num, cat = reg
    tr, te = slice(0, 200), slice(200, 300)
    for name in ("ridge", "rf", "hgb", "xgb", "stack"):
        m = make_model(name, num, cat, seed=0).fit(X.iloc[tr], y[tr])
        assert r2(y[te], m.predict(X.iloc[te])) > 0.5, name


def test_same_seed_same_predictions(reg):
    X, y, num, cat = reg
    a = make_model("xgb", num, cat, seed=3).fit(X, y).predict(X)
    b = make_model("xgb", num, cat, seed=3).fit(X, y).predict(X)
    assert np.allclose(a, b)


@pytest.mark.parametrize("name", [n for n in MODEL_NAMES if n not in ("mean", "ols", "stack")])
def test_suggested_params_accepted(reg, name):
    X, y, num, cat = reg
    trial = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0)).ask()
    params = suggest_params(name, trial)
    assert params
    make_model(name, num, cat, params=params, seed=0).fit(X, y)


def test_stack_uses_supplied_base_params(reg):
    _, _, num, cat = reg
    m = make_model("stack", num, cat, seed=0, stack_params={"ridge": {"alpha": 5.0}})
    base = dict(m.regressor.estimators)
    assert base["ridge"].named_steps["reg"].alpha == 5.0
    assert set(base) == {"ridge", "rf", "hgb", "xgb"}


def test_quantile_models_are_ordered_in_log_space(reg):
    X, y, num, cat = reg
    lo = make_quantile_model(0.05, num, cat, seed=0).fit(X.iloc[:250], np.log(y[:250]))
    hi = make_quantile_model(0.95, num, cat, seed=0).fit(X.iloc[:250], np.log(y[:250]))
    assert (hi.predict(X.iloc[250:]) >= lo.predict(X.iloc[250:])).mean() > 0.9


def test_unknown_model_raises(reg):
    _, _, num, cat = reg
    with pytest.raises(ValueError):
        make_model("nope", num, cat)
