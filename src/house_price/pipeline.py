"""Train / evaluate orchestration for one dataset. Everything written is reproducible from a seed."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price import conformal, explain, metrics, plots, runinfo
from house_price.data import load_dataset
from house_price.models import (
    MODEL_LABELS,
    MODEL_NAMES,
    make_model,
    make_quantile_model,
)
from house_price.splits import three_way_split
from house_price.tuning import nested_cv, tune_params

CV_METRICS = ("rmse", "mae", "r2", "rmsle", "mape")
ALPHA = 0.1


@dataclass
class Config:
    out: Path = Path(".")
    dataset: str = "king_county"
    data_file: Path | None = None
    data_dir: Path | None = None
    seed: int = 42
    trials: int = 15
    splits: int = 5
    inner_splits: int = 3
    jobs: int = 4
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    bootstrap: int = 1000

    @property
    def artifacts(self) -> Path:
        return self.out / "artifacts" / self.dataset

    @property
    def results(self) -> Path:
        return self.out / "results" / self.dataset

    @property
    def img(self) -> Path:
        return self.out / "docs" / "img"

    @property
    def runs(self) -> Path:
        return self.out / "docs" / "runs"


def _aggregate_nested(res, y_train: np.ndarray) -> list[dict[str, Any]]:
    folds = pd.DataFrame(res.folds)
    rows = []
    for name, oof in res.oof.items():
        sub = folds[folds["model"] == name]
        row: dict[str, Any] = {"model": name, "label": MODEL_LABELS[name]}
        for m in CV_METRICS:
            row[m] = {"mean": float(sub[m].mean()), "sd": float(sub[m].std(ddof=1))}
        row["pooled_oof"] = metrics.all_metrics(y_train, oof)
        rows.append(row)
    return rows


def select_model(rows: list[dict[str, Any]]) -> str:
    """Pre-declared rule: lowest mean nested-CV RMSLE among non-baseline models (test split unused)."""
    cands = [r for r in rows if r["model"] not in ("mean", "ols")]
    return min(cands, key=lambda r: r["rmsle"]["mean"])["model"]


def _split(cfg: Config):
    ds = load_dataset(cfg.dataset, cfg.data_file, cfg.data_dir)
    tr, ca, te = three_way_split(ds.y, cfg.seed)
    return ds, tr, ca, te


def run_train(cfg: Config) -> dict[str, Any]:
    t0 = time.time()
    cfg.results.mkdir(parents=True, exist_ok=True)
    cfg.artifacts.mkdir(parents=True, exist_ok=True)
    ds, tr, ca, te = _split(cfg)
    Xtr, ytr = ds.X.iloc[tr].reset_index(drop=True), ds.y[tr]
    numeric, categorical = ds.numeric, ds.categorical

    res = nested_cv(
        Xtr, ytr, numeric, categorical, cfg.models,
        n_splits=cfg.splits, n_trials=cfg.trials, inner_splits=cfg.inner_splits,
        seed=cfg.seed, n_jobs=cfg.jobs,
    )  # fmt: skip
    t_nested = time.time() - t0
    rows = _aggregate_nested(res, ytr)
    selected = select_model(rows)
    pd.DataFrame({"row": tr, "actual": ytr, **{k: np.round(v, 2) for k, v in res.oof.items()}}).to_csv(
        cfg.results / "oof_predictions.csv", index=False
    )

    # final models: tune on the whole training split, then fit
    t1 = time.time()
    tuned: dict[str, dict[str, Any]] = {}
    final_params: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for name in [m for m in cfg.models if m != "stack"]:
        r = tune_params(
            name, Xtr, ytr, numeric, categorical,
            seed=cfg.seed, n_trials=cfg.trials, inner_splits=cfg.inner_splits,
        )  # fmt: skip
        tuned[name] = r.params
        final_params[name] = {"params": r.params, "inner_rmsle": r.value}
        models[name] = make_model(
            name, numeric, categorical, params=r.params, seed=cfg.seed,
            log_target=name != "mean", n_jobs=cfg.jobs,
        ).fit(Xtr, ytr)  # fmt: skip
    if "stack" in cfg.models:
        models["stack"] = make_model(
            "stack", numeric, categorical, seed=cfg.seed, stack_params=tuned, n_jobs=cfg.jobs
        ).fit(Xtr, ytr)
    logy = np.log(ytr)
    models["q_lo"] = make_quantile_model(ALPHA / 2, numeric, categorical, cfg.seed).fit(Xtr, logy)
    models["q_hi"] = make_quantile_model(1 - ALPHA / 2, numeric, categorical, cfg.seed).fit(Xtr, logy)
    t_final = time.time() - t1

    joblib.dump(models, cfg.artifacts / "models.joblib")
    meta = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "numeric": numeric,
        "categorical": categorical,
        "train_rows": tr.tolist(),
        "calib_rows": ca.tolist(),
        "test_rows": te.tolist(),
        "selected": selected,
        "final_params": final_params,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "dataset_notes": ds.notes,
    }
    runinfo.dump_json(meta, cfg.artifacts / "meta.json", indent=None)
    nested = {
        "protocol": {
            "split": "random 60/20/20 train/calibration/test, stratified on price deciles",
            "outer_splits": cfg.splits,
            "inner_splits": cfg.inner_splits,
            "optuna_trials_per_model": cfg.trials,
            "tuning_objective": "mean inner-CV RMSLE (RMSE of log price), TPE sampler seeded per fold",
            "seed": cfg.seed,
            "n_train": int(len(tr)),
        },
        "selected": selected,
        "summary": rows,
        "folds": res.folds,
        "tuned_params_per_fold": res.params,
    }
    runinfo.dump_json(nested, cfg.results / "nested_cv.json")
    train_info = {
        "env": runinfo.environment(),
        "seconds_nested_cv": round(t_nested, 1),
        "seconds_final_fit": round(t_final, 1),
        "config": meta["config"],
        "final_params": final_params,
        "dataset_notes": ds.notes,
    }
    runinfo.dump_json(train_info, cfg.results / "train_run.json")
    return {"selected": selected, "rows": rows, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------------------------------


def _ci(y, p, fn, idx):
    lo, hi = metrics.ci_from_indices(y, p, fn, idx)
    return {"value": float(fn(y, p)), "lo": lo, "hi": hi}


def _interval_block(y, lo, hi, pred, n_seg: int = 5) -> dict[str, Any]:
    by_pred = pd.qcut(pd.Series(pred).rank(method="first"), q=n_seg, labels=False).to_numpy()
    by_actual = pd.qcut(pd.Series(y).rank(method="first"), q=n_seg, labels=False).to_numpy()

    def seg_table(seg, key):
        return [
            {
                "segment": s + 1,
                "n": int((seg == s).sum()),
                key + "_min": float(np.asarray(pred if key == "pred" else y)[seg == s].min()),
                key + "_max": float(np.asarray(pred if key == "pred" else y)[seg == s].max()),
                "coverage": metrics.coverage(y[seg == s], lo[seg == s], hi[seg == s]),
                "median_relative_width": metrics.median_relative_width(
                    y[seg == s], lo[seg == s], hi[seg == s]
                ),
            }
            for s in range(n_seg)
        ]

    return {
        "nominal": 1 - ALPHA,
        "coverage": metrics.coverage(y, lo, hi),
        "mean_width": metrics.mean_width(lo, hi),
        "median_relative_width": metrics.median_relative_width(y, lo, hi),
        "by_predicted_price_segment": seg_table(by_pred, "pred"),
        "by_actual_price_segment": seg_table(by_actual, "actual"),
    }


def _repeat_calibrations(
    method: str, pool_pred_log, pool_y_log, pool_lo_log, pool_hi_log, n_cal: int, n_rep: int, seed: int
) -> dict[str, float]:
    """Re-split the pooled (calibration + test) rows at random and recompute test coverage.

    The point model is fixed, so calibration and test rows stay exchangeable; this shows how much the
    realised coverage moves with the particular calibration draw."""
    rng = np.random.default_rng(seed)
    n = len(pool_y_log)
    covs = []
    for _ in range(n_rep):
        perm = rng.permutation(n)
        c, t = perm[:n_cal], perm[n_cal:]
        if method == "split_conformal":
            lo, hi = conformal.split_conformal(pool_pred_log[c], pool_y_log[c], pool_pred_log[t], ALPHA)
        else:
            lo, hi = conformal.cqr(
                pool_lo_log[c], pool_hi_log[c], pool_y_log[c], pool_lo_log[t], pool_hi_log[t], ALPHA
            )
        covs.append(metrics.coverage(pool_y_log[t], lo, hi))
    a = np.asarray(covs)
    return {
        "n_repeats": n_rep,
        "mean": float(a.mean()),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_evaluate(cfg: Config) -> dict[str, Any]:
    t0 = time.time()
    meta = runinfo.load_json(cfg.artifacts / "meta.json")
    for k in ("seed", "trials", "splits", "inner_splits"):
        setattr(cfg, k, meta["config"][k])  # evaluate reuses the protocol train used
    nested = runinfo.load_json(cfg.results / "nested_cv.json")
    train_info = runinfo.load_json(cfg.results / "train_run.json")
    models = joblib.load(cfg.artifacts / "models.joblib")
    ds = load_dataset(cfg.dataset, cfg.data_file, cfg.data_dir)
    tr, ca, te = (np.array(meta[k]) for k in ("train_rows", "calib_rows", "test_rows"))
    numeric, categorical = meta["numeric"], meta["categorical"]
    X, y = ds.X, ds.y
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]
    Xca, yca = X.iloc[ca].reset_index(drop=True), y[ca]
    Xte, yte = X.iloc[te].reset_index(drop=True), y[te]
    selected = meta["selected"]
    idx = metrics.bootstrap_indices(len(yte), cfg.bootstrap, cfg.seed)
    cv_rows = {r["model"]: r for r in nested["summary"]}

    # --- test metrics with bootstrap intervals ------------------------------------------------
    test_pred: dict[str, np.ndarray] = {}
    table: list[dict[str, Any]] = []
    for name in [m for m in MODEL_NAMES if m in models]:
        p = models[name].predict(Xte)
        test_pred[name] = p
        row = {"model": name, "label": MODEL_LABELS[name]}
        for m in ("rmse", "mae", "r2", "rmsle", "mape"):
            row[m] = _ci(yte, p, metrics.METRIC_FUNCS[m], idx)
        row["median_ape"] = float(metrics.median_ape(yte, p))
        row["bias"] = float(metrics.bias(yte, p))
        table.append(row)

    # paired bootstrap differences in RMSE (same resamples)
    def rmse_diff(a: str, b: str) -> dict[str, float]:
        d = np.array(
            [metrics.rmse(yte[i], test_pred[a][i]) - metrics.rmse(yte[i], test_pred[b][i]) for i in idx]
        )
        return {
            "rmse_diff": float(metrics.rmse(yte, test_pred[a]) - metrics.rmse(yte, test_pred[b])),
            "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)),
        }

    paired = {}
    for a, b in (
        (selected, "ridge"),
        (selected, "ols"),
        ("stack", "xgb"),
        ("stack", "hgb"),
        ("xgb", "ridge"),
    ):
        if a in test_pred and b in test_pred and a != b:
            paired[f"{a} minus {b}"] = rmse_diff(a, b)

    # --- log-target ablation (same tuned parameters, raw-price target) -------------------------
    logt = []
    for name in ("ridge", "rf", "hgb", "xgb"):
        if name not in models:
            continue
        params = meta["final_params"][name]["params"]
        raw_model = make_model(
            name, numeric, categorical, params=params, seed=cfg.seed, log_target=False, n_jobs=cfg.jobs
        ).fit(Xtr, ytr)
        p_raw = raw_model.predict(Xte)
        p_log = test_pred[name]
        logt.append(
            {
                "model": name,
                "label": MODEL_LABELS[name],
                "log_target": {
                    "rmse": metrics.rmse(yte, p_log),
                    "rmsle": metrics.rmsle(yte, p_log),
                    "mape": metrics.mape(yte, p_log),
                    "min_prediction": float(p_log.min()),
                },
                "raw_target": {
                    "rmse": metrics.rmse(yte, p_raw),
                    "rmsle": metrics.rmsle(yte, np.clip(p_raw, 1.0, None)),
                    "mape": metrics.mape(yte, p_raw),
                    "min_prediction": float(p_raw.min()),
                    "n_nonpositive": int((p_raw <= 0).sum()),
                },
            }
        )

    # --- error analysis by price segment --------------------------------------------------------
    seg = metrics.segment_report(yte, test_pred[selected], 5)

    # --- prediction intervals -------------------------------------------------------------------
    pred_ca, pred_te = models[selected].predict(Xca), test_pred[selected]
    lo_s, hi_s = conformal.split_conformal_price(pred_ca, yca, pred_te, ALPHA)
    qlo_ca, qhi_ca = models["q_lo"].predict(Xca), models["q_hi"].predict(Xca)
    qlo_te, qhi_te = models["q_lo"].predict(Xte), models["q_hi"].predict(Xte)
    lo_c_log, hi_c_log = conformal.cqr(qlo_ca, qhi_ca, np.log(yca), qlo_te, qhi_te, ALPHA)
    lo_c, hi_c = np.exp(lo_c_log), np.exp(hi_c_log)
    lo_q, hi_q = np.exp(qlo_te), np.exp(qhi_te)
    q_split = conformal.conformal_quantile(np.abs(np.log(yca) - np.log(pred_ca)), ALPHA)
    scores = np.maximum(qlo_ca - np.log(yca), np.log(yca) - qhi_ca)
    q_cqr = conformal.conformal_quantile(scores, ALPHA)
    intervals = {
        "alpha": ALPHA,
        "n_calibration": int(len(ca)),
        "n_test": int(len(te)),
        "split_conformal": {
            **_interval_block(yte, lo_s, hi_s, pred_te),
            "q_log": q_split,
            "repeated_calibration": _repeat_calibrations(
                "split_conformal",
                np.log(np.concatenate([pred_ca, pred_te])),
                np.log(np.concatenate([yca, yte])),
                None,
                None,
                len(ca),
                200,
                cfg.seed,
            ),  # fmt: skip
        },
        "cqr": {
            **_interval_block(yte, lo_c, hi_c, pred_te),
            "q_log": q_cqr,
            "repeated_calibration": _repeat_calibrations(
                "cqr",
                None,
                np.log(np.concatenate([yca, yte])),
                np.concatenate([qlo_ca, qlo_te]),
                np.concatenate([qhi_ca, qhi_te]),
                len(ca),
                200,
                cfg.seed,
            ),  # fmt: skip
        },
        "raw_quantile_band": _interval_block(yte, lo_q, hi_q, pred_te),
    }
    runinfo.dump_json(
        {"alpha": ALPHA, "q_log_split": q_split, "q_log_cqr": q_cqr, "model": selected},
        cfg.artifacts / "intervals.json",
        indent=None,
    )

    # --- SHAP (XGBoost, log-price scale) ---------------------------------------------------------
    sh = explain.shap_values(models["xgb"], Xte)
    imp = explain.mean_abs_importance(sh["values"])

    # --- figures ----------------------------------------------------------------------------------
    ds_name = cfg.dataset
    cmp_rows = [
        {
            "model": r["model"],
            "label": r["label"],
            "test_rmse": r["rmse"]["value"],
            "test_rmse_lo": r["rmse"]["lo"],
            "test_rmse_hi": r["rmse"]["hi"],
            "cv_rmse": cv_rows[r["model"]]["rmse"]["mean"],
        }
        for r in table
    ]
    plots.model_comparison(cmp_rows, cfg.img / f"{ds_name}_model_comparison.png")
    plots.predicted_vs_actual(
        yte, pred_te, lo_c, hi_c,
        f"{ds_name}: {MODEL_LABELS[selected]}, 90% CQR intervals (test)",
        cfg.img / f"{ds_name}_predicted_vs_actual.png",
    )  # fmt: skip
    plots.segment_errors(seg, cfg.img / f"{ds_name}_segment_errors.png")
    plots.coverage_by_segment(
        {
            "split conformal": pd.DataFrame(intervals["split_conformal"]["by_predicted_price_segment"]),
            "CQR": pd.DataFrame(intervals["cqr"]["by_predicted_price_segment"]),
            "raw quantile band": pd.DataFrame(intervals["raw_quantile_band"]["by_predicted_price_segment"]),
        },
        1 - ALPHA,
        cfg.img / f"{ds_name}_coverage_by_segment.png",
    )
    plots.shap_summary(sh["values"], Xte, cfg.img / f"{ds_name}_shap_summary.png")
    plots.shap_bar(imp, cfg.img / f"{ds_name}_shap_importance.png")

    # --- outputs ----------------------------------------------------------------------------------
    summary = {
        "dataset": ds_name,
        "seed": cfg.seed,
        "date": runinfo.environment()["date"],
        "n_train": int(len(tr)),
        "n_calibration": int(len(ca)),
        "n_test": int(len(te)),
        "selected": selected,
        "selection_rule": "lowest mean nested-CV RMSLE among non-baseline, non-OLS models (test split unused)",
        "bootstrap_resamples": cfg.bootstrap,
        "dataset_notes": ds.notes,
        "test": table,
        "nested_cv": nested["summary"],
        "paired_rmse_differences": paired,
        "log_target_ablation": logt,
        "segments": seg.to_dict(orient="records"),
        "intervals": intervals,
    }
    runinfo.dump_json(summary, cfg.results / "summary.json")
    runinfo.dump_json(
        {"model": "xgb", "units": "log price", "importance_test": imp.to_dict()},
        cfg.results / "shap_importance.json",
    )
    explorer = {
        "meta": {
            "dataset": ds_name,
            "seed": cfg.seed,
            "date": summary["date"],
            "n_test": int(len(te)),
            "selected": selected,
            "selected_label": MODEL_LABELS[selected],
            "alpha": ALPHA,
        },
        "models": [
            {
                "model": r["model"],
                "label": r["label"],
                "cv_rmse": cv_rows[r["model"]]["rmse"],
                "cv_rmsle": cv_rows[r["model"]]["rmsle"],
                "test_rmse": r["rmse"],
                "test_mae": r["mae"],
                "test_r2": r["r2"],
                "test_mape": r["mape"],
            }
            for r in table
        ],
        "importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in imp.items()],
        "segments": seg.to_dict(orient="records"),
        "intervals": {
            k: {
                "coverage": intervals[k]["coverage"],
                "mean_width": intervals[k]["mean_width"],
                "median_relative_width": intervals[k]["median_relative_width"],
                "by_predicted_price_segment": intervals[k]["by_predicted_price_segment"],
            }
            for k in ("split_conformal", "cqr", "raw_quantile_band")
        },
        "points": {
            "actual": np.round(yte).astype(int).tolist(),
            "pred": np.round(pred_te).astype(int).tolist(),
            "split_conformal": [np.round(lo_s).astype(int).tolist(), np.round(hi_s).astype(int).tolist()],
            "cqr": [np.round(lo_c).astype(int).tolist(), np.round(hi_c).astype(int).tolist()],
            "raw_quantile_band": [np.round(lo_q).astype(int).tolist(), np.round(hi_q).astype(int).tolist()],
        },
    }
    runinfo.dump_json(explorer, cfg.results / "explorer_data.json", indent=None)
    from house_price.report import write_run_log

    write_run_log(cfg, train_info, summary, round(time.time() - t0, 1))
    return summary
