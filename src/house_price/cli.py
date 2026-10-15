"""Command line interface: ``house-price train | evaluate | predict``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from house_price import runinfo
from house_price.data import DATASET_NAMES, cast_categoricals, clean_king_county
from house_price.models import MODEL_NAMES
from house_price.pipeline import Config, run_evaluate, run_train

DISCLAIMER = (
    "NOTE: research demo trained on old, small public data (King County 2014-2015, California 1990). "
    "Not a valuation or appraisal tool."
)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    p.add_argument("--out", type=Path, default=Path("."), help="project root for artifacts/ results/ docs/")
    p.add_argument("--data-file", type=Path, default=None, help="local raw data file (skips the download)")
    p.add_argument("--data-dir", type=Path, default=None, help="cache directory for the verified download")
    p.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="house-price", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="nested CV, tuning and final fits")
    _common(t)
    t.add_argument("--trials", type=int, default=15, help="Optuna trials per model and fold")
    t.add_argument("--splits", type=int, default=5, help="outer CV folds")
    t.add_argument("--inner-splits", type=int, default=3, help="inner CV folds used by Optuna")
    t.add_argument("--jobs", type=int, default=4, help="parallel outer folds")
    t.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)

    e = sub.add_parser("evaluate", help="test metrics, intervals, error analysis, SHAP, figures")
    _common(e)
    e.add_argument("--bootstrap", type=int, default=1000)
    e.add_argument("--jobs", type=int, default=4)

    pr = sub.add_parser("predict", help="predict prices with 90%% intervals from a CSV")
    pr.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    pr.add_argument("--out", type=Path, default=Path("."))
    pr.add_argument("--input", type=Path, required=True, help="CSV with the raw feature columns")
    pr.add_argument("--output", type=Path, default=None, help="write predictions here (default: stdout)")
    pr.add_argument("--method", choices=["cqr", "split_conformal"], default="cqr")
    return ap


def _predict(a: argparse.Namespace) -> None:
    arts = a.out / "artifacts" / a.dataset
    meta = runinfo.load_json(arts / "meta.json")
    models = joblib.load(arts / "models.joblib")
    iv = runinfo.load_json(arts / "intervals.json")
    if a.dataset == "king_county":
        raw = pd.read_csv(a.input, dtype={"zipcode": str})
        df = clean_king_county(raw, strict=False, training=False)
    else:
        df = pd.read_csv(a.input)
        if "AveRooms" not in df.columns:
            raise SystemExit("california input needs the 8 scikit-learn feature columns")
    cols = meta["numeric"] + meta["categorical"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"input is missing columns: {missing}")
    X = cast_categoricals(df[cols], meta["categorical"])
    point = models[meta["selected"]].predict(X)
    if a.method == "split_conformal":
        lo, hi = point * np.exp(-iv["q_log_split"]), point * np.exp(iv["q_log_split"])
    else:
        lo = np.exp(models["q_lo"].predict(X) - iv["q_log_cqr"])
        hi = np.exp(models["q_hi"].predict(X) + iv["q_log_cqr"])
    out = df.copy()
    out["predicted_price"] = point.round(0)
    out["lower"] = lo.round(0)
    out["upper"] = hi.round(0)
    out["interval"] = f"{1 - iv['alpha']:.0%} {a.method}"
    out["model"] = meta["selected"]
    if a.output:
        out.to_csv(a.output, index=False)
        print(f"wrote {len(out)} predictions to {a.output}")
    else:
        print(out[["predicted_price", "lower", "upper", "interval", "model"]].to_string(index=False))
    print(DISCLAIMER, file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.command == "predict":
        _predict(a)
        return
    cfg = Config(out=a.out, dataset=a.dataset, data_file=a.data_file, data_dir=a.data_dir, seed=a.seed)
    for name in ("trials", "splits", "inner_splits", "jobs", "models", "bootstrap"):
        if hasattr(a, name):
            setattr(cfg, name, getattr(a, name))
    if a.command == "train":
        r = run_train(cfg)
        print(f"{a.dataset}: selected model {r['selected']} ({r['seconds']} s)")
    elif a.command == "evaluate":
        s = run_evaluate(cfg)
        print(f"{a.dataset}: selected {s['selected']}; split conformal coverage "
              f"{s['intervals']['split_conformal']['coverage']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
