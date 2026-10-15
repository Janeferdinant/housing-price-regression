"""Static figures for the README (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

BLUE, ORANGE, GREEN, VERMILION, PURPLE, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#6b7280",
)
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def model_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    """Test RMSE (bootstrap CI) and nested-CV mean for every non-baseline model."""
    rows = [r for r in rows if r["model"] != "mean"]
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        ax.errorbar(
            r["test_rmse"], i, xerr=[[r["test_rmse"] - r["test_rmse_lo"]], [r["test_rmse_hi"] - r["test_rmse"]]],
            fmt="o", color=BLUE, capsize=3, label="test (95% bootstrap CI)" if i == 0 else None,
        )  # fmt: skip
        ax.plot(r["cv_rmse"], i, "D", color=ORANGE, label="nested CV mean" if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (US dollars)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, path)


def predicted_vs_actual(
    y: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray, title: str, path: Path, n_show: int = 400
) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(n_show, len(y)), replace=False)
    order = idx[np.argsort(y[idx])]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    inside = (y[order] >= lo[order]) & (y[order] <= hi[order])
    ax.vlines(y[order], lo[order], hi[order], color=np.where(inside, BLUE, VERMILION), alpha=0.35, lw=1)
    ax.scatter(y[order], pred[order], s=7, color=BLUE, zorder=3)
    lim = [min(y.min(), lo.min()) * 0.9, max(y.max(), hi.max()) * 1.1]
    ax.plot(lim, lim, "--", color=GREY, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("actual price (USD, log axis)")
    ax.set_ylabel("predicted price (USD, log axis)")
    ax.set_title(title, fontsize=9)
    _save(fig, path)


def segment_errors(report: pd.DataFrame, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    x = np.arange(len(report))
    labels = [f"{int(r.price_min / 1000)}-{int(r.price_max / 1000)}k" for r in report.itertuples()]
    a.bar(x, report["mape"] * 100, color=BLUE)
    a.set_ylabel("mean abs. % error")
    b.bar(x, report["bias_pct"] * 100, color=np.where(report["bias_pct"] > 0, ORANGE, VERMILION))
    b.axhline(0, color="black", lw=0.6)
    b.set_ylabel("mean signed % error (over-prediction > 0)")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_xlabel("actual price segment")
    _save(fig, path)


def coverage_by_segment(tables: dict[str, pd.DataFrame], nominal: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    w = 0.8 / max(1, len(tables))
    for k, ((name, tab), colour) in enumerate(zip(tables.items(), [BLUE, ORANGE, GREEN], strict=False)):
        x = np.arange(len(tab)) + k * w
        ax.bar(x, tab["coverage"], w, color=colour, label=name)
    ax.axhline(nominal, color="black", ls="--", lw=1)
    ax.text(len(tab) - 0.4, nominal + 0.01, f"nominal {nominal:.0%}", ha="right", fontsize=7)
    ax.set_xticks(np.arange(len(tab)) + w * (len(tables) - 1) / 2)
    ax.set_xticklabels([f"{i + 1}" for i in range(len(tab))])
    ax.set_xlabel("segment of predicted price (1 = cheapest)")
    ax.set_ylabel("empirical coverage on test")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    _save(fig, path)


def shap_summary(values: pd.DataFrame, X: pd.DataFrame, path: Path, top: int = 12) -> None:
    order = values.abs().mean().sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(6.6, 0.38 * top + 1.2))
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        v = values[f].to_numpy()
        y = i + rng.uniform(-0.28, 0.28, len(v))
        col = pd.to_numeric(X[f], errors="coerce")
        if col.notna().all() and col.nunique() > 8:
            ax.scatter(v, y, c=col.rank(pct=True).to_numpy(), cmap="coolwarm", s=6, alpha=0.7, linewidths=0)
        else:
            ax.scatter(v, y, color=GREY, s=6, alpha=0.5, linewidths=0)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (log price); continuous features coloured low (blue) to high (red)")
    ax.xaxis.label.set_size(8)
    _save(fig, path)


def shap_bar(importance: pd.Series, path: Path, top: int = 12) -> None:
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 0.3 * len(imp) + 1.0))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("mean |SHAP| (log price), test split")
    _save(fig, path)
