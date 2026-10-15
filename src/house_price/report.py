"""Markdown run logs written next to the results (docs/runs/)."""

from __future__ import annotations

from typing import Any

from house_price.runinfo import environment


def _ci(d: dict[str, float], nd: int = 0) -> str:
    return f"{d['value']:,.{nd}f} [{d['lo']:,.{nd}f}, {d['hi']:,.{nd}f}]"


def write_run_log(cfg, train_info: dict[str, Any], s: dict[str, Any], seconds_eval: float) -> None:
    env = environment()
    t_env = train_info["env"]
    pk = ", ".join(f"{k} {v}" for k, v in t_env["packages"].items())
    c = train_info["config"]
    lines = [
        f"# {s['dataset']} run, {env['date']}",
        "",
        "Produced by `house-price train` followed by `house-price evaluate`. Every number in the README and "
        "the explorer is read from the JSON files this run wrote to `results/" + s["dataset"] + "/`.",
        "",
        "## Setup",
        f"- Date: {t_env['date']}",
        f"- Hardware: {t_env['cpu']}, {t_env['cpu_count']} logical CPUs, {t_env['gpu']}; {t_env['os']}",
        f"- Python {t_env['python']}; {pk}",
        f"- Seed: {s['seed']} (split, tuning samplers, model seeds, bootstrap); per-fold tuning seed is "
        "`seed + 1000 + fold`.",
        f"- Split: {s['n_train']} train / {s['n_calibration']} calibration / {s['n_test']} test rows "
        "(random, stratified on price deciles). The calibration part is used only for the interval corrections.",
        f"- Nested CV on the training part: {c['splits']} outer folds, {c['inner_splits']} inner folds, "
        f"{c['trials']} Optuna TPE trials per model, objective inner-CV RMSLE.",
        f"- Bootstrap: {s['bootstrap_resamples']} resamples of the test rows.",
        f"- Wall clock: nested CV {train_info['seconds_nested_cv']} s, final fit {train_info['seconds_final_fit']} s, "
        f"evaluate {seconds_eval} s.",
        f"- Dataset notes: {train_info['dataset_notes']}",
        "",
        "```",
        f"house-price train --dataset {s['dataset']} --seed {s['seed']} --trials {c['trials']} "
        f"--splits {c['splits']} --inner-splits {c['inner_splits']}",
        f"house-price evaluate --dataset {s['dataset']} --bootstrap {s['bootstrap_resamples']}",
        "```",
        "",
        f"Selected model (rule: {s['selection_rule']}): `{s['selected']}`.",
        "",
        "## Nested cross-validation (training rows only, mean +/- sd over outer folds)",
        "",
        "| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["nested_cv"]:
        lines.append(
            f"| {r['label']} | {r['rmse']['mean']:,.0f} +/- {r['rmse']['sd']:,.0f} | "
            f"{r['mae']['mean']:,.0f} | {r['r2']['mean']:.3f} | {r['rmsle']['mean']:.3f} | "
            f"{r['mape']['mean']:.3f} |"
        )
    lines += [
        "",
        "## Hold-out test split (95% bootstrap intervals)",
        "",
        "| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |",
        "|---|---|---|---|---|---|",
    ]
    for r in s["test"]:
        lines.append(
            f"| {r['label']} | {_ci(r['rmse'])} | {_ci(r['mae'])} | {_ci(r['r2'], 3)} | "
            f"{_ci(r['rmsle'], 3)} | {_ci(r['mape'], 3)} |"
        )
    lines += ["", "## Paired RMSE differences on the test split (negative = first is better)", ""]
    for k, v in s["paired_rmse_differences"].items():
        lines.append(f"- {k}: {v['rmse_diff']:+,.0f} [{v['lo']:+,.0f}, {v['hi']:+,.0f}]")
    lines += [
        "",
        "## Log-target ablation (same tuned parameters, trained on raw vs log price)",
        "",
        "| Model | RMSE log / raw ($) | RMSLE log / raw | MAPE log / raw | raw non-positive predictions |",
        "|---|---|---|---|---|",
    ]
    for r in s["log_target_ablation"]:
        a, b = r["log_target"], r["raw_target"]
        lines.append(
            f"| {r['label']} | {a['rmse']:,.0f} / {b['rmse']:,.0f} | {a['rmsle']:.3f} / {b['rmsle']:.3f} | "
            f"{a['mape']:.3f} / {b['mape']:.3f} | {b['n_nonpositive']} |"
        )
    lines += [
        "",
        f"## Error by actual-price segment ({s['selected']}, test split)",
        "",
        "| Segment | n | price range ($) | MAE ($) | RMSE ($) | MAPE | mean signed error |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in s["segments"]:
        lines.append(
            f"| {g['segment']} | {g['n']} | {g['price_min']:,.0f} to {g['price_max']:,.0f} | {g['mae']:,.0f} | "
            f"{g['rmse']:,.0f} | {g['mape']:.3f} | {g['bias_pct']:+.1%} |"
        )
    iv = s["intervals"]
    lines += [
        "",
        f"## 90% prediction intervals (nominal {1 - iv['alpha']:.0%}; calibration n = {iv['n_calibration']}, "
        f"test n = {iv['n_test']})",
        "",
        "| Method | test coverage | mean width ($) | median width / price | coverage over 200 re-drawn calibration sets (mean, 5th-95th pct) |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("split_conformal", "Split conformal (log scale)"),
        ("cqr", "Conformalised quantile regression"),
        ("raw_quantile_band", "Raw 5%/95% quantile band (no correction)"),
    ):
        b = iv[key]
        rep = b.get("repeated_calibration")
        rep_s = f"{rep['mean']:.3f} ({rep['p05']:.3f} to {rep['p95']:.3f})" if rep else "n/a"
        lines.append(
            f"| {label} | {b['coverage']:.3f} | {b['mean_width']:,.0f} | {b['median_relative_width']:.3f} | {rep_s} |"
        )
    for key, label in (
        ("split_conformal", "Split conformal"),
        ("cqr", "CQR"),
        ("raw_quantile_band", "Raw band"),
    ):
        lines += [
            "",
            f"Coverage of {label} by predicted-price segment (1 = cheapest): "
            + ", ".join(f"{g['coverage']:.3f}" for g in iv[key]["by_predicted_price_segment"]),
            f"Coverage of {label} by actual-price segment: "
            + ", ".join(f"{g['coverage']:.3f}" for g in iv[key]["by_actual_price_segment"]),
        ]
    lines.append("")
    path = cfg.runs / f"{env['date']}-{s['dataset']}-run.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
