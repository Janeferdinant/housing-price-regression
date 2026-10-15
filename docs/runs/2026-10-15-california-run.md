# california run, 2026-10-15

Produced by `house-price train` followed by `house-price evaluate`. Every number in the README and the explorer is read from the JSON files this run wrote to `results/california/`.

## Setup
- Date: 2026-10-15
- Hardware: AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, 16 logical CPUs, GPU: none (CPU only); Windows 10
- Python 3.11.9; numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.2.0, optuna 5.0.0, shap 0.51.0, matplotlib 3.11.2
- Seed: 42 (split, tuning samplers, model seeds, bootstrap); per-fold tuning seed is `seed + 1000 + fold`.
- Split: 12384 train / 4128 calibration / 4128 test rows (random, stratified on price deciles). The calibration part is used only for the interval corrections.
- Nested CV on the training part: 5 outer folds, 3 inner folds, 15 Optuna TPE trials per model, objective inner-CV RMSLE.
- Bootstrap: 1000 resamples of the test rows.
- Wall clock: nested CV 266.0 s, final fit 298.6 s, evaluate 17.1 s.
- Dataset notes: {'n_rows': 20640, 'n_censored_at_500001': 965, 'target': 'median house value of a census block group, US dollars, capped at 500001'}

```
house-price train --dataset california --seed 42 --trials 15 --splits 5 --inner-splits 3
house-price evaluate --dataset california --bootstrap 1000
```

Selected model (rule: lowest mean nested-CV RMSLE among non-baseline, non-OLS models (test split unused)): `stack`.

## Nested cross-validation (training rows only, mean +/- sd over outer folds)

| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |
|---|---|---|---|---|---|
| Mean price (baseline) | 115,701 +/- 1,398 | 91,345 | -0.000 | 0.590 | 0.621 |
| Linear regression (OLS) | 66,852 +/- 1,343 | 46,056 | 0.666 | 0.321 | 0.248 |
| Ridge | 66,862 +/- 1,342 | 46,064 | 0.666 | 0.321 | 0.249 |
| Lasso | 66,869 +/- 1,339 | 46,069 | 0.666 | 0.321 | 0.249 |
| Elastic net | 66,865 +/- 1,341 | 46,065 | 0.666 | 0.321 | 0.249 |
| Random forest | 51,567 +/- 1,161 | 32,797 | 0.801 | 0.235 | 0.170 |
| HistGradientBoosting | 47,443 +/- 1,637 | 30,418 | 0.831 | 0.224 | 0.160 |
| XGBoost | 47,041 +/- 1,742 | 29,920 | 0.834 | 0.219 | 0.156 |
| Stacked ensemble | 46,493 +/- 1,640 | 29,581 | 0.838 | 0.217 | 0.154 |

## Hold-out test split (95% bootstrap intervals)

| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |
|---|---|---|---|---|---|
| Mean price (baseline) | 114,878 [112,253, 117,619] | 90,909 [88,854, 93,101] | -0.000 [-0.001, -0.000] | 0.593 [0.580, 0.606] | 0.630 [0.605, 0.654] |
| Linear regression (OLS) | 67,932 [65,557, 70,484] | 46,634 [45,248, 48,185] | 0.650 [0.628, 0.674] | 0.331 [0.319, 0.343] | 0.257 [0.246, 0.269] |
| Ridge | 67,934 [65,557, 70,485] | 46,635 [45,249, 48,186] | 0.650 [0.628, 0.674] | 0.331 [0.319, 0.343] | 0.257 [0.246, 0.269] |
| Lasso | 67,940 [65,555, 70,489] | 46,641 [45,255, 48,190] | 0.650 [0.628, 0.673] | 0.331 [0.319, 0.343] | 0.257 [0.246, 0.269] |
| Elastic net | 67,938 [65,556, 70,488] | 46,639 [45,253, 48,189] | 0.650 [0.628, 0.673] | 0.331 [0.319, 0.343] | 0.257 [0.246, 0.269] |
| Random forest | 51,843 [49,551, 54,301] | 33,035 [31,833, 34,330] | 0.796 [0.779, 0.813] | 0.246 [0.234, 0.261] | 0.179 [0.169, 0.191] |
| HistGradientBoosting | 47,721 [45,478, 50,067] | 30,504 [29,412, 31,715] | 0.827 [0.811, 0.843] | 0.229 [0.216, 0.244] | 0.165 [0.155, 0.178] |
| XGBoost | 48,005 [45,656, 50,544] | 30,505 [29,488, 31,734] | 0.825 [0.807, 0.842] | 0.228 [0.215, 0.243] | 0.164 [0.154, 0.175] |
| Stacked ensemble | 47,074 [44,775, 49,466] | 29,904 [28,846, 31,082] | 0.832 [0.816, 0.847] | 0.224 [0.211, 0.239] | 0.161 [0.151, 0.173] |

## Paired RMSE differences on the test split (negative = first is better)

- stack minus ridge: -20,860 [-22,753, -19,030]
- stack minus ols: -20,858 [-22,752, -19,028]
- stack minus xgb: -931 [-1,577, -447]
- stack minus hgb: -647 [-1,148, -97]
- xgb minus ridge: -19,929 [-22,018, -17,830]

## Log-target ablation (same tuned parameters, trained on raw vs log price)

| Model | RMSE log / raw ($) | RMSLE log / raw | MAPE log / raw | raw non-positive predictions |
|---|---|---|---|---|
| Ridge | 67,934 / 71,328 | 0.331 / 1.222 | 0.257 / 0.329 | 42 |
| Random forest | 51,843 / 51,610 | 0.246 / 0.259 | 0.179 / 0.201 | 0 |
| HistGradientBoosting | 47,721 / 46,761 | 0.229 / 0.236 | 0.165 / 0.177 | 0 |
| XGBoost | 48,005 / 46,566 | 0.228 / 0.237 | 0.164 / 0.178 | 0 |

## Error by actual-price segment (stack, test split)

| Segment | n | price range ($) | MAE ($) | RMSE ($) | MAPE | mean signed error |
|---|---|---|---|---|---|---|
| 1 | 826 | 14,999 to 107,100 | 15,915 | 28,782 | 0.245 | +19.9% |
| 2 | 825 | 107,300 to 157,300 | 20,235 | 29,994 | 0.153 | +5.3% |
| 3 | 826 | 157,300 to 209,400 | 23,151 | 33,647 | 0.128 | +1.7% |
| 4 | 825 | 209,400 to 289,900 | 31,560 | 41,608 | 0.129 | -2.2% |
| 5 | 826 | 290,000 to 500,001 | 58,648 | 80,537 | 0.148 | -9.1% |

## 90% prediction intervals (nominal 90%; calibration n = 4128, test n = 4128)

| Method | test coverage | mean width ($) | median width / price | coverage over 200 re-drawn calibration sets (mean, 5th-95th pct) |
|---|---|---|---|---|
| Split conformal (log scale) | 0.889 | 128,278 | 0.640 | 0.901 (0.890 to 0.911) |
| Conformalised quantile regression | 0.891 | 147,092 | 0.725 | 0.900 (0.889 to 0.910) |
| Raw 5%/95% quantile band (no correction) | 0.837 | 133,604 | 0.656 | n/a |

Coverage of Split conformal by predicted-price segment (1 = cheapest): 0.892, 0.857, 0.898, 0.875, 0.923
Coverage of Split conformal by actual-price segment: 0.860, 0.903, 0.908, 0.922, 0.852

Coverage of CQR by predicted-price segment (1 = cheapest): 0.890, 0.869, 0.906, 0.885, 0.903
Coverage of CQR by actual-price segment: 0.824, 0.914, 0.942, 0.949, 0.823

Coverage of Raw band by predicted-price segment (1 = cheapest): 0.849, 0.840, 0.866, 0.859, 0.770
Coverage of Raw band by actual-price segment: 0.776, 0.885, 0.908, 0.924, 0.691

