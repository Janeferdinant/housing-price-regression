# king_county run, 2026-10-15

Produced by `house-price train` followed by `house-price evaluate`. Every number in the README and the explorer is read from the JSON files this run wrote to `results/king_county/`.

## Setup
- Date: 2026-10-15
- Hardware: AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD, 16 logical CPUs, GPU: none (CPU only); Windows 10
- Python 3.11.9; numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.2.0, optuna 5.0.0, shap 0.51.0, matplotlib 3.11.2
- Seed: 42 (split, tuning samplers, model seeds, bootstrap); per-fold tuning seed is `seed + 1000 + fold`.
- Split: 12861 train / 4287 calibration / 4287 test rows (random, stratified on price deciles). The calibration part is used only for the interval corrections.
- Nested CV on the training part: 5 outer folds, 3 inner folds, 15 Optuna TPE trials per model, objective inner-CV RMSLE.
- Bootstrap: 1000 resamples of the test rows.
- Wall clock: nested CV 470.6 s, final fit 405.7 s, evaluate 18.6 s.
- Dataset notes: {'rule_repeat_sales': 'kept the latest sale of each house id', 'rule_bedrooms': 'dropped bedrooms > 15', 'n_rows': 21435, 'n_raw_rows': 21613, 'date_range': '2014-05-02 to 2015-05-27'}

```
house-price train --dataset king_county --seed 42 --trials 15 --splits 5 --inner-splits 3
house-price evaluate --dataset king_county --bootstrap 1000
```

Selected model (rule: lowest mean nested-CV RMSLE among non-baseline, non-OLS models (test split unused)): `stack`.

## Nested cross-validation (training rows only, mean +/- sd over outer folds)

| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |
|---|---|---|---|---|---|
| Mean price (baseline) | 365,532 +/- 29,342 | 233,986 | -0.001 | 0.546 | 0.532 |
| Linear regression (OLS) | 127,524 +/- 9,671 | 72,223 | 0.877 | 0.177 | 0.133 |
| Ridge | 127,609 +/- 9,581 | 72,285 | 0.877 | 0.177 | 0.133 |
| Lasso | 128,628 +/- 9,893 | 72,862 | 0.875 | 0.178 | 0.134 |
| Elastic net | 128,164 +/- 9,744 | 72,617 | 0.876 | 0.177 | 0.133 |
| Random forest | 135,642 +/- 17,462 | 70,253 | 0.863 | 0.174 | 0.125 |
| HistGradientBoosting | 118,081 +/- 15,522 | 64,219 | 0.896 | 0.161 | 0.118 |
| XGBoost | 114,764 +/- 13,199 | 63,275 | 0.901 | 0.160 | 0.116 |
| Stacked ensemble | 112,264 +/- 12,613 | 62,224 | 0.906 | 0.157 | 0.115 |

## Hold-out test split (95% bootstrap intervals)

| Model | RMSE ($) | MAE ($) | R2 | RMSLE | MAPE |
|---|---|---|---|---|---|
| Mean price (baseline) | 384,071 [340,302, 430,787] | 235,014 [225,856, 244,056] | -0.000 [-0.001, -0.000] | 0.544 [0.532, 0.556] | 0.529 [0.514, 0.544] |
| Linear regression (OLS) | 133,150 [117,861, 151,813] | 72,936 [69,746, 76,669] | 0.880 [0.861, 0.898] | 0.179 [0.173, 0.186] | 0.133 [0.129, 0.137] |
| Ridge | 132,323 [118,065, 149,281] | 72,953 [69,756, 76,596] | 0.881 [0.864, 0.898] | 0.180 [0.173, 0.186] | 0.133 [0.129, 0.138] |
| Lasso | 133,170 [118,703, 149,872] | 73,560 [70,319, 77,121] | 0.880 [0.863, 0.898] | 0.181 [0.174, 0.187] | 0.134 [0.130, 0.138] |
| Elastic net | 132,474 [118,383, 148,829] | 73,231 [70,015, 76,699] | 0.881 [0.864, 0.898] | 0.180 [0.174, 0.186] | 0.134 [0.129, 0.138] |
| Random forest | 151,780 [125,690, 181,642] | 70,635 [66,994, 74,920] | 0.844 [0.813, 0.873] | 0.176 [0.169, 0.182] | 0.124 [0.121, 0.129] |
| HistGradientBoosting | 133,024 [103,668, 166,174] | 62,741 [59,233, 66,601] | 0.880 [0.844, 0.913] | 0.160 [0.154, 0.166] | 0.114 [0.111, 0.118] |
| XGBoost | 130,554 [106,122, 158,173] | 63,819 [60,657, 67,567] | 0.884 [0.858, 0.911] | 0.161 [0.155, 0.168] | 0.116 [0.113, 0.121] |
| Stacked ensemble | 124,915 [101,673, 152,244] | 61,360 [58,202, 64,974] | 0.894 [0.868, 0.918] | 0.157 [0.151, 0.164] | 0.113 [0.109, 0.117] |

## Paired RMSE differences on the test split (negative = first is better)

- stack minus ridge: -7,408 [-21,628, +7,730]
- stack minus ols: -8,235 [-21,764, +6,646]
- stack minus xgb: -5,640 [-11,237, -1,713]
- stack minus hgb: -8,109 [-14,365, -1,576]
- xgb minus ridge: -1,768 [-17,886, +17,501]

## Log-target ablation (same tuned parameters, trained on raw vs log price)

| Model | RMSE log / raw ($) | RMSLE log / raw | MAPE log / raw | raw non-positive predictions |
|---|---|---|---|---|
| Ridge | 132,323 / 194,124 | 0.180 / 1.214 | 0.133 / 0.215 | 39 |
| Random forest | 151,780 / 139,932 | 0.176 / 0.175 | 0.124 / 0.128 | 0 |
| HistGradientBoosting | 133,024 / 142,958 | 0.160 / 0.166 | 0.114 / 0.120 | 0 |
| XGBoost | 130,554 / 130,606 | 0.161 / 0.167 | 0.116 / 0.123 | 0 |

## Error by actual-price segment (stack, test split)

| Segment | n | price range ($) | MAE ($) | RMSE ($) | MAPE | mean signed error |
|---|---|---|---|---|---|---|
| 1 | 858 | 100,000 to 299,500 | 33,166 | 48,554 | 0.151 | +8.4% |
| 2 | 857 | 299,800 to 400,000 | 35,238 | 56,297 | 0.102 | +2.4% |
| 3 | 857 | 400,000 to 520,000 | 46,325 | 65,541 | 0.101 | +0.9% |
| 4 | 857 | 520,000 to 703,000 | 58,757 | 79,863 | 0.098 | -1.8% |
| 5 | 858 | 704,000 to 7,700,000 | 133,261 | 248,555 | 0.112 | -3.8% |

## 90% prediction intervals (nominal 90%; calibration n = 4287, test n = 4287)

| Method | test coverage | mean width ($) | median width / price | coverage over 200 re-drawn calibration sets (mean, 5th-95th pct) |
|---|---|---|---|---|
| Split conformal (log scale) | 0.894 | 258,353 | 0.481 | 0.901 (0.888 to 0.910) |
| Conformalised quantile regression | 0.891 | 287,103 | 0.478 | 0.900 (0.888 to 0.909) |
| Raw 5%/95% quantile band (no correction) | 0.844 | 256,586 | 0.420 | n/a |

Coverage of Split conformal by predicted-price segment (1 = cheapest): 0.840, 0.898, 0.925, 0.928, 0.876
Coverage of Split conformal by actual-price segment: 0.818, 0.922, 0.922, 0.936, 0.871

Coverage of CQR by predicted-price segment (1 = cheapest): 0.896, 0.894, 0.882, 0.893, 0.892
Coverage of CQR by actual-price segment: 0.860, 0.918, 0.903, 0.908, 0.867

Coverage of Raw band by predicted-price segment (1 = cheapest): 0.864, 0.851, 0.831, 0.833, 0.844
Coverage of Raw band by actual-price segment: 0.823, 0.868, 0.865, 0.856, 0.810

