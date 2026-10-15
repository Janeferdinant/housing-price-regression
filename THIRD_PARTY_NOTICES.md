# Third-party notices

Neither dataset is stored in this repository. Both are downloaded at run time from the URLs below and verified against the pinned SHA-256 values in `src/house_price/data.py`.

## Data

### King County house sales (Seattle area, May 2014 to May 2015)

- **License: CC0 1.0 Universal (public domain dedication).** https://creativecommons.org/publicdomain/zero/1.0/
- **License evidence, read 2026-10-15:**
  - Kaggle dataset "House Sales in King County, USA" (creator `harlfoxem`), https://www.kaggle.com/datasets/harlfoxem/housesalesprediction. Its public metadata (https://www.kaggle.com/api/v1/datasets/view/harlfoxem/housesalesprediction) returns `"licenseNameNullable":"CC0: Public Domain"`, and the page's structured data lists the licence as `{"name":"CC0: Public Domain","url":"https://creativecommons.org/publicdomain/zero/1.0/"}`. The Kaggle description reads: "This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015."
  - OpenML dataset 42092 `house_sales` version 2, https://www.openml.org/d/42092 (API record https://www.openml.org/api/v1/json/data/42092): `"licence": "CC0 Public Domain"`, `"original_data_url": "https://www.kaggle.com/harlfoxem/housesalesprediction"`.
- **No-login source used here:** the OpenML copy, https://openml.org/data/v1/download/21578898/house_sales.arff (2,513,249 bytes; SHA-256 `25817379c3f06c584ca61eb2413a8368af72fc5548848a0ceb9bb4a885e90401`; MD5 `0208b93d87877d917afdc43987e7b4b4`, the checksum OpenML publishes for the file).
- Underlying records: house sales in King County, Washington, a public record. CC0 requires no attribution; credit is given anyway: Kaggle user `harlfoxem`, via OpenML.
- Changes made: the latest sale of each house is kept (177 earlier sales of repeat-sold houses are dropped), one record with 33 bedrooms is dropped, `id` and `date` are replaced by derived columns (`sale_month_index`, `house_age`, `years_since_update`, `was_renovated`, `has_basement`), and `zipcode` is treated as a category.

### California Housing

- Source: the file distributed with scikit-learn, https://ndownloader.figshare.com/files/5976036 (SHA-256 `aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`, the same checksum scikit-learn pins), originally from StatLib, https://lib.stat.cmu.edu/datasets/houses.zip
- Reference: Pace, R. Kelley and Barry, Ronald (1997). Sparse Spatial Autoregressions. *Statistics and Probability Letters*, 33, 291 to 297.
- Origin: 1990 U.S. Census, one row per block group.
- **License status: none stated.** The scikit-learn documentation (https://scikit-learn.org/stable/datasets/real_world.html, read 2026-10-14) names StatLib as the source and Pace and Barry as the reference, and states no license for the data. The records are statistics from the U.S. Census; no explicit terms were found. The attribution above is given in good faith, and the data is not redistributed here (the explorer embeds only the model's predictions and actual median values for held-out block groups).
- Changes made: scikit-learn's feature definitions are reproduced (`AveRooms`, `AveBedrms` and `AveOccup` are ratios to `households`); the target is kept in dollars (the median house value is capped at 500,001 in the source).

## Software (installed from PyPI, not redistributed)

| Package | License |
|---|---|
| numpy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| xgboost | Apache-2.0 |
| optuna | MIT |
| shap | MIT |
| matplotlib | Matplotlib (PSF-based) license |
| joblib | BSD-3-Clause |
| pytest | MIT |
| ruff | MIT |

Conformalised quantile regression follows Romano, Patterson and Candes (2019), "Conformalized Quantile Regression", NeurIPS; the implementation here is original. No pretrained models or third-party source code are included.
