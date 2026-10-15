"""Synthetic stand-ins for the real datasets, used only by the tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = ["area", "rooms", "age", "quality"]
CATEGORICAL = ["neighborhood", "style"]


def reg_data(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Features and a strictly positive, right-skewed price with known structure."""
    rng = np.random.default_rng(seed)
    area = np.exp(rng.normal(7.2, 0.35, n))  # skewed
    rooms = rng.integers(2, 9, n)
    age = rng.integers(0, 90, n)
    quality = rng.integers(1, 11, n)
    nb = rng.choice(["A", "B", "C", "D"], n)
    style = rng.choice(["x", "y", "z"], n)
    nb_effect = pd.Series(nb).map({"A": 0.0, "B": 0.15, "C": 0.3, "D": -0.1}).to_numpy()
    log_price = (
        10.2
        + 0.0004 * area
        + 0.08 * quality
        - 0.003 * age
        + 0.02 * rooms
        + nb_effect
        + rng.normal(0, 0.12, n)
    )
    X = pd.DataFrame(
        {"area": area, "rooms": rooms, "age": age, "quality": quality, "neighborhood": nb, "style": style}
    )
    return X, np.exp(log_price)


KC_COLUMNS = [
    "id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15",
]  # fmt: skip


def kc_like(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A small frame with the King County columns, in the form the parser produces."""
    rng = np.random.default_rng(seed)
    sale = pd.Timestamp("2014-05-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    yr_built = rng.integers(1900, 2014, n)
    renovated = np.where(rng.random(n) < 0.2, rng.integers(1990, 2014, n), 0)
    return pd.DataFrame(
        {
            "id": np.arange(1000, 1000 + n),
            "date": sale,
            "price": rng.integers(80_000, 1_500_000, n).astype(float),
            "bedrooms": rng.integers(1, 7, n),
            "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], n),
            "sqft_living": rng.integers(500, 4000, n),
            "sqft_lot": rng.integers(1000, 20000, n),
            "floors": rng.choice([1.0, 1.5, 2.0], n),
            "waterfront": rng.integers(0, 2, n),
            "view": rng.integers(0, 5, n),
            "condition": rng.integers(1, 6, n),
            "grade": rng.integers(3, 13, n),
            "sqft_above": rng.integers(500, 3000, n),
            "sqft_basement": np.where(rng.random(n) < 0.4, rng.integers(100, 1500, n), 0),
            "yr_built": yr_built,
            "yr_renovated": renovated,
            "zipcode": rng.choice(["98001", "98103", "98178"], n),
            "lat": 47.2 + rng.random(n) * 0.5,
            "long": -122.5 + rng.random(n) * 0.4,
            "sqft_living15": rng.integers(800, 3500, n),
            "sqft_lot15": rng.integers(1000, 20000, n),
        }
    )


def kc_arff_text(df: pd.DataFrame) -> str:
    """Render a frame as an OpenML-style ARFF file (comma-separated @DATA section)."""
    lines = ["% synthetic test file", "@RELATION house_sales", ""]
    for c in KC_COLUMNS:
        lines.append(f"@ATTRIBUTE {c} REAL")
    lines += ["", "@DATA"]
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y%m%dT000000")
    for row in out[KC_COLUMNS].itertuples(index=False):
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines) + "\n"


def california_text(n: int = 500, seed: int = 0) -> str:
    """Raw cal_housing.data layout: lon, lat, age, rooms, bedrooms, pop, households, income, value."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        hh = int(rng.integers(50, 1500))
        rooms = int(hh * rng.uniform(3, 8))
        bed = int(hh * rng.uniform(0.8, 1.4))
        pop = int(hh * rng.uniform(1.5, 4.5))
        income = float(np.round(rng.lognormal(1.2, 0.5), 4))
        value = float(min(500001, np.round(40000 + 45000 * income + rng.normal(0, 25000), 0)))
        value = max(value, 15000.0)
        rows.append(
            f"{-124 + rng.uniform(0, 10):.2f},{32 + rng.uniform(0, 9):.2f},{int(rng.integers(1, 52))}.0,"
            f"{rooms}.0,{bed}.0,{pop}.0,{hh}.0,{income},{value}"
        )
    return "\n".join(rows) + "\n"
