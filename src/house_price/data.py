"""Download, verify and clean the two housing datasets.

* King County house sales (Seattle area, May 2014 to May 2015), CC0 public domain. No-login copy hosted
  by OpenML (dataset 42092, file 21578898), from the Kaggle dataset "House Sales in King County, USA".
* California Housing: Pace, R. K. and Barry, R. (1997). Sparse Spatial Autoregressions. Statistics and
  Probability Letters 33, 291-297 (1990 U.S. Census block groups); the file scikit-learn distributes.

Files are fetched at run time and checked against pinned SHA-256 values; raw data is never committed.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KC_URL = "https://openml.org/data/v1/download/21578898/house_sales.arff"
KC_SHA256 = "25817379c3f06c584ca61eb2413a8368af72fc5548848a0ceb9bb4a885e90401"
KC_MD5_OPENML = "0208b93d87877d917afdc43987e7b4b4"  # the checksum OpenML publishes for this file
KC_NAME = "house_sales.arff"
CAL_URL = "https://ndownloader.figshare.com/files/5976036"
CAL_SHA256 = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
CAL_NAME = "cal_housing.tgz"
CAL_MEMBER = "CaliforniaHousing/cal_housing.data"

CAL_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
CAL_CENSOR = 500001  # median house values are capped at this value in the source

EXPECTED_KC_COLUMNS = [
    "id", "date", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15",
]  # fmt: skip
KC_MAX_BEDROOMS = 15  # drops one record with 33 bedrooms in 1,620 sq ft, an evident entry error
KC_NOMINAL = ("zipcode",)


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its pinned SHA-256."""


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray  # sale price in US dollars
    numeric: list[str]
    categorical: list[str]
    notes: dict[str, Any] = field(default_factory=dict)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_data_dir() -> Path:
    return Path(os.environ.get("HOUSE_PRICE_DATA_DIR", "data"))


def _urlopen_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "house-price-study/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (fixed https URL)
        return resp.read()


def download_file(
    dest_dir: Path | str | None,
    url: str,
    sha256: str,
    filename: str,
    opener: Callable[[str], bytes] | None = None,
) -> Path:
    """Fetch ``url`` into ``dest_dir/filename`` unless a copy with the right SHA-256 is cached."""
    dest = Path(dest_dir) if dest_dir is not None else default_data_dir()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename
    if path.exists() and sha256_file(path) != sha256:
        path.unlink()  # corrupt or stale cache: fetch again
    if not path.exists():
        payload = (opener or _urlopen_bytes)(url)
        got = sha256_bytes(payload)
        if got != sha256:
            raise ChecksumError(f"SHA-256 mismatch for {url}: expected {sha256}, got {got}")
        path.write_bytes(payload)
    return path


# ------------------------------------------------------------------------------ King County


def parse_arff_table(text: str) -> pd.DataFrame:
    """Read an ARFF file whose data section is plain comma-separated rows (as OpenML's is)."""
    names = re.findall(r"(?im)^@ATTRIBUTE\s+(\S+)", text)
    head, sep, body = text.partition("@DATA")
    if not sep or not names:
        raise ValueError("not an ARFF file with @ATTRIBUTE and @DATA sections")
    df = pd.read_csv(io.StringIO(body.strip()), header=None, names=names, dtype={"zipcode": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    return df


def clean_king_county(df: pd.DataFrame, strict: bool = True, training: bool = True) -> pd.DataFrame:
    """Apply documented, row-wise cleaning to the King County sales table.

    Training only: keep the latest sale of each house (``id``), because the same house sold twice would
    otherwise sit in both the training and the test split, and drop the one record with an impossible
    bedroom count. Both are fixed rules that do not use the price or any split.
    Always: derive ``sale_month_index``, ``house_age``, ``years_since_update``, ``was_renovated`` and
    ``has_basement`` from the sale date and the build and renovation years; drop ``id`` and ``date``.
    """
    if strict and set(EXPECTED_KC_COLUMNS) - set(df.columns):
        raise ValueError("unexpected King County columns; the file does not match the documented schema")
    out = df.copy()
    if training:
        if {"id", "date"} <= set(out.columns):
            out = (
                out.sort_values("date", kind="stable").drop_duplicates(subset="id", keep="last").sort_index()
            )
        if "bedrooms" in out:
            out = out[out["bedrooms"] <= KC_MAX_BEDROOMS]
    date = pd.to_datetime(out["date"])
    out["sale_month_index"] = (date.dt.year - 2014) * 12 + date.dt.month - 5
    out["house_age"] = date.dt.year - out["yr_built"]
    updated = np.maximum(out["yr_built"], out["yr_renovated"])
    out["years_since_update"] = date.dt.year - updated
    out["was_renovated"] = (out["yr_renovated"] > 0).astype(int)
    out["has_basement"] = (out["sqft_basement"] > 0).astype(int)
    return out.drop(columns=[c for c in ("id", "date") if c in out.columns])


def cast_categoricals(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast category columns (including numeric codes such as zip codes) to text; keep missing as NaN."""
    X = X.copy()
    for c in categorical:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))
    return X


def dataset_from_king_county_frame(df: pd.DataFrame) -> Dataset:
    y = df["price"].to_numpy(dtype=float)
    X = df.drop(columns=["price"]).copy()
    categorical = [c for c in X.columns if c in KC_NOMINAL or not pd.api.types.is_numeric_dtype(X[c])]
    X = cast_categoricals(X, categorical)
    numeric = [c for c in X.columns if c not in categorical]
    return Dataset(
        "king_county",
        X.reset_index(drop=True),
        y,
        numeric,
        categorical,
        {
            "rule_repeat_sales": "kept the latest sale of each house id",
            "rule_bedrooms": f"dropped bedrooms > {KC_MAX_BEDROOMS}",
            "n_rows": len(X),
        },
    )


def load_king_county(data_file: Path | str | None = None, data_dir: Path | str | None = None) -> Dataset:
    path = Path(data_file) if data_file else download_file(data_dir, KC_URL, KC_SHA256, KC_NAME)
    raw = parse_arff_table(path.read_text(encoding="utf-8"))
    ds = dataset_from_king_county_frame(clean_king_county(raw, strict=True))
    ds.notes["n_raw_rows"] = len(raw)
    ds.notes["date_range"] = f"{raw['date'].min():%Y-%m-%d} to {raw['date'].max():%Y-%m-%d}"
    return ds


# ------------------------------------------------------------------------------ California


def parse_california(text: str) -> Dataset:
    """Parse raw ``cal_housing.data`` (no header) into scikit-learn's feature set; target in dollars."""
    raw = pd.read_csv(
        io.StringIO(text),
        header=None,
        names=[
            "longitude",
            "latitude",
            "housingMedianAge",
            "totalRooms",
            "totalBedrooms",
            "population",
            "households",
            "medianIncome",
            "medianHouseValue",
        ],
    )
    X = pd.DataFrame(
        {
            "MedInc": raw["medianIncome"],
            "HouseAge": raw["housingMedianAge"],
            "AveRooms": raw["totalRooms"] / raw["households"],
            "AveBedrms": raw["totalBedrooms"] / raw["households"],
            "Population": raw["population"],
            "AveOccup": raw["population"] / raw["households"],
            "Latitude": raw["latitude"],
            "Longitude": raw["longitude"],
        }
    )
    y = raw["medianHouseValue"].to_numpy(dtype=float)
    notes = {
        "n_rows": len(X),
        "n_censored_at_500001": int((y >= CAL_CENSOR).sum()),
        "target": "median house value of a census block group, US dollars, capped at 500001",
    }
    return Dataset("california", X, y, list(CAL_FEATURES), [], notes)


def load_california(data_file: Path | str | None = None, data_dir: Path | str | None = None) -> Dataset:
    path = Path(data_file) if data_file else download_file(data_dir, CAL_URL, CAL_SHA256, CAL_NAME)
    if path.suffix in (".tgz", ".gz"):
        with tarfile.open(path) as t:
            member = t.extractfile(CAL_MEMBER)
            if member is None:
                raise ValueError(f"{CAL_MEMBER} not found in {path}")
            text = member.read().decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return parse_california(text)


DATASET_NAMES = ("king_county", "california")


def load_dataset(
    name: str, data_file: Path | str | None = None, data_dir: Path | str | None = None
) -> Dataset:
    if name == "king_county":
        return load_king_county(data_file, data_dir)
    if name == "california":
        return load_california(data_file, data_dir)
    raise ValueError(f"unknown dataset {name!r}; choose from {DATASET_NAMES}")
