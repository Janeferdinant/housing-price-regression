from __future__ import annotations

import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from house_price import data
from synthetic import california_text, kc_arff_text, kc_like


def test_parse_arff_reads_columns_dates_and_zipcodes():
    df = kc_like(30)
    parsed = data.parse_arff_table(kc_arff_text(df))
    assert list(parsed.columns) == list(df.columns)
    assert len(parsed) == 30
    assert str(parsed["date"].dtype).startswith("datetime64")
    assert parsed["date"].iloc[0] == df["date"].iloc[0]
    assert parsed["zipcode"].map(type).eq(str).all()


def test_parse_arff_rejects_other_text():
    with pytest.raises(ValueError, match="ARFF"):
        data.parse_arff_table("a,b,c\n1,2,3\n")


def test_clean_keeps_latest_sale_of_a_repeat_sold_house():
    df = kc_like(20)
    early = df.iloc[[0]].copy()
    early["date"] = pd.Timestamp("2014-05-02")
    early["price"] = 111_111.0
    late = df.iloc[[0]].copy()
    late["date"] = pd.Timestamp("2015-04-01")
    late["price"] = 222_222.0
    both = pd.concat([df.iloc[1:], early, late], ignore_index=True)
    out = data.clean_king_county(both, strict=False)
    assert len(out) == 20
    assert 222_222.0 in set(out["price"]) and 111_111.0 not in set(out["price"])


def test_clean_drops_the_impossible_bedroom_count():
    df = kc_like(20)
    df.loc[3, "bedrooms"] = 33
    out = data.clean_king_county(df, strict=False)
    assert len(out) == 19 and out["bedrooms"].max() <= data.KC_MAX_BEDROOMS


def test_clean_derived_features_and_dropped_identifiers():
    df = kc_like(40)
    out = data.clean_king_county(df, strict=False)
    src = df.loc[out.index]
    year = src["date"].dt.year
    assert "id" not in out.columns and "date" not in out.columns
    assert (out["house_age"] == year - src["yr_built"]).all()
    assert (out["was_renovated"] == (src["yr_renovated"] > 0).astype(int)).all()
    assert (out["has_basement"] == (src["sqft_basement"] > 0).astype(int)).all()
    updated = np.maximum(src["yr_built"], src["yr_renovated"])
    assert (out["years_since_update"] == year - updated).all()
    month_index = (year - 2014) * 12 + src["date"].dt.month - 5
    assert (out["sale_month_index"] == month_index).all()
    assert out["sale_month_index"].min() >= 0


def test_prediction_mode_keeps_every_row_and_needs_no_price_or_id():
    df = kc_like(15)
    df.loc[2, "bedrooms"] = 33
    out = data.clean_king_county(df.drop(columns=["price", "id"]), strict=False, training=False)
    assert len(out) == 15 and "house_age" in out.columns


def test_dataset_split_of_features_and_target():
    out = data.clean_king_county(kc_like(50), strict=False)
    ds = data.dataset_from_king_county_frame(out)
    assert ds.categorical == ["zipcode"]
    assert ds.X["zipcode"].map(type).eq(str).all()
    assert "price" not in ds.X.columns and (ds.y > 0).all()
    assert set(ds.numeric).isdisjoint(ds.categorical)
    assert len(ds.X) == len(ds.y)


def test_strict_schema_check_rejects_unexpected_columns():
    with pytest.raises(ValueError, match="schema"):
        data.clean_king_county(kc_like(10).drop(columns=["grade"]), strict=True)


def test_load_king_county_from_a_local_file(tmp_path):
    df = kc_like(60)
    f = tmp_path / "house_sales.arff"
    f.write_text(kc_arff_text(df))
    ds = data.load_dataset("king_county", data_file=f)
    assert len(ds.X) == 60 and ds.notes["n_raw_rows"] == 60


def test_cast_categoricals_keeps_missing_and_stringifies_codes():
    X = pd.DataFrame({"zipcode": [98001, 98103, None], "other": [1, 2, 3]})
    out = data.cast_categoricals(X, ["zipcode"])
    assert out["zipcode"].iloc[0] in ("98001", "98001.0")
    assert out["zipcode"].isna().iloc[2]
    assert out["other"].tolist() == [1, 2, 3]


def test_california_parsing_and_derived_columns():
    ds = data.parse_california(california_text(200, 1))
    assert len(ds.X) == 200
    assert list(ds.X.columns) == data.CAL_FEATURES
    assert (ds.y > 0).all()
    assert np.isfinite(ds.X["AveRooms"]).all()
    assert ds.numeric == data.CAL_FEATURES and ds.categorical == []


def test_california_average_columns_match_definition():
    line = "-122.23,37.88,41.0,880.0,129.0,322.0,126.0,8.3252,452600.0\n"
    ds = data.parse_california(line)
    row = ds.X.iloc[0]
    assert row["AveRooms"] == pytest.approx(880 / 126)
    assert row["AveBedrms"] == pytest.approx(129 / 126)
    assert row["AveOccup"] == pytest.approx(322 / 126)
    assert row["MedInc"] == pytest.approx(8.3252)
    assert ds.y[0] == pytest.approx(452600.0)


def test_california_flags_censored_prices():
    ds = data.parse_california(california_text(2000, 3))
    assert ds.notes["n_censored_at_500001"] == int((ds.y >= 500001).sum())


def _targz(text: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        raw = text.encode()
        info = tarfile.TarInfo("CaliforniaHousing/cal_housing.data")
        info.size = len(raw)
        t.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def test_download_checksum_cache_and_failure(tmp_path):
    payload = b"hello"
    calls = []

    def opener(url):
        calls.append(url)
        return payload

    sha = data.sha256_bytes(payload)
    p = data.download_file(tmp_path, "http://x/f", sha, "f.bin", opener=opener)
    assert p.read_bytes() == payload
    data.download_file(tmp_path, "http://x/f", sha, "f.bin", opener=opener)
    assert len(calls) == 1
    with pytest.raises(data.ChecksumError):
        data.download_file(tmp_path / "other", "http://x/f", "0" * 64, "f.bin", opener=opener)
    assert not (tmp_path / "other" / "f.bin").exists()
    (tmp_path / "f.bin").write_bytes(b"corrupt")
    data.download_file(tmp_path, "http://x/f", sha, "f.bin", opener=opener)
    assert (tmp_path / "f.bin").read_bytes() == payload


def test_load_california_from_local_archive(tmp_path):
    f = tmp_path / "cal.tgz"
    f.write_bytes(_targz(california_text(50, 2)))
    ds = data.load_dataset("california", data_file=f)
    assert len(ds.X) == 50


def test_load_unknown_dataset_raises():
    with pytest.raises(ValueError):
        data.load_dataset("boston")


def test_pinned_checksums_well_formed():
    for h in (data.KC_SHA256, data.CAL_SHA256):
        assert len(h) == 64 and int(h, 16) >= 0
    assert data.KC_URL.startswith("https://openml.org/")
