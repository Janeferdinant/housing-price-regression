from __future__ import annotations

import json

import pytest

from house_price import cli
from synthetic import california_text


def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("train", "evaluate", "predict"):
        assert cmd in out


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        cli.main([])


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    f = tmp_path / "cal_housing.data"
    f.write_text(california_text(600, seed=4))
    out = tmp_path / "out"
    common = ["--dataset", "california", "--data-file", str(f), "--out", str(out)]
    train = ["--trials", "2", "--splits", "3", "--inner-splits", "2", "--jobs", "1"]
    cli.main(["train", *common, *train, "--models", "mean", "ridge", "hgb", "xgb"])
    cli.main(["evaluate", *common, "--bootstrap", "40"])
    summary = json.loads((out / "results" / "california" / "summary.json").read_text())
    assert summary["seed"] == 42
    iv = summary["intervals"]["split_conformal"]
    assert 0 < iv["coverage"] <= 1 and iv["nominal"] == 0.9

    inp = tmp_path / "homes.csv"
    inp.write_text(
        "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude\n"
        "3.5,20,5.0,1.0,1500,3.0,34.0,-118.0\n"
    )
    pred = tmp_path / "pred.csv"
    cli.main(
        ["predict", "--dataset", "california", "--out", str(out), "--input", str(inp), "--output", str(pred)]
    )
    text = pred.read_text()
    assert "predicted_price" in text and "lower" in text and "upper" in text
