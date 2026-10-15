from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import CATEGORICAL, NUMERIC, reg_data  # noqa: E402


@pytest.fixture(scope="session")
def reg():
    X, y = reg_data(300, seed=1)
    return X, y, NUMERIC, CATEGORICAL


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end runs excluded from CI")
