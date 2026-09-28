"""Shared fixtures: a small synthetic dataset (no Kaggle download needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import ColumnMapping, standardise
from src.synthetic import generate_sales


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def raw_sales() -> pd.DataFrame:
    """3 stores x 4 items x 3 years, Kaggle-like columns with ``quantity``."""
    return generate_sales(n_stores=3, n_items=4, start="2014-01-01", end="2016-12-31", seed=3).rename(
        columns={"sales": "quantity"})


@pytest.fixture(scope="session")
def data(raw_sales, cfg):
    mapping = ColumnMapping(date="date", item="item", quantity="quantity", store="store")
    return standardise(raw_sales, mapping, cfg.input, cfg.prices, cfg.inventory, source="demo")


@pytest.fixture(scope="session")
def dm(data):
    return data.matrix()
