"""Synthetic daily sales with the same shape as the Kaggle demo.

Used by the tests, by CI (no Kaggle download there) and for development. The
data is *not* used for any reported result.

Demand = store level x item level x trend x weekly pattern x yearly pattern,
with Poisson noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sales(n_stores: int = 10, n_items: int = 50, start: str = "2013-01-01",
                   end: str = "2017-12-31", seed: int = 0, yearly_growth: float = 0.07,
                   base: float = 30.0) -> pd.DataFrame:
    """Return ``[date, store, item, sales]`` with integer sales."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    t = np.arange(len(dates))
    store_level = rng.uniform(0.6, 1.4, n_stores)
    item_level = rng.lognormal(0.0, 0.45, n_items)
    weekly = np.array([0.80, 0.93, 0.95, 1.00, 1.07, 1.12, 1.18])  # Mon .. Sun
    weekly = weekly / weekly.mean()
    doy = dates.dayofyear.to_numpy()
    yearly = 1 + 0.22 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    trend = (1 + yearly_growth) ** (t / 365.25)
    season = trend * weekly[dates.dayofweek.to_numpy()] * yearly            # (T,)
    level = base * np.outer(store_level, item_level).ravel()                # (S,)
    lam = level[:, None] * season[None, :]
    sales = rng.poisson(lam)
    n_series = n_stores * n_items
    return pd.DataFrame({
        "date": np.tile(dates.to_numpy(), n_series),
        "store": np.repeat(np.repeat(np.arange(1, n_stores + 1), n_items), len(dates)).astype(str),
        "item": np.repeat(np.tile(np.arange(1, n_items + 1), n_stores), len(dates)).astype(str),
        "sales": sales.ravel().astype("float64"),
    })


def write_kaggle_like_csv(path, **kwargs) -> pd.DataFrame:
    """Write a synthetic file with the exact Kaggle columns (``date,store,item,sales``)."""
    df = generate_sales(**kwargs)
    out = df.assign(date=df["date"].dt.strftime("%Y-%m-%d"), sales=df["sales"].astype(int))
    out.to_csv(path, index=False)
    return df
