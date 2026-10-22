"""Dense (series x day) representation of demand, shared by models and simulation.

Long format ``[date, store, item, sales]`` is convenient for SQL and pandas; the
forecasting models and the inventory simulation work much faster on a dense
matrix: one row per ``(store, item)`` series, one column per calendar day.
Days before a series' first sale are ``NaN`` (the series did not exist yet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def natural_key(value: object) -> tuple:
    """Sort key that puts "2" before "10" (IDs are strings)."""
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def natural_sorted(values) -> list[str]:
    return sorted((str(v) for v in values), key=natural_key)


@dataclass
class DemandMatrix:
    keys: pd.DataFrame            # columns [store, item]; row i = series i
    dates: pd.DatetimeIndex       # consecutive days
    values: np.ndarray            # float64 (n_series, n_days); NaN before the series' start
    start: np.ndarray             # int (n_series,): index of the first observed day
    promo: np.ndarray | None = None  # float (n_series, n_days): discount of the day, 0 = no promotion

    # ------------------------------------------------------------------ build
    @classmethod
    def from_long(cls, sales: pd.DataFrame, end: pd.Timestamp | None = None,
                  promo: pd.DataFrame | None = None) -> DemandMatrix:
        """Pivot ``[date, store, item, sales]`` into a dense matrix.

        Series are ordered by natural (store, item) order. Days after a series'
        first observation that have no row are treated as zero sales. ``promo``
        (``[date, store, item, discount]``, promotion days only) fills ``promo``.
        """
        if sales.empty:
            raise ValueError("No sales rows to build the demand matrix from.")
        df = sales[["date", "store", "item", "sales"]].copy()
        df["store"] = df["store"].astype(str)
        df["item"] = df["item"].astype(str)
        first = df["date"].min()
        last = pd.Timestamp(end) if end is not None else df["date"].max()
        dates = pd.date_range(first, last, freq="D")

        keys = (
            df[["store", "item"]]
            .drop_duplicates()
            .assign(_s=lambda d: d["store"].map(natural_key), _i=lambda d: d["item"].map(natural_key))
            .sort_values(["_s", "_i"])
            .drop(columns=["_s", "_i"])
            .reset_index(drop=True)
        )
        key_index = pd.MultiIndex.from_frame(keys)
        row = key_index.get_indexer(pd.MultiIndex.from_frame(df[["store", "item"]]))
        col = ((df["date"] - first).dt.days).to_numpy()
        keep = col < len(dates)
        values = np.zeros((len(keys), len(dates)), dtype="float64")
        np.add.at(values, (row[keep], col[keep]), df["sales"].to_numpy(dtype="float64")[keep])

        start = np.full(len(keys), len(dates), dtype=int)
        np.minimum.at(start, row[keep], col[keep])
        values[np.arange(len(dates))[None, :] < start[:, None]] = np.nan
        from src.promotions import promo_matrix

        return cls(keys=keys, dates=dates, values=values, start=start, promo=promo_matrix(promo, keys, dates))

    # ------------------------------------------------------------------ helpers
    @property
    def n_series(self) -> int:
        return self.values.shape[0]

    @property
    def n_days(self) -> int:
        return self.values.shape[1]

    def date_index(self, day) -> int:
        """Column index of a calendar day (raises KeyError if outside the calendar)."""
        idx = (pd.Timestamp(day) - self.dates[0]).days
        if not 0 <= idx < self.n_days:
            raise KeyError(f"{day} is outside the calendar {self.dates[0].date()} – {self.dates[-1].date()}")
        return idx

    def series_index(self, store: str, item: str) -> int:
        mask = (self.keys["store"] == str(store)) & (self.keys["item"] == str(item))
        hits = np.flatnonzero(mask.to_numpy())
        if len(hits) == 0:
            raise KeyError(f"Series ({store}, {item}) not found")
        return int(hits[0])

    def subset(self, rows) -> DemandMatrix:
        rows = np.asarray(rows)
        return DemandMatrix(
            keys=self.keys.iloc[rows].reset_index(drop=True),
            dates=self.dates,
            values=self.values[rows],
            start=self.start[rows],
            promo=None if self.promo is None else self.promo[rows],
        )

    def history_before(self, day_index: int) -> np.ndarray:
        """Number of observed days strictly before ``day_index`` for each series."""
        return np.clip(day_index - self.start, 0, None)

    def to_long(self) -> pd.DataFrame:
        n, t = self.values.shape
        out = pd.DataFrame(
            {
                "date": np.tile(self.dates.to_numpy(), n),
                "store": np.repeat(self.keys["store"].to_numpy(), t),
                "item": np.repeat(self.keys["item"].to_numpy(), t),
                "sales": self.values.ravel(),
            }
        )
        return out.dropna(subset=["sales"]).reset_index(drop=True)

    def window(self, start: int, end: int) -> np.ndarray:
        """Values for day indices ``start..end`` inclusive (NaN-padded outside the calendar)."""
        out = np.full((self.n_series, end - start + 1), np.nan)
        lo, hi = max(start, 0), min(end, self.n_days - 1)
        if lo <= hi:
            out[:, lo - start : hi - start + 1] = self.values[:, lo : hi + 1]
        return out
