"""Leakage-safe features for the global LightGBM model.

Every demand-based feature at target day ``t`` uses only data up to
``t - min_lag`` with ``min_lag = horizon``. One model can therefore forecast every
horizon h = 1..H from any origin ``o >= t - H`` without recursion (the *direct,
single-model* approach, ADR-0002). Promotion features describe the promotion
planned on day ``t`` itself: the promotion calendar is decided in advance, so it is
known at the origin (it is not demand data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORICAL = ("store_code", "item_code")


def shift(values: np.ndarray, k: int) -> np.ndarray:
    """``out[:, t] = values[:, t - k]`` (NaN where ``t < k``)."""
    out = np.full(values.shape, np.nan, dtype="float64")
    if k < values.shape[1]:
        out[:, k:] = values[:, : values.shape[1] - k]
    return out


def rolling_stats(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and sample std over ``window`` columns, ignoring NaN (min 1 value)."""
    x = np.nan_to_num(values, nan=0.0)
    ok = (~np.isnan(values)).astype("float64")
    zeros = np.zeros((values.shape[0], 1))
    cs = np.concatenate([zeros, np.cumsum(x, axis=1)], axis=1)
    cs2 = np.concatenate([zeros, np.cumsum(x * x, axis=1)], axis=1)
    cn = np.concatenate([zeros, np.cumsum(ok, axis=1)], axis=1)
    t = np.arange(values.shape[1]) + 1
    lo = np.clip(t - window, 0, None)
    s = cs[:, t] - cs[:, lo]
    s2 = cs2[:, t] - cs2[:, lo]
    n = cn[:, t] - cn[:, lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 1, (s2 - s * s / np.where(n > 0, n, 1)) / (n - 1), np.nan)
    return mean, np.sqrt(np.clip(var, 0, None))


@dataclass
class FeatureSet:
    names: list[str]
    series_features: dict[str, np.ndarray]    # (n_series, n_days) arrays
    calendar_features: dict[str, np.ndarray]  # (n_days,) arrays
    codes: dict[str, np.ndarray]              # (n_series,) integer codes
    n_series: int
    n_days: int

    @property
    def categorical(self) -> list[str]:
        return [c for c in CATEGORICAL if c in self.names]

    def rows(self, series_idx: np.ndarray, day_idx: np.ndarray) -> np.ndarray:
        """Feature matrix (float32) for the (series, day) pairs."""
        X = np.empty((len(series_idx), len(self.names)), dtype="float32")
        for j, name in enumerate(self.names):
            if name in self.series_features:
                X[:, j] = self.series_features[name][series_idx, day_idx]
            elif name in self.calendar_features:
                X[:, j] = self.calendar_features[name][day_idx]
            else:
                X[:, j] = self.codes[name][series_idx]
        return X


def build_features(values: np.ndarray, dates: pd.DatetimeIndex, store_codes: np.ndarray,
                   item_codes: np.ndarray, lags, rolling_windows, min_lag: int,
                   yearly: bool = True, promo: np.ndarray | None = None, post_promo_days: int = 7) -> FeatureSet:
    """Build every feature over the (possibly extended) calendar ``dates``.

    ``values`` must have ``len(dates)`` columns; future days are NaN. ``promo``
    (same shape, discount of the day) adds the promotion features. Raises
    ``ValueError`` if a lag is shorter than ``min_lag`` (leakage guard).
    """
    lags = sorted(set(int(x) for x in lags))
    if min(lags) < min_lag:
        raise ValueError(f"Lag {min(lags)} < min_lag {min_lag}: features would leak future information")
    if values.shape[1] != len(dates):
        raise ValueError("values and dates have different lengths")

    series: dict[str, np.ndarray] = {}
    for k in lags:
        series[f"lag_{k}"] = shift(values, k)
    base = shift(values, min_lag)
    for w in sorted(set(int(x) for x in rolling_windows)):
        mean, std = rolling_stats(base, w)
        series[f"roll_mean_{w}"] = mean
        if w >= 28:
            series[f"roll_std_{w}"] = std
    # Same weekday over the last 4 available weeks (captures the weekly profile).
    same_dow = [f"lag_{k}" for k in lags if k % 7 == 0][:4]
    if same_dow:
        stack = np.stack([series[c] for c in same_dow])
        count = (~np.isnan(stack)).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            series["same_dow_mean"] = np.where(count > 0, np.nansum(stack, axis=0) / count, np.nan)
    if "roll_mean_28" in series and "roll_mean_91" in series:
        with np.errstate(divide="ignore", invalid="ignore"):
            series["trend_28_91"] = series["roll_mean_28"] / series["roll_mean_91"]
    if promo is not None:
        from src.promotions import post_promo_mask, promo_day_counter

        series["promo_discount"] = promo
        series["promo_day"] = promo_day_counter(promo)
        series["post_promo"] = post_promo_mask(promo, post_promo_days).astype("float64")

    cal: dict[str, np.ndarray] = {
        "dow": dates.dayofweek.to_numpy(dtype="float64"),
        "dom": dates.day.to_numpy(dtype="float64"),
        "month": dates.month.to_numpy(dtype="float64"),
        "is_weekend": (dates.dayofweek >= 5).astype("float64"),
        "is_month_end": dates.is_month_end.astype("float64"),
    }
    if yearly:
        cal["dayofyear"] = dates.dayofyear.to_numpy(dtype="float64")
        cal["weekofyear"] = dates.isocalendar().week.to_numpy(dtype="float64")

    codes = {"store_code": np.asarray(store_codes, dtype="float64"),
             "item_code": np.asarray(item_codes, dtype="float64")}
    names = list(series) + list(cal) + list(codes)
    return FeatureSet(names=names, series_features=series, calendar_features=cal, codes=codes,
                      n_series=values.shape[0], n_days=values.shape[1])
