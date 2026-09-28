"""Adapt the analysis to the history: periods, features, models, exclusions.

Rules (docs/08_methodology.md §1):

* test length ``T = min(365, max(28, round(0.25 * H)))`` where ``H`` = history in days;
* validation length ``V = T`` if ``H - 2T >= 56``, else ``max(14, H - T - 56)``;
* train = everything before validation (at least 14 days);
* yearly features (lag 364, day of year) only with >= 400 training days;
* LightGBM only with >= 2,000 training rows; Holt-Winters skipped in fast mode
  above ``fast_mode.holt_winters_max_series`` series;
* series with fewer than ``input.min_history_days`` days before the test period
  are excluded (and listed).

Explicit dates in ``config.periods`` override the automatic split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import Config
from src.series import DemandMatrix

MIN_TRAIN_DAYS = 14
MIN_VALID_DAYS = 14
YEARLY_MIN_TRAIN_DAYS = 400
LIGHTGBM_MIN_ROWS = 2_000


@dataclass(frozen=True)
class DataProfile:
    data_start: date
    data_end: date
    history_days: int
    train: tuple[date, date]
    valid: tuple[date, date]
    test: tuple[date, date]
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    yearly_features: bool
    models: tuple[str, ...]
    excluded_series: tuple[tuple[str, str, str], ...]
    low_confidence: bool
    fast: bool
    notes: tuple[str, ...]

    @property
    def test_days(self) -> int:
        return (self.test[1] - self.test[0]).days + 1

    @property
    def valid_days(self) -> int:
        return (self.valid[1] - self.valid[0]).days + 1

    @property
    def train_days(self) -> int:
        return (self.train[1] - self.train[0]).days + 1

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("data_start", "data_end"):
            d[key] = str(d[key])
        for key in ("train", "valid", "test"):
            d[key] = [str(x) for x in d[key]]
        d["excluded_series"] = [list(x) for x in self.excluded_series]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DataProfile:
        def _d(x):
            return date.fromisoformat(x)

        return cls(
            data_start=_d(d["data_start"]),
            data_end=_d(d["data_end"]),
            history_days=int(d["history_days"]),
            train=tuple(_d(x) for x in d["train"]),
            valid=tuple(_d(x) for x in d["valid"]),
            test=tuple(_d(x) for x in d["test"]),
            lags=tuple(d["lags"]),
            rolling_windows=tuple(d["rolling_windows"]),
            yearly_features=bool(d["yearly_features"]),
            models=tuple(d["models"]),
            excluded_series=tuple(tuple(x) for x in d["excluded_series"]),
            low_confidence=bool(d["low_confidence"]),
            fast=bool(d["fast"]),
            notes=tuple(d["notes"]),
        )


def _start_before(end_exclusive: date, days: int) -> date:
    """Start of a period of ``days`` days ending the day before ``end_exclusive``.

    A 365-day period is taken as one calendar year, so that leap years are
    handled naturally (the demo gives exactly 2016 and 2017).
    """
    if days == 365:
        return (pd.Timestamp(end_exclusive) - pd.DateOffset(years=1)).date()
    return end_exclusive - timedelta(days=days)


def split_periods(data_start: date, data_end: date, cfg: Config) -> tuple[tuple, tuple, tuple]:
    """Return (train, valid, test) as (start, end) date tuples."""
    history = (data_end - data_start).days + 1
    p = cfg.periods
    if p.test_start or p.test_end:
        test_end = p.test_end or data_end
        test_start = p.test_start or _start_before(test_end + timedelta(days=1), p.test_days or 365)
    else:
        t_len = p.test_days or min(365, max(28, round(0.25 * history)))
        test_end = data_end
        test_start = _start_before(data_end + timedelta(days=1), t_len)
    if not data_start < test_start <= test_end <= data_end:
        raise ValueError(f"Test period {test_start} → {test_end} is not inside the data ({data_start} → {data_end}).")
    t_len = min((test_end - test_start).days + 1, 365)

    if p.valid_start:
        valid_start = p.valid_start
    else:
        before = (test_start - data_start).days
        if p.valid_days:
            v_len = p.valid_days
        elif before - t_len >= 56:
            v_len = t_len
        else:
            v_len = max(MIN_VALID_DAYS, before - 56)
        valid_start = _start_before(test_start, v_len)
    valid_end = test_start - timedelta(days=1)
    if not valid_start <= valid_end:
        raise ValueError("The validation period is empty.")
    train_end = valid_start - timedelta(days=1)
    if (train_end - data_start).days + 1 < MIN_TRAIN_DAYS:
        raise ValueError(
            f"Not enough history before the validation period: at least {MIN_TRAIN_DAYS} training days are needed.")
    return (data_start, train_end), (valid_start, valid_end), (test_start, test_end)


def profile_history(dm: DemandMatrix, cfg: Config, fast: bool = False) -> DataProfile:
    """Derive periods, features, models and exclusions from the demand matrix."""
    data_start = dm.dates[0].date()
    data_end = dm.dates[-1].date()
    history = (data_end - data_start).days + 1
    train, valid, test = split_periods(data_start, data_end, cfg)
    train_days = (train[1] - train[0]).days + 1
    valid_days = (valid[1] - valid[0]).days + 1
    notes: list[str] = [
        f"History: {history:,} days ({data_start} → {data_end}).",
        f"Train {train[0]} → {train[1]} ({train_days:,} days) · validation {valid[0]} → {valid[1]} "
        f"({valid_days:,} days) · test {test[0]} → {test[1]} ({(test[1] - test[0]).days + 1:,} days).",
    ]

    fc = cfg.forecast
    min_lag = fc.horizon
    yearly = train_days >= YEARLY_MIN_TRAIN_DAYS
    lags = tuple(lag for lag in fc.lightgbm.lags if lag < 365 or yearly)
    lags = tuple(lag for lag in lags if lag + 7 <= train_days) or (min_lag,)
    windows = tuple(w for w in fc.lightgbm.rolling_windows if w + min_lag <= train_days) or (7,)
    if yearly:
        notes.append("Yearly seasonality is modelled (lag 364, day of year).")
    else:
        notes.append(f"Less than {YEARLY_MIN_TRAIN_DAYS} training days: yearly seasonality is not modelled.")

    # Exclusions: not enough history before the test period.
    test_idx = (pd.Timestamp(test[0]) - dm.dates[0]).days
    history_before = dm.history_before(test_idx)
    excluded = []
    for i in np.flatnonzero(history_before < cfg.input.min_history_days):
        excluded.append((str(dm.keys.at[i, "store"]), str(dm.keys.at[i, "item"]),
                         f"only {int(history_before[i])} days of history before the test period "
                         f"(minimum {cfg.input.min_history_days})"))
    if excluded:
        notes.append(f"{len(excluded)} series excluded for short history.")
    kept = history_before >= cfg.input.min_history_days
    n_kept = int(kept.sum())

    models = list(fc.models)
    valid_idx = (pd.Timestamp(valid[0]) - dm.dates[0]).days
    train_rows = int(np.clip(valid_idx - (dm.start[kept] + min_lag), 0, None).sum())
    if "lightgbm" in models and train_rows < LIGHTGBM_MIN_ROWS:
        models.remove("lightgbm")
        notes.append(f"LightGBM skipped: only {train_rows:,} training rows (minimum {LIGHTGBM_MIN_ROWS:,}).")
    if "holt_winters" in models and fast and n_kept > cfg.fast_mode.holt_winters_max_series:
        models.remove("holt_winters")
        notes.append(f"Holt-Winters skipped in fast mode: {n_kept:,} series "
                     f"(limit {cfg.fast_mode.holt_winters_max_series:,}).")
    if fc.baseline not in models:
        models.insert(0, fc.baseline)

    low_conf = valid_days < 28 or train_days < 56
    if low_conf:
        notes.append("Short history: validation or training period is very short, results have low confidence.")
    if fast:
        notes.append("Fast mode: models are fitted once per period, fewer LightGBM trees.")

    return DataProfile(
        data_start=data_start,
        data_end=data_end,
        history_days=history,
        train=train,
        valid=valid,
        test=test,
        lags=lags,
        rolling_windows=windows,
        yearly_features=yearly,
        models=tuple(models),
        excluded_series=tuple(excluded),
        low_confidence=low_conf,
        fast=fast,
        notes=tuple(notes),
    )


def apply_exclusions(dm: DemandMatrix, profile: DataProfile) -> DemandMatrix:
    """Drop the excluded series from the matrix."""
    if not profile.excluded_series:
        return dm
    excluded = {(s, i) for s, i, _ in profile.excluded_series}
    keep = [k for k, (s, i) in enumerate(zip(dm.keys["store"], dm.keys["item"])) if (s, i) not in excluded]
    if not keep:
        raise ValueError("Every series was excluded: the history is too short.")
    return dm.subset(keep)
