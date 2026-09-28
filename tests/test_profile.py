"""Adaptive periods (T-P1 .. T-P4)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.profile import apply_exclusions, profile_history, split_periods
from src.series import DemandMatrix


def _dm(days: int, start: str = "2013-01-01", n: int = 2) -> DemandMatrix:
    dates = pd.date_range(start, periods=days, freq="D")
    keys = pd.DataFrame({"store": ["1"] * n, "item": [str(i) for i in range(1, n + 1)]})
    return DemandMatrix(keys, dates, np.full((n, days), 10.0), np.zeros(n, dtype=int))


def test_demo_length_gives_calendar_years():  # T-P1
    cfg = load_config()
    train, valid, test = split_periods(date(2013, 1, 1), date(2017, 12, 31), cfg)
    assert test == (date(2017, 1, 1), date(2017, 12, 31))
    assert valid == (date(2016, 1, 1), date(2016, 12, 31))
    assert train == (date(2013, 1, 1), date(2015, 12, 31))
    prof = profile_history(_dm(1826), cfg)
    assert prof.yearly_features and 364 in prof.lags and not prof.low_confidence


def test_two_hundred_days():  # T-P2
    cfg = load_config()
    prof = profile_history(_dm(200), cfg)
    assert prof.test_days == 50 and prof.valid_days == 50 and prof.train_days == 100
    assert not prof.yearly_features and 364 not in prof.lags


def test_very_short_history_is_low_confidence():  # T-P3
    cfg = load_config()
    prof = profile_history(_dm(60), cfg)
    assert prof.test_days == 28 and prof.valid_days == 14 and prof.low_confidence
    assert "lightgbm" not in prof.models  # not enough training rows


def test_explicit_dates_override():  # T-P4
    cfg = load_config(overrides={"periods": {"test_start": "2017-07-01", "test_end": "2017-12-31",
                                             "valid_start": "2017-01-01"}})
    train, valid, test = split_periods(date(2013, 1, 1), date(2017, 12, 31), cfg)
    assert test == (date(2017, 7, 1), date(2017, 12, 31))
    assert valid == (date(2017, 1, 1), date(2017, 6, 30))
    assert train[1] == date(2016, 12, 31)


def test_short_series_are_excluded_and_listed():
    cfg = load_config()
    dm = _dm(400, n=2)
    dm.start[1] = 380  # starts 20 days before the end -> too little history
    dm.values[1, :380] = np.nan
    prof = profile_history(dm, cfg)
    assert [(s, i) for s, i, _ in prof.excluded_series] == [("1", "2")]
    assert apply_exclusions(dm, prof).n_series == 1


def test_bad_explicit_period_raises():
    cfg = load_config(overrides={"periods": {"test_start": "2019-01-01", "test_end": "2019-02-01"}})
    with pytest.raises(ValueError):
        split_periods(date(2013, 1, 1), date(2017, 12, 31), cfg)
