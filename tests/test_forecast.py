"""Models, back-test and leakage guards (T-L1 .. T-L3)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, rolling_stats, shift
from src.forecast import (
    HoltWinters,
    LightGBMForecaster,
    MovingAverage,
    SeasonalNaive,
    lead_time_demand,
    make_origins,
    run_backtest,
    seasonal_naive_forecast,
)
from src.series import DemandMatrix


def test_leakage_guard_rejects_short_lags():  # T-L1
    v = np.arange(40, dtype=float)[None, :]
    dates = pd.date_range("2020-01-01", periods=40)
    with pytest.raises(ValueError):
        build_features(v, dates, [0], [0], lags=[7, 14], rolling_windows=[7], min_lag=14)


def test_features_only_use_data_up_to_t_minus_min_lag():  # T-L2
    rng = np.random.default_rng(0)
    v = rng.poisson(20, size=(2, 120)).astype(float)
    dates = pd.date_range("2020-01-01", periods=120)
    t = 90
    base = build_features(v, dates, [0, 1], [0, 1], lags=[14, 21], rolling_windows=[7, 28], min_lag=14)
    v2 = v.copy()
    v2[:, t - 13:] = 999.0  # change everything after t - 14
    changed = build_features(v2, dates, [0, 1], [0, 1], lags=[14, 21], rolling_windows=[7, 28], min_lag=14)
    rows = (np.array([0, 1]), np.array([t, t]))
    np.testing.assert_array_equal(base.rows(*rows), changed.rows(*rows))


@pytest.mark.parametrize("model", [SeasonalNaive(), MovingAverage(28), HoltWinters(n_jobs=1)])
def test_forecasts_ignore_data_after_the_origin(model, dm):  # T-L3
    origin = dm.n_days - 60
    fit_end = origin - 30
    model.fit(dm, fit_end)
    a = model.predict(dm, np.array([origin]), 14)
    changed = DemandMatrix(dm.keys, dm.dates, dm.values.copy(), dm.start)
    changed.values[:, origin + 1:] *= 3.0
    b = model.predict(changed, np.array([origin]), 14)
    np.testing.assert_allclose(a, b)


def test_lightgbm_ignores_data_after_the_origin(dm):
    model = LightGBMForecaster({"objective": "tweedie", "verbose": -1, "seed": 1, "num_leaves": 7},
                               num_boost_round=20, horizon=14, lags=[14, 21, 28], rolling_windows=[7, 28],
                               yearly=False, n_jobs=1)
    origin = dm.n_days - 60
    model.fit(dm, origin - 30)
    a = model.predict(dm, np.array([origin]), 14)
    changed = DemandMatrix(dm.keys, dm.dates, dm.values.copy(), dm.start)
    changed.values[:, origin + 1:] *= 3.0
    b = model.predict(changed, np.array([origin]), 14)
    np.testing.assert_allclose(a, b)


def test_seasonal_naive_repeats_last_week():
    v = np.arange(30, dtype=float)[None, :]
    f = seasonal_naive_forecast(v, np.array([20]), 14)
    assert f[0, 0].tolist() == list(range(14, 21)) + list(range(14, 21))


def test_rolling_and_shift_helpers():
    v = np.array([[1.0, 2.0, 3.0, np.nan, 5.0]])
    mean, std = rolling_stats(v, 2)
    np.testing.assert_allclose(mean[0], [1.0, 1.5, 2.5, 3.0, 5.0])
    assert np.isnan(shift(v, 2)[0, :2]).all() and shift(v, 2)[0, 2] == 1.0


def test_holt_winters_filter_matches_statsmodels():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    rng = np.random.default_rng(1)
    t = np.arange(400)
    y = 40 + 0.02 * t + 6 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 400)
    dates = pd.date_range("2020-01-01", periods=400)
    dm = DemandMatrix(pd.DataFrame({"store": ["1"], "item": ["1"]}), dates, y[None, :], np.array([0]))
    model = HoltWinters(n_jobs=1).fit(dm, 399)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                   initialization_method="estimated").fit(use_brute=False).forecast(14)
    ours = model.predict(dm, np.array([399]), 14)[0, 0]
    np.testing.assert_allclose(ours, np.clip(ref, 0, None), rtol=1e-8)


def test_backtest_shapes_and_actuals(dm):
    ps, pe = dm.n_days - 60, dm.n_days - 1
    bt = run_backtest([SeasonalNaive(), MovingAverage()], dm, "test", ps, pe, 14, 7)
    assert bt.origins[0] == ps - 1
    assert bt.forecasts["seasonal_naive"].shape == (dm.n_series, len(make_origins(ps, pe, 7)), 14)
    tgt = bt.targets()
    assert np.isnan(bt.actual[:, tgt > pe]).all()
    view = bt.rolling_view()
    assert set(view["h"]) <= set(range(1, 8))
    assert view.groupby("model").size().eq(dm.n_series * 60).all()


def test_lead_time_demand_sums_and_extends_weekly():
    F = np.arange(1, 15, dtype=float)[None, None, :]   # one origin, forecasts 1..14
    origins = np.array([0])
    ltd = lead_time_demand(F, origins, np.array([0, 6]), np.array([3]))
    assert ltd.tolist() == [[1 + 2 + 3, 7 + 8 + 9]]
    # L = 10 from day 6 needs h = 7..16: 15 and 16 reuse h = 8 and 9 (same weekday)
    ltd10 = lead_time_demand(F, origins, np.array([6]), np.array([10]))
    assert ltd10[0, 0] == sum(range(7, 15)) + 8 + 9
