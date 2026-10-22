"""Forecasting models and the rolling-origin back-test.

Every model follows the same small interface on a :class:`DemandMatrix`:

* ``fit(dm, fit_end)`` estimates parameters on days ``<= fit_end`` only;
* ``predict(dm, origins, horizon)`` returns an array ``(n_series, n_origins, horizon)``
  where the forecast for origin ``o`` uses data up to day ``o`` only.

Between two refits the parameters stay fixed but the forecasts still use the
latest actuals (Holt-Winters states are updated, LightGBM lag features read the
new data): this is how a planner would run the models every week.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from src.features import build_features, rolling_stats
from src.metrics import error_sums
from src.series import DemandMatrix

MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "moving_average": "Moving average",
    "holt_winters": "Holt-Winters",
    "lightgbm": "LightGBM",
}


class Forecaster(Protocol):
    name: str

    def fit(self, dm: DemandMatrix, fit_end: int) -> Forecaster: ...

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray: ...


# --------------------------------------------------------------------------- baselines
def seasonal_naive_forecast(values: np.ndarray, origins: np.ndarray, horizon: int,
                            season: int = 7) -> np.ndarray:
    """``F(o + h) = y(o + h − season·⌈h/season⌉)``: the same weekday of the last observed week."""
    h = np.arange(1, horizon + 1)
    back = season * np.ceil(h / season).astype(int)
    src = np.asarray(origins)[:, None] + h[None, :] - back[None, :]
    out = values[:, np.clip(src, 0, values.shape[1] - 1)]
    out[:, src < 0] = np.nan
    return np.nan_to_num(out, nan=0.0)


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def fit(self, dm: DemandMatrix, fit_end: int) -> SeasonalNaive:
        return self

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray:
        return seasonal_naive_forecast(dm.values, origins, horizon, self.season_length)


class MovingAverage:
    name = "moving_average"

    def __init__(self, window: int = 28):
        self.window = window

    def fit(self, dm: DemandMatrix, fit_end: int) -> MovingAverage:
        return self

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray:
        mean, _ = rolling_stats(dm.values, self.window)
        level = np.nan_to_num(mean[:, np.asarray(origins)], nan=0.0)
        return np.repeat(level[:, :, None], horizon, axis=2)


# --------------------------------------------------------------------------- Holt-Winters
N_HW_FIXED = 6  # alpha, beta, gamma, phi, initial level, initial trend


def _fit_holt_winters_one(y: np.ndarray, m: int) -> np.ndarray | None:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=m,
                initialization_method="estimated",
            ).fit(use_brute=False)
        p = res.params
        arr = np.array([p["smoothing_level"], p["smoothing_trend"], p["smoothing_seasonal"],
                        p["damping_trend"], p["initial_level"], p["initial_trend"],
                        *np.asarray(p["initial_seasons"], dtype="float64")], dtype="float64")
        return arr if np.all(np.isfinite(arr)) else None
    except Exception:  # statsmodels can fail on degenerate series: fall back
        return None


class HoltWinters:
    """Additive damped trend + additive weekly seasonality, one model per series.

    Parameters are estimated with statsmodels at each refit; forecasts come from a
    vectorised filter that reproduces statsmodels' recursion exactly (tested) and
    keeps updating the states with new actuals between refits. Series that cannot
    be fitted fall back to the seasonal naive forecast.
    """

    name = "holt_winters"

    def __init__(self, seasonal_periods: int = 7, n_jobs: int = -1):
        self.m = seasonal_periods
        self.n_jobs = n_jobs
        self.params_: np.ndarray | None = None
        self.ok_: np.ndarray | None = None

    def fit(self, dm: DemandMatrix, fit_end: int) -> HoltWinters:
        from joblib import Parallel, delayed

        m = self.m
        rows, series = [], []
        for i in range(dm.n_series):
            y = dm.values[i, dm.start[i]: fit_end + 1]
            if len(y) >= 2 * m + 7 and np.nanstd(y) > 0:
                rows.append(i)
                series.append(np.asarray(y, dtype="float64"))
        if len(series) > 20 and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs, batch_size=8)(
                delayed(_fit_holt_winters_one)(y, m) for y in series)
        else:
            results = [_fit_holt_winters_one(y, m) for y in series]
        params = np.full((dm.n_series, N_HW_FIXED + m), np.nan)
        ok = np.zeros(dm.n_series, dtype=bool)
        for i, r in zip(rows, results):
            if r is not None:
                params[i] = r
                ok[i] = True
        self.params_, self.ok_ = params, ok
        return self

    @property
    def n_fallback(self) -> int:
        return int((~self.ok_).sum()) if self.ok_ is not None else 0

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("fit() must be called before predict()")
        origins = np.asarray(origins)
        out = seasonal_naive_forecast(dm.values, origins, horizon, self.m)
        idx = np.flatnonzero(self.ok_)
        if len(idx) == 0:
            return out
        m = self.m
        p = self.params_[idx]
        alpha, beta, gamma, phi, l0, b0 = (p[:, k] for k in range(N_HW_FIXED))
        s0 = p[:, N_HW_FIXED:]
        Y = dm.values[idx]
        start = dm.start[idx]
        k = len(idx)
        level = np.zeros(k)
        trend = np.zeros(k)
        seas = np.zeros((k, m))
        hh = np.arange(1, horizon + 1)
        damp = np.cumsum(phi[:, None] ** hh[None, :], axis=1)
        origin_pos = {int(o): j for j, o in enumerate(origins)}
        rows_k = np.arange(k)
        for t in range(int(start.min()), int(origins.max()) + 1):
            init = start == t
            if init.any():
                level[init] = l0[init]
                trend[init] = b0[init]
                for j in range(m):
                    seas[init, (t + j) % m] = s0[init, j]
            active = start <= t
            y = Y[:, t]
            pos = t % m
            s_cur = seas[:, pos]
            base = level + phi * trend
            with np.errstate(invalid="ignore"):
                l_new = alpha * (y - s_cur) + (1 - alpha) * base
                b_new = beta * (l_new - level) + (1 - beta) * phi * trend
                s_new = gamma * (y - base) + (1 - gamma) * s_cur
            level = np.where(active, l_new, level)
            trend = np.where(active, b_new, trend)
            seas[:, pos] = np.where(active, s_new, s_cur)
            j = origin_pos.get(t)
            if j is not None:
                started = rows_k[active]
                fc = level[:, None] + damp * trend[:, None] + seas[:, (t + hh) % m]
                out[idx[started], j, :] = fc[started]
        return np.clip(out, 0.0, None)


# --------------------------------------------------------------------------- LightGBM
class LightGBMForecaster:
    """One global LightGBM model on all series (direct approach, ADR-0002)."""

    name = "lightgbm"

    def __init__(self, params: dict, num_boost_round: int, horizon: int, lags, rolling_windows,
                 yearly: bool = True, n_jobs: int = -1, use_promo: bool = True, post_promo_days: int = 7):
        self.params = dict(params)
        self.num_boost_round = num_boost_round
        self.horizon = horizon
        self.lags = tuple(lags)
        self.rolling_windows = tuple(rolling_windows)
        self.yearly = yearly
        self.n_jobs = n_jobs
        self.use_promo = use_promo
        self.post_promo_days = post_promo_days
        self._cache_key = None
        self._features = None
        self._values = None
        self.booster_ = None
        self.importance_: pd.Series | None = None

    def _feature_set(self, dm: DemandMatrix):
        key = (id(dm), dm.values.shape)
        if self._cache_key != key:
            n, t = dm.values.shape
            values = np.concatenate([dm.values, np.full((n, self.horizon), np.nan)], axis=1)
            dates = pd.date_range(dm.dates[0], periods=t + self.horizon, freq="D")
            store_codes = pd.Categorical(dm.keys["store"]).codes
            item_codes = pd.Categorical(dm.keys["item"]).codes
            promo = None
            if self.use_promo and dm.promo is not None and (dm.promo > 0).any():
                # promotions are planned in advance: none is known beyond the data
                promo = np.concatenate([dm.promo, np.zeros((n, self.horizon))], axis=1)
            self._features = build_features(values, dates, store_codes, item_codes, self.lags,
                                            self.rolling_windows, min_lag=self.horizon, yearly=self.yearly,
                                            promo=promo, post_promo_days=self.post_promo_days)
            self._values = values
            self._cache_key = key
        return self._features

    def fit(self, dm: DemandMatrix, fit_end: int) -> LightGBMForecaster:
        import lightgbm as lgb

        fs = self._feature_set(dm)
        day = np.arange(self._values.shape[1])[None, :]
        mask = (day >= (dm.start[:, None] + self.horizon)) & (day <= fit_end) & ~np.isnan(self._values)
        ii, tt = np.nonzero(mask)
        if len(ii) == 0:
            raise ValueError("No training rows for LightGBM")
        X = fs.rows(ii, tt)
        y = self._values[ii, tt]
        params = {"verbose": -1, **self.params, "num_threads": 0 if self.n_jobs in (-1, None) else self.n_jobs}
        ds = lgb.Dataset(X, y, feature_name=fs.names, categorical_feature=fs.categorical, free_raw_data=True)
        self.booster_ = lgb.train(params, ds, num_boost_round=self.num_boost_round)
        self.importance_ = pd.Series(self.booster_.feature_importance("gain"), index=fs.names)
        return self

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("fit() must be called before predict()")
        if horizon > self.horizon:
            raise ValueError(f"horizon {horizon} > model horizon {self.horizon} (features would leak)")
        fs = self._feature_set(dm)
        origins = np.asarray(origins)
        d0, d1 = int(origins.min()) + 1, int(origins.max()) + horizon
        days = np.arange(d0, d1 + 1)
        n = dm.n_series
        ii = np.repeat(np.arange(n), len(days))
        tt = np.tile(days, n)
        pred = self.booster_.predict(fs.rows(ii, tt)).reshape(n, len(days))
        idx = origins[:, None] + np.arange(1, horizon + 1)[None, :] - d0
        return np.clip(pred[:, idx], 0.0, None)


# --------------------------------------------------------------------------- back-test
def make_origins(period_start: int, period_end: int, step: int = 7) -> np.ndarray:
    """Weekly origins: the day before the period starts, then every ``step`` days."""
    return np.arange(period_start - 1, period_end, step)


def actual_array(dm: DemandMatrix, origins: np.ndarray, horizon: int, period_end: int) -> np.ndarray:
    tgt = np.asarray(origins)[:, None] + np.arange(1, horizon + 1)[None, :]
    out = dm.values[:, np.clip(tgt, 0, dm.n_days - 1)].copy()
    out[:, tgt > period_end] = np.nan
    return out


@dataclass
class BacktestResult:
    period: str
    period_start: int
    period_end: int
    origins: np.ndarray
    horizon: int
    dates: pd.DatetimeIndex
    keys: pd.DataFrame
    actual: np.ndarray
    forecasts: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, dict] = field(default_factory=dict)

    @property
    def models(self) -> list[str]:
        return list(self.forecasts)

    def targets(self) -> np.ndarray:
        return self.origins[:, None] + np.arange(1, self.horizon + 1)[None, :]

    def subset(self, rows) -> BacktestResult:
        rows = np.asarray(rows)
        return BacktestResult(self.period, self.period_start, self.period_end, self.origins, self.horizon,
                              self.dates, self.keys.iloc[rows].reset_index(drop=True), self.actual[rows],
                              {k: v[rows] for k, v in self.forecasts.items()}, self.info)

    def error_table(self) -> pd.DataFrame:
        """Additive error sums per model x series x horizon."""
        frames = []
        n, _, H = self.actual.shape
        for model, F in self.forecasts.items():
            sums = error_sums(self.actual, F, axis=1)  # (n, H)
            df = pd.DataFrame({k: v.ravel() for k, v in sums.items()})
            df.insert(0, "h", np.tile(np.arange(1, H + 1), n))
            df.insert(0, "item", np.repeat(self.keys["item"].to_numpy(), H))
            df.insert(0, "store", np.repeat(self.keys["store"].to_numpy(), H))
            df.insert(0, "model", model)
            df.insert(0, "period", self.period)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def latest_origin(self, day: np.ndarray) -> np.ndarray:
        """Index of the latest origin strictly before each target day."""
        return np.searchsorted(self.origins, np.asarray(day) - 1, side="right") - 1

    def rolling_view(self, models=None, rows=None) -> pd.DataFrame:
        """Each day of the period forecast from the latest origin (h = 1..7)."""
        models = list(models or self.models)
        rows = np.arange(len(self.keys)) if rows is None else np.asarray(rows)
        days = np.arange(self.period_start, self.period_end + 1)
        j = self.latest_origin(days)
        h = days - self.origins[j]
        frames = []
        for model in models:
            F = self.forecasts[model][rows][:, j, h - 1]
            A = self.actual[rows][:, j, h - 1]
            frames.append(pd.DataFrame({
                "model": model,
                "store": np.repeat(self.keys["store"].to_numpy()[rows], len(days)),
                "item": np.repeat(self.keys["item"].to_numpy()[rows], len(days)),
                "date": np.tile(self.dates[days].to_numpy(), len(rows)),
                "h": np.tile(h, len(rows)),
                "actual": A.ravel(),
                "forecast": F.ravel(),
            }))
        return pd.concat(frames, ignore_index=True)

    def paths_frame(self, model: str, rows=None) -> pd.DataFrame:
        """Long table of every forecast path: [store, item, origin, h, forecast]."""
        rows = np.arange(len(self.keys)) if rows is None else np.asarray(rows)
        F = self.forecasts[model][rows]
        n, J, H = F.shape
        return pd.DataFrame({
            "store": np.repeat(self.keys["store"].to_numpy()[rows], J * H),
            "item": np.repeat(self.keys["item"].to_numpy()[rows], J * H),
            "origin": np.tile(np.repeat(self.dates[self.origins].to_numpy(), H), n),
            "h": np.tile(np.arange(1, H + 1), n * J),
            "forecast": F.ravel().astype("float32"),
        })


def run_backtest(models: list, dm: DemandMatrix, period: str, period_start: int, period_end: int,
                 horizon: int, step: int = 7, refit_every: dict[str, int] | None = None,
                 progress=None) -> BacktestResult:
    """Rolling-origin back-test of several models over one period."""
    refit_every = refit_every or {}
    origins = make_origins(period_start, period_end, step)
    result = BacktestResult(period, period_start, period_end, origins, horizon, dm.dates, dm.keys,
                            actual_array(dm, origins, horizon, period_end))
    for k, model in enumerate(models):
        if progress:
            progress(f"{period}: {MODEL_LABELS.get(model.name, model.name)}", k / max(len(models), 1))
        every = max(1, refit_every.get(model.name, len(origins)))
        F = np.empty((dm.n_series, len(origins), horizon))
        t0 = time.perf_counter()
        n_fits = 0
        fallbacks = []
        for b in range(0, len(origins), every):
            block = origins[b: b + every]
            model.fit(dm, int(block[0]))
            n_fits += 1
            F[:, b: b + len(block)] = model.predict(dm, block, horizon)
            if hasattr(model, "n_fallback"):
                fallbacks.append(model.n_fallback)
        result.forecasts[model.name] = F
        result.info[model.name] = {"seconds": round(time.perf_counter() - t0, 2), "n_fits": n_fits,
                                   "fallback_series": max(fallbacks) if fallbacks else 0}
    return result


def lead_time_demand(forecast: np.ndarray, origins: np.ndarray, decision_days: np.ndarray,
                     lead_time: np.ndarray, season: int = 7) -> np.ndarray:
    """Forecast demand over the lead time for each decision day.

    At the end of day ``t`` the planner uses the latest origin ``o <= t`` and sums
    the forecasts for days ``t+1 .. t+L``. Horizons beyond the forecast horizon
    ``H`` reuse the same weekday of the last forecast week (weekly-periodic
    extension), so any lead time can be served. Returns ``(n_series, n_days)``.
    """
    origins = np.asarray(origins)
    decision_days = np.asarray(decision_days)
    lead_time = np.asarray(lead_time, dtype=int)
    n, _, H = forecast.shape
    j = np.searchsorted(origins, decision_days, side="right") - 1
    if (j < 0).any():
        raise ValueError("A decision day is before the first forecast origin")
    d = decision_days - origins[j]
    out = np.zeros((n, len(decision_days)))
    for L in np.unique(lead_time):
        rows = np.flatnonzero(lead_time == L)
        F = forecast[rows]
        acc = np.zeros((len(rows), len(decision_days)))
        for k in range(1, int(L) + 1):
            h = d + k
            over = h > H
            h = np.where(over, h - season * np.ceil((h - H) / season).astype(int), h)
            acc += F[:, j, h - 1]
        out[rows] = acc
    return out


def get_model(name: str, cfg, profile, fast: bool = False, n_jobs: int = -1, promotions: bool = False):
    """Factory for the configured models.

    ``promotions=True`` (the data has promotion days) wraps the moving average and
    Holt-Winters in :class:`~src.promotions.PromoAware`; LightGBM reads the
    promotion features itself; the seasonal-naive baseline stays promotion-blind.
    """
    from src.promotions import PROMO_ADJUSTED, PromoAware

    fc = cfg.forecast
    if name == "seasonal_naive":
        model = SeasonalNaive(7)
    elif name == "moving_average":
        model = MovingAverage(fc.ma_window)
    elif name == "holt_winters":
        model = HoltWinters(fc.holt_winters.seasonal_periods, n_jobs=n_jobs)
    elif name == "lightgbm":
        rounds = cfg.fast_mode.lightgbm_num_boost_round if fast else fc.lightgbm.num_boost_round
        return LightGBMForecaster(fc.lightgbm.params, rounds, fc.horizon, profile.lags,
                                  profile.rolling_windows, profile.yearly_features, n_jobs=n_jobs,
                                  use_promo=promotions, post_promo_days=cfg.promotions.post_promo_days)
    else:
        raise ValueError(f"Unknown model {name!r}")
    if promotions and name in PROMO_ADJUSTED:
        return PromoAware(model, cfg.promotions)
    return model


def refit_schedule(cfg, fast: bool = False) -> dict[str, int]:
    if fast:
        every = cfg.fast_mode.refit_every
        return {"holt_winters": every, "lightgbm": every}
    return {"holt_winters": cfg.forecast.holt_winters.refit_every,
            "lightgbm": cfg.forecast.lightgbm.refit_every}
