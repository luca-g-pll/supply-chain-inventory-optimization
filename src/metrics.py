"""Forecast error metrics: WAPE (main), MAPE, bias, RMSE.

All metrics ignore pairs where the actual or the forecast is missing. Metrics are
also available as *additive sums* (``error_sums``), so that any subset of series
(a dashboard scenario) can be scored exactly by summing rows first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUM_COLUMNS = ("abs_err", "err", "actual", "sq_err", "n", "ape_sum", "n_pos")


def _pair(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype="float64").ravel()
    f = np.asarray(forecast, dtype="float64").ravel()
    if a.shape != f.shape:
        raise ValueError(f"actual and forecast have different sizes: {a.shape} vs {f.shape}")
    m = ~(np.isnan(a) | np.isnan(f))
    return a[m], f[m]


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: Σ|A − F| / Σ|A|."""
    a, f = _pair(actual, forecast)
    denom = np.abs(a).sum()
    return float(np.abs(a - f).sum() / denom) if denom > 0 else float("nan")


def mape(actual, forecast) -> float:
    """Mean absolute percentage error over actuals > 0."""
    a, f = _pair(actual, forecast)
    pos = a > 0
    return float(np.mean(np.abs(a[pos] - f[pos]) / a[pos])) if pos.any() else float("nan")


def bias(actual, forecast) -> float:
    """Σ(F − A) / ΣA; positive = over-forecast."""
    a, f = _pair(actual, forecast)
    denom = a.sum()
    return float((f - a).sum() / denom) if denom != 0 else float("nan")


def rmse(actual, forecast) -> float:
    a, f = _pair(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2))) if len(a) else float("nan")


def error_sums(actual: np.ndarray, forecast: np.ndarray, axis=None) -> dict[str, np.ndarray]:
    """Additive error sums along ``axis`` (NaN pairs ignored)."""
    a = np.asarray(actual, dtype="float64")
    f = np.asarray(forecast, dtype="float64")
    m = ~(np.isnan(a) | np.isnan(f))
    a0 = np.where(m, a, 0.0)
    f0 = np.where(m, f, 0.0)
    e = f0 - a0
    pos = m & (a > 0)
    ape = np.where(pos, np.abs(e) / np.where(pos, a, 1.0), 0.0)
    return {
        "abs_err": np.abs(e).sum(axis=axis),
        "err": e.sum(axis=axis),
        "actual": a0.sum(axis=axis),
        "sq_err": (e ** 2).sum(axis=axis),
        "n": m.sum(axis=axis),
        "ape_sum": ape.sum(axis=axis),
        "n_pos": pos.sum(axis=axis),
    }


def metrics_from_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``wape``, ``mape``, ``bias`` and ``rmse`` columns computed from the sums."""
    out = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["wape"] = np.where(out["actual"] > 0, out["abs_err"] / out["actual"], np.nan)
        out["mape"] = np.where(out["n_pos"] > 0, out["ape_sum"] / out["n_pos"], np.nan)
        out["bias"] = np.where(out["actual"] != 0, out["err"] / out["actual"], np.nan)
        out["rmse"] = np.where(out["n"] > 0, np.sqrt(out["sq_err"] / out["n"]), np.nan)
    return out


def evaluate(errors: pd.DataFrame, by) -> pd.DataFrame:
    """Aggregate an error-sums table by ``by`` and compute the metrics."""
    by = list(by)
    grouped = errors.groupby(by, as_index=False, observed=True)[list(SUM_COLUMNS)].sum()
    return metrics_from_sums(grouped)


def improvement(leaderboard: pd.DataFrame, baseline: str = "seasonal_naive", metric: str = "wape",
                model_col: str = "model") -> pd.Series:
    """Relative improvement of each model vs. the baseline: 1 − metric / metric_baseline."""
    lb = leaderboard.set_index(model_col)[metric]
    if baseline not in lb.index:
        raise ValueError(f"Baseline {baseline!r} not in the leaderboard")
    return 1 - lb / lb[baseline]
