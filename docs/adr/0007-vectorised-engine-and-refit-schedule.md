# ADR-0007 — Vectorised simulation engine and quarterly refits

**Status:** Accepted · **Date:** 2026-09-28

## Context
The analysis simulates 500 series × 365 days × 3 policies × 2 channels × 2 variants, and the calibrated variant
needs a bisection (up to 40 simulations per policy). The back-test forecasts 53 weekly origins × 14 days × 500
series for two periods with four models. A naive implementation (one Python loop per series, one model fit per
origin) would take hours, and the dashboard must re-run an uploaded file in under a minute.

## Decision
1. **Dense demand matrix** (`src.series.DemandMatrix`): one row per series, one column per day.
2. **Simulation vectorised across series**: the engine loops over days only; each step updates every series at
   once with NumPy. The bisection is vectorised too (one simulation call per iteration for all series).
3. **Forecast once, use many times**: models expose `fit(dm, fit_end)` + `predict(dm, origins, horizon)`.
   Holt-Winters parameters come from statsmodels at each refit, then a vectorised filter (tested to reproduce
   statsmodels' recursion) updates the states with every new actual. LightGBM is one global model whose features
   are all ≥ 14 days old, so a fitted model predicts any origin in its block without retraining.
4. **Refit every 13 origins** (quarterly) in full mode, once per period in fast mode.

## Consequences
- ➕ Full demo run ≈ 90 s on a laptop (≈ 85 s forecasting, ≈ 2 s for all calibrated simulations); an 8-series
  upload runs in seconds in the dashboard.
- ➕ The same engine serves the pipeline, the notebooks and the dashboard's one-series what-if (milliseconds).
- ➖ Parameters are 1–13 weeks old at each forecast. This is realistic (planners rarely re-estimate models every
  week) and is stated in the methodology.
- ➖ The Holt-Winters filter duplicates statsmodels' recursion; `tests/test_forecast.py` guards it.
