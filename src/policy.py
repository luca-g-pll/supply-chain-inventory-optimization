"""Build and simulate the replenishment policies P0 / P1 / P2 for every series.

* **P0 naive** — every Monday order the mean weekly demand of the validation period.
* **P1 static (s, Q)** — ``s = d̄·L + z·σ_d·√L`` from validation-period statistics.
* **P2 forecast-driven (s_t, Q)** — ``s_t = Σ forecast over (t, t+L] + z·σ_E`` where
  ``σ_E`` is the RMSE of the L-day cumulative forecast error of the best model
  on the validation back-test.

Both (s, Q) policies share the same EOQ lot ``Q``. Each policy is run in two
variants: **fixed** (z from the service target, set in advance) and
**calibrated** (knob tuned per series until the simulated fill rate reaches the
target — the iso-service comparison of ADR-0004).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ChannelParams, Config
from src.forecast import BacktestResult, lead_time_demand
from src.inventory import (
    ForecastSQPolicy,
    NaiveWeeklyPolicy,
    StaticSQPolicy,
    lot_size,
    safety_factor,
    safety_factor_fill_rate,
)
from src.series import DemandMatrix
from src.simulation import BatchResult, calibrate_to_fill_rate, simulate_batch

KNOB_RANGE = {"P0_naive": (0.0, 4.0), "P1_static_sq": (-12.0, 8.0), "P2_forecast_sq": (-12.0, 8.0)}
POLICY_LABELS = {"P0_naive": "P0 · Naive weekly order", "P1_static_sq": "P1 · Static (s, Q)",
                 "P2_forecast_sq": "P2 · Forecast-driven (s, Q)"}


@dataclass
class PolicyInputs:
    keys: pd.DataFrame
    test_dates: pd.DatetimeIndex
    test_days: np.ndarray          # day indices of the test period
    demand: np.ndarray             # (n, T) test-period actuals
    mean_daily: np.ndarray         # validation-period mean
    sigma_daily: np.ndarray        # validation-period std
    unit_cost: np.ndarray
    unit_price: np.ndarray
    item_lead_time: np.ndarray     # item-level lead time (NaN = use the channel default)
    fc_test: np.ndarray            # best model paths (n, J, H), test back-test
    origins_test: np.ndarray
    fc_valid: np.ndarray           # best model paths, validation back-test
    origins_valid: np.ndarray
    valid_days: np.ndarray
    valid_actual: np.ndarray       # (n, Tv)
    best_model: str

    @property
    def n(self) -> int:
        return len(self.keys)

    def lead_time(self, channel: ChannelParams, override: int | None = None) -> np.ndarray:
        if override is not None:
            return np.full(self.n, int(override))
        lt = np.where(np.isnan(self.item_lead_time), channel.inventory.lead_time_days, self.item_lead_time)
        return np.maximum(lt.astype(int), 1)

    def subset(self, rows) -> PolicyInputs:
        rows = np.asarray(rows)
        return PolicyInputs(
            self.keys.iloc[rows].reset_index(drop=True), self.test_dates, self.test_days, self.demand[rows],
            self.mean_daily[rows], self.sigma_daily[rows], self.unit_cost[rows], self.unit_price[rows],
            self.item_lead_time[rows], self.fc_test[rows], self.origins_test, self.fc_valid[rows],
            self.origins_valid, self.valid_days, self.valid_actual[rows], self.best_model)


def build_policy_inputs(dm: DemandMatrix, bt_valid: BacktestResult, bt_test: BacktestResult, best_model: str,
                        item_master: pd.DataFrame) -> PolicyInputs:
    """Collect everything the policies need, aligned on the series order of ``dm``."""
    vs, ve = bt_valid.period_start, bt_valid.period_end
    ts, te = bt_test.period_start, bt_test.period_end
    valid = dm.values[:, vs: ve + 1]
    with np.errstate(invalid="ignore"):
        count = (~np.isnan(valid)).sum(axis=1)
        mean = np.where(count > 0, np.nansum(valid, axis=1) / np.maximum(count, 1), 0.0)
        centred = np.where(np.isnan(valid), 0.0, valid - mean[:, None])
        sigma = np.where(count > 1, np.sqrt((centred ** 2).sum(axis=1) / np.maximum(count - 1, 1)), 0.0)
    master = item_master.set_index("item")
    items = dm.keys["item"]
    lead = pd.to_numeric(master["lead_time_days"], errors="coerce").astype("float64")
    return PolicyInputs(
        keys=dm.keys.reset_index(drop=True),
        test_dates=dm.dates[ts: te + 1],
        test_days=np.arange(ts, te + 1),
        demand=np.nan_to_num(dm.values[:, ts: te + 1], nan=0.0),
        mean_daily=mean,
        sigma_daily=sigma,
        unit_cost=master["unit_cost"].reindex(items).to_numpy(dtype="float64"),
        unit_price=master["unit_price"].reindex(items).to_numpy(dtype="float64"),
        item_lead_time=lead.reindex(items).to_numpy(dtype="float64"),
        fc_test=bt_test.forecasts[best_model],
        origins_test=bt_test.origins,
        fc_valid=bt_valid.forecasts[best_model],
        origins_valid=bt_valid.origins,
        valid_days=np.arange(vs, ve + 1),
        valid_actual=np.nan_to_num(valid, nan=0.0),
        best_model=best_model,
    )


def forecast_error_sigma(inp: PolicyInputs, lead_time: np.ndarray) -> np.ndarray:
    """RMSE of the cumulative forecast error over the lead time (validation period).

    For each decision day ``t`` with ``t + L`` inside the validation period:
    ``E_t = Σ actual(t+1..t+L) − Σ forecast(t+1..t+L)``. Falls back to ``σ_d·√L``
    when the validation period is too short for a series.
    """
    ltd = lead_time_demand(inp.fc_valid, inp.origins_valid, inp.valid_days, lead_time)
    n, tv = inp.valid_actual.shape
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(inp.valid_actual, axis=1)], axis=1)
    k = np.arange(tv)
    end = np.clip(k[None, :] + 1 + lead_time[:, None], 0, tv)
    actual_l = csum[np.arange(n)[:, None], end] - csum[:, 1: tv + 1]
    valid_mask = (k[None, :] + lead_time[:, None]) <= tv - 1
    err = np.where(valid_mask, actual_l - ltd, 0.0)
    count = valid_mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(count >= 7, np.sqrt((err ** 2).sum(axis=1) / np.maximum(count, 1)), np.nan)
    fallback = inp.sigma_daily * np.sqrt(lead_time)
    return np.where(np.isnan(sigma), fallback, sigma)


@dataclass
class PolicyParams:
    lead_time: np.ndarray
    Q: np.ndarray
    sigma_L_hist: np.ndarray       # σ_d·√L (P1)
    sigma_L_fc: np.ndarray         # forecast-error σ over L (P2)
    z_p1: np.ndarray               # safety factors from the service target (fixed variant)
    z_p2: np.ndarray
    base_weekly_qty: np.ndarray    # P0 quantity before the multiplier
    ltd: np.ndarray                # (n, T) P2 lead-time demand forecast per decision day
    initial_on_hand: np.ndarray
    holding_rate: float
    service_level: float


def policy_parameters(inp: PolicyInputs, channel: ChannelParams, cfg: Config,
                      lead_time_override: int | None = None,
                      service_level: float | None = None) -> PolicyParams:
    inv = channel.inventory
    sl = service_level if service_level is not None else inv.service_level
    L = inp.lead_time(channel, lead_time_override)
    annual = inp.mean_daily * inv.days_per_year
    h = inv.holding_rate * inp.unit_cost
    Q = lot_size(np.maximum(annual, 0.0), inv.order_cost, h)
    if inv.max_cover_days:
        Q = np.minimum(Q, np.maximum(np.ceil(inp.mean_daily * inv.max_cover_days - 1e-9), 1.0))
    sig_hist = inp.sigma_daily * np.sqrt(L)
    sig_fc = forecast_error_sigma(inp, L)
    if inv.service_type == "fill_rate":
        z1 = safety_factor_fill_rate(sl, Q, sig_hist)
        z2 = safety_factor_fill_rate(sl, Q, sig_fc)
    else:
        z = safety_factor(sl)
        z1 = np.full(inp.n, z)
        z2 = np.full(inp.n, z)
    ltd = lead_time_demand(inp.fc_test, inp.origins_test, inp.test_days, L)
    s1 = inp.mean_daily * L + z1 * sig_hist
    initial = np.maximum(np.ceil(s1 + Q / 2), 0.0)
    return PolicyParams(lead_time=L, Q=Q, sigma_L_hist=sig_hist, sigma_L_fc=sig_fc, z_p1=z1, z_p2=z2,
                        base_weekly_qty=7.0 * inp.mean_daily, ltd=ltd, initial_on_hand=initial,
                        holding_rate=inv.holding_rate, service_level=sl)


def make_policy(name: str, p: PolicyParams, inp: PolicyInputs, knob: np.ndarray, weekday: int = 0):
    """Policy object for a knob: quantity multiplier (P0) or safety factor z (P1, P2)."""
    knob = np.broadcast_to(np.asarray(knob, dtype="float64"), (inp.n,))
    if name == "P0_naive":
        return NaiveWeeklyPolicy(qty=knob * p.base_weekly_qty, order_weekday=weekday)
    if name == "P1_static_sq":
        return StaticSQPolicy(s=inp.mean_daily * p.lead_time + knob * p.sigma_L_hist, Q=p.Q)
    if name == "P2_forecast_sq":
        return ForecastSQPolicy(ltd=p.ltd, ss=knob * p.sigma_L_fc, Q=p.Q)
    raise ValueError(f"Unknown policy {name!r}")


def default_knob(name: str, p: PolicyParams) -> np.ndarray:
    if name == "P0_naive":
        return np.ones_like(p.Q)
    return p.z_p1 if name == "P1_static_sq" else p.z_p2


@dataclass
class PolicyRun:
    policy: str
    variant: str                   # fixed | calibrated
    channel: str
    knob: np.ndarray
    feasible: np.ndarray
    result: BatchResult
    params: PolicyParams


def run_policy(name: str, variant: str, inp: PolicyInputs, channel: ChannelParams, cfg: Config,
               params: PolicyParams | None = None, record_daily: bool = False) -> PolicyRun:
    p = params or policy_parameters(inp, channel, cfg)
    wd = cfg.simulation.naive_order_weekday
    kwargs = dict(lead_time=p.lead_time, initial_on_hand=p.initial_on_hand,
                  backorder_share=channel.backorder_share)
    if variant == "calibrated":
        lo, hi = KNOB_RANGE[name]
        knob, _, feasible = calibrate_to_fill_rate(
            lambda k: make_policy(name, p, inp, k, wd), inp.demand, inp.test_dates,
            target=cfg.simulation.target_fill_rate, lo=lo, hi=hi, **kwargs)
    else:
        knob = default_knob(name, p)
        feasible = np.ones(inp.n, dtype=bool)
    result = simulate_batch(inp.demand, inp.test_dates, make_policy(name, p, inp, knob, wd),
                            record_daily=record_daily, **kwargs)
    return PolicyRun(name, variant, channel.name, knob, feasible, result, p)


def run_frame(run: PolicyRun, inp: PolicyInputs, channel: ChannelParams) -> pd.DataFrame:
    """One row per series with parameters and additive simulation sums."""
    p = run.params
    df = inp.keys.copy()
    df["channel"] = run.channel
    df["policy"] = run.policy
    df["variant"] = run.variant
    df["knob"] = run.knob
    df["calibration_ok"] = run.feasible
    df["lead_time"] = p.lead_time
    df["mean_daily"] = inp.mean_daily
    df["order_qty"] = p.Q if run.policy != "P0_naive" else np.ceil(run.knob * p.base_weekly_qty - 1e-9)
    if run.policy == "P0_naive":
        df["sigma_L"] = np.nan
        df["safety_stock"] = np.nan
        df["reorder_point"] = np.nan
    elif run.policy == "P1_static_sq":
        df["sigma_L"] = p.sigma_L_hist
        df["safety_stock"] = run.knob * p.sigma_L_hist
        df["reorder_point"] = inp.mean_daily * p.lead_time + df["safety_stock"]
    else:
        df["sigma_L"] = p.sigma_L_fc
        df["safety_stock"] = run.knob * p.sigma_L_fc
        df["reorder_point"] = p.ltd.mean(axis=1) + df["safety_stock"]   # average dynamic s_t
    df["initial_on_hand"] = p.initial_on_hand
    df["unit_cost"] = inp.unit_cost
    df["unit_price"] = inp.unit_price
    df["holding_rate"] = channel.inventory.holding_rate
    df["order_cost"] = channel.inventory.order_cost
    df["backorder_cost"] = channel.backorder_cost
    df["days_per_year"] = channel.inventory.days_per_year
    sums = run.result.sums_frame()
    return pd.concat([df.reset_index(drop=True), sums.reset_index(drop=True)], axis=1)


def run_all_policies(inp: PolicyInputs, cfg: Config, progress=None) -> pd.DataFrame:
    """Every channel x policy x variant: the simulation summary table."""
    frames = []
    variants = ["fixed"] + (["calibrated"] if cfg.simulation.calibrate_to_fill_rate else [])
    combos = [(c, p, v) for c in cfg.simulation.channels for p in cfg.simulation.policies for v in variants]
    params_cache: dict[str, PolicyParams] = {}
    for k, (ch_name, pol, var) in enumerate(combos):
        if progress:
            progress(f"Simulating {POLICY_LABELS[pol]} · {ch_name} · {var}", k / len(combos))
        channel = cfg.channel(ch_name)
        if ch_name not in params_cache:
            params_cache[ch_name] = policy_parameters(inp, channel, cfg)
        run = run_policy(pol, var, inp, channel, cfg, params_cache[ch_name])
        frames.append(run_frame(run, inp, channel))
    return pd.concat(frames, ignore_index=True)


Z_GRID = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
MULTIPLIER_GRID = (0.9, 0.95, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.15, 1.2, 1.3)


def tradeoff_curves(inp: PolicyInputs, cfg: Config, channel_name: str = "physical") -> pd.DataFrame:
    """Fill rate vs. inventory for each policy over a grid of its knob (same knob for all series)."""
    from src.kpi import SUM_COLUMNS, add_costs, derive_kpis

    channel = cfg.channel(channel_name)
    p = policy_parameters(inp, channel, cfg)
    wd = cfg.simulation.naive_order_weekday
    rows = []
    for pol in cfg.simulation.policies:
        grid = MULTIPLIER_GRID if pol == "P0_naive" else Z_GRID
        for knob in grid:
            k = np.full(inp.n, float(knob))
            res = simulate_batch(inp.demand, inp.test_dates, make_policy(pol, p, inp, k, wd),
                                 lead_time=p.lead_time, initial_on_hand=p.initial_on_hand,
                                 backorder_share=channel.backorder_share)
            run = PolicyRun(pol, "grid", channel.name, k, np.ones(inp.n, dtype=bool), res, p)
            frame = add_costs(run_frame(run, inp, channel))
            agg = frame[list(SUM_COLUMNS)].sum()
            agg["n_days"] = frame["n_days"].max()
            agg["policy"] = pol
            agg["knob"] = float(knob)
            rows.append(agg)
    return derive_kpis(pd.DataFrame(rows).reset_index(drop=True))


def series_inputs(test_dates: pd.DatetimeIndex, valid_dates: pd.DatetimeIndex, actual_valid: np.ndarray,
                  actual_test: np.ndarray, paths_valid: np.ndarray, origins_valid: pd.DatetimeIndex,
                  paths_test: np.ndarray, origins_test: pd.DatetimeIndex, unit_cost: float, unit_price: float,
                  item_lead_time: float, store: str, item: str, best_model: str) -> PolicyInputs:
    """PolicyInputs for a single series, rebuilt from dated arrays (dashboard what-if).

    Day indices are counted from the day before the validation period.
    """
    day0 = valid_dates[0] - pd.Timedelta(days=1)

    def _idx(dates) -> np.ndarray:
        return np.asarray((pd.DatetimeIndex(dates) - day0).days, dtype=int)

    valid = np.asarray(actual_valid, dtype="float64")
    mean = float(np.mean(valid)) if len(valid) else 0.0
    sigma = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    return PolicyInputs(
        keys=pd.DataFrame({"store": [str(store)], "item": [str(item)]}),
        test_dates=pd.DatetimeIndex(test_dates),
        test_days=_idx(test_dates),
        demand=np.asarray(actual_test, dtype="float64")[None, :],
        mean_daily=np.array([mean]),
        sigma_daily=np.array([sigma]),
        unit_cost=np.array([float(unit_cost)]),
        unit_price=np.array([float(unit_price)]),
        item_lead_time=np.array([float(item_lead_time) if item_lead_time is not None else np.nan]),
        fc_test=np.asarray(paths_test, dtype="float64")[None, :, :],
        origins_test=_idx(origins_test),
        fc_valid=np.asarray(paths_valid, dtype="float64")[None, :, :],
        origins_valid=_idx(origins_valid),
        valid_days=_idx(valid_dates),
        valid_actual=valid[None, :],
        best_model=best_model,
    )


def sensitivity(inp: PolicyInputs, cfg: Config, channel_name: str = "physical") -> pd.DataFrame:
    """Lead time x service level grid for P1 and P2 (fixed variant), summed over series."""
    from src.kpi import SUM_COLUMNS, add_costs, derive_kpis

    channel = cfg.channel(channel_name)
    rows = []
    for L in cfg.sensitivity.lead_times:
        for sl in cfg.sensitivity.service_levels:
            p = policy_parameters(inp, channel, cfg, lead_time_override=L, service_level=sl)
            for pol in ("P1_static_sq", "P2_forecast_sq"):
                run = run_policy(pol, "fixed", inp, channel, cfg, p)
                frame = add_costs(run_frame(run, inp, channel))
                agg = frame[list(SUM_COLUMNS)].sum()
                agg["n_days"] = frame["n_days"].max()
                agg["policy"] = pol
                agg["lead_time"] = L
                agg["service_level"] = sl
                rows.append(agg)
    return derive_kpis(pd.DataFrame(rows).reset_index(drop=True))
