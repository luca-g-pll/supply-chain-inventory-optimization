"""Series-level computations for the Inventory page (UC-05, UC-06).

Everything here works on ONE series and runs in milliseconds, so the what-if can
recompute safety stock, reorder point and EOQ and re-simulate the whole test
period on every slider move.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from common import ActiveData

from src.config import ChannelParams, Config
from src.kpi import add_costs
from src.policy import (
    KNOB_RANGE,
    PolicyInputs,
    PolicyRun,
    make_policy,
    policy_parameters,
    run_frame,
    series_inputs,
)
from src.simulation import calibrate_to_fill_rate, simulate_batch

POLICIES = ("P0_naive", "P1_static_sq", "P2_forecast_sq")


def detail_series(data: ActiveData) -> pd.DataFrame:
    """(store, item) pairs with daily detail available."""
    return data.t("detail_actuals")[["store", "item"]].drop_duplicates()


def build_inputs(data: ActiveData, store: str, item: str) -> PolicyInputs:
    act = data.t("detail_actuals")
    act = act[(act["store"] == store) & (act["item"] == item)].sort_values("date")
    v, t = act[act["period"] == "validation"], act[act["period"] == "test"]
    paths = data.t("detail_paths")
    paths = paths[(paths["store"] == store) & (paths["item"] == item)]

    def _matrix(period: str):
        p = paths[paths["period"] == period].pivot(index="origin", columns="h", values="forecast").sort_index()
        return p.to_numpy(dtype="float64"), pd.DatetimeIndex(p.index)

    fv, ov = _matrix("validation")
    ft, ot = _matrix("test")
    m = data.t("item_master").set_index("item").loc[item]
    lt = m["lead_time_days"]
    return series_inputs(
        test_dates=pd.DatetimeIndex(t["date"]), valid_dates=pd.DatetimeIndex(v["date"]),
        actual_valid=v["sales"].to_numpy(), actual_test=t["sales"].to_numpy(),
        paths_valid=fv, origins_valid=ov, paths_test=ft, origins_test=ot,
        unit_cost=float(m["unit_cost"]), unit_price=float(m["unit_price"]),
        item_lead_time=float(lt) if pd.notna(lt) else np.nan, store=store, item=item,
        best_model=data.best_model)


def channel_with(cfg: Config, name: str, **inventory_changes) -> ChannelParams:
    ch = cfg.channel(name)
    backorder_share = inventory_changes.pop("backorder_share", ch.backorder_share)
    inv = dataclasses.replace(ch.inventory, **inventory_changes)
    return dataclasses.replace(ch, inventory=inv, backorder_share=backorder_share)


def simulate_series(inp: PolicyInputs, channel: ChannelParams, cfg: Config, knobs: dict[str, float] | None = None,
                    calibrate: bool = False, lead_time: int | None = None,
                    service_level: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run P0/P1/P2 on one series. Returns (per-policy KPI rows, daily trajectory)."""
    p = policy_parameters(inp, channel, cfg, lead_time_override=lead_time, service_level=service_level)
    wd = cfg.simulation.naive_order_weekday
    kwargs = dict(lead_time=p.lead_time, initial_on_hand=p.initial_on_hand, backorder_share=channel.backorder_share)
    rows, daily = [], []
    for pol in POLICIES:
        if calibrate:
            lo, hi = KNOB_RANGE[pol]
            knob, _, feasible = calibrate_to_fill_rate(
                lambda k, pol=pol: make_policy(pol, p, inp, k, wd), inp.demand, inp.test_dates,
                target=cfg.simulation.target_fill_rate, lo=lo, hi=hi, **kwargs)
        elif knobs and pol in knobs:
            knob, feasible = np.array([knobs[pol]]), np.array([True])
        else:
            knob = np.ones(1) if pol == "P0_naive" else (p.z_p1 if pol == "P1_static_sq" else p.z_p2)
            feasible = np.array([True])
        res = simulate_batch(inp.demand, inp.test_dates, make_policy(pol, p, inp, knob, wd), record_daily=True,
                             **kwargs)
        rows.append(add_costs(run_frame(PolicyRun(pol, "what-if", channel.name, knob, feasible, res, p), inp,
                                        channel)))
        d = pd.DataFrame({f: v[0] for f, v in res.daily.items()})
        d.insert(0, "date", inp.test_dates)
        d.insert(0, "policy", pol)
        daily.append(d)
    return pd.concat(rows, ignore_index=True), pd.concat(daily, ignore_index=True)
