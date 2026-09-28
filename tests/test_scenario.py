"""Scenario filter and aggregation (T-C1 .. T-C6)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.kpi import SUM_COLUMNS, aggregate_kpis
from src.scenario import Scenario, apply_scenario, build_scenario, select_items, select_stores

REVENUE = pd.Series({"1": 10.0, "2": 50.0, "3": 30.0, "10": 40.0})


@pytest.mark.parametrize("kwargs", [
    dict(stores=(), items=("1",)),
    dict(stores=("1",), items=()),
    dict(stores=("1",), items=("1",), online_stores=frozenset({"2"})),
])
def test_invalid_scenarios(kwargs):  # T-C1
    with pytest.raises(ValueError):
        Scenario(**kwargs)
    with pytest.raises(ValueError):
        select_stores(["1", "2"], n_stores=3)
    with pytest.raises(ValueError):
        select_items(9, "top_revenue", REVENUE)


def test_top_revenue_selection():  # T-C2
    assert select_items(2, "top_revenue", REVENUE) == ("2", "10")


def test_random_selection_is_seeded():  # T-C3
    assert select_items(2, "random", REVENUE, seed=1) == select_items(2, "random", REVENUE, seed=1)


def _summary():
    rows = []
    for store in ("1", "2", "10"):
        for item in ("1", "2"):
            for channel in ("physical", "online"):
                for policy in ("P0_naive", "P2_forecast_sq"):
                    rows.append({**{c: 0.0 for c in SUM_COLUMNS},
                                 "store": store, "item": item, "channel": channel, "policy": policy,
                                 "variant": "calibrated", "demand": 10.0, "sales": 9.0 if channel == "physical" else 8.0,
                                 "n_days": 365, "inv_unit_days": 365.0, "inv_value_days": 3650.0,
                                 "total_cost": 100.0 if channel == "physical" else 50.0})
    return pd.DataFrame(rows)


def test_apply_scenario_takes_each_store_in_its_channel():  # T-C4
    sc = Scenario(("1", "10"), ("1", "2"), frozenset({"10"}))
    out = apply_scenario(_summary(), sc)
    assert len(out) == 2 * 2 * 2  # stores x items x policies
    assert set(out.loc[out["store"] == "10", "channel"]) == {"online"}
    assert set(out.loc[out["store"] == "1", "channel"]) == {"physical"}


def test_default_scenario_equals_full_physical_run():  # T-C5
    s = _summary()
    sc = build_scenario(["1", "2", "10"], pd.Series({"1": 1.0, "2": 2.0}))
    a = aggregate_kpis(apply_scenario(s, sc), ["policy"])
    b = aggregate_kpis(s[s["channel"] == "physical"], ["policy"])
    np.testing.assert_allclose(a["total_cost"], b["total_cost"])


def test_scenario_kpis_equal_direct_simulation(dm, cfg):  # T-C6
    from src.forecast import MovingAverage, SeasonalNaive, run_backtest
    from src.kpi import add_costs
    from src.policy import build_policy_inputs, run_all_policies
    from src.profile import profile_history

    prof = profile_history(dm, cfg)
    vs, ve = dm.date_index(prof.valid[0]), dm.date_index(prof.valid[1])
    ts, te = dm.date_index(prof.test[0]), dm.date_index(prof.test[1])
    btv = run_backtest([SeasonalNaive(), MovingAverage()], dm, "validation", vs, ve, 14, 7)
    btt = run_backtest([SeasonalNaive(), MovingAverage()], dm, "test", ts, te, 14, 7)
    master = pd.DataFrame({"item": sorted(dm.keys["item"].unique()), "unit_price": 10.0, "unit_cost": 6.0,
                           "lead_time_days": pd.array([pd.NA] * dm.keys["item"].nunique(), dtype="Int64")})
    cfg_fast = cfg.replace(simulation=dataclasses.replace(cfg.simulation, calibrate_to_fill_rate=False))
    inp = build_policy_inputs(dm, btv, btt, "moving_average", master)
    full = add_costs(run_all_policies(inp, cfg_fast))
    rows = np.flatnonzero(dm.keys["store"].isin(["1", "2"]).to_numpy())
    sub = add_costs(run_all_policies(inp.subset(rows), cfg_fast))
    sc = Scenario(("1", "2"), tuple(sorted(dm.keys["item"].unique())), frozenset({"2"}))
    a = aggregate_kpis(apply_scenario(full, sc), ["policy", "variant"]).sort_values(["policy", "variant"])
    b = aggregate_kpis(apply_scenario(sub, sc), ["policy", "variant"]).sort_values(["policy", "variant"])
    np.testing.assert_allclose(a["total_cost"].to_numpy(), b["total_cost"].to_numpy())
    np.testing.assert_allclose(a["fill_rate"].to_numpy(), b["fill_rate"].to_numpy())
