"""KPI aggregation (T-K1, T-K2)."""

import pandas as pd
import pytest

from src.kpi import add_costs, aggregate_kpis, headline
from src.simulation import SUM_FIELDS


def _rows():
    base = {f: 0.0 for f in SUM_FIELDS}
    a = {**base, "demand": 100.0, "sales": 99.0, "lost": 1.0, "inv_unit_days": 3650.0, "n_orders": 10.0}
    b = {**base, "demand": 10.0, "sales": 5.0, "lost": 5.0, "inv_unit_days": 365.0, "n_orders": 2.0}
    df = pd.DataFrame([a, b])
    df["n_days"] = 365
    df["unit_cost"], df["unit_price"] = [6.0, 10.0], [10.0, 20.0]
    df["holding_rate"], df["order_cost"], df["backorder_cost"], df["days_per_year"] = 0.25, 50.0, 0.0, 365
    df["policy"], df["variant"] = "P2_forecast_sq", "calibrated"
    return add_costs(df)


def test_fill_rate_is_ratio_of_sums_not_mean_of_ratios():  # T-K1
    agg = aggregate_kpis(_rows(), [])
    assert agg.at[0, "fill_rate"] == pytest.approx(104 / 110)
    assert agg.at[0, "fill_rate"] != pytest.approx((0.99 + 0.5) / 2)


def test_total_cost_is_the_sum_of_components():  # T-K2
    r = _rows()
    assert (r["total_cost"] - (r["ordering_cost"] + r["holding_cost"] + r["shortage_cost"])).abs().max() < 1e-9
    assert r.at[0, "holding_cost"] == pytest.approx(3650 * 6 * 0.25 / 365)
    assert r.at[0, "shortage_cost"] == pytest.approx(1 * (10 - 6))
    assert r.at[0, "avg_inventory_units"] == pytest.approx(10.0)


def test_headline_reductions():
    r = _rows()
    p0 = r.copy()
    p0["policy"] = "P0_naive"
    p0["inv_value_days"] *= 2
    p0 = aggregate_kpis(p0, ["policy", "variant"]).assign(variant="calibrated")
    both = pd.concat([r, p0], ignore_index=True)
    h = headline(both)
    assert h["inventory_reduction_pct"] == pytest.approx(0.5)
