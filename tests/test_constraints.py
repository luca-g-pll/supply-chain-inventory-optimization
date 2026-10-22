"""Supplier terms and limits in the purchase plan (T-C1 .. T-C10)."""

import numpy as np
import pandas as pd
import pytest

from src.constraints import (
    Constraints,
    SupplierTerms,
    parse_weekdays,
    project_constrained,
    read_supplier_terms,
    terms_from_frame,
)
from src.ingest import read_any
from src.replenishment import orders_frame, project

DAYS = 28
START = pd.Timestamp("2026-10-26")          # a Monday


def _setup(n=1, demand=10.0, L=3, Q=40.0, on_hand=40.0, cost=2.0, supplier="S", store="A"):
    F = np.full((n, DAYS + 30), demand)
    dates = pd.date_range(START, periods=F.shape[1])
    keys = pd.DataFrame({"store": [store] * n, "item": [f"X{i}" for i in range(n)], "supplier": [supplier] * n})
    args = dict(F_ext=F, days=DAYS, dates=dates, keys=keys, lead_time=np.full(n, L), Q=np.full(n, Q),
                safety_stock=np.zeros(n), on_hand=np.full(n, on_hand), backlog=np.zeros(n),
                arrivals=np.zeros((n, L + 1)), backorder_share=np.zeros(n), unit_cost=np.full(n, cost))
    return args


def test_parse_weekdays():  # T-C1
    assert parse_weekdays("lun mer ven") == (0, 2, 4)
    assert parse_weekdays("Monday, Thu") == (0, 3)
    assert parse_weekdays("Lunedì;Giovedì") == (0, 3)
    assert parse_weekdays("1 4") == (0, 3)
    assert parse_weekdays("") == tuple(range(7)) and parse_weekdays(None) == tuple(range(7))
    assert parse_weekdays(["1", "3"]) == (1, 3)


def test_supplier_terms_file():  # T-C2
    t = read_supplier_terms(read_any("data/samples/supplier_terms.csv"))
    g = t.set_index("supplier").loc["Grocery wholesaler"]
    assert g["order_days"] == ["1", "3"] and g["min_order_value"] == 300
    assert g["discount_qty"] == 120 and g["discount_pct"] == pytest.approx(0.03)
    terms = terms_from_frame(t)
    assert terms["Household goods supplier"].order_days == (1,)
    assert terms["Beverage distributor (regional)"].delivery_days == (0, 1, 2, 3, 4, 5)


def test_no_constraints_matches_the_plain_rule():  # T-C3
    a = _setup(n=3, demand=7.0, L=4, Q=50.0, on_hand=30.0)
    r = project_constrained(**a)
    keys = a["keys"][["store", "item"]]
    p = project(a["F_ext"], DAYS, a["dates"], a["lead_time"], a["Q"], a["safety_stock"], a["on_hand"],
                a["backlog"], a["arrivals"], 0.0)
    o = orders_frame(p, keys, a["dates"][:DAYS], a["lead_time"], a["F_ext"])
    assert np.allclose(r.daily["order_qty"], p.daily["order_qty"])
    assert len(r.orders) == len(o) and r.orders["notes"].eq("").all()


def test_pack_size_and_minimum_quantity():  # T-C4
    a = _setup(Q=25.0)
    r = project_constrained(**a, cons=Constraints(pack_size={"X0": 12}, min_order_qty={"X0": 30}))
    assert (r.orders["qty"] % 12 == 0).all() and (r.orders["qty"] >= 30).all()
    assert r.orders["notes"].str.contains("moq").all() and r.orders["notes"].str.contains("pack").all()


def test_order_days_and_coverage():  # T-C5
    a = _setup(L=2, Q=30.0, on_hand=60.0)
    r = project_constrained(**a, cons=Constraints(suppliers={"S": SupplierTerms(order_days=(1, 3))}))
    assert set(r.orders["date"].dt.dayofweek) <= {1, 3}
    # the reorder point covers the gap to the next order day: no expected stock-out after the start
    assert r.daily["short"][:, 7:].sum() == 0


def test_delivery_days():  # T-C6
    a = _setup(L=3, Q=30.0, on_hand=60.0)
    r = project_constrained(**a, cons=Constraints(suppliers={"S": SupplierTerms(delivery_days=(4,))}))
    assert len(r.orders) and set(r.orders["arrival"].dt.dayofweek) == {4}
    assert set(r.orders["date"].dt.dayofweek) == {1}                    # Tuesday + 3 days = Friday


def test_minimum_order_value():  # T-C7
    a = _setup(n=2, demand=5.0, L=2, Q=10.0, on_hand=100.0, cost=1.0)
    a["on_hand"] = np.array([12.0, 100.0])
    r = project_constrained(**a, cons=Constraints(suppliers={"S": SupplierTerms(min_order_value=50)}))
    by_day = r.orders.groupby("date")["value"].sum()
    assert (by_day >= 50 - 1e-9).all()
    assert r.orders["notes"].str.contains("topup").any() or (r.deferred["reason"] == "min_value").any()
    assert r.daily["short"].sum() == pytest.approx(0)


def test_weekly_budget():  # T-C8
    a = _setup(n=4, demand=10.0, L=2, Q=70.0, on_hand=15.0, cost=1.0)
    r = project_constrained(**a, cons=Constraints(weekly_budget=150.0))
    week = r.orders["date"].dt.isocalendar().week
    assert (r.orders.groupby(week)["value"].sum() <= 150 + 1e-9).all()
    assert (r.deferred["reason"] == "budget").any()


def test_warehouse_space():  # T-C9
    a = _setup(n=3, demand=10.0, L=3, Q=100.0, on_hand=20.0)
    r = project_constrained(**a, cons=Constraints(store_capacity={"A": 200}))
    stock_plus_order = r.daily["on_hand_end"].sum(axis=0)
    assert stock_plus_order.max() <= 200 + 1e-9


def test_quantity_discount():  # T-C10
    a = _setup(demand=10.0, L=2, Q=40.0, on_hand=30.0, cost=2.0)
    terms = SupplierTerms(discount_qty=60, discount_pct=0.05)
    r = project_constrained(**a, cons=Constraints(suppliers={"S": terms}, break_extra_days=3))
    assert (r.orders["qty"] >= 60).all()                                 # 20 units = 2 days away: raised
    assert r.orders["unit_cost"].iloc[0] == pytest.approx(1.9)
    assert r.orders["saving"].iloc[0] == pytest.approx(60 * 0.1)
    far = project_constrained(**a, cons=Constraints(suppliers={"S": terms}, break_extra_days=1))
    assert (far.orders["qty"] == 40).all()                               # too far: not raised
