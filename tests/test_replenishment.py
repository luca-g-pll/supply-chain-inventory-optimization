"""Purchase plan on the current calendar (T-R1 .. T-R7)."""

import numpy as np
import pandas as pd
import pytest

from src.ingest import InputError
from src.replenishment import (
    apply_planned_promotions,
    extend_weekly,
    lead_time_sums,
    orders_frame,
    plan_calendar,
    project,
    read_stock_file,
    stockout_risks,
)


def test_calendar_moves_data_by_whole_weeks():  # T-R1
    cal = plan_calendar("2024-12-31", "2026-10-22")
    assert cal.shift_days % 7 == 0
    assert cal.start.dayofweek == (pd.Timestamp("2025-01-01") + pd.Timedelta(days=cal.shift_days)).dayofweek
    assert 0 <= cal.gap_days < 7 and cal.today == pd.Timestamp("2026-10-22")
    recent = plan_calendar("2026-10-20", "2026-10-22")             # data up to date: no shift
    assert recent.shift_days == 0 and recent.gap_days == 1
    off = plan_calendar("2024-12-31", "2026-10-22", bring_to_today=False)
    assert off.shift_days == 0 and off.start == pd.Timestamp("2025-01-01")


def test_extend_weekly_reuses_the_last_week():  # T-R2
    F = np.arange(1, 15, dtype=float)[None, :]                    # h = 1..14
    out = extend_weekly(F, 21)
    assert out[0, :14].tolist() == F[0].tolist()
    assert out[0, 14:].tolist() == F[0, 7:14].tolist()             # h = 15..21 -> h = 8..14


def test_lead_time_sums():  # T-R3
    F = np.ones((2, 20))
    s = lead_time_sums(F, np.array([3, 5]), 10)
    assert (s[0] == 3).all() and (s[1] == 5).all()


def test_projection_orders_before_running_out():  # T-R4
    days = 28
    F = np.full((1, days + 10), 10.0)
    dates = pd.date_range("2026-10-22", periods=F.shape[1])
    r = project(F, days, dates, np.array([5]), np.array([70.0]), np.array([0.0]), np.array([60.0]),
                np.array([0.0]), np.zeros((1, 6)))
    o = orders_frame(r, pd.DataFrame({"store": ["A"], "item": ["X"]}), dates[:days], np.array([5]), F)
    # first order when stock + on order <= 5 days of demand (50): at the end of day 1 (60 - 10 = 50)
    assert o["date"].iloc[0] == dates[0] and o["qty"].iloc[0] == 70
    assert o["arrival"].iloc[0] == dates[5]
    assert stockout_risks(r, pd.DataFrame({"store": ["A"], "item": ["X"]}), dates[:days]).empty


def test_projection_flags_stock_that_cannot_last():  # T-R5
    F = np.full((1, 40), 10.0)
    dates = pd.date_range("2026-10-22", periods=40)
    r = project(F, 28, dates, np.array([7]), np.array([100.0]), np.array([0.0]), np.array([20.0]),
                np.array([0.0]), np.zeros((1, 8)))
    risk = stockout_risks(r, pd.DataFrame({"store": ["A"], "item": ["X"]}), dates[:28])
    assert len(risk) == 1 and risk["first_day"].iloc[0] == dates[2]
    assert risk["units_short"].iloc[0] == pytest.approx(50)            # days 2..6 before the first delivery on day 7


def test_orders_already_on_the_way_are_counted():  # T-R6
    F = np.full((1, 40), 10.0)
    dates = pd.date_range("2026-10-22", periods=40)
    arrivals = np.zeros((1, 8))
    arrivals[0, 3] = 100
    r = project(F, 28, dates, np.array([7]), np.array([100.0]), np.array([0.0]), np.array([40.0]),
                np.array([0.0]), arrivals)
    assert r.daily["receipt"][0, 3] == 100
    assert r.daily["order_qty"][0, 0] == 0                              # 40 + 100 on order covers the lead time


def test_planned_promotions_and_stock_file():  # T-R7
    keys = pd.DataFrame({"store": ["A", "B"], "item": ["X", "X"]})
    dates = pd.date_range("2026-11-02", periods=21)
    plan = pd.DataFrame({"item": ["X"], "store": ["A"], "start": [dates[7]], "end": [dates[13]], "discount": [25.0]})
    out = apply_planned_promotions(np.ones((2, 21)), keys, dates, plan, {"X": np.log(1.8) / 0.25}, {"X": 0.85}, 7)
    assert out[0, 7:14] == pytest.approx(1.8) and out[0, 14:21] == pytest.approx(0.85)
    assert (out[1] == 1).all() and (out[0, :7] == 1).all()
    raw = pd.DataFrame({"Negozio": ["A"], "Articolo": ["X"], "Giacenza": ["12"], "Ordinato": ["30"],
                        "Data arrivo": ["05/11/2026"]})
    stock = read_stock_file(raw)
    assert stock.loc[0, "on_hand"] == 12 and stock.loc[0, "on_order"] == 30
    assert stock.loc[0, "arrival"] == pd.Timestamp("2026-11-05")
    with pytest.raises(InputError):
        read_stock_file(pd.DataFrame({"Articolo": ["X"], "Giacenza": [1]}))   # several stores, no store column
