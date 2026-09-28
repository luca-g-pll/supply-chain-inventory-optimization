"""Simulation invariants (T-S1 .. T-S11), with random demand from hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.inventory import NaiveWeeklyPolicy, StaticSQPolicy
from src.simulation import calibrate_to_fill_rate, simulate, simulate_batch

DATES = pd.date_range("2017-01-01", periods=60, freq="D")


def _demand(seed: int, n: int = 4, lam: float = 20.0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(lam, size=(n, len(DATES))).astype(float)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), s=st.floats(0, 200), q=st.floats(1, 300), lt=st.integers(1, 10),
       share=st.sampled_from([0.0, 0.5, 1.0]))
def test_stock_balance_and_bounds(seed, s, q, lt, share):  # T-S1, T-S2, T-S3, T-S9
    d = _demand(seed)
    n = d.shape[0]
    res = simulate_batch(d, DATES, StaticSQPolicy(np.full(n, s), np.full(n, float(np.ceil(q)))), lt, 100.0,
                         backorder_share=share, record_daily=True)
    y = res.daily
    begin = np.concatenate([np.full((n, 1), 100.0), y["on_hand_end"][:, :-1]], axis=1)
    served_bo = begin + y["receipt"] - y["sales"] - y["on_hand_end"]
    # on hand end = begin + receipts - sales - backlog served
    assert np.all(served_bo >= -1e-9)
    assert np.all(y["on_hand_end"] >= -1e-9)                               # no negative stock
    assert np.all((y["sales"] >= -1e-9) & (y["sales"] <= y["demand"] + 1e-9))
    assert np.allclose(y["sales"] + y["lost"] + y["backordered"], y["demand"])
    if share == 0.0:
        assert np.allclose(served_bo, 0.0) and np.allclose(y["backordered"], 0.0)
    assert np.allclose(res.sums["demand"], d.sum(axis=1))


def test_orders_arrive_after_exactly_the_lead_time():  # T-S4
    d = np.full(len(DATES), 10.0)
    daily, _ = simulate(d, DATES, NaiveWeeklyPolicy(np.array([70.0]), order_weekday=0), 5, 0.0)
    order_days = np.flatnonzero(daily["order_qty"].to_numpy() > 0)
    receipt_days = np.flatnonzero(daily["receipt"].to_numpy() > 0)
    assert len(order_days) > 3
    assert np.array_equal(order_days[: len(receipt_days)] + 5, receipt_days)


def test_zero_demand_gives_full_fill_rate():  # T-S5
    d = np.zeros((2, len(DATES)))
    res = simulate_batch(d, DATES, StaticSQPolicy(np.array([-1.0, -1.0]), np.array([10.0, 10.0])), 3, 5.0)
    assert np.all(res.fill_rate == 1.0)
    assert np.all(res.sums["n_orders"] == 0)


def test_huge_reorder_point_serves_everything_and_negative_one_never_orders():  # T-S6
    d = _demand(1)
    n = d.shape[0]
    full = simulate_batch(d, DATES, StaticSQPolicy(np.full(n, 1e6), np.full(n, 1e5)), 1, 1e6)
    assert np.allclose(full.fill_rate, 1.0)
    never = simulate_batch(d, DATES, StaticSQPolicy(np.full(n, -1e9), np.full(n, 10.0)), 1, 50.0)
    assert np.all(never.sums["n_orders"] == 0)


def test_calibration_reaches_the_target():  # T-S7
    d = _demand(2, n=6)
    n = d.shape[0]
    mean = d.mean(axis=1)

    def policy_for(z):
        return StaticSQPolicy(s=mean * 3 + z * np.sqrt(3) * d.std(axis=1), Q=np.full(n, 200.0))

    knob, res, feasible = calibrate_to_fill_rate(policy_for, d, DATES, 3, mean * 5, 0.0, 0.95, -4, 8)
    assert feasible.all()
    assert np.all(res.fill_rate >= 0.95)
    # a slightly smaller knob would miss the target
    lower = simulate_batch(d, DATES, policy_for(knob - 0.05), 3, mean * 5).fill_rate
    assert np.all(lower <= res.fill_rate + 1e-12)


def test_deterministic():  # T-S8
    d = _demand(5)
    pol = StaticSQPolicy(np.full(4, 60.0), np.full(4, 100.0))
    a = simulate_batch(d, DATES, pol, 4, 80.0).sums
    b = simulate_batch(d, DATES, pol, 4, 80.0).sums
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_backorder_share_zero_equals_lost_sales():  # T-S10
    d = _demand(6)
    pol = StaticSQPolicy(np.full(4, 30.0), np.full(4, 60.0))
    a = simulate_batch(d, DATES, pol, 4, 40.0, backorder_share=0.0).sums
    b = simulate_batch(d, DATES, pol, 4, 40.0).sums
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_backorders_reduce_lost_units():  # T-S11
    d = _demand(7)
    pol = StaticSQPolicy(np.full(4, 30.0), np.full(4, 60.0))
    lost_only = simulate_batch(d, DATES, pol, 4, 40.0, backorder_share=0.0).sums["lost"]
    with_bo = simulate_batch(d, DATES, pol, 4, 40.0, backorder_share=0.8).sums["lost"]
    assert np.all(with_bo <= lost_only + 1e-9)


def test_invalid_arguments():
    d = _demand(8)
    pol = StaticSQPolicy(np.full(4, 30.0), np.full(4, 60.0))
    with pytest.raises(ValueError):
        simulate_batch(d, DATES, pol, 0, 10.0)
    with pytest.raises(ValueError):
        simulate_batch(d, DATES, pol, 2, 10.0, backorder_share=1.5)
