import numpy as np
import pandas as pd
import pytest

from src.inventory import (
    ForecastSQPolicy,
    NaiveWeeklyPolicy,
    StaticSQPolicy,
    eoq,
    lot_size,
    normal_loss,
    reorder_point,
    safety_factor,
    safety_factor_fill_rate,
    safety_stock,
    sigma_lead_time,
)


def test_safety_factor_95():  # T-I1
    assert safety_factor(0.95) == pytest.approx(1.6449, abs=1e-4)


def test_eoq_textbook():  # T-I2
    assert eoq(1000, 50, 2) == pytest.approx(np.sqrt(50_000))


def test_sigma_lead_time_fixed_lead_time():  # T-I3
    assert sigma_lead_time(10, 4) == pytest.approx(20)


def test_sigma_lead_time_with_random_lead_time():
    assert sigma_lead_time(10, 4, mean_daily=5, sigma_lead_time_days=2) == pytest.approx(np.sqrt(400 + 100))


def test_reorder_point_and_safety_stock():  # T-I4
    assert reorder_point(5, 7, 10) == pytest.approx(45)
    assert safety_stock(2.0, 15) == pytest.approx(30)


def test_normal_loss_at_zero():  # T-I5
    assert normal_loss(0.0) == pytest.approx(0.3989, abs=1e-4)


@pytest.mark.parametrize("beta,Q,sigma", [(0.95, 500, 60), (0.99, 200, 80), (0.9, 50, 10)])
def test_fill_rate_factor_solves_the_loss_equation(beta, Q, sigma):  # T-I6
    k = safety_factor_fill_rate(beta, Q, sigma)
    assert normal_loss(k) == pytest.approx((1 - beta) * Q / sigma, rel=1e-6)


def test_fill_rate_factor_vectorised_and_zero_sigma():
    k = safety_factor_fill_rate(0.95, np.array([500.0, 500.0]), np.array([60.0, 0.0]))
    assert k.shape == (2,) and k[1] == 0.0


@pytest.mark.parametrize("call", [
    lambda: safety_factor(1.0),
    lambda: safety_factor(0.5),
    lambda: eoq(1000, 50, 0),
    lambda: eoq(1000, 0, 2),
    lambda: safety_factor_fill_rate(1.0, 100, 10),
])
def test_invalid_inputs_raise(call):  # T-I7
    with pytest.raises(ValueError):
        call()


def test_lot_size_is_whole_and_at_least_one():
    assert lot_size(0.0, 50, 2) == 1.0
    assert lot_size(1000, 50, 2) == np.ceil(np.sqrt(50_000))


def test_sq_rule_orders_enough_lots():
    pol = StaticSQPolicy(s=np.array([100.0, 100.0]), Q=np.array([50.0, 50.0]))
    q = pol.order(0, 0, np.array([120.0, 10.0]))
    # IP 120 > s -> nothing; IP 10 -> floor(90/50) + 1 = 2 lots
    assert q.tolist() == [0.0, 100.0]


def test_naive_orders_only_on_its_weekday():
    pol = NaiveWeeklyPolicy(qty=np.array([10.4]), order_weekday=0)
    assert pol.order(0, 0, np.array([0.0])).tolist() == [11.0]
    assert pol.order(1, 3, np.array([0.0])).tolist() == [0.0]


def test_forecast_policy_uses_the_daily_reorder_point():
    pol = ForecastSQPolicy(ltd=np.array([[10.0, 100.0]]), ss=np.array([5.0]), Q=np.array([20.0]))
    assert pol.order(0, 0, np.array([50.0])).tolist() == [0.0]      # s_0 = 15
    assert pol.order(1, 0, np.array([50.0])).tolist() == [60.0]     # s_1 = 105 -> 3 lots


def test_lot_size_is_capped_at_max_cover_days():
    from src.config import load_config
    from src.policy import PolicyInputs, policy_parameters

    cfg = load_config()
    n, days = 1, 30
    inp = PolicyInputs(
        keys=pd.DataFrame({"store": ["1"], "item": ["A"]}),
        test_dates=pd.date_range("2024-01-01", periods=days), test_days=np.arange(8, 8 + days),
        demand=np.full((n, days), 20.0), mean_daily=np.array([20.0]), sigma_daily=np.array([5.0]),
        unit_cost=np.array([0.6]), unit_price=np.array([1.0]), item_lead_time=np.array([np.nan]),
        fc_test=np.full((n, 6, 14), 20.0), origins_test=np.arange(7, 7 + 42, 7), fc_valid=np.full((n, 2, 14), 20.0),
        origins_valid=np.array([0, 7]), valid_days=np.arange(1, 8), valid_actual=np.full((n, 7), 20.0),
        best_model="seasonal_naive")
    capped = policy_parameters(inp, cfg.channel("physical"), cfg)
    # cheap fast mover: the raw EOQ would be months of demand; the lot stops at 28 days
    assert capped.Q[0] == 28 * 20
    uncapped_cfg = load_config(overrides={"inventory": {"max_cover_days": None}})
    assert policy_parameters(inp, uncapped_cfg.channel("physical"), uncapped_cfg).Q[0] > 28 * 20
