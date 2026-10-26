"""New products planned from a similar product (T-N1 .. T-N8)."""

import numpy as np
import pandas as pd
import pytest

from src.cold_start import (
    blend_forecasts,
    handover_weight,
    launch_growth,
    name_similarity,
    new_series_params,
    suggest_analogs,
)
from src.config import ColdStartConfig, load_config
from src.ingest import ColumnMapping, standardise
from src.pipeline import run_in_memory
from src.series import DemandMatrix

CFG = ColdStartConfig()
T, NEW_DAYS = 400, 30


def _world(named: bool = True, seed: int = 3):
    """Items A (weekly pattern, a mid-life step), B (smooth season), C (flat); N = 0.5 × A launched 30 days ago."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    dow = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 0.8])[t % 7]
    a = 20 * dow * np.where(t > T - 20, 1.4, 1.0)
    b = 15 * (1 + 0.5 * np.sin(2 * np.pi * t / 90))
    c = np.full(T, 12.0)
    stores, rows, keys, starts = ["S1", "S2"], [], [], []
    for s, f in zip(stores, (1.0, 0.7)):
        for item, mu in (("A", a), ("B", b), ("C", c)):
            rows.append(rng.poisson(mu * f).astype(float))
            keys.append((s, item))
            starts.append(0)
        new = np.full(T, np.nan)
        new[T - NEW_DAYS:] = rng.poisson(0.5 * a[T - NEW_DAYS:] * f)
        rows.append(new)
        keys.append((s, "N"))
        starts.append(T - NEW_DAYS)
    dm = DemandMatrix(pd.DataFrame(keys, columns=["store", "item"]), pd.date_range("2024-01-01", periods=T),
                      np.array(rows), np.array(starts))
    master = pd.DataFrame({"item": ["A", "B", "C", "N"], "unit_price": [0.99, 1.19, 3.29, 1.09],
                           "unit_cost": [0.6, 0.7, 2.0, 0.7], "lead_time_days": [4, 4, 4, 4]})
    if named:
        master["item_name"] = ["Penne rigate 500g", "Passata di pomodoro", "Caffè macinato", "Penne integrali 500g"]
        master["category"] = ["Pasta", "Sauces", "Coffee", "Pasta"]
        master["supplier"] = ["Grocery", "Grocery", "Coffee Co", "Grocery"]
    new_rows = np.flatnonzero(dm.keys["item"].to_numpy() == "N")
    return dm, master, new_rows


def test_name_similarity():  # T-N1
    assert name_similarity("Penne integrali 500g", "Penne rigate 500g") > 0.25
    assert name_similarity("Penne integrali 500g", "Passata di pomodoro 700g") < 0.1
    assert name_similarity("", "Penne") == 0.0


@pytest.mark.parametrize("named", [True, False])
def test_the_true_similar_product_is_suggested(named):  # T-N2
    dm, master, new_rows = _world(named)
    items, series = suggest_analogs(dm, new_rows, master, CFG)
    row = items.set_index("item").loc["N"]
    assert row["analog"] == "A"
    assert row["scale"] == pytest.approx(0.5, rel=0.25)
    assert "early_sales" in row["reasons"]
    if named:
        assert {"name", "category", "supplier"} <= set(row["reasons"].split(","))
        assert row["confidence"] == "high"
    assert set(series["analog"]) == {"A"} and len(series) == 2


def test_new_in_one_store_uses_the_same_product_elsewhere():  # T-N3
    dm, master, _ = _world()
    rows = np.flatnonzero((dm.keys["store"] == "S2").to_numpy() & (dm.keys["item"] == "A").to_numpy())
    dm.values[rows, : T - 40] = np.nan
    dm.start[rows] = T - 40
    _, series = suggest_analogs(dm, rows, master, CFG)
    s = series.iloc[0]
    assert s["rule"] == "same_item" and s["analog"] == "A"
    assert s["scale"] == pytest.approx(0.7, rel=0.2)                   # S2 sells 70% of S1


def test_handover_and_launch_growth():  # T-N4
    assert handover_weight(0, 28) == 0 and handover_weight(28, 28) == pytest.approx(0.5)
    assert handover_weight(84, 28) == pytest.approx(0.875)
    flat = np.full(14, 10.0)
    growing = np.r_[np.full(7, 10.0), np.full(7, 13.0)]
    assert launch_growth(growing, flat, 30, CFG) == pytest.approx(1.3)
    assert launch_growth(np.r_[np.full(7, 10.0), np.full(7, 40.0)], flat, 30, CFG) == CFG.launch_growth_cap
    assert launch_growth(growing, flat, 200, CFG) == 1.0                # no longer a launch
    assert launch_growth(growing, growing, 30, CFG) == 1.0              # same growth as the similar product: season


def test_blend_moves_to_own_sales():  # T-N5
    est = pd.DataFrame({"store": "S1", "item": "A", "h": np.arange(1, 8), "forecast": 10.0})
    own = pd.DataFrame({"store": "S1", "item": "N", "h": np.arange(1, 8), "forecast": 2.0})
    young = pd.DataFrame({"store": ["S1"], "item": ["N"], "age_days": [0], "analog": ["A"], "scale": [0.5],
                          "rule": ["suggested"], "growth": [1.0]})
    f0 = blend_forecasts(young, own, est, 28)
    assert f0["forecast"].eq(5.0).all() and f0["weight"].eq(0).all()     # all from the similar product
    f28 = blend_forecasts(young.assign(age_days=28), own, est, 28)
    assert f28["forecast"].iloc[0] == pytest.approx(0.5 * 2 + 0.5 * 5)
    assert blend_forecasts(young.assign(growth=1.2), own, est, 28)["forecast"].iloc[0] == pytest.approx(6.0)


def test_new_products_get_cautious_parameters():  # T-N6
    cfg = load_config()
    fc = pd.DataFrame({"store": "S1", "item": "N", "h": np.arange(1, 15), "forecast": 10.0, "weight": 0.0})
    series = pd.DataFrame({"store": ["S1"], "item": ["N"], "analog": ["A"]})
    master = pd.DataFrame({"item": ["N"], "unit_cost": [1.0], "lead_time_days": [4]})
    cv = pd.Series({"A": 0.3})
    p0 = new_series_params(fc, series, master, cv, {"physical": cfg.channel("physical")}, cfg.cold_start).iloc[0]
    p1 = new_series_params(fc.assign(weight=1.0), series, master, cv, {"physical": cfg.channel("physical")},
                           cfg.cold_start).iloc[0]
    assert p0["order_qty"] <= 10 * cfg.cold_start.first_lot_days
    assert p0["sigma_L"] == pytest.approx(2 * p1["sigma_L"])            # double while relying on the similar product
    assert p0["sigma_L"] == pytest.approx(0.3 * 10 * 2 * np.sqrt(4))
    assert p0["on_hand"] >= 10 * 4                                     # estimated stock covers the lead time


def test_pipeline_plans_a_recent_launch():  # T-N7
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", "2024-06-30")
    rows = []
    for store in ("S1", "S2"):
        for item, mu in (("A", 20), ("B", 8), ("C", 5)):
            rows.append(pd.DataFrame({"date": dates, "store": store, "item": item, "quantity": rng.poisson(mu, len(dates))}))
        late = dates[-25:]
        rows.append(pd.DataFrame({"date": late, "store": store, "item": "N", "quantity": rng.poisson(10, len(late))}))
    raw = pd.concat(rows, ignore_index=True)
    cfg = load_config(overrides={"forecast": {"models": ["seasonal_naive", "moving_average", "lightgbm"]}})
    data = standardise(raw, ColumnMapping("date", "item", "quantity", "store"), cfg.input, cfg.prices, cfg.inventory)
    tb = run_in_memory(data, cfg).tables()
    assert list(tb["new_items"]["item"]) == ["N"]
    assert set(tb["new_series"]["store"]) == {"S1", "S2"}
    assert set(tb["new_forecast_own"]["item"]) == {"N"} and "N" not in set(tb["plan_forecast"]["item"])
    assert "N" not in set(tb["policy_summary"]["item"])                # not in the comparison on the past


def test_cold_start_can_be_switched_off():  # T-N8
    dm, master, new_rows = _world()
    cfg = load_config(overrides={"cold_start": {"enabled": False}})
    assert cfg.cold_start.enabled is False
    assert len(suggest_analogs(dm, np.array([], dtype=int), master, CFG)[0]) == 0
