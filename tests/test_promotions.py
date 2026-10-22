"""Promotions: reading, detection from prices, effect estimate, promotion-aware forecasts (T-P1 .. T-P10)."""

import numpy as np
import pandas as pd
import pytest

from src.config import PromotionsConfig
from src.features import build_features
from src.forecast import MovingAverage, run_backtest
from src.ingest import ColumnMapping, guess_mapping, standardise
from src.promotions import (
    PromoAware,
    estimate_effects,
    parse_promo_values,
    plan_promotion,
    post_promo_mask,
    promo_accuracy,
    promo_day_counter,
    promo_events,
    read_promo_calendar,
)
from src.series import DemandMatrix

PCFG = PromotionsConfig()


def _promo_shop(days: int = 730, lift: float = 1.8, dip: float = 0.85, cut: float = 0.25, seed: int = 1,
                stores=("A", "B")) -> pd.DataFrame:
    """Daily rows of one item with a 7-day promotion every 6 weeks (price −cut, sales ×lift, dip after)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    promo = np.zeros(days, dtype=bool)
    for a in range(30, days - 10, 42):
        promo[a: a + 7] = True
    after = np.zeros(days, dtype=bool)
    for a in range(37, days - 3, 42):
        after[a: a + 7] = True
    after &= ~promo
    list_price = np.where(dates < "2023-01-01", 10.0, 10.5)          # +5% list price: not a promotion
    price = np.round(list_price * np.where(promo, 1 - cut, 1.0), 2)
    mu = 20 * np.where(promo, lift, np.where(after, dip, 1.0))
    frames = [pd.DataFrame({"date": dates, "store": s, "item": "X1", "quantity": rng.poisson(mu),
                            "unit_price": price}) for s in stores]
    out = pd.concat(frames, ignore_index=True)
    out.attrs["promo"] = promo
    return out


def _std(raw, cfg, mapping=None):
    m = mapping or guess_mapping(raw.columns)
    return standardise(raw, ColumnMapping.from_dict(m), cfg.input, cfg.prices, cfg.inventory, promotions=cfg.promotions)


def test_parse_promo_values():  # T-P1
    assert parse_promo_values(pd.Series(["1", "0", "1"])).isna().tolist() == [True, False, True]
    v = parse_promo_values(pd.Series(["sì", "no", "", "Volantino"]))
    assert v.isna().tolist() == [True, False, False, True]
    d = parse_promo_values(pd.Series(["20%", "0", "-30%", "15"]))
    assert d.round(2).tolist() == [0.20, 0.0, 0.30, 0.15]
    assert parse_promo_values(pd.Series([0.25, 0.0, 0.1])).round(2).tolist() == [0.25, 0.0, 0.1]


def test_promo_column_is_recognised():  # T-P2
    m = guess_mapping(["Data", "Articolo", "Quantità", "Promozione"])
    assert m["promo"] == "Promozione"
    assert guess_mapping(["date", "sku", "qty", "sconto"])["promo"] == "sconto"


def test_promotions_found_from_prices(cfg):  # T-P3
    raw = _promo_shop()
    data = _std(raw, cfg)
    assert data.notes["promo_source"] == "price"
    truth = pd.Series(raw.attrs["promo"], index=pd.date_range("2022-01-01", periods=len(raw.attrs["promo"])))
    found = data.promo[data.promo["store"] == "A"].set_index("date")["discount"]
    assert set(found.index) == set(truth[truth].index)          # every promotion, and the list-price rise is not one
    assert found.round(2).eq(0.25).all()


def test_promo_column_wins_over_prices(cfg):  # T-P4
    shop = _promo_shop()
    raw = shop.drop(columns="unit_price")
    raw["in_promo"] = np.tile(np.where(shop.attrs["promo"], "si", "no"), 2)
    data = _std(raw, cfg)
    assert data.notes["promo_source"] == "file"
    assert data.promo["discount"].eq(cfg.promotions.default_discount).all()   # flag without price: default depth


def test_no_prices_no_promotions(data):  # T-P5
    assert data.promo is None and data.notes["promo_source"] is None
    assert data.matrix().promo is None


def test_effects_recover_lift_and_dip(cfg):  # T-P6
    dm = _std(_promo_shop(days=1095, stores=("A", "B", "C")), cfg).matrix()
    eff = estimate_effects(dm, dm.n_days - 1, PCFG)
    lift_25 = np.exp(eff.beta[0] * 0.25)
    assert lift_25 == pytest.approx(1.8, rel=0.12)
    assert eff.dip[0] == pytest.approx(0.85, abs=0.07)
    assert eff.item_table.loc[0, "promo_days"] > 0


def test_effects_use_only_training_days(cfg):  # T-P7
    dm = _std(_promo_shop(), cfg).matrix()
    cut = 400
    changed = DemandMatrix(dm.keys, dm.dates, dm.values.copy(), dm.start, dm.promo)
    changed.values[:, cut + 1:] *= 3
    a, b = estimate_effects(dm, cut, PCFG), estimate_effects(changed, cut, PCFG)
    assert np.allclose(a.beta, b.beta) and np.allclose(a.dip, b.dip)


def test_promo_aware_model_beats_blind_on_promotion_days(cfg):  # T-P8
    dm = _std(_promo_shop(), cfg).matrix()
    a, b = 500, dm.n_days - 1
    blind = run_backtest([MovingAverage(28)], dm, "test", a, b, 14)
    aware = run_backtest([PromoAware(MovingAverage(28), PCFG)], dm, "test", a, b, 14)
    acc_b = promo_accuracy(blind, dm, PCFG).groupby("day_type")[["abs_error", "actual"]].sum()
    acc_a = promo_accuracy(aware, dm, PCFG).groupby("day_type")[["abs_error", "actual"]].sum()
    wape_b = acc_b["abs_error"] / acc_b["actual"]
    wape_a = acc_a["abs_error"] / acc_a["actual"]
    assert wape_a["promo"] < 0.6 * wape_b["promo"]
    assert wape_a["normal"] <= wape_b["normal"] * 1.02


def test_promo_features_describe_the_target_day():  # T-P9
    n, t = 1, 60
    values = np.ones((n, t))
    promo = np.zeros((n, t))
    promo[0, 20:27] = 0.3
    fs = build_features(values, pd.date_range("2024-01-01", periods=t), np.zeros(n), np.zeros(n), [14, 21],
                        [7], min_lag=14, yearly=False, promo=promo, post_promo_days=7)
    assert fs.series_features["promo_discount"][0, 20] == 0.3
    assert promo_day_counter(promo)[0, 20:27].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert post_promo_mask(promo, 7)[0, 27:34].all() and not post_promo_mask(promo, 7)[0, 34]
    # demand features still ignore the last 13 days
    changed = values.copy()
    changed[0, 47:] = 99
    fs2 = build_features(changed, pd.date_range("2024-01-01", periods=t), np.zeros(n), np.zeros(n), [14, 21],
                         [7], min_lag=14, yearly=False, promo=promo)
    assert np.allclose(np.nan_to_num(fs.series_features["lag_14"][0, :61]),
                       np.nan_to_num(fs2.series_features["lag_14"][0, :61]))


def test_events_and_planner(cfg):  # T-P10
    dm = _std(_promo_shop(), cfg).matrix()
    ev = promo_events(dm, PCFG)
    assert (ev["days"] == 7).all() and ev["discount"].round(2).eq(0.25).all()
    sub = dm.subset([0])
    assert sub.promo is not None and sub.promo.shape == sub.values.shape
    base = pd.Series({d: 10.0 for d in range(7)})
    r = plan_promotion(base, pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-09"), 0.25, np.log(1.8) / 0.25,
                       0.85, 7, price=10.0, cost=6.0)
    assert r["baseline_units"] == 70 and r["promo_units"] == pytest.approx(126)
    assert r["extra_units"] == pytest.approx(56) and r["dip_units"] == pytest.approx(10.5)
    # margin: 126 × (7.5 − 6) + 59.5 × 4 − (70 + 70) × 4
    assert r["margin_change"] == pytest.approx(126 * 1.5 + 59.5 * 4 - 140 * 4)


def test_promo_calendar_file():  # T-P11
    from src.ingest import read_any

    cal = read_promo_calendar(read_any("data/samples/promo_calendar_2025.csv"), 0.2)
    assert cal["item"].tolist() == ["GRO-002", "BEV-003", "HOM-001", "GRO-006"]
    assert cal["store"].tolist() == ["", "", "", "MI01"]
    assert cal["discount"].round(2).tolist() == [0.25, 0.30, 0.20, 0.25]
    assert (cal["end"] >= cal["start"]).all()
    it = pd.DataFrame({"Articolo": ["A1"], "Dal": ["03/02/2025"], "Al": ["09/02/2025"], "Promo": ["sì"]})
    r = read_promo_calendar(it, 0.2)
    assert r.at[0, "start"] == pd.Timestamp("2025-02-03") and r.at[0, "discount"] == 0.2
