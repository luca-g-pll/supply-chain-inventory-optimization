"""Hypothetical sample datasets: deterministic, realistic patterns, readable by the input adapter."""

import io

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.ingest import ColumnMapping, current_item_values, guess_mapping, read_any, standardise, validate
from src.sample_data import (
    CHAIN_PRODUCTS,
    CHAIN_STORES,
    ecommerce_order_lines,
    generate_chain,
    italian_holidays,
    single_store_italian,
)

SMALL_STORES = tuple(s for s in CHAIN_STORES if s.code in ("MI02", "TO01", "NA01", "WEB"))
SMALL_PRODUCTS = tuple(p for p in CHAIN_PRODUCTS if p.code in ("BEV-001", "GRO-001", "HOM-005", "HOM-006",
                                                                 "SEA-001", "SEA-003", "APP-001", "BEV-008"))


@pytest.fixture(scope="module")
def chain():
    return generate_chain(stores=SMALL_STORES, products=SMALL_PRODUCTS, seed=1)


def test_generator_is_deterministic(chain):
    again = generate_chain(stores=SMALL_STORES, products=SMALL_PRODUCTS, seed=1)
    pd.testing.assert_frame_equal(chain[0], again[0])
    other = generate_chain(stores=SMALL_STORES, products=SMALL_PRODUCTS, seed=2)
    assert not chain[0]["quantity"].equals(other[0]["quantity"])


def test_calendar_and_store_rules(chain):
    sales = chain[0].assign(date=lambda d: pd.to_datetime(d["date"]))
    physical = sales[sales["channel"] == "physical"]
    assert not (physical["date"].dt.strftime("%m-%d") == "12-25").any()          # closed on Christmas
    assert not ((sales["store"] == "MI02") & (sales["date"].dt.dayofweek == 6)).any()   # closed on Sundays
    renovation = (sales["store"] == "TO01") & sales["date"].between("2023-08-07", "2023-08-20")
    assert not renovation.any()
    assert sales.loc[sales["store"] == "NA01", "date"].min() >= pd.Timestamp("2022-06-01")
    assert pd.Timestamp("2024-04-01") in italian_holidays([2024])                # Pasquetta 2024


def test_product_life_cycle_and_seasons(chain):
    sales = chain[0].assign(date=lambda d: pd.to_datetime(d["date"]))
    old, new = sales[sales["item"] == "HOM-005"], sales[sales["item"] == "HOM-006"]
    assert old["date"].max() <= pd.Timestamp("2023-09-30")
    assert new["date"].min() >= pd.Timestamp("2023-10-01")
    panettone = sales[sales["item"] == "SEA-001"]
    assert set(panettone["date"].dt.month) <= {11, 12, 1}
    colomba = sales[sales["item"] == "SEA-003"]
    assert set(colomba["date"].dt.month) <= {2, 3, 4}


def test_promotions_lower_the_price(chain):
    s = chain[0]
    pasta = s[s["item"] == "GRO-001"]
    list_prices = pasta.groupby(pasta["date"].str[:4])["unit_price"].max()
    assert (pasta["unit_price"] < pasta["date"].str[:4].map(list_prices) - 0.05).any()


def test_chain_sample_passes_the_input_contract(chain):
    cfg = load_config()
    sales, items, _ = chain
    raw = read_any(io.BytesIO(sales.to_csv(index=False).encode()), filename="chain.csv")
    data = standardise(raw, ColumnMapping.from_dict(guess_mapping(raw.columns)), cfg.input, cfg.prices,
                       cfg.inventory)
    report = validate(data, cfg.input, enforce_limits=True)
    assert report.ok and report.n_series == len(SMALL_STORES) * len(SMALL_PRODUCTS)
    assert set(data.stores.loc[data.stores["channel"] == "online", "store"]) == {"WEB"}


def test_italian_and_excel_samples_are_read_correctly():
    cfg = load_config()
    it = single_store_italian()
    raw = read_any(io.BytesIO(it.to_csv(sep=";", index=False).encode("utf-8-sig")), filename="negozio.csv")
    m = guess_mapping(raw.columns)
    assert (m["date"], m["item"], m["quantity"], m["unit_price"]) == ("Data", "Codice", "Quantità", "Prezzo")
    data = standardise(raw, ColumnMapping.from_dict(m), cfg.input, cfg.prices, cfg.inventory)
    assert data.date_min == pd.Timestamp("2023-07-01") and data.sales["store"].unique().tolist() == ["ALL"]

    lines = ecommerce_order_lines()
    assert (lines["Qtà"] < 0).any() and lines["ID ordine"].duplicated().any()   # returns, multi-line orders
    buf = io.BytesIO()
    lines.to_excel(buf, index=False)
    raw = read_any(io.BytesIO(buf.getvalue()), filename="ordini.xlsx")
    data = standardise(raw, ColumnMapping.from_dict(guess_mapping(raw.columns)), cfg.input, cfg.prices,
                       cfg.inventory)
    assert data.stores.at[0, "channel"] == "online"
    assert any("negative quantities" in w for w in data.notes["warnings"])


def test_item_price_ignores_a_promotion_at_the_end_of_the_history():
    dates = pd.date_range("2024-01-01", periods=120)
    price = np.full(120, 10.0)
    price[-10:] = 7.0                      # promotion running when the history ends
    attrs = pd.DataFrame({"date": dates, "item": "A", "unit_price": price, "unit_cost": np.nan,
                          "lead_time_days": np.nan})
    assert current_item_values(attrs).at["A", "unit_price"] == 10.0
