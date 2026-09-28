"""Input adapter on messy files (T-G1 .. T-G10)."""

import dataclasses
import io

import numpy as np
import pandas as pd
import pytest

from src.data import demo_mapping
from src.ingest import (
    ColumnMapping,
    InputError,
    guess_mapping,
    list_sheets,
    merge_item_master,
    parse_dates,
    parse_numbers,
    read_any,
    standardise,
    validate,
)


def _std(raw, mapping, cfg):
    return standardise(raw, ColumnMapping.from_dict(mapping), cfg.input, cfg.prices, cfg.inventory)


def test_guess_mapping_italian_columns():  # T-G1
    m = guess_mapping(["Data", "Articolo", "Qtà", "Negozio", "Prezzo unitario"])
    assert (m["date"], m["item"], m["quantity"], m["store"]) == ("Data", "Articolo", "Qtà", "Negozio")
    assert m["unit_price"] == "Prezzo unitario"


def test_guess_mapping_prefers_exact_names_and_cost_over_price():
    m = guess_mapping(["date", "store", "item", "sales", "prezzo_acquisto", "prezzo"])
    assert m["quantity"] == "sales" and m["unit_cost"] == "prezzo_acquisto" and m["unit_price"] == "prezzo"


def test_semicolon_csv_with_day_first_dates_and_decimal_commas(cfg):  # T-G2
    text = "Data;Articolo;Qtà\n15/03/2024;A1;1,5\n16/03/2024;A1;2\n17/03/2024;A1;1.234,5\n"
    raw = read_any(io.BytesIO(text.encode()), filename="vendite.csv")
    assert list(raw.columns) == ["Data", "Articolo", "Qtà"]
    data = _std(raw, guess_mapping(raw.columns), cfg)
    assert data.sales["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-03-15", "2024-03-16", "2024-03-17"]
    assert data.sales["sales"].tolist() == [1.5, 2.0, 1234.5]


def test_excel_sheet_selection(tmp_path):  # T-G3
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"x": [1]}).to_excel(xw, sheet_name="Notes", index=False)
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "sku": ["A"] * 3, "qty": [1, 2, 3]}).to_excel(
            xw, sheet_name="Vendite", index=False)
    assert list_sheets(path) == ["Notes", "Vendite"]
    raw = read_any(path, sheet="Vendite")
    assert list(raw.columns) == ["date", "sku", "qty"]


def test_transaction_rows_are_summed(cfg):  # T-G4
    raw = pd.DataFrame({"date": ["2024-01-01"] * 3 + ["2024-01-02"], "item": ["A"] * 4, "qty": [1, 2, 3, 4]})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    assert data.sales["sales"].tolist() == [6.0, 4.0]
    assert any("summed" in w for w in data.notes["warnings"])


def test_missing_days_are_zero_filled_only_after_the_first_sale(cfg):  # T-G5
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-05", "2024-01-03"], "item": ["A", "A", "B"],
                        "qty": [1, 1, 2]})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    a = data.sales[data.sales["item"] == "A"]
    b = data.sales[data.sales["item"] == "B"]
    assert len(a) == 5 and a["sales"].tolist() == [1, 0, 0, 0, 1]
    assert b["date"].min() == pd.Timestamp("2024-01-03") and len(b) == 3


def test_negative_quantities_clip_or_reject(cfg):  # T-G6
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "item": ["A", "A"], "qty": [5, -2]})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    assert data.sales["sales"].tolist() == [5.0, 0.0]
    strict = dataclasses.replace(cfg.input, negative_quantity="reject")
    with pytest.raises(InputError):
        standardise(raw, ColumnMapping("date", "item", "qty"), strict, cfg.prices, cfg.inventory)


def test_no_store_column_means_one_location_and_no_price_means_assumed(cfg):  # T-G7, T-G8
    raw = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=70).astype(str), "item": "A", "qty": 3})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    assert data.sales["store"].unique().tolist() == ["ALL"]
    assert data.item_master.at[0, "price_source"] == "assumed"
    report = validate(data, cfg.input, prices=cfg.prices, inventory=cfg.inventory)
    assert report.ok
    assert any("Selling price missing" in a for a in report.assumptions)


def test_blocking_problems_are_reported_not_raised(cfg):  # T-G9
    raw = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30).astype(str), "item": "A", "qty": 1})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    report = validate(data, cfg.input)
    assert not report.ok and "56 days" in report.errors[0]
    with pytest.raises(InputError):
        ColumnMapping.from_dict({"date": "date", "item": None, "quantity": "qty"})
    with pytest.raises(InputError):
        read_any(io.BytesIO(b"whatever"), filename="data.xls")
    with pytest.raises(InputError):
        _std(pd.DataFrame({"date": ["x", "y"], "item": ["A", "B"], "qty": [1, 2]}),
             {"date": "date", "item": "item", "quantity": "qty"}, cfg)


def test_demo_columns_go_through_the_same_adapter(cfg, raw_sales):  # T-G10
    data = standardise(raw_sales, demo_mapping(), cfg.input, cfg.prices, cfg.inventory, source="demo")
    direct = raw_sales.groupby(["store", "item"])["quantity"].sum().sort_index()
    ours = data.sales.groupby(["store", "item"])["sales"].sum().sort_index()
    np.testing.assert_allclose(ours.to_numpy(), direct.to_numpy())
    assert len(data.sales) == len(raw_sales)


def test_item_master_overrides_assumed_prices(cfg):
    raw = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60).astype(str), "item": "A", "qty": 3})
    data = _std(raw, {"date": "date", "item": "item", "quantity": "qty"}, cfg)
    master = pd.DataFrame({"sku": ["A"], "prezzo": ["12,50"], "lead_time": [4]})
    merged = merge_item_master(data, master, cfg.prices, cfg.inventory)
    row = merged.item_master.iloc[0]
    assert row["unit_price"] == 12.5 and row["price_source"] == "item_master"
    assert row["unit_cost"] == pytest.approx(12.5 * cfg.inventory.cost_to_price_ratio)
    assert int(row["lead_time_days"]) == 4


def test_parsers():
    d, _ = parse_dates(pd.Series(["2024-03-01 10:22", "2024-03-02T08:00"]))
    assert d.dt.day.tolist() == [1, 2]
    d, notes = parse_dates(pd.Series(["03/04/2024", "05/06/2024"]))
    assert d.dt.month.tolist() == [4, 6] and notes  # ambiguous -> day-first, with a note
    d, _ = parse_dates(pd.Series(["03/13/2024"]))
    assert d.dt.month.tolist() == [3]  # 13 cannot be a month -> month-first
    assert parse_numbers(pd.Series(["1,234", "12,345"])).tolist() == [1234.0, 12345.0]
    assert parse_numbers(pd.Series(["€ 3,5", "1,234.5"])).tolist() == [3.5, 1234.5]
