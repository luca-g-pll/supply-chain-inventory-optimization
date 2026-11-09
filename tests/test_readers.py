"""File formats: every reader returns the same table, and odd spreadsheets are repaired."""

import io
import json

import pandas as pd
import pytest

from src.ingest import (
    ColumnMapping,
    InputError,
    detect_granularity,
    expand_periods,
    guess_mapping,
    read_any,
    standardise,
)
from src.readers import SUPPORTED, grid_header_row, period_of, text_header_row, unpivot_wide

ROWS = [{"date": "2024-01-01", "item": "A1", "quantity": 3}, {"date": "2024-01-02", "item": "A1", "quantity": 5},
        {"date": "2024-01-02", "item": "B2", "quantity": 1}]


def _cells(df: pd.DataFrame) -> list[list[str]]:
    return [[str(v) for v in row] for row in df[["date", "item", "quantity"]].itertuples(index=False)]


EXPECTED = [[r["date"], r["item"], str(r["quantity"])] for r in ROWS]


def test_json_records_wrapped_and_lines():
    plain = read_any(json.dumps(ROWS).encode(), "a.json")
    wrapped = read_any(json.dumps({"meta": {"n": 3}, "data": {"rows": ROWS}}).encode(), "a.json")
    lines = read_any("\n".join(json.dumps(r) for r in ROWS).encode(), "a.jsonl")
    columnar = read_any(json.dumps({k: [r[k] for r in ROWS] for k in ROWS[0]}).encode(), "a.json")
    for df in (plain, wrapped, lines, columnar):
        assert _cells(df) == EXPECTED


def test_json_nested_order_lines_are_opened():
    orders = [{"order_date": "2024-01-01", "id": 1, "lines": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]},
              {"order_date": "2024-01-02", "id": 2, "lines": [{"sku": "A1", "qty": 4}]}]
    df = read_any(json.dumps(orders).encode(), "orders.json")
    assert len(df) == 3 and {"sku", "qty", "order_date"} <= set(df.columns)
    assert df.attrs["import_notes"]
    mapping = guess_mapping(df.columns)
    assert mapping["date"] == "order_date" and mapping["item"] == "sku" and mapping["quantity"] == "qty"


def test_markdown_table_largest_is_read():
    text = ("# Sales\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\nText in between.\n\n"
            "| date | item | quantity |\n|:-----|:----:|---------:|\n"
            + "\n".join(f"| {r['date']} | {r['item']} | {r['quantity']} |" for r in ROWS) + "\n")
    df = read_any(text.encode(), "notes.md")
    assert _cells(df) == EXPECTED
    with pytest.raises(InputError):
        read_any(b"# Only a title\n\nNo table here.\n", "empty.md")


def test_html_and_fake_xls():
    html = ("<html><body><table><tr><th>date</th><th>item</th><th>quantity</th></tr>"
            + "".join(f"<tr><td>{r['date']}</td><td>{r['item']}</td><td>{r['quantity']}</td></tr>" for r in ROWS)
            + "</table></body></html>").encode()
    assert _cells(read_any(html, "export.html")) == EXPECTED
    renamed = read_any(html, "export.xls")                      # an HTML export saved with the .xls extension
    assert _cells(renamed) == EXPECTED and "HTML" in renamed.attrs["import_notes"][0]
    text_xls = read_any(b"date\titem\tquantity\n2024-01-01\tA1\t3\n", "export.xls")
    assert text_xls.loc[0, "item"] == "A1"


def test_xml_records():
    xml = ("<rows>" + "".join(f"<row><date>{r['date']}</date><item>{r['item']}</item><quantity>{r['quantity']}"
                              "</quantity></row>" for r in ROWS) + "</rows>").encode()
    assert _cells(read_any(xml, "export.xml")) == EXPECTED


@pytest.mark.parametrize("ext, engine", [(".xlsx", "openpyxl"), (".ods", "odf")])
def test_spreadsheet_with_title_rows_and_empty_first_sheet(ext, engine):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=engine) as book:
        pd.DataFrame().to_excel(book, sheet_name="Cover", index=False)
        pd.DataFrame(ROWS).to_excel(book, sheet_name="Data", index=False, startrow=3)
    df = read_any(buf.getvalue(), "report" + ext)
    assert list(df.columns) == ["date", "item", "quantity"] and len(df) == 3
    assert any("title line" in n for n in df.attrs["import_notes"])
    assert df["quantity"].tolist() == [3, 5, 1]


def test_csv_title_lines_are_skipped():
    text = "Sales report 2024\nprinted on 01/02/2024\n\ndata;articolo;qta\n01/01/2024;A1;3\n02/01/2024;A1;5\n"
    df = read_any(text.encode("cp1252"), "report.csv")
    assert list(df.columns) == ["data", "articolo", "qta"] and len(df) == 2
    assert text_header_row("a,b,c\n1,2,3\n", ",") == 0


def test_header_row_of_a_grid():
    grid = pd.DataFrame([["Report", None, None], [None, None, None], ["date", "item", "qty"], ["2024-01-01", "A", 3]])
    assert grid_header_row(grid) == 2
    assert grid_header_row(pd.DataFrame([["date", "item"], ["2024-01-01", "A"]])) == 0


@pytest.mark.parametrize("label, expected", [("2024-03", "2024-03-01"), ("mar 24", "2024-03-01"), ("Mag-2023", "2023-05-01"),
                                             ("03/2024", "2024-03-01"), ("15/03/2024", "2024-03-15"),
                                             ("2024-03-15", "2024-03-15"), ("2024-03-15 00:00:00", "2024-03-15")])
def test_period_labels(label, expected):
    assert period_of(label) == pd.Timestamp(expected)


@pytest.mark.parametrize("label", ["item", "quantity", "2024", "Q1", "price", "store 12"])
def test_non_period_labels(label):
    assert period_of(label) is None


def test_wide_table_is_unpivoted_and_long_table_is_left_alone():
    wide = pd.DataFrame({"Articolo": ["A1", "B2"], "gen 24": [10, 0], "feb 24": [12, 3], "mar 24": [8, 4],
                         "apr 24": [9, None]})
    long = unpivot_wide(wide)
    assert set(long.columns) == {"Articolo", "date", "quantity"} and len(long) == 7
    assert long["date"].min() == pd.Timestamp("2024-01-01")
    plain = pd.DataFrame(ROWS)
    assert unpivot_wide(plain) is plain


def test_unsupported_and_empty_files():
    assert ".json" in SUPPORTED and ".md" in SUPPORTED and ".xls" in SUPPORTED
    with pytest.raises(InputError):
        read_any(b"%PDF-1.4", "report.pdf")
    with pytest.raises(InputError):
        read_any(b"   ", "empty.csv")
    with pytest.raises(InputError):
        read_any(b"{not json", "broken.json")


def test_granularity_detection():
    days = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))
    assert detect_granularity(days) == "daily"
    assert detect_granularity(days[days.dt.dayofweek < 5]) == "daily"          # closed at weekends
    assert detect_granularity(pd.Series(pd.date_range("2024-01-01", periods=12, freq="W-MON"))) == "weekly"
    assert detect_granularity(pd.Series(pd.date_range("2023-01-01", periods=12, freq="MS"))) == "monthly"
    assert detect_granularity(pd.Series(pd.date_range("2023-01-31", periods=12, freq="ME"))) == "monthly"


def test_period_totals_are_spread_and_kept(cfg):
    month = pd.DataFrame({"store": ["S"] * 2, "item": ["A"] * 2, "date": pd.to_datetime(["2024-02-01", "2024-03-15"]),
                          "sales": [29.0, 62.0]})
    daily = expand_periods(month, "monthly")
    assert len(daily) == 29 + 31 and daily["sales"].sum() == pytest.approx(91.0)
    assert daily["date"].min() == pd.Timestamp("2024-02-01") and daily["date"].max() == pd.Timestamp("2024-03-31")
    week = expand_periods(month.assign(date=pd.to_datetime(["2024-02-05", "2024-02-12"])), "weekly")
    assert len(week) == 14 and week["sales"].sum() == pytest.approx(91.0)

    raw = pd.DataFrame({"mese": ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"], "articolo": ["A"] * 6,
                        "qta": [310, 280, 310, 300, 310, 300]})
    data = standardise(raw, ColumnMapping.from_dict(guess_mapping(raw.columns) | {"date": "mese"}), cfg.input,
                       cfg.prices, cfg.inventory)
    assert data.notes["granularity"] == "monthly"
    assert data.sales["sales"].sum() == pytest.approx(1810.0)
    assert data.sales["date"].nunique() == 181 and data.promo is None
    assert any("spread evenly" in w for w in data.notes["warnings"])
