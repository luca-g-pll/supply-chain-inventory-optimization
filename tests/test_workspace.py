"""Company folder: files recognised by their columns, joined, analysed and reopened; purchases; exports; launcher."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from src import workspace as ws
from src.exports import orders_workbook, sheet_name
from src.ingest import ColumnMapping, InputError, guess_mapping, standardise
from src.purchases import apply_purchases, purchase_columns, read_purchases
from src.replenishment import stock_health


def _sales(days: int = 120, items=("A1", "B2"), stores=("S1",), start="2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    rng = np.random.default_rng(1)
    rows = [{"date": d.strftime("%Y-%m-%d"), "store": s, "item": i, "quantity": int(rng.poisson(6)), "unit_price": 4.0}
            for d in dates for s in stores for i in items]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- purchases
def _purchases() -> pd.DataFrame:
    return pd.DataFrame({
        "Data ordine": ["01/02/2024", "15/02/2024", "01/03/2024", "20/03/2024", "02/02/2024", "25/03/2024"],
        "Data arrivo": ["06/02/2024", "19/02/2024", "07/03/2024", "", "12/02/2024", ""],
        "Codice": ["A1", "A1", "A1", "A1", "B2", "B2"],
        "Fornitore": ["Rossi srl"] * 4 + ["Bianchi spa"] * 2,
        "Quantità": [24, 48, 36, 24, 10, 15],
        "Costo unitario": ["2,50", "2,50", "2,70", "2,70", "1,20", "1,30"]})


def test_purchase_columns_are_recognised():
    cols = purchase_columns(_purchases().columns)
    assert cols == {"receipt_date": "Data arrivo", "order_date": "Data ordine", "item": "Codice",
                    "supplier": "Fornitore", "store": None, "unit_cost": "Costo unitario", "amount": None,
                    "quantity": "Quantità"}
    with pytest.raises(InputError):
        read_purchases(pd.DataFrame({"x": [1]}))


def test_purchases_give_cost_supplier_lead_time_pack_and_open_orders():
    summary = read_purchases(_purchases(), default_lead_time=7)
    items = summary.items.set_index("item")
    assert items.loc["A1", "unit_cost"] == pytest.approx(2.6)            # median of the last 90 days
    assert items.loc["A1", "supplier"] == "Rossi srl"
    assert items.loc["A1", "lead_time_days"] == 5                        # median of 5, 4, 6 days
    assert items.loc["A1", "pack_size"] == 12                            # 24, 48, 36, 24
    assert items.loc["B2", "lead_time_days"] == 10 or pd.isna(items.loc["B2", "lead_time_days"])
    assert pd.isna(items.loc["B2", "pack_size"])                         # only two purchases: no pack is guessed
    open_orders = summary.open_orders.set_index("item")
    assert open_orders.loc["A1", "on_order"] == 24
    assert open_orders.loc["A1", "arrival"] == pd.Timestamp("2024-03-25")  # order date + the item's lead time


def test_unit_cost_from_line_amount():
    raw = pd.DataFrame({"data": ["2024-01-05", "2024-01-20"], "articolo": ["A1", "A1"], "qta": [10, 20],
                        "importo": [30.0, 60.0]})
    assert read_purchases(raw).items.loc[0, "unit_cost"] == pytest.approx(3.0)


def test_purchases_replace_the_assumed_cost(cfg):
    raw = _sales()
    data = standardise(raw, ColumnMapping.from_dict(guess_mapping(raw.columns)), cfg.input, cfg.prices, cfg.inventory)
    assert (data.item_master["cost_source"] == "derived").all()
    out = apply_purchases(data, read_purchases(_purchases()), cfg.prices, cfg.inventory)
    master = out.item_master.set_index("item")
    assert master.loc["A1", "unit_cost"] == pytest.approx(2.6) and master.loc["A1", "cost_source"] == "purchases"
    assert master.loc["A1", "lead_time_days"] == 5 and master.loc["A1", "supplier"] == "Rossi srl"
    assert master.loc["A1", "pack_size"] == 12
    assert any("Purchases read" in a for a in out.notes["assumptions"])


# --------------------------------------------------------------------------- recognising files
@pytest.mark.parametrize("columns, name, role", [
    (["Data", "Articolo", "Qta", "Negozio", "Prezzo"], "export.xlsx", "sales"),
    (["Data ordine", "Data arrivo", "Codice", "Fornitore", "Quantità", "Costo unitario"], "export.csv", "purchases"),
    (["data", "articolo", "quantita", "fornitore", "costo"], "acquisti_2024.csv", "purchases"),
    (["data", "articolo", "quantita", "prezzo"], "vendite.csv", "sales"),
    (["articolo", "negozio", "giacenza"], "x.csv", "stock"),
    (["item", "unit_price", "unit_cost", "supplier", "lead_time_days"], "x.csv", "item_master"),
    (["fornitore", "giorni_ordine", "ordine_minimo"], "x.csv", "supplier_terms"),
    (["articolo", "inizio", "fine", "sconto"], "x.csv", "promo_calendar"),
    (["foo", "bar"], "x.csv", "unknown"),
])
def test_files_are_recognised_by_columns_and_name(columns, name, role):
    assert ws.classify(columns, name) == role


@pytest.fixture
def company(tmp_path):
    """A company folder with two sales files in different formats, purchases, stock and a file that is not data."""
    folder = ws.ensure(tmp_path / "Company")
    sales = _sales(days=150, stores=("S1", "S2"))
    first, second = sales[sales["date"] < "2024-03-01"], sales[sales["date"] >= "2024-03-01"]
    (folder / "data" / "vendite_gen_feb.json").write_text(json.dumps({"rows": first.to_dict("records")}))
    italian = second.rename(columns={"date": "Data", "store": "Negozio", "item": "Articolo", "quantity": "Qta",
                                     "unit_price": "Prezzo"})
    italian["Data"] = pd.to_datetime(italian["Data"]).dt.strftime("%d/%m/%Y")
    italian.to_csv(folder / "data" / "Vendite da marzo.csv", sep=";", index=False, decimal=",")
    _purchases().to_csv(folder / "data" / "acquisti.csv", sep=";", index=False)
    pd.DataFrame({"articolo": ["A1", "B2", "A1", "B2"], "negozio": ["S1", "S1", "S2", "S2"],
                  "giacenza": [40, 5, 400, 0]}).to_csv(folder / "data" / "giacenze.csv", index=False)
    (folder / "data" / "notes.csv").write_text("foo,bar\n1,2\n")
    (folder / "data" / "~$lock.xlsx").write_text("temporary office file")
    return folder


def test_folder_structure_and_settings(tmp_path):
    folder = ws.ensure(tmp_path / "New")
    assert (folder / "data").is_dir() and (folder / "results").is_dir() and (folder / ws.SETTINGS).exists()
    assert ws.load_settings(folder)["overrides"] == {}
    ws.save_settings(folder, {"inventory": {"order_cost": 80.0, "holding_rate": 0.3, "hacked": 1},
                              "paths": {"raw_train": "/etc/passwd"}}, {"name": "Rossi & figli"})
    settings = ws.load_settings(folder)
    assert settings["overrides"] == {"inventory": {"order_cost": 80.0, "holding_rate": 0.3}}   # only known keys
    assert settings["company"]["name"] == "Rossi & figli"
    cfg = ws.company_config(folder)
    assert cfg.inventory.order_cost == 80.0 and cfg.inventory.holding_rate == 0.3
    (folder / ws.SETTINGS).write_text("inventory: [broken")
    with pytest.raises(InputError):
        ws.load_settings(folder)


def test_scan_and_fingerprint(company):
    files = {f.name: f for f in ws.scan(company)}
    assert {f.role for f in files.values()} == {"sales", "purchases", "stock", "unknown"}
    assert "~$lock.xlsx" not in files and files["notes.csv"].message.startswith("Columns not recognised")
    before = ws.fingerprint(company)
    assert before == ws.fingerprint(company)
    (company / "data" / "extra.csv").write_text("date,item,quantity\n2024-01-01,A1,1\n")
    assert ws.fingerprint(company) != before


def test_company_files_are_joined(company, cfg):
    loaded = ws.load_company(company, cfg)
    data = loaded.data
    assert data.n_series == 4 and data.date_min == pd.Timestamp("2024-01-01")
    assert data.date_max == pd.Timestamp("2024-05-29")                      # both files, both date formats
    assert data.sales["sales"].sum() == _sales(days=150, stores=("S1", "S2"))["quantity"].sum()
    assert any("2 sales files were joined" in n for n in loaded.notes)
    assert data.item_master.set_index("item").loc["A1", "cost_source"] == "purchases"
    assert loaded.stock is not None and len(loaded.stock) == 4
    assert any("no store" in n for n in loaded.notes)                       # open orders cannot be placed in a store
    assert set(loaded.tables()) == {"ws_files", "ws_stock"}


def test_open_orders_are_added_for_a_single_store():
    stock = pd.DataFrame({"store": ["S1", "S1"], "item": ["A1", "B2"], "on_hand": [5.0, 9.0], "on_order": 0.0,
                          "arrival": pd.NaT})
    orders = pd.DataFrame({"store": ["", ""], "item": ["A1", "Z9"], "on_order": [24.0, 3.0],
                           "arrival": pd.to_datetime(["2024-03-25", "2024-03-26"])})
    merged = ws._with_open_orders(stock, orders, "S1", [])
    assert merged.set_index("item")["on_order"].to_dict() == {"A1": 24.0, "B2": 0.0}
    already = stock.assign(on_order=[3.0, 0.0])
    assert ws._with_open_orders(already, orders, "S1", []) is already        # the stock file already says it


def test_no_sales_file_is_an_error(tmp_path, cfg):
    folder = ws.ensure(tmp_path / "Empty")
    assert ws.scan(folder) == []
    with pytest.raises(InputError):
        ws.load_company(folder, cfg)


def test_analysis_is_saved_and_reopened(company):
    seen = []
    tables, meta, loaded = ws.analyse(company, progress=lambda message, share: seen.append(share))
    assert seen and meta["workspace"]["fingerprint"] == ws.fingerprint(company)
    reopened = ws.load_results(company)
    assert reopened is not None
    saved_tables, saved_meta = reopened
    assert set(saved_tables) == set(tables) and saved_meta["best_model"] == meta["best_model"]
    pd.testing.assert_frame_equal(saved_tables["policy_summary"], tables["policy_summary"])
    plan = ws.reorder_parameters(saved_tables)
    assert len(plan) == 4 and (plan["order_qty"] > 0).all()
    assert ws.main(["--folder", str(company), "--status"]) == 0
    assert ws.load_results(company.parent / "nowhere") is None


def test_workspace_env_variable(monkeypatch, tmp_path):
    monkeypatch.delenv(ws.ENV_VAR, raising=False)
    assert ws.workspace_dir() is None
    monkeypatch.setenv(ws.ENV_VAR, str(tmp_path))
    assert ws.workspace_dir() == tmp_path


# --------------------------------------------------------------------------- exports and stock health
def test_orders_workbook_has_a_sheet_per_supplier():
    orders = pd.DataFrame({"supplier": ["Rossi: srl/1", "Rossi: srl/1", "Bianchi"], "store": ["S1", "S2", "S1"],
                           "item": ["A1", "A1", "B2"], "qty": [24.0, 12.0, 10.0], "value": [60.0, 30.0, 12.0]})
    book = pd.read_excel(io.BytesIO(orders_workbook(orders, {"qty": "Quantità", "units": "Pezzi"}, "Riepilogo")), sheet_name=None)
    assert list(book) == ["Riepilogo", "Rossi  srl 1", "Bianchi"]
    assert book["Riepilogo"]["Pezzi"].tolist() == [36, 10] and len(book["Rossi  srl 1"]) == 2
    assert "Quantità" in book["Bianchi"].columns and "supplier" not in book["Bianchi"].columns
    used: set[str] = set()
    names = [sheet_name("x" * 40, used), sheet_name("x" * 40, used), sheet_name("", used)]
    assert len(set(names)) == 3 and all(len(n) <= 31 for n in names)


def test_stock_health_flags_excess_and_dead_stock():
    keys = pd.DataFrame({"store": ["S"] * 4, "item": ["ok", "excess", "dead", "empty"]})
    forecast = np.vstack([np.full(60, 2.0), np.full(60, 2.0), np.zeros(60), np.zeros(60)])
    out = stock_health(keys, np.array([100.0, 500.0, 30.0, 0.0]), forecast, np.array([1.0, 2.0, 3.0, 4.0]),
                       excess_cover_days=90, horizon_days=28).set_index("item")
    assert list(out.index) == ["excess", "dead"]                             # sorted by value, healthy series left out
    assert out.loc["excess", "excess_units"] == 320 and out.loc["excess", "excess_value"] == 640
    assert out.loc["excess", "cover_days"] == 250
    assert out.loc["dead", "status"] == "dead" and out.loc["dead", "excess_units"] == 30


# --------------------------------------------------------------------------- launcher
def test_launcher_helpers(tmp_path, monkeypatch):
    import launcher

    monkeypatch.delenv(launcher.ENV_VAR, raising=False)
    folder = launcher.company_folder(str(tmp_path / "Co"))
    assert (folder / "data").is_dir() and (folder / "results").is_dir()
    port = launcher.free_port(49500)
    assert 49500 <= port < 49500 + launcher.PORT_TRIES and not launcher.is_running(port)
    assert len(launcher.requirements_hash()) == 64
    assert launcher.last_port(folder) is None
    launcher.remember_port(folder, port)
    assert launcher.last_port(folder) == port
    assert launcher.ENV_VAR == ws.ENV_VAR and launcher.FOLDER_NAME == ws.load_config().workspace.folder_name
