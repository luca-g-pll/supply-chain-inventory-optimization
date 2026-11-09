"""Purchase history: real unit costs, suppliers, lead times and orders on their way.

A sales file says what was sold; the purchase file (supplier orders, delivery notes, invoices)
says what the company paid, to whom and how long the goods took to arrive. Read here, these
values replace the assumptions of ``config/params.yaml`` item by item:

* **unit cost** – median cost of the item's purchases in the last ``cost_window_days``;
* **supplier** – the supplier of the most recent purchase;
* **lead time** – median days between order and receipt (the item's own when it has enough
  receipts, otherwise its supplier's);
* **pack size** – the common multiple of the quantities bought, when there is one;
* **open orders** – recent orders with no receipt date: stock on its way for the purchase plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import InventoryParams, PricesConfig, PurchasesConfig
from src.ingest import FIELDS, SYNONYMS, StandardisedData, merge_item_master, normalise_name, parse_dates, parse_numbers
from src.readers import InputError

PURCHASE_COLUMNS: dict[str, set[str]] = {
    "order_date": {"order_date", "data_ordine", "data", "date", "data_documento", "data_acquisto", "purchase_date",
                   "po_date", "data_ordine_fornitore", "data_fattura", "invoice_date", "data_doc"},
    "receipt_date": {"receipt_date", "received", "received_date", "data_arrivo", "data_consegna", "data_ricevimento",
                     "data_carico", "data_ddt", "delivery_date", "arrival", "arrival_date", "data_entrata",
                     "data_ricezione", "ricevuto_il", "consegnato_il"},
    "item": set(SYNONYMS["item"]),
    "quantity": {"quantity", "qty", "qta", "quantita", "quantita_ordinata", "qta_ordinata", "ordered_qty",
                 "quantita_acquistata", "qta_acquistata", "purchased_qty", "pezzi", "pz", "units", "qty_ordered",
                 "quantita_ricevuta", "qta_ricevuta", "received_qty", "carico"},
    "unit_cost": {"unit_cost", "cost", "costo", "costo_unitario", "prezzo", "prezzo_unitario", "prezzo_acquisto",
                  "purchase_price", "price", "unit_price", "net_price", "prezzo_netto", "costo_acquisto", "buy_price"},
    "amount": {"amount", "total", "totale", "importo", "importo_riga", "line_total", "valore", "imponibile",
               "net_amount", "totale_riga"},
    "supplier": set(SYNONYMS["supplier"]),
    "store": set(SYNONYMS["store"]),
}
REQUIRED = ("order_date", "item", "quantity")
MAX_LEAD_TIME_DAYS = 365


@dataclass
class PurchaseSummary:
    items: pd.DataFrame          # [item, unit_cost, supplier, lead_time_days, pack_size, n_orders, last_order]
    open_orders: pd.DataFrame    # [store, item, on_order, arrival]; store is "" when the file has no store
    notes: list[str] = field(default_factory=list)
    n_rows: int = 0


def purchase_columns(columns) -> dict[str, str | None]:
    """Standard purchase field -> column of the file (EN / IT names), each column used once."""
    names = {str(c): normalise_name(c) for c in columns}
    out: dict[str, str | None] = {}
    for fld in ("receipt_date", "order_date", "item", "supplier", "store", "unit_cost", "amount", "quantity"):
        out[fld] = next((c for c, n in names.items() if n in PURCHASE_COLUMNS[fld] and c not in out.values()), None)
    return out


def read_purchases(raw: pd.DataFrame, cfg: PurchasesConfig | None = None, default_lead_time: int = 7) -> PurchaseSummary:
    """Summarise a purchase file. Raises ``InputError`` when date, item or quantity are missing."""
    cfg = cfg or PurchasesConfig()
    cols = purchase_columns(raw.columns)
    missing = [f for f in REQUIRED if cols[f] is None]
    if missing:
        raise InputError("Missing column(s): " + ", ".join(missing))
    ordered, _ = parse_dates(raw[cols["order_date"]])
    qty = parse_numbers(raw[cols["quantity"]])
    df = pd.DataFrame({
        "item": raw[cols["item"]].astype("string").str.strip(),
        "ordered": ordered,
        "received": parse_dates(raw[cols["receipt_date"]])[0] if cols["receipt_date"] else pd.NaT,
        "qty": qty,
        "cost": parse_numbers(raw[cols["unit_cost"]]) if cols["unit_cost"] else np.nan,
        "supplier": raw[cols["supplier"]].astype("string").str.strip() if cols["supplier"] else pd.NA,
        "store": raw[cols["store"]].astype("string").str.strip() if cols["store"] else "",
    })
    if cols["unit_cost"] is None and cols["amount"]:
        df["cost"] = parse_numbers(raw[cols["amount"]]) / qty.where(qty > 0)
    df = df[df["item"].notna() & (df["item"] != "") & df["ordered"].notna() & (df["qty"] > 0)]
    if df.empty:
        raise InputError("No usable rows after reading dates, items and quantities.")
    df = df.sort_values("ordered").reset_index(drop=True)
    df.loc[~(df["cost"] > 0), "cost"] = np.nan
    notes: list[str] = []

    last = df.groupby("item")["ordered"].transform("max")
    recent = df[df["ordered"] > last - pd.Timedelta(days=cfg.cost_window_days)]
    items = pd.DataFrame({"unit_cost": recent.groupby("item")["cost"].median()})
    items["unit_cost"] = items["unit_cost"].combine_first(df.dropna(subset=["cost"]).groupby("item")["cost"].last())
    items["supplier"] = df.dropna(subset=["supplier"]).groupby("item")["supplier"].last()
    items["n_orders"] = df.groupby("item").size()
    items["last_order"] = df.groupby("item")["ordered"].max()

    days = (df["received"] - df["ordered"]).dt.days
    done = df[days.between(0, MAX_LEAD_TIME_DAYS)].assign(days=days.clip(lower=1))
    own = done.groupby("item")["days"].agg(["median", "size"])
    by_supplier = done.dropna(subset=["supplier"]).groupby("supplier")["days"].median()
    lead = own["median"].where(own["size"] >= cfg.min_lead_time_orders).reindex(items.index)
    items["lead_time_days"] = lead.combine_first(items["supplier"].map(by_supplier)).round()
    if len(done):
        notes.append(f"Lead time measured on {len(done):,} received order line(s): "
                     f"median {done['days'].median():.0f} days.")

    whole = df[np.isclose(df["qty"], df["qty"].round())]
    pack = whole.groupby("item")["qty"].agg(lambda q: np.gcd.reduce(q.round().astype(int).to_numpy())
                                            if len(q) >= cfg.min_orders_for_pack else 1)
    items["pack_size"] = pack.where(pack >= 2).reindex(items.index)

    open_rows = df[df["received"].isna()] if cols["receipt_date"] and df["received"].notna().any() else df.iloc[:0]
    open_rows = open_rows[open_rows["ordered"] > df["ordered"].max() - pd.Timedelta(days=cfg.open_order_max_days)]
    lt = open_rows["item"].map(items["lead_time_days"]).fillna(default_lead_time)
    open_orders = (open_rows.assign(arrival=open_rows["ordered"] + pd.to_timedelta(lt, unit="D"))
                   .groupby(["store", "item"], as_index=False).agg(on_order=("qty", "sum"), arrival=("arrival", "min")))
    if len(open_orders):
        notes.append(f"{len(open_rows):,} order line(s) have no receipt date yet (open orders).")
    n_cost = int(items["unit_cost"].notna().sum())
    notes.insert(0, f"Purchases read: {len(df):,} lines, {len(items):,} item(s); real unit cost for {n_cost:,} of them.")
    return PurchaseSummary(items.reset_index(), open_orders.reset_index(drop=True), notes, int(len(df)))


def apply_purchases(data: StandardisedData, summary: PurchaseSummary, prices: PricesConfig,
                    inventory: InventoryParams) -> StandardisedData:
    """Replace assumed costs, suppliers, lead times and pack sizes with the ones of the purchase history."""
    frame = summary.items[["item", "unit_cost", "supplier", "lead_time_days", "pack_size"]].copy()
    frame["supplier"] = frame["supplier"].astype("string").fillna("")
    mapping = {f: (f if f in frame.columns else None) for f in FIELDS}
    out = merge_item_master(data, frame, prices, inventory, mapping=mapping, source="purchases")
    out.notes["assumptions"] = list(out.notes.get("assumptions", [])) + summary.notes
    return out
