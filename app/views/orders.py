"""Orders: what to order today and in the next 4 weeks, by supplier, on the current calendar."""

import io
from datetime import date
from pathlib import Path

import charts
import numpy as np
import pandas as pd
import streamlit as st
from common import (
    active_data,
    channel_label,
    config,
    csv_download,
    euro,
    no_data_message,
    plan_calendar_for,
    scenario_badge,
)
from i18n import is_it, tx

from src.constraints import Constraints, project_constrained, read_supplier_terms, terms_from_frame
from src.ingest import InputError, read_any
from src.replenishment import (
    PLAN_DAYS,
    apply_planned_promotions,
    extend_weekly,
    lead_time_sums,
    read_stock_file,
    risks_from_short,
)
from src.series import natural_sorted

NUM = st.column_config.NumberColumn
SAMPLE_TERMS = Path(__file__).resolve().parents[2] / "data" / "samples" / "supplier_terms.csv"
DATE = st.column_config.DateColumn
DAYS = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "it": ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]}
LONG_DAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
             "it": ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]}
LANG = "it" if is_it() else "en"


def day_label(d: pd.Timestamp, long: bool = False) -> str:
    return f"{(LONG_DAYS if long else DAYS)[LANG][d.dayofweek]} {d:%d/%m}" + (f"/{d:%Y}" if long else "")


data = active_data()
st.title(tx("Orders", "Ordini"))
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
cfg = config()
fc_tbl, state_tbl, pipe_tbl = data.t("plan_forecast"), data.t("plan_state"), data.t("plan_pipeline")
if fc_tbl.empty or state_tbl.empty:
    st.info(tx("The purchase plan needs the results of an analysis run with this version of the software: re-run the "
               "analysis on the Data page.",
               "Il piano acquisti richiede i risultati di un'analisi fatta con questa versione del software: rifai "
               "l'analisi nella pagina Dati."))
    st.stop()

top = st.container()          # headline and tiles, filled once the plan is computed
# --------------------------------------------------------------------------- calendar
meta = data.meta.get("plan") or {}
data_end = pd.Timestamp(meta["data_end"])
c1, c2 = st.columns([1, 2])
today_in = c1.date_input(tx("Today", "Oggi"), value=st.session_state.get("plan_today") or date.today(),
                         format="DD/MM/YYYY", key="ord_today")
st.session_state["plan_today"] = today_in
gap_real = (pd.Timestamp(today_in) - data_end).days - 1
if gap_real >= 7:
    bring = c2.toggle(tx("Bring the sales history to today", "Porta lo storico vendite a oggi"),
                      value=st.session_state.get("plan_bring", True), key="ord_bring",
                      help=tx("The data end in the past: every date is moved forward by whole weeks (weekdays stay "
                              "the same) so that the history ends just before today.",
                              "I dati finiscono nel passato: ogni data viene spostata avanti di settimane intere (i "
                              "giorni della settimana restano uguali) così lo storico finisce poco prima di oggi."))
    st.session_state["plan_bring"] = bring
cal = plan_calendar_for(data)
if cal.gap_days > PLAN_DAYS:
    st.warning(tx(f"The data end on {data_end:%d/%m/%Y}: without bringing them to today the plan starts on "
                  f"{cal.start:%d/%m/%Y}.",
                  f"I dati finiscono il {data_end:%d/%m/%Y}: senza portarli a oggi il piano parte dal "
                  f"{cal.start:%d/%m/%Y}."))
    cal = type(cal)(cal.data_end, cal.shift_days, cal.start, cal.start)
if cal.shift_days:
    shifted_end = data_end + pd.Timedelta(days=cal.shift_days)
    st.caption(tx(
        f":material/update: Your sales end on **{data_end:%d/%m/%Y}**. To plan on today's calendar every date is moved "
        f"forward by **{cal.shift_days // 7} weeks**: the last day of sales becomes {shifted_end:%d/%m/%Y} (same "
        f"weekday). Seasons move with it: the sales of {data_end:%B %Y} now play the role of the last weeks. "
        "Upload sales up to yesterday to plan on real data.",
        f":material/update: Le tue vendite finiscono il **{data_end:%d/%m/%Y}**. Per pianificare sul calendario di oggi "
        f"ogni data è spostata avanti di **{cal.shift_days // 7} settimane**: l'ultimo giorno di vendite diventa il "
        f"{shifted_end:%d/%m/%Y} (stesso giorno della settimana). Le stagioni si spostano con lui: le vendite di fine "
        "dati fanno da ultime settimane. Carica le vendite fino a ieri per pianificare su dati reali."))

# --------------------------------------------------------------------------- series of the scenario
modes = {"calibrated": tx("Minimum for the target", "Minima per l'obiettivo"),
         "fixed": tx("Cautious", "Prudente")}
target = data.target
mode = st.segmented_control(
    tx("Safety stock", "Scorta di sicurezza"), list(modes), format_func=modes.get, default="calibrated", key="ord_mode",
    help=tx(f"Minimum: the smallest safety stock that served {target:.0%} of demand in the test period (never below "
            "zero). Cautious: the textbook safety factor of the service target (more stock, fewer stock-outs).",
            f"Minima: la scorta di sicurezza più piccola che ha servito il {target:.0%} della domanda nel periodo di test "
            "(mai sotto zero). Prudente: il fattore di sicurezza da manuale del livello di servizio (più scorta, meno "
            "rotture).")) or "calibrated"
P2 = data.t("policy_summary")
P2 = P2[(P2["policy"] == "P2_forecast_sq") & (P2["variant"] == mode)]
keys = pd.DataFrame([(s, i) for s in natural_sorted(sc.stores) for i in natural_sorted(sc.items)], columns=["store", "item"])
keys["channel"] = keys["store"].map(sc.channel_of)
par = keys.merge(P2[["store", "item", "channel", "knob", "order_qty", "sigma_L", "lead_time", "unit_cost"]],
                 on=["store", "item", "channel"], how="inner")
if par.empty:
    st.info(tx("No series in this scenario.", "Nessuna serie in questo scenario."))
    st.stop()
master = data.t("item_master").set_index("item")
par["supplier"] = par["item"].map(master.get("supplier", pd.Series(dtype=str))).fillna("").replace(
    "", tx("Supplier not set", "Fornitore non indicato"))
par["name"] = par["item"].map(master.get("item_name", pd.Series(dtype=str))).fillna("")
n = len(par)
L = par["lead_time"].to_numpy(dtype=int)
Q = par["order_qty"].to_numpy(dtype="float64")
# a negative safety stock reaches the fill rate on average over a year, but in a day-by-day plan it would
# schedule a stock-out at the end of every cycle: the plan never goes below zero
ss = np.maximum(par["knob"] * par["sigma_L"], 0.0).to_numpy(dtype="float64")
cost = par["unit_cost"].to_numpy(dtype="float64")

# --------------------------------------------------------------------------- starting stock
st.subheader(tx("Stock today", "Giacenze di oggi"))
src_opts = {"estimated": tx("Estimated by the software", "Stimate dal software"),
            "file": tx("Upload my stock", "Carico le mie giacenze")}
stock_src = st.segmented_control(tx("Stock", "Giacenze"), list(src_opts), format_func=src_opts.get, default="estimated",
                                 key="ord_stock_src", label_visibility="collapsed") or "estimated"
st_ = par[["store", "item", "channel"]].merge(state_tbl, on=["store", "item", "channel"], how="left")
on_hand = st_["on_hand"].fillna(0).to_numpy(dtype="float64")
backlog = st_["backlog"].fillna(0).to_numpy(dtype="float64")
arrivals = np.zeros((n, int(L.max()) + 1))
pipe = par[["store", "item", "channel"]].reset_index().merge(pipe_tbl, on=["store", "item", "channel"])
np.add.at(arrivals, (pipe["index"].to_numpy(), pipe["day"].to_numpy(dtype=int)), pipe["qty"].to_numpy())
start = cal.start
if stock_src == "estimated":
    st.caption(tx(
        f"Stock and orders on the way that the forecast-driven policy (P2) would have at the end of the data "
        f"({start - pd.Timedelta(days=1):%d/%m/%Y} on this calendar). The days from then to yesterday are projected "
        "with the forecast. Replace them with your real stock for a real order list.",
        f"Giacenze e ordini in arrivo che la politica guidata dalla previsione (P2) avrebbe alla fine dei dati "
        f"({start - pd.Timedelta(days=1):%d/%m/%Y} su questo calendario). I giorni da allora a ieri sono proiettati "
        "con la previsione. Sostituiscile con le tue giacenze reali per una lista d'ordini reale."))
else:
    st.caption(tx("Columns: item, store, stock on hand; optional: quantity on order and its arrival date. "
                  "Italian or English names (giacenza, ordinato, data arrivo…).",
                  "Colonne: articolo, negozio, giacenza; facoltative: quantità ordinata e data di arrivo. "
                  "Nomi in italiano o inglese (giacenza, ordinato, data arrivo…)."))
    up = st.file_uploader(tx("Stock file", "File giacenze"), type=["csv", "xlsx"], key="ord_stock_file")
    if up is not None:
        try:
            stock = read_stock_file(read_any(io.BytesIO(up.getvalue()), filename=up.name),
                                    sc.stores[0] if len(data.stores) == 1 else None)
        except InputError as exc:
            st.error(tx("Could not read the stock file: ", "Impossibile leggere il file giacenze: ") + str(exc))
            st.stop()
        m = par[["store", "item"]].merge(stock, on=["store", "item"], how="left")
        found = m["on_hand"].notna().to_numpy()
        on_hand = np.where(found, m["on_hand"].fillna(0), 0.0)
        backlog = np.zeros(n)
        start = cal.today                               # real stock is today's: nothing to project before
        arrivals = np.zeros((n, int(L.max()) + 1))
        if "on_order" in m:
            qty = m["on_order"].fillna(0).to_numpy(dtype="float64")
            due = ((pd.to_datetime(m["arrival"]) - start).dt.days.fillna(-1)).to_numpy()
            due = np.where(due < 0, np.ceil(L / 2), due).astype(int).clip(0, arrivals.shape[1] - 1)
            np.add.at(arrivals, (np.arange(n), due), qty)
        st.caption(tx(f"{int(found.sum())} of {n} series found in the file; the others start from zero stock.",
                      f"{int(found.sum())} serie su {n} trovate nel file; le altre partono da giacenza zero."))
    else:
        st.info(tx("Upload a stock file, or switch back to the estimated stock.",
                   "Carica un file giacenze, oppure torna alle giacenze stimate."))
        st.stop()

with st.expander(tx("Check or correct the stock", "Controlla o correggi le giacenze")):
    edit = pd.DataFrame({"store": par["store"], "item": par["item"], "name": par["name"], "on_hand": np.round(on_hand)})
    edited = st.data_editor(edit, hide_index=True, width="stretch", disabled=["store", "item", "name"],
                            key=f"ord_edit_{stock_src}_{data.meta.get('file_name', '')}", column_config={
                                "store": tx("Store", "Negozio"), "item": tx("Item", "Articolo"),
                                "name": tx("Description", "Descrizione"),
                                "on_hand": NUM(tx(f"Stock on {start:%d/%m}", f"Giacenza al {start:%d/%m}"), min_value=0,
                                               step=1, format="%.0f")})
    on_hand = edited["on_hand"].fillna(0).to_numpy(dtype="float64")

# --------------------------------------------------------------------------- projection
k_today = int((cal.today - start).days)
D = k_today + PLAN_DAYS
fc = fc_tbl.pivot_table(index=["store", "item"], columns="h", values="forecast")
F = fc.reindex(pd.MultiIndex.from_frame(par[["store", "item"]])).fillna(0).to_numpy(dtype="float64")
skip = int((start - cal.start).days)            # uploaded stock starts today: drop the projected gap
F_ext = extend_weekly(F, skip + D + int(L.max()) + 9)[:, skip:]
dates_ext = pd.date_range(start, periods=F_ext.shape[1], freq="D")
plan_promos = st.session_state.get("pr_plan")
if plan_promos is not None and len(plan_promos):
    win = plan_promos[(pd.to_datetime(plan_promos["end"]) >= start)
                      & (pd.to_datetime(plan_promos["start"]) <= dates_ext[-1])]
    if len(win) and st.toggle(tx(f"Include the {len(win)} planned promotion(s) (Promotions page)",
                                 f"Includi le {len(win)} promozioni pianificate (pagina Promozioni)"),
                              value=True, key="ord_promos"):
        pm = data.meta.get("promotions") or {}
        items_tbl = data.t("promo_items")
        beta = items_tbl.set_index("item")["beta"].to_dict() if len(items_tbl) else {}
        dip = items_tbl.set_index("item")["post_dip"].to_dict() if len(items_tbl) else {}
        default_beta = float(pm.get("beta_global") or np.log(1.5) / 0.2)
        beta = {i: beta.get(i, default_beta) for i in par["item"].unique()}
        F_ext = apply_planned_promotions(F_ext, par, dates_ext, win, beta, dip, cfg.promotions.post_promo_days)

# --------------------------------------------------------------------------- supplier terms and limits
st.subheader(tx("Supplier terms and limits", "Vincoli fornitori e limiti"))
suppliers = natural_sorted(par["supplier"].unique())
day_opts = [str(d) for d in range(7)]
sample_terms = SAMPLE_TERMS.exists() and data.meta.get("file_name", "") == "retail_chain_sales.csv"


def _blank_terms() -> pd.DataFrame:
    return pd.DataFrame({"supplier": suppliers, "order_days": [day_opts] * len(suppliers),
                         "delivery_days": [day_opts] * len(suppliers), "min_order_value": 0.0,
                         "discount_qty": 0.0, "discount_pct": 0.0})


def _set_terms(df: pd.DataFrame) -> None:
    base = _blank_terms().set_index("supplier")
    df = df.assign(discount_pct=np.where(df["discount_pct"] <= 1, df["discount_pct"] * 100, df["discount_pct"]))
    df = df.set_index("supplier")
    common = base.index.intersection(df.index)
    for col in base.columns:
        base.loc[common, col] = df.loc[common, col]
    st.session_state["ord_terms"] = base.reset_index()
    st.session_state["ord_terms_ver"] = st.session_state.get("ord_terms_ver", 0) + 1


terms_key = f"{data.source}:{data.meta.get('file_name', '')}"
if st.session_state.get("ord_terms_key") != terms_key:
    st.session_state["ord_terms_key"] = terms_key
    st.session_state["ord_terms"] = _blank_terms()
    items_base = pd.DataFrame({"item": natural_sorted(par["item"].unique())})
    items_base["name"] = items_base["item"].map(master.get("item_name", pd.Series(dtype=str))).fillna("")
    items_base["supplier"] = items_base["item"].map(par.drop_duplicates("item").set_index("item")["supplier"])
    for col in ("pack_size", "min_order_qty"):
        items_base[col] = items_base["item"].map(master[col]) if col in master else np.nan
    st.session_state["ord_items"] = items_base
    st.session_state["ord_caps"] = pd.DataFrame({"store": natural_sorted(par["store"].unique()), "capacity": np.nan})
    st.session_state["plan_budget"] = 0.0
    if sample_terms:
        _set_terms(read_supplier_terms(read_any(SAMPLE_TERMS)))
terms_df = st.session_state["ord_terms"]
terms_df = terms_df[terms_df["supplier"].isin(suppliers)] if len(terms_df) else _blank_terms()
DAY_NAMES = DAYS[LANG]

with st.expander(tx("Order days, minimums, packs, discounts, budget, space",
                    "Giorni d'ordine, minimi, confezioni, sconti, budget, spazio"),
                 expanded=st.session_state.get("ord_cons_open", False)):
    t_sup, t_item, t_lim = st.tabs([tx("Suppliers", "Fornitori"), tx("Items", "Articoli"),
                                    tx("Budget and space", "Budget e spazio")])
    with t_sup:
        u1, u2 = st.columns([3, 1])
        up_terms = u1.file_uploader(tx("Load the terms from a file (optional)", "Carica le condizioni da file (facoltativo)"),
                                    type=["csv", "xlsx"], key="ord_terms_file",
                                    help=tx("Columns: supplier, order days, delivery days, minimum order €, discount "
                                            "from units, discount %. Italian or English names.",
                                            "Colonne: fornitore, giorni ordine, giorni consegna, ordine minimo €, sconto "
                                            "da pezzi, sconto %. Nomi in italiano o inglese."))
        if up_terms is not None and st.session_state.get("ord_terms_loaded") != (up_terms.name, up_terms.size):
            try:
                _set_terms(read_supplier_terms(read_any(io.BytesIO(up_terms.getvalue()), filename=up_terms.name)))
                st.session_state["ord_terms_loaded"] = (up_terms.name, up_terms.size)
                st.session_state["ord_cons_open"] = True
                st.rerun()
            except InputError as exc:
                st.error(tx("Could not read the terms: ", "Impossibile leggere le condizioni: ") + str(exc))
        if u2.button(tx("No constraints", "Nessun vincolo"), width="stretch", key="ord_terms_reset"):
            st.session_state["ord_terms"] = _blank_terms()
            st.session_state["ord_terms_ver"] = st.session_state.get("ord_terms_ver", 0) + 1
            st.session_state["ord_cons_open"] = True
            st.rerun()
        edited_terms = st.data_editor(
            terms_df, hide_index=True, width="stretch", num_rows="fixed", disabled=["supplier"],
            key=f"ord_terms_ed_{st.session_state.get('ord_terms_ver', 0)}", column_config={
                "supplier": tx("Supplier", "Fornitore"),
                "order_days": st.column_config.MultiselectColumn(
                    tx("Order days", "Giorni d'ordine"), options=day_opts, format_func=lambda d: DAY_NAMES[int(d)],
                    help=tx("Days on which the supplier accepts orders", "Giorni in cui il fornitore accetta ordini")),
                "delivery_days": st.column_config.MultiselectColumn(
                    tx("Delivery days", "Giorni di consegna"), options=day_opts, format_func=lambda d: DAY_NAMES[int(d)],
                    help=tx("Days on which it delivers: an order is placed only if order day + lead time is one of "
                            "them", "Giorni in cui consegna: si ordina solo se giorno d'ordine + tempo di consegna cade "
                            "in uno di questi")),
                "min_order_value": NUM(tx("Minimum order €", "Ordine minimo €"), min_value=0.0, step=10.0,
                                       format="%.0f", help=tx("Per store and order", "Per negozio e per ordine")),
                "discount_qty": NUM(tx("Discount from units", "Sconto da pezzi"), min_value=0.0, step=1.0,
                                    format="%.0f", help=tx("Units per order line from which the discount applies",
                                                           "Pezzi per riga d'ordine da cui vale lo sconto")),
                "discount_pct": NUM(tx("Discount %", "Sconto %"), min_value=0.0, max_value=90.0, step=0.5,
                                    format="%.1f%%")})
        st.session_state["ord_terms"] = edited_terms
        csv_download(edited_terms.assign(order_days=edited_terms["order_days"].map(
            lambda v: " ".join(DAY_NAMES[int(d)] for d in (v or []))),
            delivery_days=edited_terms["delivery_days"].map(lambda v: " ".join(DAY_NAMES[int(d)] for d in (v or [])))),
            tx("Download the terms (CSV)", "Scarica le condizioni (CSV)"), "supplier_terms.csv", "dl_terms")
    with t_item:
        edited_items = st.data_editor(
            st.session_state["ord_items"], hide_index=True, width="stretch", num_rows="fixed",
            disabled=["item", "name", "supplier"], key="ord_items_ed", column_config={
                "item": tx("Item", "Articolo"), "name": tx("Description", "Descrizione"),
                "supplier": tx("Supplier", "Fornitore"),
                "pack_size": NUM(tx("Units per pack", "Pezzi per confezione"), min_value=1, step=1, format="%.0f",
                                 help=tx("Orders are rounded up to whole packs", "Gli ordini sono arrotondati a confezioni intere")),
                "min_order_qty": NUM(tx("Minimum units per line", "Minimo pezzi per riga"), min_value=0, step=1,
                                     format="%.0f")})
        st.session_state["ord_items"] = edited_items
        st.caption(tx("Values read from the item master (columns pack size / minimum order) can be changed here.",
                      "I valori letti dall'anagrafica articoli (colonne confezione / minimo d'ordine) si possono "
                      "cambiare qui."))
    with t_lim:
        budget = st.number_input(tx("Weekly purchase budget, € (0 = no limit)",
                                    "Budget acquisti settimanale, € (0 = nessun limite)"), min_value=0.0, step=500.0,
                                 value=float(st.session_state.get("plan_budget", 0.0)), key="ord_budget")
        st.session_state["plan_budget"] = budget
        caps = st.session_state["ord_caps"].copy()
        caps["now"] = caps["store"].map(pd.Series(on_hand, index=par.index).groupby(par["store"]).sum()).round()
        edited_caps = st.data_editor(caps, hide_index=True, width="stretch", num_rows="fixed",
                                     disabled=["store", "now"], key="ord_caps_ed", column_config={
                                         "store": tx("Store", "Negozio"),
                                         "capacity": NUM(tx("Max units in stock", "Massimo pezzi a magazzino"),
                                                         min_value=0, step=100, format="%.0f",
                                                         help=tx("Stock + on order never above this; empty = no limit",
                                                                 "Giacenza + ordinato mai sopra questo valore; vuoto = "
                                                                 "nessun limite")),
                                         "now": NUM(tx("Units now", "Pezzi ora"), format="%.0f")})
        st.session_state["ord_caps"] = edited_caps[["store", "capacity"]]
        extra_days = st.slider(tx("Raise a line to the discount threshold if it is at most … days of sales away",
                                  "Porta una riga alla soglia di sconto se mancano al massimo … giorni di vendite"),
                               0, 21, 7, key="ord_break_days")

cons = Constraints(
    suppliers=terms_from_frame(edited_terms.assign(discount_pct=edited_terms["discount_pct"].fillna(0) / 100)),
    pack_size={r.item: r.pack_size for r in edited_items.itertuples() if pd.notna(r.pack_size) and r.pack_size >= 1},
    min_order_qty={r.item: r.min_order_qty for r in edited_items.itertuples()
                   if pd.notna(r.min_order_qty) and r.min_order_qty > 0},
    store_capacity={r.store: float(r.capacity) for r in edited_caps.itertuples()
                    if pd.notna(r.capacity) and r.capacity > 0},
    weekly_budget=budget or None, break_extra_days=float(extra_days))
active = [lbl for flag, lbl in (
    (any(set(t.order_days) != set(range(7)) or set(t.delivery_days) != set(range(7)) for t in cons.suppliers.values()),
     tx("order/delivery days", "giorni d'ordine/consegna")),
    (any(t.min_order_value > 0 for t in cons.suppliers.values()), tx("minimum orders", "ordini minimi")),
    (any(t.discount_qty > 0 for t in cons.suppliers.values()), tx("quantity discounts", "sconti quantità")),
    (bool(cons.pack_size) or bool(cons.min_order_qty), tx("packs and minimum quantities", "confezioni e minimi")),
    (bool(cons.weekly_budget), tx("weekly budget", "budget settimanale")),
    (bool(cons.store_capacity), tx("warehouse space", "spazio a magazzino"))) if flag]
st.caption((tx("Active: ", "Attivi: ") + ", ".join(active)) if active else
           tx("No constraint: orders on any day, in lots of any size.",
              "Nessun vincolo: ordini in qualsiasi giorno, a lotti di qualsiasi dimensione."))

# --------------------------------------------------------------------------- run
share = par["channel"].map(lambda c: cfg.channel(c).backorder_share).to_numpy(dtype="float64")
run_args = dict(F_ext=F_ext, days=D, dates=dates_ext, keys=par[["store", "item", "supplier"]], lead_time=L, Q=Q,
                safety_stock=ss, on_hand=on_hand, backlog=backlog, arrivals=arrivals, backorder_share=share,
                unit_cost=cost)
plan = project_constrained(**run_args, cons=cons)
free = project_constrained(**run_args) if cons.active else plan
daily = plan.daily
dates = dates_ext[:D]
orders = plan.orders.assign(store=par["store"].to_numpy()[plan.orders["row"]], item=par["item"].to_numpy()[plan.orders["row"]],
                            supplier=par["supplier"].to_numpy()[plan.orders["row"]],
                            name=par["name"].to_numpy()[plan.orders["row"]],
                            channel=par["channel"].to_numpy()[plan.orders["row"]])
risk = risks_from_short(daily["short"], par, dates)
NOTES = {"moq": tx("minimum quantity", "quantità minima"), "pack": tx("whole packs", "confezioni intere"),
         "discount_up": tx("raised to the discount", "portato alla soglia di sconto"),
         "discount": tx("discounted", "scontato"), "space": tx("cut for space", "ridotto per spazio"),
         "topup": tx("added to reach the minimum order", "aggiunto per l'ordine minimo")}
orders["notes_txt"] = orders["notes"].map(lambda v: ", ".join(NOTES.get(c, c) for c in v.split(",") if c))
k_today_ = int((cal.today - start).days)


def _summary(r) -> dict:
    o = r.orders[r.orders["date"] >= cal.today]
    stock = (r.daily["on_hand_end"][:, k_today_:] * np.nan_to_num(cost)[:, None]).sum(axis=0).mean()
    return {"lines": len(o), "purchases": o["value"].sum(), "stock": stock,
            "short": r.daily["short"][:, k_today_:].sum(), "saving": o["saving"].sum(),
            "deferred": int((r.deferred["date"] >= cal.today).sum()) if len(r.deferred) else 0}


if cons.active:
    a_, b_ = _summary(free), _summary(plan)
    labels = {"lines": tx("Order lines", "Righe d'ordine"), "purchases": tx("Purchases (€)", "Acquisti (€)"),
              "stock": tx("Average stock (€)", "Scorta media (€)"),
              "short": tx("Units not served", "Pezzi non serviti"),
              "saving": tx("Discounts obtained (€)", "Sconti ottenuti (€)"),
              "deferred": tx("Lines postponed", "Righe rinviate")}
    imp = pd.DataFrame({"what": list(labels.values()), "free": [a_[k] for k in labels], "cons": [b_[k] for k in labels]})
    imp["change"] = imp["cons"] - imp["free"]
    st.markdown(tx(f"**What the constraints change** over the next {PLAN_DAYS} days",
                   f"**Cosa cambiano i vincoli** nei prossimi {PLAN_DAYS} giorni"))
    st.dataframe(imp, hide_index=True, column_config={
        "what": "", "free": NUM(tx("Without constraints", "Senza vincoli"), format="%,.0f"),
        "cons": NUM(tx("With constraints", "Con i vincoli"), format="%,.0f"),
        "change": NUM(tx("Difference", "Differenza"), format="%+,.0f")})
    extra = plan.daily["short"][:, k_today_:].sum(axis=0) - free.daily["short"][:, k_today_:].sum(axis=0)
    if extra.sum() > 1:
        early = extra[:14].sum() / extra.sum()
        if early >= 0.8:
            st.caption(tx(
                f"{early:.0%} of the extra units not served fall in the first two weeks: suppliers that take orders "
                "only on some days cannot deliver earlier, and the stock of today was built without those limits. "
                "From then on the plan covers them; check the **Risks** tab for an extra order.",
                f"Il {early:.0%} dei pezzi non serviti in più cade nelle prime due settimane: i fornitori che accettano "
                "ordini solo in certi giorni non possono consegnare prima, e la giacenza di oggi è nata senza questi "
                "limiti. Da lì in poi il piano li copre; guarda la scheda **Rischi** per un ordine straordinario."))
        else:
            st.caption(tx("The constraints leave part of the demand uncovered for the whole plan: check the budget, the "
                          "warehouse space and the minimum orders (Risks and postponed tab).",
                          "I vincoli lasciano scoperta parte della domanda per tutto il piano: controlla budget, spazio a "
                          "magazzino e ordini minimi (scheda Rischi e rinvii)."))
today = cal.today
today_orders = orders[orders["date"] == today]
past_orders = orders[orders["date"] < today]
week = orders[(orders["date"] >= today) & (orders["date"] < today + pd.Timedelta(days=7))]
risk_soon = risk[(risk["first_day"] >= today) & (risk["first_day"] < today + pd.Timedelta(days=14))]
stock_value = float((daily["on_hand_end"][:, k_today - 1] if k_today > 0 else on_hand) @ np.nan_to_num(cost))

# --------------------------------------------------------------------------- headline
n_sup = today_orders["supplier"].nunique()
with top:
    if len(today_orders):
        st.markdown(tx(
            f"#### Today, {day_label(today, True)}: order **{len(today_orders)} item(s)** from **{n_sup} supplier(s)**, "
            f"for **{euro(today_orders['value'].sum())}** at cost.",
            f"#### Oggi, {day_label(today, True)}: ordina **{len(today_orders)} articol{'o' if len(today_orders) == 1 else 'i'}** "
            f"da **{n_sup} fornitor{'e' if n_sup == 1 else 'i'}**, per **{euro(today_orders['value'].sum())}** al costo."))
    else:
        nxt = orders[orders["date"] > today]["date"].min()
        st.markdown(tx(f"#### Today, {day_label(today, True)}: nothing to order."
                       + (f" Next order: {day_label(nxt)}." if pd.notna(nxt) else ""),
                       f"#### Oggi, {day_label(today, True)}: niente da ordinare."
                       + (f" Prossimo ordine: {day_label(nxt)}." if pd.notna(nxt) else "")))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tx("Lines today", "Righe oggi"), f"{len(today_orders)}")
    m2.metric(tx("Next 7 days", "Prossimi 7 giorni"), euro(week["value"].sum()),
              help=tx(f"{len(week)} lines, today included", f"{len(week)} righe, oggi compreso"))
    m3.metric(tx("Stock-out risks", "Rischi rottura"), f"{len(risk_soon)}",
              help=tx("Series whose expected stock runs out in the next 14 days even with the planned orders",
                      "Serie la cui scorta prevista finisce nei prossimi 14 giorni anche con gli ordini pianificati"))
    m4.metric(tx("Stock value", "Valore scorte"), euro(stock_value),
              help=tx("Stock on hand today, at unit cost", "Giacenza di oggi, al costo unitario"))
    if len(past_orders):
        st.info(tx(
            f"Between the end of the data and yesterday the plan assumes **{len(past_orders)} order(s)** were already "
            f"placed ({euro(past_orders['value'].sum())}): check them in the calendar.",
            f"Tra la fine dei dati e ieri il piano presume **{len(past_orders)} ordini** già fatti "
            f"({euro(past_orders['value'].sum())}): controllali nel calendario."))

COLS = {
    "store": tx("Store", "Negozio"), "item": tx("Item", "Articolo"), "name": tx("Description", "Descrizione"),
    "qty": NUM(tx("Quantity", "Quantità"), format="%.0f"), "unit_cost": NUM(tx("Unit cost", "Costo unit."), format="euro"),
    "value": NUM(tx("Value", "Valore"), format="euro"),
    "arrival": DATE(tx("Arrives", "Arriva"), format="DD/MM/YYYY"),
    "on_hand": NUM(tx("Stock after today", "Scorta a fine giornata"), format="%.0f"),
    "on_order": NUM(tx("Already on order", "Già in arrivo"), format="%.0f"),
    "cover_days": NUM(tx("Days of cover", "Giorni di copertura"), format="%.1f"),
    "supplier": tx("Supplier", "Fornitore"), "date": DATE(tx("Order on", "Ordina il"), format="DD/MM/YYYY"),
    "channel": tx("Channel", "Canale"),
    "notes_txt": tx("Adjusted for", "Modificato per"),
}
LINE_COLS = ["store", "item", "name", "qty", "notes_txt", "unit_cost", "value", "arrival", "on_hand", "on_order",
             "cover_days"]

tab_today, tab_cal, tab_risk, tab_item = st.tabs([tx("Today's orders", "Ordini di oggi"), tx("Calendar", "Calendario"),
                                                  tx("Risks and postponed", "Rischi e rinvii"),
                                                  tx("By item", "Per articolo")])
with tab_today:
    if today_orders.empty:
        st.info(tx("No order today: every item is above its reorder point.",
                   "Nessun ordine oggi: tutti gli articoli sono sopra il punto di riordino."))
    else:
        by_sup = today_orders.groupby("supplier").agg(lines=("qty", "size"), value=("value", "sum"),
                                                      arrival=("arrival", "min")).sort_values("value", ascending=False)
        for sup, g in by_sup.iterrows():
            lines = today_orders[today_orders["supplier"] == sup].sort_values(["store", "item"])
            terms = cons.suppliers.get(sup)
            minimum = ""
            if terms is not None and terms.min_order_value > 0:
                per_store = lines.groupby("store")["value"].sum()
                ok = (per_store >= terms.min_order_value - 1e-6).all()
                minimum = (" · " + tx("minimum", "minimo") + f" {euro(terms.min_order_value)} "
                           + (":green[✓]" if ok else ":orange[" + tx("urgent, below minimum", "urgente, sotto minimo") + "]"))
            with st.expander(f"**{sup}** · {int(g['lines'])} " + tx("lines", "righe") + f" · {euro(g['value'])} · "
                             + tx("arrives", "arriva") + f" {day_label(g['arrival'])}" + minimum,
                             expanded=len(by_sup) <= 3):
                st.dataframe(lines[LINE_COLS], hide_index=True, column_config=COLS)
                csv_download(lines[["supplier"] + LINE_COLS], tx("Download this order (CSV)", "Scarica questo ordine (CSV)"),
                             f"order_{today:%Y%m%d}_{sup[:20].replace(' ', '_')}.csv", f"dl_ord_{sup}")
        csv_download(today_orders[["supplier"] + LINE_COLS], tx("Download all of today's orders (CSV)",
                                                                "Scarica tutti gli ordini di oggi (CSV)"),
                     f"orders_{today:%Y%m%d}.csv", "dl_ord_all")
    st.caption(tx(
        "Rule (P2, calibrated to the target fill rate): order when stock on hand + on order falls to the forecast of "
        "the lead time + safety stock; the quantity is the economic lot, in multiples. Values at unit cost.",
        "Regola (P2, calibrata sul fill rate obiettivo): ordina quando giacenza + ordinato scende alla previsione del "
        "tempo di consegna + scorta di sicurezza; la quantità è il lotto economico, in multipli. Valori al costo "
        "unitario."))

with tab_cal:
    by_day = pd.DataFrame({"date": dates})
    ov = orders.groupby("date").agg(n_orders=("qty", "size"), order_value=("value", "sum"),
                                    suppliers=("supplier", "nunique"))
    dv = pd.DataFrame({"date": dates, "delivery_value": (daily["receipt"] * np.nan_to_num(cost)[:, None]).sum(axis=0),
                       "deliveries": (daily["receipt"] > 0).sum(axis=0)}).set_index("date")
    by_day = by_day.set_index("date").join(ov).join(dv).fillna(0).reset_index()
    view = by_day[by_day["date"] >= today - pd.Timedelta(days=today.dayofweek)]
    charts.show(charts.plan_order_values(view[view["date"] < today + pd.Timedelta(days=PLAN_DAYS)], today),
                key="ord_values")
    st.caption(tx("Click a day to see its orders and deliveries.", "Clicca un giorno per vedere ordini e consegne."))
    first = today - pd.Timedelta(days=today.dayofweek)
    head = st.columns(7)
    for j in range(7):
        head[j].markdown(f"**{DAYS[LANG][j]}**")
    sel = pd.Timestamp(st.session_state.get("ord_day") or today)
    info = by_day.set_index("date")
    for w in range(5):
        cols = st.columns(7, gap="small")
        for j in range(7):
            d = first + pd.Timedelta(days=7 * w + j)
            with cols[j].container(border=True):
                inside = d in info.index
                label = ("● " if d == sel else "") + (f"**{d:%d/%m}**" if d == today else f"{d:%d/%m}")
                if st.button(label, key=f"ord_d_{d:%Y%m%d}", disabled=not inside, width="stretch", type="tertiary",
                             help=tx("today", "oggi") if d == today else None):
                    st.session_state["ord_day"] = d
                    st.rerun()
                if not inside:
                    st.caption("&nbsp;\n\n&nbsp;")
                    continue
                r = info.loc[d]
                orders_txt = f":material/shopping_cart: {int(r['n_orders'])}" if r["n_orders"] else "&nbsp;"
                value_txt = euro(r["order_value"]) if r["n_orders"] else "&nbsp;"
                deliv_txt = f":material/local_shipping: {int(r['deliveries'])}" if r["deliveries"] else "&nbsp;"
                st.caption(f"{orders_txt}  \n{value_txt}  \n{deliv_txt}")
    st.markdown(f"##### {day_label(sel, True)}")
    o = orders[orders["date"] == sel]
    if sel < today and len(o):
        st.caption(tx("Day before today without data: orders the plan assumes were placed.",
                      "Giorno prima di oggi senza dati: ordini che il piano presume già fatti."))
    with st.container():
        st.markdown(tx(f"**Orders** · {len(o)} · {euro(o['value'].sum())}", f"**Ordini** · {len(o)} · {euro(o['value'].sum())}"))
        if len(o):
            st.dataframe(o.sort_values(["supplier", "store", "item"])[["supplier", "store", "item", "qty", "value", "arrival"]],
                         hide_index=True, column_config=COLS)
    with st.container():
        j = int((sel - start).days)
        rec_rows = np.flatnonzero(daily["receipt"][:, j] > 0) if 0 <= j < D else np.array([], dtype=int)
        st.markdown(tx(f"**Deliveries** · {len(rec_rows)}", f"**Consegne** · {len(rec_rows)}"))
        if len(rec_rows):
            st.dataframe(pd.DataFrame({"supplier": par["supplier"].to_numpy()[rec_rows],
                                       "store": par["store"].to_numpy()[rec_rows], "item": par["item"].to_numpy()[rec_rows],
                                       "qty": daily["receipt"][rec_rows, j]}), hide_index=True, column_config=COLS)
    csv_download(orders[orders["date"] >= today].sort_values(["date", "supplier", "store", "item"])[
        ["date", "supplier", "channel"] + LINE_COLS], tx("Download the 4-week order calendar (CSV)",
                                                         "Scarica il calendario ordini di 4 settimane (CSV)"),
        f"order_calendar_{today:%Y%m%d}.csv", "dl_ord_cal")

with tab_risk:
    rs = risk[risk["first_day"] >= today].copy()
    if rs.empty:
        st.success(tx("No expected stock-out in the next 4 weeks with the planned orders.",
                      "Nessuna rottura prevista nelle prossime 4 settimane con gli ordini pianificati."))
    else:
        nxt = []
        for r in rs.itertuples():
            rec = np.flatnonzero(daily["receipt"][r.row, k_today:] > 0)
            nxt.append(dates[k_today + rec[0]] if len(rec) else pd.NaT)
        rs["next_delivery"] = nxt
        rs["name"] = par["name"].to_numpy()[rs["row"]]
        rs["supplier"] = par["supplier"].to_numpy()[rs["row"]]
        rs["lead_time"] = L[rs["row"]]
        st.dataframe(rs.sort_values("first_day")[["first_day", "store", "item", "name", "supplier", "units_short",
                                                  "next_delivery", "lead_time"]], hide_index=True, column_config={
            **COLS, "first_day": DATE(tx("Runs out on", "Finisce il"), format="DD/MM/YYYY"),
            "units_short": NUM(tx("Units missing", "Pezzi mancanti"), format="%.0f"),
            "next_delivery": DATE(tx("Next delivery", "Prossima consegna"), format="DD/MM/YYYY"),
            "lead_time": tx("Lead time (d)", "Consegna (gg)")})
        st.caption(tx(
            "Expected stock-outs even with the orders above: usually the stock today is too low for the supplier's "
            "lead time. Options: an urgent order, a transfer from another store, or a faster supplier.",
            "Rotture previste anche con gli ordini qui sopra: di solito la giacenza di oggi è troppo bassa per il tempo "
            "di consegna del fornitore. Possibili azioni: un ordine urgente, un trasferimento da un altro negozio o un "
            "fornitore più rapido."))
    dfr = plan.deferred[plan.deferred["date"] >= today]
    if len(dfr):
        reasons = {"min_value": tx("below the supplier's minimum order", "sotto l'ordine minimo del fornitore"),
                   "budget": tx("weekly budget used up", "budget settimanale esaurito"),
                   "space": tx("no room in the warehouse", "nessuno spazio a magazzino")}
        g = dfr.groupby(["row", "reason"]).agg(first=("date", "min"), last=("date", "max"), days=("date", "nunique"),
                                               qty=("qty", "max")).reset_index()
        g = g.assign(store=par["store"].to_numpy()[g["row"]], item=par["item"].to_numpy()[g["row"]],
                     supplier=par["supplier"].to_numpy()[g["row"]], reason=g["reason"].map(reasons))
        st.markdown(tx(f"**Postponed lines** · {len(g)}", f"**Righe rinviate** · {len(g)}"))
        st.dataframe(g.sort_values("first")[["first", "last", "days", "store", "item", "supplier", "qty", "reason"]],
                     hide_index=True, column_config={
                         **COLS, "first": DATE(tx("First time", "Prima volta"), format="DD/MM/YYYY"),
                         "last": DATE(tx("Last time", "Ultima volta"), format="DD/MM/YYYY"),
                         "days": tx("Days", "Giorni"), "qty": NUM(tx("Units wanted", "Pezzi voluti"), format="%.0f"),
                         "reason": tx("Why", "Perché")})
        st.caption(tx("A postponed line is proposed again on the next possible day; if it becomes urgent (stock would "
                      "run out before the next delivery) it is ordered anyway.",
                      "Una riga rinviata viene riproposta il primo giorno possibile; se diventa urgente (la scorta "
                      "finirebbe prima della prossima consegna) viene ordinata comunque."))

with tab_item:
    i1, i2 = st.columns(2)
    store = i1.selectbox(tx("Store", "Negozio"), natural_sorted(par["store"].unique()), key="ord_store")
    items = par.loc[par["store"] == store, "item"]
    default = today_orders.loc[today_orders["store"] == store, "item"]
    item = i2.selectbox(tx("Item", "Articolo"), natural_sorted(items),
                        index=natural_sorted(items).index(default.iloc[0]) if len(default) else 0, key="ord_item",
                        format_func=lambda i: f"{i} · {par.loc[par['item'] == i, 'name'].iloc[0]}".rstrip(" ·"))
    row = int(np.flatnonzero(((par["store"] == store) & (par["item"] == item)).to_numpy())[0])
    rop = lead_time_sums(F_ext[[row]], L[[row]], D)[0] + ss[row]
    d = pd.DataFrame({"date": dates, "demand": daily["demand"][row], "on_hand": daily["on_hand_end"][row],
                      "order": daily["order_qty"][row], "receipt": daily["receipt"][row]})
    charts.show(charts.plan_projection(d, today, pd.Series(rop)), key="ord_proj")
    p = par.iloc[row]
    st.caption(tx(
        f"{channel_label(p['channel'])} store · supplier {p['supplier']} · lead time {int(p['lead_time'])} days · lot "
        f"{int(p['order_qty'])} units · safety stock {ss[row]:.0f} units · expected sales {F[row, :7].sum():.0f} units "
        "in the next 7 days.",
        f"Negozio {channel_label(p['channel'])} · fornitore {p['supplier']} · consegna in {int(p['lead_time'])} giorni · "
        f"lotto {int(p['order_qty'])} pezzi · scorta di sicurezza {ss[row]:.0f} pezzi · vendite previste "
        f"{F[row, :7].sum():.0f} pezzi nei prossimi 7 giorni."))
