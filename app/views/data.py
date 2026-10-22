"""Data: switch between the demo and your own sales history (UC-12)."""

import hashlib
import io

import pandas as pd
import streamlit as st
from common import ROOT, ActiveData, demo_available
from i18n import model_name, translate_message, tx

from src.config import load_config
from src.ingest import (
    FIELDS,
    REQUIRED,
    ColumnMapping,
    InputError,
    guess_mapping,
    list_sheets,
    merge_item_master,
    read_any,
    standardise,
    validate,
)
from src.pipeline import run_in_memory
from src.profile import profile_history

st.title(tx("Data", "Dati"))

SAMPLE_DIR = ROOT / "data" / "samples"


def samples() -> dict:
    """file -> (label, description, item master, Excel sheet)."""
    return {
        "retail_chain_sales.csv": (
            tx("Retail chain · 8 stores (1 online) × 32 products × 3 years",
               "Catena di negozi · 8 negozi (1 online) × 32 prodotti × 3 anni"),
            tx("Promotions, holidays, seasonal products, a new store, a launch and a discontinued item",
               "Promozioni, festività, prodotti stagionali, un negozio nuovo, un lancio e un prodotto dismesso"),
            "retail_chain_items.csv", None),
        "single_store_it.csv": (
            tx("Neighbourhood shop · Italian CSV · 18 months", "Negozio di quartiere · CSV italiano · 18 mesi"),
            tx("`;` separator, dd/mm/yyyy dates, decimal commas, closed on Sundays and in August",
               "separatore `;`, date gg/mm/aaaa, virgole decimali, chiuso la domenica e ad agosto"),
            None, None),
        "ecommerce_order_lines.xlsx": (
            tx("Online shop · Excel order lines with returns · 2024",
               "Negozio online · righe d'ordine Excel con resi · 2024"),
            tx("One row per order line with timestamps; returns as negative quantities",
               "Una riga per riga d'ordine con data e ora; resi come quantità negative"),
            None, "Ordini"),
    }


LABELS = {
    "date": tx("Date", "Data"), "item": tx("Item / SKU", "Articolo / codice"),
    "quantity": tx("Quantity sold", "Quantità venduta"), "store": tx("Store (optional)", "Negozio (facoltativo)"),
    "unit_price": tx("Selling price (optional)", "Prezzo di vendita (facoltativo)"),
    "unit_cost": tx("Unit cost (optional)", "Costo unitario (facoltativo)"),
    "channel": tx("Channel physical/online (optional)", "Canale fisico/online (facoltativo)"),
    "lead_time_days": tx("Lead time in days (optional)", "Tempo di consegna in giorni (facoltativo)"),
    "promo": tx("Promotion: yes/no or discount % (optional)", "Promozione: sì/no o sconto % (facoltativo)"),
    "supplier": tx("Supplier (optional)", "Fornitore (facoltativo)"),
    "item_name": tx("Item description (optional)", "Descrizione articolo (facoltativo)"),
    "pack_size": tx("Pack size, units (optional)", "Pezzi per confezione (facoltativo)"),
    "min_order_qty": tx("Minimum order quantity (optional)", "Quantità minima d'ordine (facoltativo)"),
}
PROGRESS = {
    "Validating and profiling the data": "Controllo e profilo dei dati",
    "Running the SQL KPIs": "Calcolo dei KPI in SQL",
    "Classifying items (ABC / XYZ)": "Classificazione degli articoli (ABC / XYZ)",
    "Trade-off curves and sensitivity": "Curve di confronto e sensibilità",
    "Purchase plan: forecast from the last day of data": "Piano acquisti: previsione dall'ultimo giorno dei dati",
    "Done": "Fatto",
}

# ---- which dataset is active
active = st.session_state.get("active_source", "demo")
has_upload = st.session_state.get("upload_results") is not None
choices = {"demo": tx("Demo dataset (Kaggle)", "Dataset demo (Kaggle)"),
           "upload": tx("My sales history", "Il mio storico vendite")}
options = (["demo"] if demo_available() else []) + ["upload"]
current = "upload" if (active == "upload" and has_upload) or not demo_available() else "demo"
choice = st.segmented_control(tx("Dataset shown on every page", "Dataset mostrato in tutte le pagine"), options,
                              default=current, format_func=choices.get, key="data_choice")
if choice == "demo" and active != "demo":
    st.session_state["active_source"] = "demo"
    st.rerun()
if choice == "upload" and has_upload and active != "upload":
    st.session_state["active_source"] = "upload"
    st.rerun()

if choice != "upload":
    st.markdown(tx("""
The **demo** uses the Kaggle *Store Item Demand Forecasting Challenge*: 5 years of daily sales, 10 stores × 50 items.
Its results were computed offline with the full pipeline and are the ones reported in the README.

Switch to **My sales history** to run the same analysis on your own data or on a sample dataset.
""", """
La **demo** usa il dataset Kaggle *Store Item Demand Forecasting Challenge*: 5 anni di vendite giornaliere, 10 negozi ×
50 articoli. I risultati sono calcolati in anticipo con la pipeline completa e sono quelli riportati nel README.

Passa a **Il mio storico vendite** per fare la stessa analisi sui tuoi dati o su un dataset di esempio.
"""))
    st.stop()

st.info(tx("🔒 Your file stays in this session's memory: it is never saved to disk or sent anywhere else. This public "
           "demo runs on shared hosting, so for confidential data run the app on your own computer "
           "(`streamlit run app/streamlit_app.py`, see the README).",
           "🔒 Il tuo file resta nella memoria di questa sessione: non viene mai salvato su disco né inviato altrove. "
           "La demo pubblica gira su un server condiviso: per dati riservati usa l'app sul tuo computer "
           "(`streamlit run app/streamlit_app.py`, vedi il README)."), icon=":material/lock:")

template = (ROOT / "data" / "templates" / "sales_history_template.csv").read_bytes()
st.download_button(tx("Download a template (CSV)", "Scarica un modello (CSV)"), template,
                   "sales_history_template.csv", "text/csv", icon=":material/download:")
st.markdown(tx("""
**What the file needs:** one row per day (or per transaction) with a **date**, an **item** code and the
**quantity** sold. Optional: store, selling price, unit cost, channel (physical/online), lead time in days.
Column names can be anything: you map them below. Daily data, at least 8 weeks.
""", """
**Cosa serve nel file:** una riga per giorno (o per transazione) con una **data**, un **codice articolo** e la
**quantità** venduta. Facoltativi: negozio, prezzo di vendita, costo unitario, canale (fisico/online), tempo di
consegna in giorni. I nomi delle colonne possono essere qualsiasi: li associ qui sotto. Dati giornalieri, almeno
8 settimane.
"""))

cfg0 = load_config()
limits = cfg0.input.limits
SAMPLES = samples()
sources = {"upload": tx("Upload a file", "Carica un file"), "sample": tx("Use a sample dataset", "Usa un dataset di esempio")}
source = st.radio(tx("Source", "Origine"), list(sources), format_func=sources.get, horizontal=True, key="data_source",
                  help=tx("The sample datasets are invented (a fictional Italian retailer): use them to try every "
                          "feature without your own data.",
                          "I dataset di esempio sono inventati (un rivenditore italiano immaginario): usali per provare "
                          "tutte le funzioni senza dati tuoi."))
master_name, master_bytes, default_sheet = None, None, None
if source == "sample":
    available = [k for k in SAMPLES if (SAMPLE_DIR / k).exists()]
    if not available:
        st.warning(tx("No sample files found: create them with `python -m src.sample_data` (or `make sample-data`).",
                      "Nessun file di esempio trovato: creali con `python -m src.sample_data` (o `make sample-data`)."))
        st.stop()
    file_name = st.selectbox(tx("Sample dataset", "Dataset di esempio"), available,
                             format_func=lambda k: SAMPLES[k][0], key="sample_name")
    st.caption(SAMPLES[file_name][1] + tx(" · invented data, see `data/samples/README.md`.",
                                          " · dati inventati, vedi `data/samples/README.md`."))
    raw_bytes = (SAMPLE_DIR / file_name).read_bytes()
    default_sheet = SAMPLES[file_name][3]
    if SAMPLES[file_name][2] and st.checkbox(
            tx(f"Also load the item master `{SAMPLES[file_name][2]}` (prices, costs and lead times)",
               f"Carica anche l'anagrafica articoli `{SAMPLES[file_name][2]}` (prezzi, costi e tempi di consegna)"),
            value=True, key="sample_master"):
        master_name = SAMPLES[file_name][2]
        master_bytes = (SAMPLE_DIR / master_name).read_bytes()
else:
    upload = st.file_uploader(tx("Sales history (.csv, .xlsx or .parquet)", "Storico vendite (.csv, .xlsx o .parquet)"),
                              type=["csv", "txt", "tsv", "xlsx", "xlsm", "parquet"], key="up_file")
    if upload is None:
        if has_upload:
            fname = st.session_state["upload_results"].meta.get("file_name", "")
            st.success(tx(f"Showing results for **{fname}**. Upload another file to replace them.",
                          f"Stai vedendo i risultati di **{fname}**. Carica un altro file per sostituirli."))
        st.stop()
    file_name = upload.name
    raw_bytes = upload.getvalue()
if len(raw_bytes) > limits.max_file_mb * 1024 * 1024:
    st.error(tx(f"The file is larger than {limits.max_file_mb:.0f} MB. Run the analysis locally: "
                "`python -m src.pipeline --all --input your_file.csv` (see the README).",
                f"Il file supera {limits.max_file_mb:.0f} MB. Esegui l'analisi in locale: "
                "`python -m src.pipeline --all --input tuo_file.csv` (vedi il README)."))
    st.stop()
digest = hashlib.sha256(raw_bytes).hexdigest()

sheet = None
if file_name.lower().endswith((".xlsx", ".xlsm")):
    try:
        sheets = list_sheets(io.BytesIO(raw_bytes))
    except InputError as exc:
        st.error(translate_message(str(exc)))
        st.stop()
    default_index = sheets.index(default_sheet) if default_sheet in sheets else 0
    sheet = st.selectbox(tx("Sheet", "Foglio"), sheets, index=default_index) if len(sheets) > 1 else sheets[0]

try:
    raw = read_any(io.BytesIO(raw_bytes), filename=file_name, sheet=sheet)
except InputError as exc:
    st.error(translate_message(str(exc)))
    st.stop()
st.caption(tx(f"{len(raw):,} rows · {len(raw.columns)} columns · preview:",
              f"{len(raw):,} righe · {len(raw.columns)} colonne · anteprima:"))
st.dataframe(raw.head(8), hide_index=True)

st.subheader(tx("1 · Map your columns", "1 · Associa le colonne"))
guess = guess_mapping(raw.columns)
cols = [None, *raw.columns]
mapping = {}
grid = st.columns(4)
for k, fld in enumerate(FIELDS):
    default = cols.index(guess[fld]) if guess[fld] in cols else 0
    mapping[fld] = grid[k % 4].selectbox(LABELS[fld] + (" *" if fld in REQUIRED else ""), cols, index=default,
                                         key=f"map_{fld}_{digest[:8]}")
f1, f2 = st.columns(2)
date_formats = {"auto": tx("Detect automatically", "Riconosci automaticamente"), "ISO": "2024-03-15",
                "dayfirst": tx("15/03/2024 (day first)", "15/03/2024 (giorno/mese)"),
                "monthfirst": tx("03/15/2024 (month first)", "03/15/2024 (mese/giorno)")}
date_format = f1.selectbox(tx("Date format", "Formato data"), list(date_formats), format_func=date_formats.get)
negatives = {"clip": tx("Set to 0", "Porta a 0"), "reject": tx("Stop with an error", "Fermati con un errore")}
negative = f2.selectbox(tx("Negative quantities (returns)", "Quantità negative (resi)"), list(negatives),
                        format_func=negatives.get)
if source == "upload":
    master_file = st.file_uploader(tx("Optional item master (item, price, cost, lead time)",
                                      "Anagrafica articoli facoltativa (articolo, prezzo, costo, tempo di consegna)"),
                                   type=["csv", "xlsx", "parquet"], key="up_master")
    if master_file is not None:
        master_name, master_bytes = master_file.name, master_file.getvalue()

with st.expander(tx("2 · Cost assumptions (used where your data has no values)",
                    "2 · Ipotesi di costo (usate dove i tuoi dati non hanno valori)")):
    a1, a2, a3, a4, a5 = st.columns(5)
    lead = a1.number_input(tx("Lead time (days)", "Tempo di consegna (giorni)"), 1, 60, cfg0.inventory.lead_time_days)
    order_cost = a2.number_input(tx("Ordering cost (€/order)", "Costo per ordine (€)"), 1.0, 5000.0,
                                 cfg0.inventory.order_cost, 5.0)
    holding = a3.number_input(tx("Holding rate (%/year)", "Costo di mantenimento (%/anno)"), 1.0, 100.0,
                              cfg0.inventory.holding_rate * 100, 1.0)
    ratio = a4.number_input(tx("Unit cost (% of price)", "Costo unitario (% del prezzo)"), 5.0, 100.0,
                            cfg0.inventory.cost_to_price_ratio * 100, 5.0)
    target = a5.number_input(tx("Target fill rate (%)", "Fill rate obiettivo (%)"), 80.0, 99.5,
                             cfg0.simulation.target_fill_rate * 100, 0.5)
overrides = {
    "input": {"date_format": date_format, "negative_quantity": negative},
    "inventory": {"lead_time_days": int(lead), "order_cost": float(order_cost), "holding_rate": holding / 100,
                  "cost_to_price_ratio": ratio / 100},
    "simulation": {"target_fill_rate": target / 100},
}
cfg = load_config(overrides=overrides)

missing = [LABELS[f] for f in REQUIRED if not mapping.get(f)]
if missing:
    st.warning(tx("Map the required columns: ", "Associa le colonne obbligatorie: ") + ", ".join(missing))
    st.stop()

st.subheader(tx("3 · Check the data", "3 · Controllo dei dati"))
try:
    data = standardise(raw, ColumnMapping.from_dict(mapping), cfg.input, cfg.prices, cfg.inventory,
                       source="file", file_name=file_name, file_hash_value=digest, promotions=cfg.promotions)
    if master_bytes is not None:
        data = merge_item_master(data, read_any(io.BytesIO(master_bytes), filename=master_name),
                                 cfg.prices, cfg.inventory)
except InputError as exc:
    st.error(translate_message(str(exc)))
    st.stop()
report = validate(data, cfg.input, enforce_limits=True, prices=cfg.prices, inventory=cfg.inventory)
m1, m2, m3, m4 = st.columns(4)
m1.metric(tx("Period", "Periodo"), tx(f"{report.history_days:,} days", f"{report.history_days:,} giorni"),
          help=f"{report.date_min} → {report.date_max}")
m2.metric(tx("Stores · items", "Negozi · articoli"), f"{report.n_stores} · {report.n_items}")
m3.metric(tx("Series (store × item)", "Serie (negozio × articolo)"), f"{report.n_series:,}")
m4.metric(tx("Days with zero sales", "Giorni con vendite zero"), f"{report.pct_zero_days:.0%}")
st.caption(tx(f"From {report.date_min} to {report.date_max} · {report.n_rows_raw:,} rows in the file.",
              f"Dal {report.date_min} al {report.date_max} · {report.n_rows_raw:,} righe nel file."))
for e in report.errors:
    st.error(translate_message(e))
for w in report.warnings:
    st.warning(translate_message(w))
for a in report.assumptions:
    st.info(translate_message(a), icon=":material/info:")
if not report.ok:
    st.stop()
try:
    prof = profile_history(data.matrix(), cfg, fast=True)
except ValueError as exc:
    st.error(translate_message(str(exc)))
    st.stop()
st.markdown(tx("**How the history will be used**", "**Come verrà usato lo storico**"))
for note in prof.notes:
    st.markdown(f"- {translate_message(note)}")
st.caption(tx("Models: ", "Modelli: ") + ", ".join(model_name(m) for m in prof.models) + ".")

st.subheader(tx("4 · Run the analysis", "4 · Avvia l'analisi"))
if st.button(tx("Run analysis", "Avvia l'analisi"), type="primary", icon=":material/play_arrow:"):
    bar = st.progress(0.0, text=tx("Starting…", "Avvio…"))

    def _progress(msg: str, frac: float) -> None:
        if tx("en", "it") == "it":
            msg = PROGRESS.get(msg, msg.replace("Forecasting", "Previsioni").replace("Simulating", "Simulazione"))
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        res = run_in_memory(data, cfg, fast=True, progress=_progress)
    except (InputError, ValueError) as exc:
        st.error(tx("The analysis stopped: ", "L'analisi si è fermata: ") + translate_message(str(exc)))
        st.stop()
    st.session_state["upload_results"] = ActiveData(
        "upload", file_name, res.tables(), res.meta() | {"file_name": file_name})
    st.session_state["active_source"] = "upload"
    st.session_state.pop("scenario_key", None)
    bar.progress(1.0, text=tx("Done", "Fatto"))
    st.success(tx("Analysis complete: every page now shows your data.",
                  "Analisi completata: ora tutte le pagine mostrano i tuoi dati."))

if st.session_state.get("upload_results") is not None and st.session_state.get("active_source") == "upload":
    up = st.session_state["upload_results"]
    s = up.t("policy_summary")
    channel_of = up.t("stores").set_index("store")["channel"]
    plan = s[(s["variant"] == "calibrated") & (s["policy"] == "P2_forecast_sq")]
    plan = plan[plan["channel"] == plan["store"].map(channel_of)][
        ["store", "item", "channel", "lead_time", "mean_daily", "reorder_point", "order_qty", "safety_stock",
         "fill_rate", "avg_inventory_units", "total_cost"]]
    c1, c2 = st.columns(2)
    c1.download_button(tx("Reorder plan per series (CSV)", "Piano di riordino per serie (CSV)"),
                       plan.to_csv(index=False).encode(), "reorder_plan.csv", "text/csv", icon=":material/download:")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        plan.to_excel(xw, sheet_name="reorder_plan", index=False)
        up.t("classification").to_excel(xw, sheet_name="abc_xyz", index=False)
        s[s["variant"] == "calibrated"].to_excel(xw, sheet_name="policy_results", index=False)
    c2.download_button(tx("All results (Excel)", "Tutti i risultati (Excel)"), buf.getvalue(), "inventory_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:")
    if st.button(tx("Go to the overview", "Vai alla panoramica"), icon=":material/insights:"):
        st.switch_page("views/overview.py")
