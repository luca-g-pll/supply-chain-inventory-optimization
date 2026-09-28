"""Data: switch between the demo and your own sales history (UC-12)."""

import hashlib
import io

import pandas as pd
import streamlit as st
from common import ROOT, ActiveData, demo_available

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

st.title("Data")

LABELS = {"date": "Date", "item": "Item / SKU", "quantity": "Quantity sold", "store": "Store (optional)",
          "unit_price": "Selling price (optional)", "unit_cost": "Unit cost (optional)",
          "channel": "Channel physical/online (optional)", "lead_time_days": "Lead time in days (optional)"}

# ---- which dataset is active
active = st.session_state.get("active_source", "demo")
has_upload = st.session_state.get("upload_results") is not None
options = (["Demo dataset (Kaggle)"] if demo_available() else []) + ["My sales history"]
current = "My sales history" if (active == "upload" and has_upload) or not demo_available() else "Demo dataset (Kaggle)"
choice = st.segmented_control("Dataset shown on every page", options, default=current, key="data_choice")
if choice == "Demo dataset (Kaggle)" and active != "demo":
    st.session_state["active_source"] = "demo"
    st.rerun()
if choice == "My sales history" and has_upload and active != "upload":
    st.session_state["active_source"] = "upload"
    st.rerun()

if choice != "My sales history":
    st.markdown("""
The **demo** uses the Kaggle *Store Item Demand Forecasting Challenge*: 5 years of daily sales, 10 stores × 50 items.
Its results were computed offline with the full pipeline and are the ones reported in the README.

Switch to **My sales history** to run the same analysis on your own data.
""")
    st.stop()

st.info("🔒 Your file stays in this session's memory: it is never saved to disk or sent anywhere else. "
        "This public demo runs on shared hosting, so for confidential data run the app on your own computer "
        "(`streamlit run app/streamlit_app.py`, see the README).", icon=":material/lock:")

template = (ROOT / "data" / "templates" / "sales_history_template.csv").read_bytes()
st.download_button("Download a template (CSV)", template, "sales_history_template.csv", "text/csv",
                   icon=":material/download:")
st.markdown("""
**What the file needs:** one row per day (or per transaction) with a **date**, an **item** code and the
**quantity** sold. Optional: store, selling price, unit cost, channel (physical/online), lead time in days.
Column names can be anything: you map them below. Daily data, at least 8 weeks.
""")

cfg0 = load_config()
limits = cfg0.input.limits
upload = st.file_uploader("Sales history (.csv, .xlsx or .parquet)", type=["csv", "txt", "tsv", "xlsx", "xlsm", "parquet"],
                          key="up_file")
if upload is None:
    if has_upload:
        st.success(f"Showing results for **{st.session_state['upload_results'].meta.get('file_name', 'your file')}**. "
                   "Upload another file to replace them.")
    st.stop()

raw_bytes = upload.getvalue()
if len(raw_bytes) > limits.max_file_mb * 1024 * 1024:
    st.error(f"The file is larger than {limits.max_file_mb:.0f} MB. Run the analysis locally: "
             "`python -m src.pipeline --all --input your_file.csv` (see the README).")
    st.stop()
digest = hashlib.sha256(raw_bytes).hexdigest()

sheet = None
if upload.name.lower().endswith((".xlsx", ".xlsm")):
    try:
        sheets = list_sheets(io.BytesIO(raw_bytes))
    except InputError as exc:
        st.error(str(exc))
        st.stop()
    sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else sheets[0]

try:
    raw = read_any(io.BytesIO(raw_bytes), filename=upload.name, sheet=sheet)
except InputError as exc:
    st.error(str(exc))
    st.stop()
st.caption(f"{len(raw):,} rows · {len(raw.columns)} columns · preview:")
st.dataframe(raw.head(8), hide_index=True)

st.subheader("1 · Map your columns")
guess = guess_mapping(raw.columns)
cols = [None, *raw.columns]
mapping = {}
grid = st.columns(4)
for k, fld in enumerate(FIELDS):
    default = cols.index(guess[fld]) if guess[fld] in cols else 0
    mapping[fld] = grid[k % 4].selectbox(LABELS[fld] + (" *" if fld in REQUIRED else ""), cols, index=default,
                                         key=f"map_{fld}_{digest[:8]}")
f1, f2 = st.columns(2)
date_format = f1.selectbox("Date format", ["auto", "ISO", "dayfirst", "monthfirst"],
                           format_func=lambda x: {"auto": "Detect automatically", "ISO": "2024-03-15",
                                                  "dayfirst": "15/03/2024 (day first)",
                                                  "monthfirst": "03/15/2024 (month first)"}[x])
negative = f2.selectbox("Negative quantities (returns)", ["clip", "reject"],
                        format_func=lambda x: {"clip": "Set to 0", "reject": "Stop with an error"}[x])
master_file = st.file_uploader("Optional item master (item, price, cost, lead time)", type=["csv", "xlsx", "parquet"],
                               key="up_master")

with st.expander("2 · Cost assumptions (used where your data has no values)"):
    a1, a2, a3, a4, a5 = st.columns(5)
    lead = a1.number_input("Lead time (days)", 1, 60, cfg0.inventory.lead_time_days)
    order_cost = a2.number_input("Ordering cost (€/order)", 1.0, 5000.0, cfg0.inventory.order_cost, 5.0)
    holding = a3.number_input("Holding rate (%/year)", 1.0, 100.0, cfg0.inventory.holding_rate * 100, 1.0)
    ratio = a4.number_input("Unit cost (% of price)", 5.0, 100.0, cfg0.inventory.cost_to_price_ratio * 100, 5.0)
    target = a5.number_input("Target fill rate (%)", 80.0, 99.5, cfg0.simulation.target_fill_rate * 100, 0.5)
overrides = {
    "input": {"date_format": date_format, "negative_quantity": negative},
    "inventory": {"lead_time_days": int(lead), "order_cost": float(order_cost), "holding_rate": holding / 100,
                  "cost_to_price_ratio": ratio / 100},
    "simulation": {"target_fill_rate": target / 100},
}
cfg = load_config(overrides=overrides)

missing = [LABELS[f] for f in REQUIRED if not mapping.get(f)]
if missing:
    st.warning("Map the required columns: " + ", ".join(missing))
    st.stop()

st.subheader("3 · Check the data")
try:
    data = standardise(raw, ColumnMapping.from_dict(mapping), cfg.input, cfg.prices, cfg.inventory,
                       source="file", file_name=upload.name, file_hash_value=digest)
    if master_file is not None:
        data = merge_item_master(data, read_any(io.BytesIO(master_file.getvalue()), filename=master_file.name),
                                 cfg.prices, cfg.inventory)
except InputError as exc:
    st.error(str(exc))
    st.stop()
report = validate(data, cfg.input, enforce_limits=True, prices=cfg.prices, inventory=cfg.inventory)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Period", f"{report.history_days:,} days", help=f"{report.date_min} → {report.date_max}")
m2.metric("Stores · items", f"{report.n_stores} · {report.n_items}")
m3.metric("Series (store × item)", f"{report.n_series:,}")
m4.metric("Days with zero sales", f"{report.pct_zero_days:.0%}")
st.caption(f"From {report.date_min} to {report.date_max} · {report.n_rows_raw:,} rows in the file.")
for e in report.errors:
    st.error(e)
for w in report.warnings:
    st.warning(w)
for a in report.assumptions:
    st.info(a, icon=":material/info:")
if not report.ok:
    st.stop()
try:
    prof = profile_history(data.matrix(), cfg, fast=True)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
st.markdown("**How the history will be used**")
for note in prof.notes:
    st.markdown(f"- {note}")
st.caption(f"Models: {', '.join(prof.models)}.")

st.subheader("4 · Run the analysis")
if st.button("Run analysis", type="primary", icon=":material/play_arrow:"):
    bar = st.progress(0.0, text="Starting…")

    def _progress(msg: str, frac: float) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        res = run_in_memory(data, cfg, fast=True, progress=_progress)
    except (InputError, ValueError) as exc:
        st.error(f"The analysis stopped: {exc}")
        st.stop()
    st.session_state["upload_results"] = ActiveData(
        "upload", f"Your data · {upload.name}", res.tables(), res.meta() | {"file_name": upload.name})
    st.session_state["active_source"] = "upload"
    st.session_state.pop("scenario_key", None)
    bar.progress(1.0, text="Done")
    st.success("Analysis complete: every page now shows your data.")

if st.session_state.get("upload_results") is not None and st.session_state.get("active_source") == "upload":
    up = st.session_state["upload_results"]
    s = up.t("policy_summary")
    channel_of = up.t("stores").set_index("store")["channel"]
    plan = s[(s["variant"] == "calibrated") & (s["policy"] == "P2_forecast_sq")]
    plan = plan[plan["channel"] == plan["store"].map(channel_of)][
        ["store", "item", "channel", "lead_time", "mean_daily", "reorder_point", "order_qty", "safety_stock",
         "fill_rate", "avg_inventory_units", "total_cost"]]
    c1, c2 = st.columns(2)
    c1.download_button("Reorder plan per series (CSV)", plan.to_csv(index=False).encode(), "reorder_plan.csv",
                       "text/csv", icon=":material/download:")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        plan.to_excel(xw, sheet_name="reorder_plan", index=False)
        up.t("classification").to_excel(xw, sheet_name="abc_xyz", index=False)
        s[s["variant"] == "calibrated"].to_excel(xw, sheet_name="policy_results", index=False)
    c2.download_button("All results (Excel)", buf.getvalue(), "inventory_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:")
    if st.button("Go to the overview", icon=":material/insights:"):
        st.switch_page("views/overview.py")
