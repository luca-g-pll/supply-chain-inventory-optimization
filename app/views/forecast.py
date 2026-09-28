"""Forecast: leaderboard for the scenario and a series explorer (UC-03, UC-04)."""

import charts
import pandas as pd
import streamlit as st
from common import MODEL_LABELS, active_data, csv_download, no_data_message, scenario_badge, scenario_leaderboard

from src.metrics import bias, mape, wape
from src.series import natural_sorted

data = active_data()
st.title("Forecast")
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
best, base = data.best_model, "seasonal_naive"

lb = scenario_leaderboard(data, sc)
lb_valid = scenario_leaderboard(data, sc, period="validation")
if lb.empty:
    st.info("No series in this scenario.")
    st.stop()
lbi = lb.set_index("model")
gain = 1 - lbi.at[best, "wape"] / lbi.at[base, "wape"]
st.markdown(f"#### {MODEL_LABELS.get(best, best)} makes **{gain:.0%} less error** than the seasonal-naive "
            f"baseline (WAPE {lbi.at[best, 'wape']:.1%} vs {lbi.at[base, 'wape']:.1%}).")
st.caption("Rolling-origin back-test: a new forecast every 7 days, 1–14 days ahead. The best model was chosen on "
           "the validation period; the numbers here are on the test period.")

c1, c2 = st.columns([1, 1])
with c1:
    charts.show(charts.leaderboard_bars(lb, best), key="fc_lb")
with c2:
    table = lb[["model", "wape", "mape", "bias", "rmse"]].copy()
    table["vs baseline"] = 1 - table["wape"] / lbi.at[base, "wape"]
    table = table.merge(lb_valid[["model", "wape"]].rename(columns={"wape": "WAPE validation"}), on="model", how="left")
    table["model"] = table["model"].map(lambda m: MODEL_LABELS.get(m, m))
    st.dataframe(table.sort_values("wape"), hide_index=True, column_config={
        "model": "Model", "wape": st.column_config.NumberColumn("WAPE", format="percent"),
        "mape": st.column_config.NumberColumn("MAPE", format="percent"),
        "bias": st.column_config.NumberColumn("Bias", format="percent"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        "vs baseline": st.column_config.NumberColumn("Error vs baseline", format="percent"),
        "WAPE validation": st.column_config.NumberColumn(format="percent")})
    st.caption("WAPE = Σ|A − F| / ΣA · MAPE on days with sales · bias = Σ(F − A) / ΣA (positive = over-forecast).")
    csv_download(table, "Download the leaderboard (CSV)", "leaderboard.csv", "dl_lb")

t1, t2, t3 = st.tabs(["By horizon", "By ABC class", "By store"])
with t1:
    charts.show(charts.wape_by(scenario_leaderboard(data, sc, by=("model", "h")), "h", best, base,
                               "WAPE by days ahead", "Days ahead"), key="fc_h")
with t2:
    by_abc = scenario_leaderboard(data, sc, by=("model", "abc"))
    pv = by_abc.pivot(index="abc", columns="model", values="wape")
    pv.columns = [MODEL_LABELS.get(m, m) for m in pv.columns]
    st.dataframe(pv.style.format("{:.1%}"))
with t3:
    by_store = scenario_leaderboard(data, sc, by=("model", "store"))
    pv = by_store.pivot(index="store", columns="model", values="wape")
    pv = pv.reindex(natural_sorted(pv.index))
    pv.columns = [MODEL_LABELS.get(m, m) for m in pv.columns]
    st.dataframe(pv.style.format("{:.1%}"))

st.subheader("Inspect one series")
view = data.t("forecasts")
if view.empty:
    st.info("No daily forecast detail available.")
else:
    avail = view[["store", "item"]].drop_duplicates()
    avail = avail[avail["store"].isin(sc.stores) & avail["item"].isin(sc.items)]
    if avail.empty:
        st.info("None of the series in this scenario has daily detail. " + data.detail_note)
    else:
        a1, a2, a3 = st.columns([1, 1, 2])
        store = a1.selectbox("Store", natural_sorted(avail["store"].unique()), key="fc_store")
        item = a2.selectbox("Item", natural_sorted(avail.loc[avail["store"] == store, "item"]), key="fc_item")
        models = [m for m in ["lightgbm", "holt_winters", "moving_average", "seasonal_naive"] if m in set(view["model"])]
        chosen = a3.multiselect("Models", models, default=[best, base] if base != best else [best],
                                format_func=lambda m: MODEL_LABELS.get(m, m), key="fc_models")
        if chosen:
            v = view[(view["store"] == store) & (view["item"] == item) & view["model"].isin(chosen)]
            charts.show(charts.forecast_series(v, chosen, best), key="fc_series")
            rows = []
            for m in chosen:
                d = v[v["model"] == m]
                rows.append({"Model": MODEL_LABELS.get(m, m), "WAPE": wape(d["actual"], d["forecast"]),
                             "MAPE": mape(d["actual"], d["forecast"]), "Bias": bias(d["actual"], d["forecast"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="percent") for c in ("WAPE", "MAPE", "Bias")})
            st.caption("Each day is forecast from the latest weekly origin (1–7 days ahead).")
        if data.detail_note:
            st.caption(data.detail_note)

imp = data.t("feature_importance")
if len(imp):
    with st.expander("What drives the LightGBM forecast"):
        charts.show(charts.importance(imp), key="fc_imp")
        st.caption("lag_k = sales k days earlier; roll_mean_w = average of the w days ending 14 days before; "
                   "every feature uses data at least 14 days old, so one model serves every horizon without leakage.")
