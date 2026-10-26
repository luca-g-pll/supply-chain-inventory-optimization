"""Forecast: leaderboard for the scenario and a series explorer (UC-03, UC-04)."""

import charts
import pandas as pd
import streamlit as st
from common import active_data, csv_download, detail_note, no_data_message, scenario_badge, scenario_leaderboard
from i18n import model_name, tx

from src.metrics import bias, mape, wape
from src.series import natural_sorted

data = active_data()
st.title(tx("Forecast", "Previsioni"))
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
best, base = data.best_model, "seasonal_naive"

lb = scenario_leaderboard(data, sc)
lb_valid = scenario_leaderboard(data, sc, period="validation")
if lb.empty:
    st.info(tx("No series in this scenario.", "Nessuna serie in questo scenario."))
    st.stop()
lbi = lb.set_index("model")
gain = 1 - lbi.at[best, "wape"] / lbi.at[base, "wape"]
wb, wbase = lbi.at[best, "wape"], lbi.at[base, "wape"]
st.markdown(tx(
    f"#### {model_name(best)} makes **{gain:.0%} less error** than the seasonal-naive baseline "
    f"(WAPE {wb:.1%} vs {wbase:.1%}).",
    f"#### {model_name(best)} sbaglia **il {gain:.0%} in meno** della regola \"come la settimana scorsa\" "
    f"(WAPE {wb:.1%} contro {wbase:.1%})."))
st.caption(tx(
    "Rolling-origin back-test: a new forecast every 7 days, 1–14 days ahead. The best model was chosen on the "
    "validation period; the numbers here are on the test period.",
    "Back-test a origine mobile: una nuova previsione ogni 7 giorni, da 1 a 14 giorni avanti. Il modello migliore è "
    "stato scelto sul periodo di validazione; i numeri qui sono sul periodo di test."))
if (data.meta.get("promotions") or {}).get("source"):
    st.caption(tx(
        ":material/sell: Promotions found in the data: the moving average, Holt-Winters and LightGBM use the "
        "promotion calendar (shaded in the chart below); the baseline does not. Details on the **Promotions** page.",
        ":material/sell: Nei dati ci sono promozioni: media mobile, Holt-Winters e LightGBM usano il calendario delle "
        "promozioni (in evidenza nel grafico qui sotto); la regola di base no. Dettagli nella pagina **Promozioni**."))

c1, c2 = st.columns([1, 1])
with c1:
    charts.show(charts.leaderboard_bars(lb, best), key="fc_lb")
with c2:
    table = lb[["model", "wape", "mape", "bias", "rmse"]].copy()
    table["vs baseline"] = 1 - table["wape"] / wbase
    table = table.merge(lb_valid[["model", "wape"]].rename(columns={"wape": "wape_valid"}), on="model", how="left")
    table["model"] = table["model"].map(model_name)
    st.dataframe(table.sort_values("wape"), hide_index=True, column_config={
        "model": tx("Model", "Modello"), "wape": st.column_config.NumberColumn("WAPE", format="percent"),
        "mape": st.column_config.NumberColumn("MAPE", format="percent"),
        "bias": st.column_config.NumberColumn("Bias", format="percent"),
        "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
        "vs baseline": st.column_config.NumberColumn(tx("Error vs baseline", "Errore in meno vs base"),
                                                     format="percent"),
        "wape_valid": st.column_config.NumberColumn(tx("WAPE validation", "WAPE validazione"), format="percent")})
    st.caption(tx("WAPE = Σ|A − F| / ΣA · MAPE on days with sales · bias = Σ(F − A) / ΣA (positive = over-forecast).",
                  "WAPE = Σ|V − P| / ΣV · MAPE sui giorni con vendite · bias = Σ(P − V) / ΣV (positivo = previsioni "
                  "troppo alte). V = venduto, P = previsto."))
    csv_download(table, tx("Download the leaderboard (CSV)", "Scarica la classifica (CSV)"), "leaderboard.csv", "dl_lb")

t1, t2, t3 = st.tabs([tx("By horizon", "Per orizzonte"), tx("By ABC class", "Per classe ABC"),
                      tx("By store", "Per negozio")])
with t1:
    charts.show(charts.wape_by(scenario_leaderboard(data, sc, by=("model", "h")), "h", best, base,
                               tx("WAPE by days ahead", "WAPE per giorni di anticipo"),
                               tx("Days ahead", "Giorni di anticipo")), key="fc_h")
with t2:
    by_abc = scenario_leaderboard(data, sc, by=("model", "abc"))
    pv = by_abc.pivot(index="abc", columns="model", values="wape")
    pv.columns = [model_name(m) for m in pv.columns]
    pv.index.name = tx("Class", "Classe")
    st.dataframe(pv.style.format("{:.1%}"))
with t3:
    by_store = scenario_leaderboard(data, sc, by=("model", "store"))
    pv = by_store.pivot(index="store", columns="model", values="wape")
    pv = pv.reindex(natural_sorted(pv.index))
    pv.columns = [model_name(m) for m in pv.columns]
    pv.index.name = tx("Store", "Negozio")
    st.dataframe(pv.style.format("{:.1%}"))

st.subheader(tx("Inspect one series", "Analizza una serie"))
view = data.t("forecasts")
note = detail_note(data)
if view.empty:
    st.info(tx("No daily forecast detail available.", "Nessun dettaglio giornaliero delle previsioni disponibile."))
else:
    avail = view[["store", "item"]].drop_duplicates()
    avail = avail[avail["store"].isin(sc.stores) & avail["item"].isin(sc.items)]
    if avail.empty:
        st.info(tx("None of the series in this scenario has daily detail. ",
                   "Nessuna serie di questo scenario ha il dettaglio giornaliero. ") + note)
    else:
        a1, a2, a3 = st.columns([1, 1, 2])
        store = a1.selectbox(tx("Store", "Negozio"), natural_sorted(avail["store"].unique()), key="fc_store")
        item = a2.selectbox(tx("Item", "Articolo"), natural_sorted(avail.loc[avail["store"] == store, "item"]),
                            key="fc_item")
        models = [m for m in ["lightgbm", "holt_winters", "moving_average", "seasonal_naive"] if m in set(view["model"])]
        chosen = a3.multiselect(tx("Models", "Modelli"), models, default=[best, base] if base != best else [best],
                                format_func=model_name, key="fc_models")
        if chosen:
            v = view[(view["store"] == store) & (view["item"] == item) & view["model"].isin(chosen)]
            charts.show(charts.forecast_series(v, chosen, best), key="fc_series",
                        alt=tx(f"Actual vs forecast sales per day, store {store}, item {item}",
                               f"Vendite reali e previste al giorno, negozio {store}, articolo {item}"))
            rows = []
            for m in chosen:
                d = v[v["model"] == m]
                rows.append({tx("Model", "Modello"): model_name(m), "WAPE": wape(d["actual"], d["forecast"]),
                             "MAPE": mape(d["actual"], d["forecast"]), "Bias": bias(d["actual"], d["forecast"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="percent") for c in ("WAPE", "MAPE", "Bias")})
            st.caption(tx("Each day is forecast from the latest weekly origin (1–7 days ahead).",
                          "Ogni giorno è previsto dall'ultima previsione settimanale (da 1 a 7 giorni prima)."))
        if note:
            st.caption(note)

imp = data.t("feature_importance")
if len(imp):
    with st.expander(tx("What drives the LightGBM forecast", "Cosa guida la previsione di LightGBM")):
        charts.show(charts.importance(imp), key="fc_imp")
        st.caption(tx(
            "lag_k = sales k days earlier; roll_mean_w = average of the w days ending 14 days before; every feature "
            "uses data at least 14 days old, so one model serves every horizon without leakage.",
            "lag_k = vendite di k giorni prima; roll_mean_w = media dei w giorni che finiscono 14 giorni prima; ogni "
            "variabile usa dati vecchi di almeno 14 giorni, così un solo modello serve ogni orizzonte senza usare "
            "informazioni dal futuro."))
