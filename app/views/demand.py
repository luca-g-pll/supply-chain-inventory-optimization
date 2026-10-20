"""Demand: trend, seasonality, ABC/XYZ (UC-01, UC-02)."""

import charts
import streamlit as st
from common import ROOT, active_data, csv_download, no_data_message, scenario_badge
from i18n import tx

from src.classification import abc_xyz_matrix
from src.scenario import apply_scenario
from src.sql_runner import parse_queries

DAYS = {"en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "it": ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]}
MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
          "it": ["Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"]}

data = active_data()
st.title(tx("Demand", "Domanda"))
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
lang = "it" if tx("en", "it") == "it" else "en"

monthly = apply_scenario(data.t("demand_monthly"), sc)
dow = apply_scenario(data.t("demand_dow"), sc)
if monthly.empty:
    st.info(tx("No series in this scenario.", "Nessuna serie in questo scenario."))
    st.stop()

charts.show(charts.monthly_trend(monthly), key="dm_trend")

# Day-of-week index: average units per calendar day on that weekday / overall average.
d = dow.groupby("dow").agg(units=("units", "sum"), days=("n_days", "max"))
per_day = d["units"] / d["days"]
dow_index = per_day / (d["units"].sum() / d["days"].sum())
# Month-of-year index from the monthly table.
m = monthly.assign(moy=monthly["month"].dt.month)
days_per_month = m.groupby(["month"])["n_days"].max().rename("days").reset_index()
days_per_month["moy"] = days_per_month["month"].dt.month
units_moy = m.groupby("moy")["units"].sum()
days_moy = days_per_month.groupby("moy")["days"].sum()
month_index = (units_moy / days_moy) / (units_moy.sum() / days_moy.sum())

c1, c2 = st.columns([7, 12])
with c1:
    charts.show(charts.index_bars([DAYS[lang][i] for i in dow_index.index], dow_index.to_numpy(),
                                  tx("Day-of-week index", "Indice per giorno della settimana")), key="dm_dow")
with c2:
    charts.show(charts.index_bars([MONTHS[lang][i - 1] for i in month_index.index], month_index.to_numpy(),
                                  tx("Month-of-year index", "Indice per mese dell'anno")), key="dm_moy")
st.caption(tx("1.0 = an average day. Blue = busier than average, red = quieter.",
              "1,0 = un giorno medio. Blu = più vendite della media, rosso = meno."))

st.subheader(tx("Which products matter: ABC × XYZ", "Quali prodotti contano: ABC × XYZ"))
cls = data.t("classification")
cls_sc = cls[cls["item"].isin(sc.items)]
c3, c4 = st.columns([3, 2])
with c3:
    charts.show(charts.pareto(cls_sc), key="dm_pareto")
with c4:
    charts.show(charts.abc_xyz_heatmap(abc_xyz_matrix(cls_sc)), key="dm_matrix")
st.caption(tx(
    "Classes are computed on the validation period over all stores: **ABC** by revenue share (A = first 80%, "
    "B = up to 95%), **XYZ** by the variability of weekly demand (coefficient of variation < 0.5 = X, < 1.0 = Y, "
    "else Z).",
    "Le classi sono calcolate sul periodo di validazione su tutti i negozi: **ABC** per quota di fatturato "
    "(A = primo 80%, B = fino al 95%), **XYZ** per variabilità della domanda settimanale (coefficiente di "
    "variazione < 0,5 = X, < 1,0 = Y, altrimenti Z)."))
show = cls_sc[["item", "rank", "units", "revenue", "revenue_share", "abc", "weekly_cv", "xyz", "abc_volume"]]
st.dataframe(show, hide_index=True, column_config={
    "item": tx("Item", "Articolo"),
    "rank": tx("Rank", "Posizione"),
    "revenue": st.column_config.NumberColumn(tx("Revenue", "Fatturato"), format="euro"),
    "revenue_share": st.column_config.NumberColumn(tx("Share", "Quota"), format="percent"),
    "units": st.column_config.NumberColumn(tx("Units", "Pezzi"), format="localized"),
    "abc": "ABC",
    "weekly_cv": st.column_config.NumberColumn(tx("Weekly CV", "CV settimanale"), format="%.2f"),
    "xyz": "XYZ",
    "abc_volume": tx("ABC by volume", "ABC per volume"),
})
csv_download(show, tx("Download the classification (CSV)", "Scarica la classificazione (CSV)"), "abc_xyz.csv", "dl_cls")
master = data.t("item_master")
if len(master) and (master["price_source"] == "assumed").any():
    n = int((master["price_source"] == "assumed").sum())
    st.info(tx(f"{n} item(s) have no price in the data: revenue uses assumed prices. The *ABC by volume* column "
               "does not depend on prices.",
               f"{n} articolo/i non hanno un prezzo nei dati: il fatturato usa prezzi ipotizzati. La colonna "
               "*ABC per volume* non dipende dai prezzi."))

st.subheader(tx("SQL KPIs (DuckDB)", "KPI calcolati in SQL (DuckDB)"))
years = data.t("sql_sales_by_year")
if len(years):
    st.dataframe(years, hide_index=True, column_config={
        "year": tx("Year", "Anno"),
        "n_days": tx("Days", "Giorni"),
        "units": st.column_config.NumberColumn(tx("Units", "Pezzi"), format="localized"),
        "revenue": st.column_config.NumberColumn(tx("Revenue", "Fatturato"), format="euro"),
        "units_per_day": st.column_config.NumberColumn(tx("Units per day", "Pezzi al giorno"), format="%.0f"),
        "yoy_growth_pct": st.column_config.NumberColumn(tx("Growth vs previous year", "Crescita sull'anno prima"),
                                                        format="%.1f%%")})
    st.caption(tx("Whole dataset (all stores and items), computed with SQL.",
                  "Intero dataset (tutti i negozi e articoli), calcolato in SQL."))
with st.expander(tx("See the SQL behind the KPIs", "Vedi l'SQL dietro ai KPI")):
    queries = parse_queries(ROOT / "sql" / "kpi_queries.sql")
    name = st.selectbox(tx("Query", "Query"), list(queries), index=list(queries).index("abc_classification"))
    st.code(queries[name], language="sql")
    tbl = data.t(f"sql_{name}")
    if len(tbl):
        st.dataframe(tbl.head(200), hide_index=True)
