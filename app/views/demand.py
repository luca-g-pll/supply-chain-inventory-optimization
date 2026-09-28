"""Demand: trend, seasonality, ABC/XYZ (UC-01, UC-02)."""

import calendar

import charts
import streamlit as st
from common import ROOT, active_data, csv_download, no_data_message, scenario_badge

from src.classification import abc_xyz_matrix
from src.scenario import apply_scenario
from src.sql_runner import parse_queries

data = active_data()
st.title("Demand")
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)

monthly = apply_scenario(data.t("demand_monthly"), sc)
dow = apply_scenario(data.t("demand_dow"), sc)
if monthly.empty:
    st.info("No series in this scenario.")
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
    charts.show(charts.index_bars([calendar.day_abbr[i] for i in dow_index.index], dow_index.to_numpy(),
                                  "Day-of-week index"), key="dm_dow")
with c2:
    charts.show(charts.index_bars([calendar.month_abbr[i] for i in month_index.index], month_index.to_numpy(),
                                  "Month-of-year index"), key="dm_moy")
st.caption("1.0 = an average day. Blue = busier than average, red = quieter.")

st.subheader("Which products matter: ABC × XYZ")
cls = data.t("classification")
cls_sc = cls[cls["item"].isin(sc.items)]
c3, c4 = st.columns([3, 2])
with c3:
    charts.show(charts.pareto(cls_sc), key="dm_pareto")
with c4:
    charts.show(charts.abc_xyz_heatmap(abc_xyz_matrix(cls_sc)), key="dm_matrix")
st.caption("Classes are computed on the validation period over all stores: **ABC** by revenue share "
           "(A = first 80%, B = up to 95%), **XYZ** by the variability of weekly demand "
           "(coefficient of variation < 0.5 = X, < 1.0 = Y, else Z).")
show = cls_sc[["item", "rank", "units", "revenue", "revenue_share", "abc", "weekly_cv", "xyz", "abc_volume"]]
st.dataframe(show, hide_index=True, column_config={
    "revenue": st.column_config.NumberColumn("Revenue", format="euro"),
    "revenue_share": st.column_config.NumberColumn("Share", format="percent"),
    "units": st.column_config.NumberColumn("Units", format="localized"),
    "weekly_cv": st.column_config.NumberColumn("Weekly CV", format="%.2f"),
    "abc_volume": "ABC by volume",
})
csv_download(show, "Download the classification (CSV)", "abc_xyz.csv", "dl_cls")
master = data.t("item_master")
if len(master) and (master["price_source"] == "assumed").any():
    st.info(f"{int((master['price_source'] == 'assumed').sum())} item(s) have no price in the data: revenue uses "
            "assumed prices. The *ABC by volume* column does not depend on prices.")

st.subheader("SQL KPIs (DuckDB)")
years = data.t("sql_sales_by_year")
if len(years):
    st.dataframe(years, hide_index=True, column_config={
        "units": st.column_config.NumberColumn("Units", format="localized"),
        "revenue": st.column_config.NumberColumn("Revenue", format="euro"),
        "units_per_day": st.column_config.NumberColumn("Units per day", format="%.0f"),
        "yoy_growth_pct": st.column_config.NumberColumn("Growth vs previous year", format="%.1f%%")})
    st.caption("Whole dataset (all stores and items), computed with SQL.")
with st.expander("See the SQL behind the KPIs"):
    queries = parse_queries(ROOT / "sql" / "kpi_queries.sql")
    name = st.selectbox("Query", list(queries), index=list(queries).index("abc_classification"))
    st.code(queries[name], language="sql")
    tbl = data.t(f"sql_{name}")
    if len(tbl):
        st.dataframe(tbl.head(200), hide_index=True)
