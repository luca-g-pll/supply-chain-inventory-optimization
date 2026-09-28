"""Overview: headline, KPI tiles, policy comparison (UC-07, UC-10)."""

import charts
import streamlit as st
from common import (
    active_data,
    csv_download,
    euro,
    no_data_message,
    policy_kpis,
    policy_label,
    scenario_badge,
    scenario_leaderboard,
    scenario_summary,
)

data = active_data()
st.title("Demand forecasting & inventory optimization")
if data is None:
    no_data_message()
    st.stop()

sc = st.session_state["scenario"]
scenario_badge(data, sc)
summary = scenario_summary(data, sc)
k = policy_kpis(summary)
if k.empty:
    st.info("No series in this scenario.")
    st.stop()
kp = k.set_index("policy")
target = data.target
best = data.best_model
lb = scenario_leaderboard(data, sc).set_index("model")

# ---- headline
p0, p2 = kp.loc["P0_naive"], kp.loc["P2_forecast_sq"]
inv_red = 1 - p2["avg_inventory_value"] / p0["avg_inventory_value"]
cost_red = 1 - p2["total_cost"] / p0["total_cost"]
wape_best, wape_base = lb.at[best, "wape"], lb.at["seasonal_naive", "wape"]
verb = "less" if inv_red >= 0 else "more"
st.markdown(
    f"#### The forecast-driven policy needs **{abs(inv_red):.0%} {verb} inventory** than ordering the same "
    f"quantity every week, at the same **{target:.0%} fill rate**. "
    f"The best forecast ({charts.MODEL_NAMES.get(best, best)}) makes **{1 - wape_best / wape_base:.0%} less error** "
    f"than the seasonal-naive baseline.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast error (WAPE)", f"{wape_best:.1%}", f"{wape_best / wape_base - 1:+.0%} vs seasonal naive",
          delta_color="inverse", border=True, help="Σ|actual − forecast| / Σ actual on the test period.")
c2.metric("Average inventory · P2", euro(p2["avg_inventory_value"]), f"{-inv_red:+.0%} vs P0",
          delta_color="inverse", border=True, help="Average value of the stock on hand (unit cost × units).")
c3.metric("Total cost · P2", euro(p2["total_cost"]), f"{-cost_red:+.0%} vs P0", delta_color="inverse", border=True,
          help="Ordering + holding + lost margin (+ backorder penalty online) over the test period.")
c4.metric("Fill rate · P2", f"{p2['fill_rate']:.1%}", f"target {target:.0%}", delta_color="off", border=True,
          help="Share of demand served from stock. Every policy is calibrated per series to reach the target.")

left, right = st.columns(2)
with left:
    charts.show(charts.inventory_at_target(k, target), key="ov_inv")
with right:
    charts.show(charts.cost_breakdown(k), key="ov_cost")

# ---- table view
st.subheader("Policy comparison at the same fill rate")
cols = {"fill_rate": "Fill rate", "avg_inventory_value": "Avg inventory (€)", "avg_inventory_units": "Avg inventory (units)",
        "days_of_supply": "Days of supply", "turns": "Turns / year", "stockout_days": "Stock-out series-days",
        "n_orders": "Orders", "ordering_cost": "Ordering (€)", "holding_cost": "Holding (€)",
        "shortage_cost": "Shortage (€)", "total_cost": "Total cost (€)"}
table = k.assign(policy=k["policy"].map(policy_label))[["policy", *cols]].rename(columns={"policy": "Policy", **cols})
money = [c for c in table.columns if "€" in c]
st.dataframe(table, hide_index=True, column_config={
    "Fill rate": st.column_config.NumberColumn(format="percent"),
    **{c: st.column_config.NumberColumn(format="euro") for c in money},
    "Avg inventory (units)": st.column_config.NumberColumn(format="%.0f"),
    "Days of supply": st.column_config.NumberColumn(format="%.1f"),
    "Turns / year": st.column_config.NumberColumn(format="%.1f"),
})
csv_download(table, "Download the comparison (CSV)", "policy_comparison.csv", "dl_ov")

with st.expander("Same policies with the safety factor set in advance (z from the 95% cycle service level)"):
    kf = policy_kpis(summary, variant="fixed")
    tf = kf.assign(policy=kf["policy"].map(policy_label))[["policy", *cols]].rename(columns={"policy": "Policy", **cols})
    st.dataframe(tf, hide_index=True, column_config={
        "Fill rate": st.column_config.NumberColumn(format="percent"),
        **{c: st.column_config.NumberColumn(format="euro") for c in money}})
    st.caption("With z = 1.645 set in advance the (s, Q) policies over-deliver (fill rate far above 95%): a 95% "
               "*cycle* service level is not a 95% *fill rate*. The calibrated view above compares every policy "
               "at the same fill rate instead.")

b1, b2 = st.columns(2)
with b1:
    st.subheader("By ABC class")
    by_abc = policy_kpis(summary, by=("abc", "policy"))
    pv = by_abc.pivot(index="abc", columns="policy", values="avg_inventory_value")
    pv = pv[[p for p in ("P0_naive", "P1_static_sq", "P2_forecast_sq") if p in pv.columns]]
    pv.columns = [policy_label(p) for p in pv.columns]
    st.dataframe(pv.style.format("€{:,.0f}"), column_config={"abc": "Class"})
    st.caption("Average inventory value at the target fill rate.")
with b2:
    st.subheader("By channel")
    by_ch = policy_kpis(summary, by=("channel", "policy"))
    pc = by_ch.pivot(index="channel", columns="policy", values="total_cost")
    pc = pc[[p for p in ("P0_naive", "P1_static_sq", "P2_forecast_sq") if p in pc.columns]]
    pc.columns = [policy_label(p) for p in pc.columns]
    st.dataframe(pc.style.format("€{:,.0f}"))
    st.caption("Total cost. Mark stores as online in the sidebar to compare channels.")

with st.expander("How to read this page"):
    st.markdown(f"""
- **P0 · naive weekly**: every Monday order the average weekly demand of the validation period.
- **P1 · static (s, Q)**: reorder point `s = d̄·L + z·σ_d·√L` from last period's statistics, lot = EOQ.
- **P2 · forecast-driven (s, Q)**: reorder point `s_t = forecast demand over the lead time + z·σ_forecast error`,
  updated every day from the best model's weekly forecasts; same EOQ lot.
- Every series is simulated day by day over the test period; the knob of each policy (weekly quantity multiplier
  for P0, safety factor z for P1/P2) is **calibrated per series** so that all policies reach a {target:.0%} fill
  rate. Only then is the stock compared.
- Costs, prices and lead times are assumptions (see the README); change them in the **Inventory** what-if.
""")
