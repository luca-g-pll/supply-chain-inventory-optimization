"""Inventory: trade-off curves, reorder parameters of one series and the what-if (UC-05, UC-06)."""

import charts
import pandas as pd
import streamlit as st
from common import active_data, config, euro, no_data_message, policy_kpis, policy_label, scenario_badge
from tools import build_inputs, channel_with, detail_series, simulate_series

from src.series import natural_sorted

data = active_data()
st.title("Inventory")
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
cfg = config()
target = data.target

tab1, tab2, tab3 = st.tabs(["Policies", "One series & what-if", "Sensitivity"])

with tab1:
    k = policy_kpis(data.t("policy_summary")[data.t("policy_summary")["channel"] == "physical"])
    charts.show(charts.tradeoff(data.t("tradeoff"), k, target), key="inv_trade")
    charts.show(charts.tradeoff(data.t("tradeoff"), k, target, metric="total_cost",
                                title="Total cost vs. fill rate for each policy"), key="inv_cost")
    st.caption("All series, physical stores. Each line moves one knob for every series at once (weekly quantity "
               "for P0, safety factor z for P1/P2); big dots = each series calibrated to the target. Under the "
               "assumed costs a lost sale (40% margin) costs far more than holding a unit for a year (25% of cost), "
               "so total cost keeps falling almost up to 100% fill rate: 95% is a conservative target.")

def series_tab() -> None:
    avail = detail_series(data)
    avail = avail[avail["store"].isin(sc.stores) & avail["item"].isin(sc.items)]
    if avail.empty:
        st.info("None of the series in this scenario has daily detail. " + data.detail_note)
        return
    a1, a2, a3 = st.columns([1, 1, 2])
    store = a1.selectbox("Store", natural_sorted(avail["store"].unique()), key="inv_store")
    item = a2.selectbox("Item", natural_sorted(avail.loc[avail["store"] == store, "item"]), key="inv_item")
    channel_name = sc.channel_of(store)
    a3.markdown(f"<br>Channel: **{channel_name}** (change it in the sidebar)", unsafe_allow_html=True)
    inp = build_inputs(data, store, item)

    s = data.t("policy_summary")
    rows = s[(s["store"] == store) & (s["item"] == item) & (s["channel"] == channel_name)]
    cal = rows[rows["variant"] == "calibrated"].set_index("policy")
    def _num(values, digits: int = 1) -> list[str]:
        return ["—" if pd.isna(v) else f"{v:,.{digits}f}" for v in values]

    params = pd.DataFrame({
        "Policy": [policy_label(p) for p in cal.index],
        "Reorder point s": _num(cal["reorder_point"]),
        "Order quantity Q": _num(cal["order_qty"], 0),
        "Safety stock": _num(cal["safety_stock"]),
        "Knob (z or multiplier)": cal["knob"].round(2).to_numpy(),
        "Fill rate": cal["fill_rate"].to_numpy(),
        "Avg inventory (units)": cal["avg_inventory_units"].round(0).to_numpy(),
        "Total cost (€)": cal["total_cost"].round(0).to_numpy(),
    })
    st.markdown(f"**When and how much to order — store {store}, item {item}** "
                f"(mean demand {inp.mean_daily[0]:.1f} units/day in the validation period, "
                f"lead time {int(cal['lead_time'].iloc[0])} days)")
    st.dataframe(params, hide_index=True, column_config={
        "Fill rate": st.column_config.NumberColumn(format="percent"),
        "Total cost (€)": st.column_config.NumberColumn(format="euro")})
    st.caption("P2's reorder point changes every day with the forecast: the table shows its average. "
               "Negative safety stock means the fill-rate target is reached even when the order is placed a bit "
               "late in the cycle (large EOQ lots already protect most demand).")

    knobs = {p: float(cal.at[p, "knob"]) for p in cal.index}
    _, daily = simulate_series(inp, cfg.channel(channel_name), cfg, knobs=knobs)
    charts.show(charts.trajectory(daily), key="inv_traj")

    st.subheader("What if?")
    base_ch = cfg.channel(channel_name)
    w1, w2, w3, w4 = st.columns(4)
    sl = w1.slider("Service target (sets z)", 0.80, 0.995, float(base_ch.inventory.service_level), 0.005,
                   format="%.3f", key="wi_sl")
    lt = w2.slider("Lead time (days)", 1, 21, int(cal["lead_time"].iloc[0]), key="wi_lt")
    oc = w3.slider("Ordering cost (€ per order)", 5, 500, int(base_ch.inventory.order_cost), 5, key="wi_oc")
    hr = w4.slider("Holding rate (% of cost per year)", 5, 50, int(round(base_ch.inventory.holding_rate * 100)),
                   key="wi_hr")
    calibrate = st.toggle(f"Calibrate every policy to a {target:.0%} fill rate (instead of setting z from the target)",
                          value=False, key="wi_cal")
    ch_new = channel_with(cfg, channel_name, lead_time_days=lt, order_cost=float(oc), holding_rate=hr / 100,
                          service_level=sl)
    base_rows, _ = simulate_series(inp, base_ch, cfg, calibrate=calibrate)
    new_rows, _ = simulate_series(inp, ch_new, cfg, calibrate=calibrate, lead_time=lt, service_level=sl)
    comp = []
    for pol in ("P0_naive", "P1_static_sq", "P2_forecast_sq"):
        b = base_rows[base_rows["policy"] == pol].iloc[0]
        n = new_rows[new_rows["policy"] == pol].iloc[0]
        comp.append({"Policy": policy_label(pol), "Fill rate": n["fill_rate"], "Δ fill rate (pp)": 100 * (n["fill_rate"] - b["fill_rate"]),
                     "Avg inventory (€)": n["avg_inventory_value"],
                     "Δ inventory": n["avg_inventory_value"] / b["avg_inventory_value"] - 1 if b["avg_inventory_value"] else 0,
                     "Order qty": n["order_qty"], "Reorder point": n["reorder_point"],
                     "Total cost (€)": n["total_cost"], "Δ cost": n["total_cost"] / b["total_cost"] - 1 if b["total_cost"] else 0})
    st.dataframe(pd.DataFrame(comp), hide_index=True, column_config={
        "Fill rate": st.column_config.NumberColumn(format="percent"),
        "Δ fill rate (pp)": st.column_config.NumberColumn(format="%+.1f"),
        "Avg inventory (€)": st.column_config.NumberColumn(format="euro"),
        "Δ inventory": st.column_config.NumberColumn(format="percent"),
        "Order qty": st.column_config.NumberColumn(format="%.0f"),
        "Reorder point": st.column_config.NumberColumn(format="%.1f", help="Empty for P0: it orders by the calendar"),
        "Total cost (€)": st.column_config.NumberColumn(format="euro"),
        "Δ cost": st.column_config.NumberColumn(format="percent")})
    p2 = new_rows[new_rows["policy"] == "P2_forecast_sq"].iloc[0]
    st.caption(f"Differences vs. the default assumptions. P2 with these settings: average stock "
               f"{euro(p2['avg_inventory_value'])}, fill rate {p2['fill_rate']:.1%}, "
               f"{int(p2['stockout_days'])} stock-out days, {int(p2['n_orders'])} orders.")
    if data.detail_note:
        st.caption(data.detail_note)


with tab2:
    series_tab()

with tab3:
    sens = data.t("sensitivity")
    if len(sens):
        charts.show(charts.sensitivity(sens), key="inv_sens")
        st.caption("All series, physical stores, z set from the service target (not calibrated). Longer lead times "
                   "need more safety stock; the forecast-driven policy stays below the static one.")
        st.dataframe(sens[["policy", "lead_time", "service_level", "fill_rate", "avg_inventory_value", "total_cost"]]
                     .assign(policy=lambda d: d["policy"].map(policy_label)), hide_index=True, column_config={
            "service_level": st.column_config.NumberColumn("Target", format="percent"),
            "fill_rate": st.column_config.NumberColumn("Fill rate", format="percent"),
            "avg_inventory_value": st.column_config.NumberColumn("Avg inventory", format="euro"),
            "total_cost": st.column_config.NumberColumn("Total cost", format="euro")})
