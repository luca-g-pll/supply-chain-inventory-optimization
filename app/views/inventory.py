"""Inventory: trade-off curves, reorder parameters of one series and the what-if (UC-05, UC-06)."""

import charts
import pandas as pd
import streamlit as st
from common import (
    active_data,
    channel_label,
    config,
    detail_note,
    euro,
    no_data_message,
    policy_kpis,
    policy_label,
    scenario_badge,
)
from i18n import tx
from tools import build_inputs, channel_with, detail_series, simulate_series

from src.series import natural_sorted

NUM = st.column_config.NumberColumn

data = active_data()
st.title(tx("Inventory", "Scorte"))
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
cfg = config()
target = data.target

tab1, tab2, tab3 = st.tabs([tx("Policies", "Politiche"), tx("One series & what-if", "Una serie e what-if"),
                            tx("Sensitivity", "Sensibilità")])

with tab1:
    k = policy_kpis(data.t("policy_summary")[data.t("policy_summary")["channel"] == "physical"])
    charts.show(charts.tradeoff(data.t("tradeoff"), k, target), key="inv_trade")
    charts.show(charts.tradeoff(data.t("tradeoff"), k, target, metric="total_cost",
                                title=tx("Total cost vs. fill rate for each policy",
                                         "Costo totale vs. fill rate per ogni politica")), key="inv_cost")
    st.caption(tx(
        "All series, physical stores. Each line moves one knob for every series at once (weekly quantity for P0, "
        "safety factor z for P1/P2); big dots = each series calibrated to the target. Under the assumed costs a lost "
        "sale (40% margin) costs far more than holding a unit for a year (25% of cost), so total cost keeps falling "
        "almost up to 100% fill rate: 95% is a conservative target.",
        "Tutte le serie, negozi fisici. Ogni linea muove un solo parametro per tutte le serie insieme (quantità "
        "settimanale per P0, fattore di sicurezza z per P1/P2); i pallini grandi = ogni serie calibrata "
        "sull'obiettivo. Con i costi ipotizzati una vendita persa (margine 40%) costa molto più che tenere un pezzo "
        "per un anno (25% del costo), quindi il costo totale scende quasi fino al 100% di fill rate: il 95% è un "
        "obiettivo prudente."))


def series_tab() -> None:
    note = detail_note(data)
    avail = detail_series(data)
    avail = avail[avail["store"].isin(sc.stores) & avail["item"].isin(sc.items)]
    if avail.empty:
        st.info(tx("None of the series in this scenario has daily detail. ",
                   "Nessuna serie di questo scenario ha il dettaglio giornaliero. ") + note)
        return
    a1, a2, a3 = st.columns([1, 1, 2])
    store = a1.selectbox(tx("Store", "Negozio"), natural_sorted(avail["store"].unique()), key="inv_store")
    item = a2.selectbox(tx("Item", "Articolo"), natural_sorted(avail.loc[avail["store"] == store, "item"]),
                        key="inv_item")
    channel = sc.channel_of(store)
    a3.markdown(tx(f"<br>Channel: **{channel_label(channel)}** (change it in the sidebar)",
                   f"<br>Canale: **{channel_label(channel)}** (puoi cambiarlo nella barra laterale)"),
                unsafe_allow_html=True)
    inp = build_inputs(data, store, item)

    s = data.t("policy_summary")
    rows = s[(s["store"] == store) & (s["item"] == item) & (s["channel"] == channel)]
    cal = rows[rows["variant"] == "calibrated"].set_index("policy")

    def _num(values, digits: int = 1) -> list[str]:
        return ["—" if pd.isna(v) else f"{v:,.{digits}f}" for v in values]

    params = pd.DataFrame({
        "policy": [policy_label(p) for p in cal.index],
        "rop": _num(cal["reorder_point"]),
        "q": _num(cal["order_qty"], 0),
        "ss": _num(cal["safety_stock"]),
        "knob": cal["knob"].round(2).to_numpy(),
        "fill": cal["fill_rate"].to_numpy(),
        "inv": cal["avg_inventory_units"].round(0).to_numpy(),
        "cost": cal["total_cost"].round(0).to_numpy(),
    })
    lt0 = int(cal["lead_time"].iloc[0])
    st.markdown(tx(
        f"**When and how much to order — store {store}, item {item}** (mean demand {inp.mean_daily[0]:.1f} "
        f"units/day in the validation period, lead time {lt0} days)",
        f"**Quando e quanto ordinare — negozio {store}, articolo {item}** (domanda media {inp.mean_daily[0]:.1f} "
        f"pezzi/giorno nel periodo di validazione, tempo di consegna {lt0} giorni)"))
    st.dataframe(params, hide_index=True, column_config={
        "policy": tx("Policy", "Politica"),
        "rop": tx("Reorder point s", "Punto di riordino s"),
        "q": tx("Order quantity Q", "Quantità d'ordine Q"),
        "ss": tx("Safety stock", "Scorta di sicurezza"),
        "knob": tx("Knob (z or multiplier)", "Parametro (z o moltiplicatore)"),
        "fill": NUM("Fill rate", format="percent"),
        "inv": NUM(tx("Avg inventory (units)", "Scorta media (pezzi)"), format="%.0f"),
        "cost": NUM(tx("Total cost (€)", "Costo totale (€)"), format="euro")})
    st.caption(tx(
        "P2's reorder point changes every day with the forecast: the table shows its average. Negative safety stock "
        "means the fill-rate target is reached even when the order is placed a bit late in the cycle (large lots "
        "already protect most demand).",
        "Il punto di riordino di P2 cambia ogni giorno con la previsione: la tabella ne mostra la media. Una scorta "
        "di sicurezza negativa significa che il fill rate obiettivo si raggiunge anche ordinando un po' tardi nel "
        "ciclo (i lotti grandi proteggono già gran parte della domanda)."))

    knobs = {p: float(cal.at[p, "knob"]) for p in cal.index}
    _, daily = simulate_series(inp, cfg.channel(channel), cfg, knobs=knobs)
    charts.show(charts.trajectory(daily), key="inv_traj")

    st.subheader(tx("What if?", "E se…?"))
    base_ch = cfg.channel(channel)
    w1, w2, w3, w4 = st.columns(4)
    sl = w1.slider(tx("Service target (sets z)", "Livello di servizio (determina z)"), 0.80, 0.995,
                   float(base_ch.inventory.service_level), 0.005, format="%.3f", key="wi_sl")
    lt = w2.slider(tx("Lead time (days)", "Tempo di consegna (giorni)"), 1, 21, lt0, key="wi_lt")
    oc = w3.slider(tx("Ordering cost (€ per order)", "Costo per ordine (€)"), 5, 500, int(base_ch.inventory.order_cost),
                   5, key="wi_oc")
    hr = w4.slider(tx("Holding rate (% of cost per year)", "Costo di mantenimento (% del costo all'anno)"), 5, 50,
                   int(round(base_ch.inventory.holding_rate * 100)), key="wi_hr")
    calibrate = st.toggle(tx(f"Calibrate every policy to a {target:.0%} fill rate (instead of setting z from the target)",
                             f"Calibra ogni politica su un fill rate del {target:.0%} (invece di fissare z "
                             "dall'obiettivo)"), value=False, key="wi_cal")
    ch_new = channel_with(cfg, channel, lead_time_days=lt, order_cost=float(oc), holding_rate=hr / 100,
                          service_level=sl)
    base_rows, _ = simulate_series(inp, base_ch, cfg, calibrate=calibrate)
    new_rows, _ = simulate_series(inp, ch_new, cfg, calibrate=calibrate, lead_time=lt, service_level=sl)
    comp = []
    for pol in ("P0_naive", "P1_static_sq", "P2_forecast_sq"):
        b = base_rows[base_rows["policy"] == pol].iloc[0]
        n = new_rows[new_rows["policy"] == pol].iloc[0]
        comp.append({
            "policy": policy_label(pol), "fill": n["fill_rate"], "dfill": 100 * (n["fill_rate"] - b["fill_rate"]),
            "inv": n["avg_inventory_value"],
            "dinv": n["avg_inventory_value"] / b["avg_inventory_value"] - 1 if b["avg_inventory_value"] else 0,
            "q": n["order_qty"], "rop": n["reorder_point"], "cost": n["total_cost"],
            "dcost": n["total_cost"] / b["total_cost"] - 1 if b["total_cost"] else 0})
    st.dataframe(pd.DataFrame(comp), hide_index=True, column_config={
        "policy": tx("Policy", "Politica"),
        "fill": NUM("Fill rate", format="percent"),
        "dfill": NUM(tx("Δ fill rate (pp)", "Δ fill rate (punti)"), format="%+.1f"),
        "inv": NUM(tx("Avg inventory (€)", "Scorta media (€)"), format="euro"),
        "dinv": NUM(tx("Δ inventory", "Δ scorta"), format="percent"),
        "q": NUM(tx("Order qty", "Quantità d'ordine"), format="%.0f"),
        "rop": NUM(tx("Reorder point", "Punto di riordino"), format="%.1f",
                   help=tx("Empty for P0: it orders by the calendar", "Vuoto per P0: ordina a calendario")),
        "cost": NUM(tx("Total cost (€)", "Costo totale (€)"), format="euro"),
        "dcost": NUM(tx("Δ cost", "Δ costo"), format="percent")})
    p2 = new_rows[new_rows["policy"] == "P2_forecast_sq"].iloc[0]
    st.caption(tx(
        f"Differences vs. the default assumptions. P2 with these settings: average stock "
        f"{euro(p2['avg_inventory_value'])}, fill rate {p2['fill_rate']:.1%}, {int(p2['stockout_days'])} stock-out "
        f"days, {int(p2['n_orders'])} orders.",
        f"Differenze rispetto alle ipotesi predefinite. P2 con queste impostazioni: scorta media "
        f"{euro(p2['avg_inventory_value'])}, fill rate {p2['fill_rate']:.1%}, {int(p2['stockout_days'])} giorni "
        f"senza merce, {int(p2['n_orders'])} ordini."))
    if note:
        st.caption(note)


with tab2:
    series_tab()

with tab3:
    sens = data.t("sensitivity")
    if len(sens):
        charts.show(charts.sensitivity(sens), key="inv_sens")
        st.caption(tx(
            "All series, physical stores, z set from the service target (not calibrated). Longer lead times need more "
            "safety stock; the forecast-driven policy stays below the static one.",
            "Tutte le serie, negozi fisici, z fissato dal livello di servizio (non calibrato). Tempi di consegna più "
            "lunghi richiedono più scorta di sicurezza; la politica guidata dalla previsione resta sotto quella "
            "statica."))
        st.dataframe(sens[["policy", "lead_time", "service_level", "fill_rate", "avg_inventory_value", "total_cost"]]
                     .assign(policy=lambda d: d["policy"].map(policy_label)), hide_index=True, column_config={
            "policy": tx("Policy", "Politica"),
            "lead_time": tx("Lead time (days)", "Tempo di consegna (giorni)"),
            "service_level": NUM(tx("Target", "Obiettivo"), format="percent"),
            "fill_rate": NUM("Fill rate", format="percent"),
            "avg_inventory_value": NUM(tx("Avg inventory", "Scorta media"), format="euro"),
            "total_cost": NUM(tx("Total cost", "Costo totale"), format="euro")})
