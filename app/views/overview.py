"""Overview: headline, KPI tiles, policy comparison (UC-07, UC-10)."""

import charts
import streamlit as st
from common import (
    active_data,
    channel_label,
    csv_download,
    euro,
    policy_kpis,
    policy_label,
    run_sample_analysis,
    scenario_badge,
    scenario_leaderboard,
    scenario_summary,
)
from i18n import model_name, tx
from website import track

data = active_data()


def landing() -> None:
    """First visit: what the app does, the proof, and one obvious next step."""
    st.title(tx("Order less, run out less.", "Ordina meno, resta meno senza merce."))
    st.markdown(tx(
        "#### Forecast demand for every product in every store, and know **what to order today** — with the proof, "
        "on your own past sales, of how much stock and cost you save.",
        "#### Prevedi la domanda di ogni prodotto in ogni negozio e sappi **cosa ordinare oggi** — con la prova, sulle "
        "tue vendite passate, di quanta scorta e quanti costi risparmi."))
    c1, c2, c3 = st.columns(3)
    c1.metric(tx("Less stock", "Scorta in meno"), "−19%",
              help=tx("Sample retail chain: forecast-driven reorder vs the same order every week, same 95% service",
                      "Catena di esempio: riordino guidato dalla previsione contro lo stesso ordine ogni settimana, "
                      "stesso servizio al 95%"))
    c2.metric(tx("Forecast error", "Errore di previsione"), "34% vs 50%",
              help=tx("Best model vs \"same as last week\", sample retail chain, test period",
                      "Modello migliore contro \"come la settimana scorsa\", catena di esempio, periodo di test"))
    c3.metric(tx("Time to the first result", "Tempo per il primo risultato"), "≈ 20 s",
              help=tx("Sample data, on a laptop", "Dati di esempio, su un portatile"))
    st.write("")
    b1, b2, b3 = st.columns([2.6, 2.2, 1.6])
    if b1.button(tx("Try it now with sample data", "Prova subito con i dati di esempio"), type="primary",
                 icon=":material/play_arrow:", width="stretch", key="cta_sample"):
        track("cta_click", target="sample")
        if run_sample_analysis():
            st.rerun()
    b2.page_link("views/data.py", label=tx("Upload your sales history", "Carica il tuo storico vendite"),
                 icon=":material/upload_file:", width="stretch")
    b3.page_link("views/faq.py", label=tx("Questions", "Domande frequenti"), icon=":material/help:", width="stretch")
    st.caption(tx("Sample data: an invented Italian retail chain, 8 stores × 32 products × 3 years. "
                  "Your files stay in memory for your session only · no tracking cookies · open source.",
                  "Dati di esempio: una catena italiana inventata, 8 negozi × 32 prodotti × 3 anni. "
                  "I tuoi file restano in memoria solo per la tua sessione · nessun cookie di tracciamento · codice "
                  "aperto."))
    st.divider()
    st.markdown(tx("""
**How it works** — 1. upload daily sales (date, product, quantity) · 2. the software tests four forecasting models
and three reorder rules on your past · 3. you get the reorder points, today's orders by supplier and what they save.
""", """
**Come funziona** — 1. carichi le vendite giornaliere (data, prodotto, quantità) · 2. il software prova quattro
modelli di previsione e tre regole di riordino sul tuo passato · 3. ottieni i punti di riordino, gli ordini di oggi per
fornitore e quanto fanno risparmiare.
"""))


if data is None:
    landing()
    st.stop()
st.title(tx("Demand forecasting & inventory optimization", "Previsione della domanda e ottimizzazione delle scorte"))

sc = st.session_state["scenario"]
scenario_badge(data, sc)
summary = scenario_summary(data, sc)
k = policy_kpis(summary)
if k.empty:
    st.info(tx("No series in this scenario.", "Nessuna serie in questo scenario."))
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
gain = 1 - wape_best / wape_base
st.markdown(tx(
    f"#### The forecast-driven policy needs **{abs(inv_red):.0%} {'less' if inv_red >= 0 else 'more'} inventory** "
    f"than ordering the same quantity every week, at the same **{target:.0%} fill rate**. The best forecast "
    f"({model_name(best)}) makes **{gain:.0%} less error** than the seasonal-naive baseline.",
    f"#### La politica guidata dalla previsione richiede **il {abs(inv_red):.0%} di scorta "
    f"{'in meno' if inv_red >= 0 else 'in più'}** rispetto all'ordine fisso settimanale, con lo stesso "
    f"**fill rate del {target:.0%}**. La previsione migliore ({model_name(best)}) sbaglia **il {gain:.0%} in meno** "
    f"della regola \"come la settimana scorsa\"."))

c1, c2, c3, c4 = st.columns(4)
c1.metric(tx("Forecast error (WAPE)", "Errore di previsione (WAPE)"), f"{wape_best:.1%}",
          f"{wape_best / wape_base - 1:+.0%} " + tx("vs seasonal naive", "vs naive stagionale"),
          delta_color="inverse", border=True,
          help=tx("Σ|actual − forecast| / Σ actual on the test period.",
                  "Σ|venduto − previsto| / Σ venduto nel periodo di test."))
c2.metric(tx("Average inventory · P2", "Scorta media · P2"), euro(p2["avg_inventory_value"]),
          f"{-inv_red:+.0%} vs P0", delta_color="inverse", border=True,
          help=tx("Average value of the stock on hand (unit cost × units).",
                  "Valore medio della merce a magazzino (costo unitario × pezzi)."))
c3.metric(tx("Total cost · P2", "Costo totale · P2"), euro(p2["total_cost"]), f"{-cost_red:+.0%} vs P0",
          delta_color="inverse", border=True,
          help=tx("Ordering + holding + lost margin (+ backorder penalty online) over the test period.",
                  "Ordini + mantenimento + margine perso (+ penale per attese online) nel periodo di test."))
c4.metric(tx("Fill rate · P2", "Fill rate · P2"), f"{p2['fill_rate']:.1%}", tx(f"target {target:.0%}",
          f"obiettivo {target:.0%}"), delta_color="off", border=True,
          help=tx("Share of demand served from stock. Every policy is calibrated per series to reach the target.",
                  "Quota di domanda servita dal magazzino. Ogni politica è calibrata per serie per raggiungere "
                  "l'obiettivo."))

left, right = st.columns(2)
with left:
    charts.show(charts.inventory_at_target(k, target), key="ov_inv")
with right:
    charts.show(charts.cost_breakdown(k), key="ov_cost")

# ---- table view
st.subheader(tx("Policy comparison at the same fill rate", "Confronto tra politiche a parità di fill rate"))
cols = {
    "fill_rate": "Fill rate",
    "avg_inventory_value": tx("Avg inventory (€)", "Scorta media (€)"),
    "avg_inventory_units": tx("Avg inventory (units)", "Scorta media (pezzi)"),
    "days_of_supply": tx("Days of supply", "Giorni di copertura"),
    "turns": tx("Turns / year", "Rotazioni / anno"),
    "stockout_days": tx("Stock-out series-days", "Giorni-serie in rottura"),
    "n_orders": tx("Orders", "Ordini"),
    "ordering_cost": tx("Ordering (€)", "Costo ordini (€)"),
    "holding_cost": tx("Holding (€)", "Mantenimento (€)"),
    "shortage_cost": tx("Shortage (€)", "Rotture (€)"),
    "total_cost": tx("Total cost (€)", "Costo totale (€)"),
}
policy_col = tx("Policy", "Politica")
table = k.assign(policy=k["policy"].map(policy_label))[["policy", *cols]].rename(columns={"policy": policy_col, **cols})
money = [c for c in table.columns if "€" in c]
config = {
    "Fill rate": st.column_config.NumberColumn(format="percent"),
    **{c: st.column_config.NumberColumn(format="euro") for c in money},
    cols["avg_inventory_units"]: st.column_config.NumberColumn(format="%.0f"),
    cols["days_of_supply"]: st.column_config.NumberColumn(format="%.1f"),
    cols["turns"]: st.column_config.NumberColumn(format="%.1f"),
}
st.dataframe(table, hide_index=True, column_config=config)
csv_download(table, tx("Download the comparison (CSV)", "Scarica il confronto (CSV)"), "policy_comparison.csv", "dl_ov")

with st.expander(tx("Same policies with the safety factor set in advance (z from the 95% cycle service level)",
                    "Le stesse politiche con il fattore di sicurezza fissato in anticipo (z dal 95% di servizio "
                    "per ciclo)")):
    kf = policy_kpis(summary, variant="fixed")
    tf = kf.assign(policy=kf["policy"].map(policy_label))[["policy", *cols]].rename(
        columns={"policy": policy_col, **cols})
    st.dataframe(tf, hide_index=True, column_config=config)
    st.caption(tx(
        "With z = 1.645 set in advance the (s, Q) policies over-deliver (fill rate far above 95%): a 95% *cycle* "
        "service level is not a 95% *fill rate*. The calibrated view above compares every policy at the same fill "
        "rate instead.",
        "Con z = 1,645 fissato in anticipo le politiche (s, Q) servono più del necessario (fill rate ben sopra il 95%): "
        "un 95% di servizio *per ciclo* non è un 95% di *fill rate*. La vista calibrata qui sopra confronta invece "
        "tutte le politiche allo stesso fill rate."))

b1, b2 = st.columns(2)
with b1:
    st.subheader(tx("By ABC class", "Per classe ABC"))
    by_abc = policy_kpis(summary, by=("abc", "policy"))
    pv = by_abc.pivot(index="abc", columns="policy", values="avg_inventory_value")
    pv = pv[[p for p in ("P0_naive", "P1_static_sq", "P2_forecast_sq") if p in pv.columns]]
    pv.columns = [policy_label(p) for p in pv.columns]
    pv.index.name = tx("Class", "Classe")
    st.dataframe(pv.style.format("€{:,.0f}"))
    st.caption(tx("Average inventory value at the target fill rate.",
                  "Valore medio della scorta al fill rate obiettivo."))
with b2:
    st.subheader(tx("By channel", "Per canale"))
    by_ch = policy_kpis(summary, by=("channel", "policy"))
    pc = by_ch.pivot(index="channel", columns="policy", values="total_cost")
    pc = pc[[p for p in ("P0_naive", "P1_static_sq", "P2_forecast_sq") if p in pc.columns]]
    pc.columns = [policy_label(p) for p in pc.columns]
    pc.index = [channel_label(c) for c in pc.index]
    pc.index.name = tx("Channel", "Canale")
    st.dataframe(pc.style.format("€{:,.0f}"))
    st.caption(tx("Total cost. Mark stores as online in the sidebar to compare channels.",
                  "Costo totale. Segna dei negozi come online nella barra laterale per confrontare i canali."))

with st.expander(tx("How to read this page", "Come leggere questa pagina")):
    st.markdown(tx(f"""
- **P0 · naive weekly**: every Monday order the average weekly demand of the validation period.
- **P1 · static (s, Q)**: reorder point `s = d̄·L + z·σ_d·√L` from last period's statistics, lot = EOQ.
- **P2 · forecast-driven (s, Q)**: reorder point `s_t = forecast demand over the lead time + z·σ_forecast error`,
  updated every day from the best model's weekly forecasts; same EOQ lot.
- Every series is simulated day by day over the test period; the knob of each policy (weekly quantity multiplier
  for P0, safety factor z for P1/P2) is **calibrated per series** so that all policies reach a {target:.0%} fill
  rate. Only then is the stock compared.
- Costs, prices and lead times are assumptions (see the README); change them in the **Inventory** what-if.
""", f"""
- **P0 · ordine fisso settimanale**: ogni lunedì si ordina la domanda settimanale media del periodo di validazione.
- **P1 · statica (s, Q)**: punto di riordino `s = d̄·L + z·σ_d·√L` dalle statistiche del periodo precedente,
  lotto = EOQ.
- **P2 · guidata dalla previsione (s, Q)**: punto di riordino `s_t = domanda prevista nel tempo di consegna +
  z·σ dell'errore di previsione`, aggiornato ogni giorno dalle previsioni settimanali del modello migliore; stesso
  lotto EOQ.
- Ogni serie è simulata giorno per giorno nel periodo di test; il parametro di ogni politica (moltiplicatore della
  quantità settimanale per P0, fattore di sicurezza z per P1/P2) è **calibrato per serie** in modo che tutte
  raggiungano un fill rate del {target:.0%}. Solo allora si confronta la scorta.
- Costi, prezzi e tempi di consegna sono ipotesi (vedi il README); puoi cambiarli nel what-if della pagina
  **Scorte**.
"""))

st.divider()
n1, n2, n3 = st.columns(3)
n1.page_link("views/orders.py", label=tx("Next: what to order today", "Prossimo passo: cosa ordinare oggi"),
             icon=":material/shopping_cart:")
n2.page_link("views/inventory.py", label=tx("Change the assumptions (what-if)", "Cambia le ipotesi (what-if)"),
             icon=":material/tune:")
n3.page_link("views/data.py", label=tx("Use your own data", "Usa i tuoi dati"), icon=":material/upload_file:")
