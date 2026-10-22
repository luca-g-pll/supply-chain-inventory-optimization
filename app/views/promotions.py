"""Promotions: what the data shows, how they change the forecasts, and a planner for future ones."""

import io

import charts
import numpy as np
import pandas as pd
import streamlit as st
from common import (
    active_data,
    config,
    csv_download,
    detail_note,
    euro,
    no_data_message,
    plan_calendar_for,
    scenario_badge,
)
from i18n import model_name, tx

from src.ingest import InputError, read_any
from src.promotions import plan_promotion, price_elasticity, read_promo_calendar
from src.series import natural_sorted

NUM = st.column_config.NumberColumn
ALL = "*"

data = active_data()
st.title(tx("Promotions", "Promozioni"))
if data is None:
    no_data_message()
    st.stop()
sc = st.session_state["scenario"]
scenario_badge(data, sc)
cfg = config()
meta = data.meta.get("promotions") or {}
items_tbl = data.t("promo_items")
has_promos = bool(meta.get("source")) and len(items_tbl) > 0

# --------------------------------------------------------------------------- what the data shows
if not has_promos:
    st.info(tx(
        "**No promotions found in this dataset.** The software finds them in two ways: a **promo column** in the "
        "file (yes/no or the discount, e.g. 20%) or the **selling price** of each day (a price clearly below the "
        "usual one = promotion). Upload a file with one of the two to measure their effect. The planner below still "
        "works, with an effect you set yourself.",
        "**Nessuna promozione trovata in questi dati.** Il software le trova in due modi: una **colonna promozione** "
        "nel file (sì/no o lo sconto, es. 20%) oppure il **prezzo di vendita** di ogni giorno (un prezzo chiaramente "
        "sotto quello abituale = promozione). Carica un file con una delle due per misurarne l'effetto. Il "
        "pianificatore qui sotto funziona comunque, con un effetto che imposti tu."))
else:
    source = tx("a promo column in the file", "una colonna promozione nel file") if meta["source"] == "file" else \
        tx(f"prices at least {meta['min_discount']:.0%} below the regular price",
           f"prezzi con sconto ≥ {meta['min_discount']:.0%} rispetto al prezzo normale")
    st.markdown(tx(
        f"#### During a promotion at 20% off, sales grow on average by **{meta['lift_20_global'] - 1:.0%}**; in the "
        f"week after they are **{1 - meta['dip_global']:.0%} below normal**.",
        f"#### Durante una promozione con il 20% di sconto le vendite crescono in media del "
        f"**{meta['lift_20_global'] - 1:.0%}**; nella settimana dopo sono **il {1 - meta['dip_global']:.0%} sotto il "
        f"normale**."))
    st.caption(tx(f"Promotions found from {source}. Whole history, all stores.",
                  f"Promozioni trovate da {source}. Tutto lo storico, tutti i negozi."))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(tx("Promo days", "Giorni in promo"), f"{meta['share_days']:.1%}",
              help=tx("Share of store-item-days with a promotion", "Quota di giorni negozio-articolo in promozione"))
    m2.metric(tx("Units on promo", "Pezzi in promo"), f"{meta['units_share_promo']:.1%}",
              help=tx("Share of the units sold during a promotion", "Quota dei pezzi venduti durante una promozione"))
    m3.metric(tx("At 20% off", "Sconto 20%"), f"{meta['lift_20_global'] - 1:+.0%}",
              help=tx("Average extra sales during a promotion at 20% off",
                      "Vendite in più in media durante una promozione con il 20% di sconto"))
    m4.metric(tx("Week after", "Settimana dopo"), f"{meta['dip_global'] - 1:+.0%}",
              help=tx("Sales in the week after a promotion: customers stocked up",
                      "Vendite nella settimana dopo una promozione: i clienti hanno fatto scorta"))

    tab_items, tab_history, tab_acc = st.tabs([tx("Effect by product", "Effetto per prodotto"),
                                               tx("Promotions in the history", "Promozioni nello storico"),
                                               tx("Forecast accuracy", "Precisione delle previsioni")])
    with tab_items:
        it = items_tbl[items_tbl["item"].isin(sc.items)]
        it_promo = it[it["promo_days"] > 0]
        if it_promo.empty:
            st.info(tx("No product of this scenario was ever on promotion.",
                       "Nessun prodotto di questo scenario è mai stato in promozione."))
        else:
            disc = st.slider(tx("Discount", "Sconto"), 5, 50, 20, 5, format="%d%%", key="pr_disc") / 100
            charts.show(charts.promo_lift_bars(it_promo, disc), key="pr_lift")
        show = it.assign(lift_sel=np.exp(it["beta"] * st.session_state.get("pr_disc", 20) / 100) - 1,
                         dip=it["post_dip"] - 1)
        st.dataframe(show[["item", "promo_days", "avg_discount", "observed_lift", "lift_sel", "dip", "units_share_promo",
                           "own_weight"]].sort_values("promo_days", ascending=False), hide_index=True, column_config={
            "item": tx("Item", "Articolo"),
            "promo_days": NUM(tx("Promo days", "Giorni in promo"), help=tx("Store-item-days", "Giorni negozio-articolo")),
            "avg_discount": NUM(tx("Average discount", "Sconto medio"), format="percent"),
            "observed_lift": NUM(tx("Observed sales ×", "Vendite osservate ×"), format="%.2f",
                                 help=tx("Sales on promotion days ÷ normal level, at the item's own discounts",
                                         "Vendite nei giorni di promozione ÷ livello normale, con gli sconti dell'articolo")),
            "lift_sel": NUM(tx("Extra sales at the chosen discount", "Vendite in più allo sconto scelto"), format="percent"),
            "dip": NUM(tx("Week after", "Settimana dopo"), format="percent"),
            "units_share_promo": NUM(tx("Units sold in promo", "Pezzi venduti in promo"), format="percent"),
            "own_weight": NUM(tx("Own data weight", "Peso dei dati propri"), format="percent",
                              help=tx("Items with few promotion days borrow the average effect of all items",
                                      "Gli articoli con pochi giorni di promozione prendono in prestito l'effetto medio "
                                      "di tutti gli articoli"))})
        st.caption(tx(
            "Effect model: sales × e^(β·discount), β estimated per product against the average of the previous 28 "
            "normal days. Seasonal clearance sales (e.g. Christmas items in January) show almost no extra sales: the "
            "discount follows the end of the season, it does not create demand.",
            "Modello dell'effetto: vendite × e^(β·sconto), β stimato per prodotto rispetto alla media dei 28 giorni "
            "normali precedenti. I saldi di fine stagione (es. articoli natalizi a gennaio) non mostrano quasi vendite "
            "in più: lo sconto segue la fine della stagione, non crea domanda."))
        csv_download(it, tx("Download the effects (CSV)", "Scarica gli effetti (CSV)"), "promo_effects.csv", "dl_pr_items")

    with tab_history:
        ev = data.t("promo_events")
        ev = ev[ev["store"].isin(sc.stores) & ev["item"].isin(sc.items)]
        if ev.empty:
            st.info(tx("No promotion in this scenario.", "Nessuna promozione in questo scenario."))
        else:
            ranked = ev.groupby("item")["units"].sum().sort_values(ascending=False).index.tolist()
            item = st.selectbox(tx("Item", "Articolo"), ranked, key="pr_item")
            det = data.t("detail_actuals")
            if "discount" in det:
                det = det[(det["item"] == item) & det["store"].isin(sc.stores)]
            if len(det):
                daily = det.groupby("date", as_index=False).agg(sales=("sales", "sum"), discount=("discount", "max"))
                charts.show(charts.promo_timeline(daily, item), key="pr_timeline")
                note = detail_note(data)
                if note:
                    st.caption(note)
            e = ev[ev["item"] == item].groupby(["start", "end"], as_index=False).agg(
                stores=("store", "nunique"), days=("days", "max"), discount=("discount", "mean"),
                units=("units", "sum"), baseline=("baseline", "sum"))
            e["lift"] = e["units"] / e["baseline"].where(e["baseline"] > 0) - 1
            st.dataframe(e.sort_values("start", ascending=False), hide_index=True, column_config={
                "start": st.column_config.DateColumn(tx("From", "Dal"), format="DD/MM/YYYY"),
                "end": st.column_config.DateColumn(tx("To", "Al"), format="DD/MM/YYYY"),
                "stores": tx("Stores", "Negozi"), "days": tx("Days", "Giorni"),
                "discount": NUM(tx("Discount", "Sconto"), format="percent"),
                "units": NUM(tx("Units sold", "Pezzi venduti"), format="%.0f"),
                "baseline": NUM(tx("Normal level", "Livello normale"), format="%.0f",
                                help=tx("Units at the average of the previous 28 normal days",
                                        "Pezzi alla media dei 28 giorni normali precedenti")),
                "lift": NUM(tx("Extra sales", "Vendite in più"), format="percent")})

    with tab_acc:
        acc = data.t("promo_accuracy")
        acc = acc[acc["store"].isin(sc.stores) & acc["item"].isin(sc.items)]
        if acc.empty:
            st.info(tx("No forecast to compare in this scenario.", "Nessuna previsione da confrontare in questo scenario."))
        else:
            g = acc.groupby(["model", "day_type"], as_index=False)[["actual", "abs_error", "error"]].sum()
            g["wape"] = g["abs_error"] / g["actual"].where(g["actual"] > 0)
            g["bias"] = g["error"] / g["actual"].where(g["actual"] > 0)
            charts.show(charts.promo_accuracy_bars(g, data.best_model, "seasonal_naive"), key="pr_acc")
            pv = g.pivot(index="model", columns="day_type", values="wape")
            pv = pv.reindex(columns=[c for c in ("normal", "promo", "post_promo") if c in pv.columns])
            pv.index = [model_name(m) for m in pv.index]
            pv.columns = [{"normal": tx("Normal days", "Giorni normali"), "promo": tx("Promotion days", "Giorni di promozione"),
                           "post_promo": tx("Week after", "Settimana dopo")}[c] for c in pv.columns]
            st.dataframe(pv.style.format("{:.1%}"))
            st.caption(tx(
                "Test period, WAPE. The moving average, Holt-Winters and LightGBM know the promotion calendar (it is "
                "decided in advance); the seasonal-naive baseline does not: it copies last week, so it misses the "
                "start of every promotion and repeats its peak the week after.",
                "Periodo di test, WAPE. Media mobile, Holt-Winters e LightGBM conoscono il calendario delle promozioni "
                "(è deciso in anticipo); la regola \"come la settimana scorsa\" no: copia la settimana prima, quindi "
                "manca l'inizio di ogni promozione e ne ripete il picco la settimana dopo."))

# --------------------------------------------------------------------------- planner
st.divider()
st.subheader(tx("Plan a promotion", "Pianifica una promozione"))
st.markdown(tx(
    "Add the promotions you are planning (for example next month's flyer): the software estimates the extra units "
    "to order, **by when** to order them, and whether the promotion pays.",
    "Inserisci le promozioni che stai pianificando (per esempio il volantino del mese prossimo): il software stima "
    "i pezzi in più da ordinare, **entro quando** ordinarli e se la promozione conviene."))

base = data.t("promo_baseline")
master = data.t("item_master").set_index("item")
if base.empty:
    st.info(tx("The planner needs the results of an analysis run with this version of the software: re-run the "
               "analysis on the Data page.",
               "Il pianificatore richiede i risultati di un'analisi fatta con questa versione del software: rifai "
               "l'analisi nella pagina Dati."))
    st.stop()
base = base[base["store"].isin(sc.stores)]
items_sc = [i for i in natural_sorted(sc.items) if i in set(base["item"])]
last_date = pd.Timestamp(meta.get("last_date") or data.t("detail_actuals")["date"].max())
if pd.isna(last_date):
    last_date = pd.Timestamp.today().normalize()
cal = plan_calendar_for(data)
plan_today = cal.today if cal is not None else last_date + pd.Timedelta(days=1)
betas = items_tbl.set_index("item")["beta"] if has_promos else pd.Series(dtype=float)
dips = items_tbl.set_index("item")["post_dip"] if has_promos else pd.Series(dtype=float)

if not has_promos:
    assumed = st.slider(tx("Assumed extra sales at 20% off", "Vendite in più ipotizzate con il 20% di sconto"),
                        0, 200, 50, 5, format="%d%%", key="pr_assumed")
    beta_default, dip_default = np.log(1 + assumed / 100) / 0.20, 0.85
else:
    beta_default, dip_default = float(meta["beta_global"]), float(meta["dip_global"])

with st.expander(tx("Load a promotion calendar from a file (optional)", "Carica un calendario promozioni da file (facoltativo)")):
    st.caption(tx("Columns: item, start, end, discount (store optional). CSV or Excel, Italian or English names.",
                  "Colonne: articolo, inizio, fine, sconto (negozio facoltativo). CSV o Excel, nomi in italiano o inglese."))
    up = st.file_uploader(tx("Promotion calendar", "Calendario promozioni"), type=["csv", "xlsx"], key="pr_upload")

key = f"{data.source}:{data.meta.get('file_name', '')}"
if st.session_state.get("pr_plan_key") != key:
    st.session_state["pr_plan_key"] = key
    top = items_sc[0] if items_sc else ""
    if has_promos and items_sc:
        cand = items_tbl[items_tbl["item"].isin(items_sc)].sort_values("promo_days", ascending=False)
        top = cand["item"].iloc[0] if len(cand) else top
    start = plan_today + pd.Timedelta(days=21)
    st.session_state["pr_plan"] = pd.DataFrame({"item": [top], "store": [ALL], "start": [start.date()],
                                                "end": [(start + pd.Timedelta(days=6)).date()], "discount": [25.0]})


def _read_calendar(file) -> pd.DataFrame:
    cal = read_promo_calendar(read_any(io.BytesIO(file.getvalue()), filename=file.name),
                              cfg.promotions.default_discount)
    return cal.assign(store=cal["store"].replace("", ALL), start=cal["start"].dt.date, end=cal["end"].dt.date,
                      discount=(cal["discount"] * 100).round(1))


if up is not None and st.session_state.get("pr_upload_name") != (up.name, up.size):
    try:
        st.session_state["pr_plan"] = _read_calendar(up)
        st.session_state["pr_upload_name"] = (up.name, up.size)
    except InputError as exc:
        st.error(tx("Could not read the calendar: ", "Impossibile leggere il calendario: ") + str(exc))

store_opts = [ALL, *natural_sorted(sc.stores)]
plan = st.data_editor(
    st.session_state["pr_plan"], num_rows="dynamic", width="stretch", key="pr_editor",
    column_config={
        "item": st.column_config.SelectboxColumn(tx("Item", "Articolo"), options=items_sc, required=True),
        "store": st.column_config.SelectboxColumn(
            tx("Store", "Negozio"), options=store_opts, required=True, default=ALL,
            help=tx(f"{ALL} = every store of the scenario", f"{ALL} = tutti i negozi dello scenario")),
        "start": st.column_config.DateColumn(tx("From", "Dal"), format="DD/MM/YYYY", required=True),
        "end": st.column_config.DateColumn(tx("To", "Al"), format="DD/MM/YYYY", required=True),
        "discount": NUM(tx("Discount %", "Sconto %"), min_value=1, max_value=90, step=1, format="%.0f%%",
                        required=True, default=20),
    })
st.caption(tx(f"Store `{ALL}` = all {len(sc.stores)} stores of the scenario (sidebar). The normal level is the "
              f"average of the last 8 weeks before {last_date:%d/%m/%Y}, without promotions, by weekday.",
              f"Negozio `{ALL}` = tutti i {len(sc.stores)} negozi dello scenario (barra laterale). Il livello normale è "
              f"la media delle ultime 8 settimane prima del {last_date:%d/%m/%Y}, senza promozioni, per giorno della "
              "settimana."))

rows, problems = [], []
for k, r in plan.reset_index(drop=True).iterrows():
    if pd.isna(r.get("item")) or pd.isna(r.get("start")) or pd.isna(r.get("end")):
        continue
    item = str(r["item"])
    start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
    if end < start:
        problems.append(tx(f"Row {k + 1}: the end is before the start.", f"Riga {k + 1}: la fine è prima dell'inizio."))
        continue
    stores = list(sc.stores) if r["store"] == ALL else [str(r["store"])]
    b = base[(base["item"] == item) & base["store"].isin(stores)].groupby("dow")["units"].sum()
    if b.empty:
        problems.append(tx(f"Row {k + 1}: no sales of {item} in these stores.",
                           f"Riga {k + 1}: nessuna vendita di {item} in questi negozi."))
        continue
    d = float(r["discount"]) / 100
    beta = float(betas.get(item, beta_default)) if has_promos else beta_default
    dip = float(dips.get(item, dip_default)) if has_promos else dip_default
    price = float(master.at[item, "unit_price"]) if item in master.index else np.nan
    cost = float(master.at[item, "unit_cost"]) if item in master.index else np.nan
    lt = master.at[item, "lead_time_days"] if item in master.index else None
    lt = int(lt) if lt is not None and not pd.isna(lt) else cfg.inventory.lead_time_days
    res = plan_promotion(b, start, end, d, beta, dip, cfg.promotions.post_promo_days, price, cost)
    rows.append({"row": k, "item": item, "store": r["store"], "start": start, "end": end, "discount": d,
                 "lift": res["lift"] - 1, "baseline": res["baseline_units"], "expected": res["promo_units"],
                 "extra": res["extra_units"], "dip": res["dip_units"],
                 "order_by": start - pd.Timedelta(days=lt), "lead_time": lt,
                 "revenue_normal": res["revenue_normal"], "revenue_promo": res["revenue_promo"],
                 "margin_change": res["margin_change"], "_base": b, "_beta": beta, "_dip": dip})
for p in problems:
    st.warning(p)

if rows:
    out = pd.DataFrame(rows)
    late = out["order_by"] < plan_today
    tot_extra, tot_margin = out["extra"].sum(), out["margin_change"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric(tx("Extra units to order", "Pezzi in più da ordinare"), f"{np.ceil(tot_extra):,.0f}")
    c2.metric(tx("First order by", "Primo ordine entro"), f"{out['order_by'].min():%d/%m/%Y}")
    c3.metric(tx("Margin effect", "Effetto sul margine"), euro(tot_margin),
              help=tx("Extra units × discounted margin − normal units sold at a discount − dip after the promotion",
                      "Pezzi in più × margine scontato − pezzi normali venduti scontati − calo dopo la promozione"))
    st.dataframe(out.drop(columns=["row", "_base", "_beta", "_dip"]), hide_index=True, column_config={
        "item": tx("Item", "Articolo"), "store": tx("Store", "Negozio"),
        "start": st.column_config.DateColumn(tx("From", "Dal"), format="DD/MM/YYYY"),
        "end": st.column_config.DateColumn(tx("To", "Al"), format="DD/MM/YYYY"),
        "discount": NUM(tx("Discount", "Sconto"), format="percent"),
        "lift": NUM(tx("Extra sales", "Vendite in più"), format="percent"),
        "baseline": NUM(tx("Normal units", "Pezzi normali"), format="%.0f"),
        "expected": NUM(tx("Expected units", "Pezzi attesi"), format="%.0f"),
        "extra": NUM(tx("Extra to order", "In più da ordinare"), format="%.0f"),
        "dip": NUM(tx("Fewer units the week after", "Pezzi in meno la settimana dopo"), format="%.0f"),
        "order_by": st.column_config.DateColumn(tx("Order by", "Ordina entro"), format="DD/MM/YYYY",
                                                help=tx("Start of the promotion − supplier lead time",
                                                        "Inizio della promozione − tempo di consegna del fornitore")),
        "lead_time": tx("Lead time (d)", "Consegna (gg)"),
        "revenue_normal": NUM(tx("Revenue without promo", "Fatturato senza promo"), format="euro"),
        "revenue_promo": NUM(tx("Revenue with promo", "Fatturato con promo"), format="euro"),
        "margin_change": NUM(tx("Margin change", "Variazione margine"), format="euro")})
    if late.any():
        st.warning(tx(f"{int(late.sum())} promotion(s) needed the order before today ({plan_today:%d/%m/%Y}): "
                      "check that the supplier can still deliver in time, or move the promotion.",
                      f"{int(late.sum())} promozione/i richiedevano l'ordine prima di oggi ({plan_today:%d/%m/%Y}): "
                      "verifica che il fornitore possa ancora consegnare in tempo, o sposta la promozione."))
    st.caption(tx("The planned promotions are also used by the **Orders** page: their extra units enter the order "
                  "calendar.", "Le promozioni pianificate sono usate anche dalla pagina **Ordini**: i loro pezzi in più "
                  "entrano nel calendario ordini."))
    csv_download(out.drop(columns=["row", "_base", "_beta", "_dip"]),
                 tx("Download the plan (CSV)", "Scarica il piano (CSV)"), "promotion_plan.csv", "dl_pr_plan")

    pick = st.selectbox(tx("Day-by-day view", "Vista giorno per giorno"), range(len(out)),
                        format_func=lambda j: f"{out.at[j, 'item']} · {out.at[j, 'start']:%d/%m} → {out.at[j, 'end']:%d/%m}",
                        key="pr_pick")
    r = out.iloc[pick]
    days = pd.date_range(r["start"] - pd.Timedelta(days=7), r["end"] + pd.Timedelta(days=cfg.promotions.post_promo_days))
    normal = np.array([r["_base"].get(d, 0.0) for d in days.dayofweek])
    in_promo = (days >= r["start"]) & (days <= r["end"])
    after = (days > r["end"])
    expected = np.where(in_promo, normal * np.exp(r["_beta"] * r["discount"]), np.where(after, normal * r["_dip"], normal))
    charts.show(charts.promo_plan_profile(pd.DataFrame({"date": days, "normal": normal, "expected": expected})),
                key="pr_profile")
    st.caption(tx(
        "Order the extra units together with the last normal order before the promotion. The forecast-driven policy "
        "(P2) does this by itself when the promotion is in the calendar: its reorder point rises with the forecast.",
        "Ordina i pezzi in più insieme all'ultimo ordine normale prima della promozione. La politica guidata dalla "
        "previsione (P2) lo fa da sola quando la promozione è nel calendario: il suo punto di riordino sale con la "
        "previsione."))

# --------------------------------------------------------------------------- price change
with st.expander(tx("What if I change the price for good?", "E se cambio il prezzo in modo permanente?")):
    if not items_sc:
        st.info(tx("No product in this scenario.", "Nessun prodotto in questo scenario."))
    else:
        p1, p2 = st.columns(2)
        item = p1.selectbox(tx("Item", "Articolo"), items_sc, key="pc_item")
        change = p2.slider(tx("Price change", "Variazione di prezzo"), -30, 30, 5, 1, format="%+d%%", key="pc_change") / 100
        beta = float(betas.get(item, beta_default)) if has_promos else beta_default
        # a permanent change moves demand less than a promotion (no flyer, no urgency): half the promo response
        factor = price_elasticity(beta / 2, change)
        week = float(base[base["item"] == item]["units"].sum())
        price = float(master.at[item, "unit_price"]) if item in master.index else np.nan
        cost = float(master.at[item, "unit_cost"]) if item in master.index else np.nan
        new_units = week * factor
        margin_old = week * (price - cost)
        margin_new = new_units * (price * (1 + change) - cost)
        q1, q2, q3 = st.columns(3)
        q1.metric(tx("Units per week", "Pezzi a settimana"), f"{new_units:,.0f}", f"{factor - 1:+.1%}")
        q2.metric(tx("Revenue per week", "Fatturato a settimana"), euro(new_units * price * (1 + change)),
                  f"{(new_units * price * (1 + change)) / (week * price) - 1:+.1%}" if week * price > 0 else None)
        q3.metric(tx("Margin per week", "Margine a settimana"), euro(margin_new),
                  f"{margin_new / margin_old - 1:+.1%}" if margin_old > 0 else None)
        st.caption(tx(
            "Rough estimate: the price sensitivity is derived from the product's promotions and halved, because a "
            "permanent change has no flyer and no urgency. Without promotions in the data an assumed sensitivity is "
            "used. Use it to compare options, not as a promise.",
            "Stima indicativa: la sensibilità al prezzo è ricavata dalle promozioni del prodotto e dimezzata, perché "
            "un cambio permanente non ha volantino né urgenza. Senza promozioni nei dati si usa una sensibilità "
            "ipotizzata. Usala per confrontare opzioni, non come una promessa."))
