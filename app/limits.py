"""Sidebar panel "Warnings and limits": what to keep in mind when using the results.

Two parts:

* **warnings on the active data** — built from the analysis of the dataset in use
  (assumed prices and costs, short history, models skipped, partial daily detail,
  data moved to today's calendar …);
* **limits of the software** — the fixed list below. It is the single source for the
  dashboard: update it, with the README "Limitations" section, whenever a feature
  adds, changes or removes a limit.
"""

from __future__ import annotations

import re

import streamlit as st
from i18n import translate_message, tx

# (topic EN, topic IT, text EN, text IT)
LIMITS: list[tuple[str, str, str, str]] = [
    ("Size", "Dimensioni",
     "The dashboard keeps every series in memory: up to 10,000 store-item series (≈ 1 minute). Larger files run in "
     "batch mode from the command line (python -m src.partition: 60,000 series ≈ 4.5 min, 2 million ≈ 2.5 h); "
     "their results are Parquet / CSV files, not dashboard pages.",
     "La dashboard tiene tutte le serie in memoria: fino a 10.000 serie negozio-articolo (≈ 1 minuto). File più "
     "grandi si elaborano a lotti da riga di comando (python -m src.partition: 60.000 serie ≈ 4,5 min, 2 milioni ≈ "
     "2,5 ore); i risultati sono file Parquet / CSV, non pagine della dashboard."),
    ("Data", "Dati",
     "Daily sales only, at least 8 weeks. Sales are taken as demand: days with empty shelves look like days without "
     "customers and lower the forecasts.",
     "Solo vendite giornaliere, almeno 8 settimane. Le vendite sono prese come domanda: i giorni con lo scaffale "
     "vuoto sembrano giorni senza clienti e abbassano le previsioni."),
    ("Costs and prices", "Costi e prezzi",
     "The cost of an order and the cost of holding stock never come from the sales file: they are assumptions "
     "(€50 per order, 25% of the value a year) unless you type yours on the Data page. Missing prices, unit costs "
     "and lead times are assumed too. Every euro saving is an estimate built on these values, not an accounting "
     "figure.",
     "Il costo di un ordine e il costo di tenere la merce non arrivano mai dal file vendite: sono ipotesi (€50 per "
     "ordine, 25% del valore all'anno) a meno che tu non inserisca i tuoi nella pagina Dati. Anche prezzi, costi "
     "unitari e tempi di consegna mancanti sono ipotizzati. Ogni risparmio in euro è una stima costruita su questi "
     "valori, non un dato contabile."),
    ("Forecasts", "Previsioni",
     "No outside factors (weather, local events, competitors). New products with a few weeks of history and items "
     "sold rarely are forecast poorly. Holt-Winters has no yearly season and is skipped on large datasets.",
     "Nessun fattore esterno (meteo, eventi locali, concorrenti). I prodotti nuovi con poche settimane di storico e "
     "gli articoli venduti di rado sono previsti male. Holt-Winters non ha stagionalità annuale ed è escluso sui "
     "dataset grandi."),
    ("Promotions", "Promozioni",
     "One effect per product, the same in every store; no cannibalisation between similar products; the effect of a "
     "permanent price change is only indicative.",
     "Un solo effetto per prodotto, uguale in tutti i negozi; nessuna cannibalizzazione tra prodotti simili; "
     "l'effetto di un cambio di prezzo permanente è solo indicativo."),
    ("Policy comparison", "Confronto politiche",
     "The calibration to the 95% fill rate uses the same period it evaluates: it compares the policies on equal "
     "terms, it is not a promise of the future service level.",
     "La calibrazione al 95% di fill rate usa lo stesso periodo che valuta: confronta le politiche alla pari, non è "
     "una promessa del livello di servizio futuro."),
    ("Orders", "Ordini",
     "Orders are proposed, not sent (no link to an ERP or to suppliers). Without your stock file the stock is "
     "estimated. Old data are moved to today by whole weeks: seasons move with them.",
     "Gli ordini sono proposti, non inviati (nessun collegamento a gestionali o fornitori). Senza il tuo file "
     "giacenze la giacenza è stimata. I dati vecchi sono portati a oggi a settimane intere: le stagioni si spostano "
     "con loro."),
    ("Supplier terms", "Vincoli fornitori",
     "One quantity-discount threshold per supplier; warehouse space in units, not volume; one weekly budget for all "
     "suppliers.",
     "Una sola soglia di sconto per fornitore; spazio a magazzino in pezzi, non in volume; un solo budget settimanale "
     "per tutti i fornitori."),
    ("Stores", "Negozi",
     "Each store-item is planned on its own: no transfers between stores, no central warehouse. The online channel "
     "reuses the history of a physical store with different service rules.",
     "Ogni negozio-articolo è pianificato da solo: niente trasferimenti tra negozi, niente magazzino centrale. Il "
     "canale online riusa lo storico di un negozio fisico con regole di servizio diverse."),
    ("Website", "Sito web",
     "Pages are drawn in the browser: titles and descriptions for search engines are written by a script, a missing "
     "page shows a 404 page but answers with status 200, and robots.txt / sitemap.xml reach the site root only "
     "through the web server (docs/12_deployment.md). Privacy policy and terms are templates to review.",
     "Le pagine sono disegnate nel browser: titoli e descrizioni per i motori di ricerca sono scritti da uno script, "
     "una pagina inesistente mostra una pagina 404 ma risponde con stato 200, e robots.txt / sitemap.xml arrivano "
     "alla radice del sito solo tramite il server web (docs/12_deployment.md). Informativa privacy e condizioni sono "
     "modelli da far rivedere."),
    ("Session", "Sessione",
     "Uploaded files and results live only in this browser session: reloading the page clears them. One user, no "
     "login.",
     "File caricati e risultati vivono solo in questa sessione del browser: ricaricare la pagina li cancella. Un "
     "utente, nessun login."),
]

# profile notes that are information, not warnings; item-level assumptions are in the euro section
_INFO = (r"History: ", r"Train ", r"Yearly seasonality is modelled", r"Promotions (read|found)",
         r"Selling price missing", r"Unit cost missing", r"Lead time missing")


def data_warnings(data) -> list[str]:
    """Warnings about the dataset in use, in the current language."""
    if data is None:
        return []
    out: list[str] = []
    meta = data.meta
    val = meta.get("validation") or {}
    prof = meta.get("profile") or {}
    for msg in list(val.get("assumptions", [])) + list(val.get("warnings", [])) + list(prof.get("notes", [])):
        if not any(re.match(p, msg) for p in _INFO):
            out.append(translate_message(msg))
    if prof.get("low_confidence"):
        out.append(tx("Short history: results have low confidence.", "Storico corto: risultati poco affidabili."))
    d = meta.get("detail") or {}
    if d and d.get("n_detail", 0) < d.get("n_series", 0):
        out.append(tx(f"Daily charts only for the {d['n_detail']:,} top-selling series of {d['n_series']:,}.",
                      f"Grafici giornalieri solo per le {d['n_detail']:,} serie più vendute su {d['n_series']:,}."))
    if data.source == "demo":
        out.append(tx("Demo data (Kaggle): prices and costs are synthetic.",
                      "Dati demo (Kaggle): prezzi e costi sono sintetici."))
    from common import plan_calendar_for

    cal = plan_calendar_for(data)
    if cal is not None and cal.shift_days:
        out.append(tx(f"Your sales end on {cal.data_end:%d/%m/%Y}: the Orders page moves them forward by "
                      f"{cal.shift_days // 7} weeks to plan on today's calendar.",
                      f"Le tue vendite finiscono il {cal.data_end:%d/%m/%Y}: la pagina Ordini le sposta avanti di "
                      f"{cal.shift_days // 7} settimane per pianificare sul calendario di oggi."))
    return list(dict.fromkeys(out))


def euro_checks(data) -> tuple[list[str], str | None]:
    """Cost values behind the euro results (what to confirm) and how much the saving moves with them."""
    import numpy as np
    from common import euro

    if data is None:
        return [], None
    lines = []
    ci = data.meta.get("cost_inputs") or {}

    def origin(key: str) -> str:
        d = ci.get(key, {})
        return tx("default, to confirm", "valore predefinito, da confermare") if d.get("default", True) else \
            tx("set by you", "impostato da te")

    if ci:
        lines.append(tx(f"Cost of one order: €{ci['order_cost']['value']:,.0f} ({origin('order_cost')})",
                        f"Costo di un ordine: €{ci['order_cost']['value']:,.0f} ({origin('order_cost')})"))
        lines.append(tx(f"Holding cost: {ci['holding_rate']['value']:.0%} of the stock value a year, online "
                        f"{ci['online_holding_rate']['value']:.0%} ({origin('holding_rate')})",
                        f"Costo di mantenimento: {ci['holding_rate']['value']:.0%} del valore della scorta all'anno, "
                        f"online {ci['online_holding_rate']['value']:.0%} ({origin('holding_rate')})"))
    master = data.t("item_master")
    if len(master):
        n = len(master)
        n_price = int((master.get("price_source") == "assumed").sum())
        n_cost = int((master.get("cost_source") == "derived").sum())
        n_lt = int(master["lead_time_days"].isna().sum()) if "lead_time_days" in master else 0
        ratio = (ci.get("cost_to_price_ratio") or {}).get("value", 0.6)
        lead = (ci.get("lead_time_days") or {}).get("value", 7)
        if n_price:
            lines.append(tx(f"Selling price invented for {n_price} of {n} items",
                            f"Prezzo di vendita inventato per {n_price} articoli su {n}"))
        if n_cost:
            lines.append(tx(f"Unit cost = {ratio:.0%} of the price for {n_cost} of {n} items",
                            f"Costo unitario = {ratio:.0%} del prezzo per {n_cost} articoli su {n}"))
        if n_lt:
            lines.append(tx(f"Lead time {lead:.0f} days assumed for {n_lt} of {n} items",
                            f"Tempo di consegna di {lead:.0f} giorni ipotizzato per {n_lt} articoli su {n}"))
    # how far the saving moves when the order and holding costs are between half and 1.5 times the values used
    s = data.t("policy_summary")
    spread = None
    needed = {"policy", "variant", "channel", "ordering_cost", "holding_cost", "shortage_cost"}
    if len(s) and needed <= set(s.columns):
        g = s[(s["variant"] == "calibrated") & (s["channel"] == "physical")].groupby("policy")[
            ["ordering_cost", "holding_cost", "shortage_cost"]].sum()
        if {"P1_static_sq", "P2_forecast_sq"} <= set(g.index):
            savings = []
            for fo in (0.5, 1.0, 1.5):
                for fh in (0.5, 1.0, 1.5):
                    tot = g["ordering_cost"] * fo + g["holding_cost"] * fh + g["shortage_cost"]
                    savings.append(tot["P1_static_sq"] - tot["P2_forecast_sq"])
            mid = savings[4]
            spread = tx(
                f"Saving of the forecast-driven policy vs the static one in the test period: {euro(mid)} with these "
                f"values, between {euro(min(savings))} and {euro(max(savings))} if the order and holding costs are "
                "anywhere from half to 1.5 times them (same orders).",
                f"Risparmio della politica guidata dalla previsione rispetto a quella statica nel periodo di test: "
                f"{euro(mid)} con questi valori, tra {euro(min(savings))} e {euro(max(savings))} se i costi di ordine "
                "e di mantenimento sono tra la metà e una volta e mezza (stessi ordini).")
            if not np.isfinite(mid):
                spread = None
    return lines, spread


def limits_panel(data) -> None:
    """The sidebar panel, under the page list."""
    warnings = data_warnings(data)
    euro_lines, spread = euro_checks(data)
    n = len(warnings) + (1 if euro_lines else 0)
    label = tx("Warnings and limits", "Avvisi e limiti") + (f" · {n}" if n else "")
    with st.sidebar.expander(label, icon=":material/warning:"):
        if euro_lines:
            st.markdown(tx("**:orange[Euro results: to verify]**", "**:orange[Risultati in euro: da verificare]**"))
            st.caption(tx("Savings and costs are computed with these values. Confirm them before showing the "
                          "results to anyone:",
                          "Risparmi e costi sono calcolati con questi valori. Confermali prima di mostrare i "
                          "risultati a qualcuno:"))
            st.markdown("\n".join(f"- {x}" for x in euro_lines))
            if spread:
                st.caption(spread)
            st.caption(tx("Set your order and holding costs on the **Data** page (section 2, cost assumptions) and upload real unit "
                          "costs (item master) to turn the estimates into your numbers.",
                          "Inserisci i tuoi costi di ordine e di mantenimento nella pagina **Dati** (sezione 2, ipotesi di costo) e carica "
                          "i costi unitari reali (anagrafica articoli) per trasformare le stime nei tuoi numeri."))
        if warnings:
            st.markdown(tx("**On your data**", "**Sui tuoi dati**"))
            st.markdown("\n".join(f"- {w}" for w in warnings))
        st.markdown(tx("**Limits of the software**", "**Limiti del software**"))
        for en_t, it_t, en, it in LIMITS:
            st.caption(f"**{tx(en_t, it_t)}** — {tx(en, it)}")
        st.caption(tx("Full list: README §12 and docs/08_methodology.md §7.",
                      "Elenco completo: README §12 e docs/08_methodology.md §7."))
