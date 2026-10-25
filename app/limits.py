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
     "Missing prices, costs and lead times are replaced by assumptions (order €50, holding 25% a year, cost 60% of "
     "the price, 7 days): euro results are estimates until your data provide them.",
     "Prezzi, costi e tempi di consegna mancanti sono sostituiti da ipotesi (ordine €50, mantenimento 25% annuo, "
     "costo 60% del prezzo, 7 giorni): i risultati in euro sono stime finché non li fornisci."),
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
    ("Session", "Sessione",
     "Uploaded files and results live only in this browser session: reloading the page clears them. One user, no "
     "login.",
     "File caricati e risultati vivono solo in questa sessione del browser: ricaricare la pagina li cancella. Un "
     "utente, nessun login."),
]

# profile notes that are information, not warnings
_INFO = (r"History: ", r"Train ", r"Yearly seasonality is modelled", r"Promotions (read|found)")


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


def limits_panel(data) -> None:
    """The sidebar panel, under the page list."""
    warnings = data_warnings(data)
    label = tx("Warnings and limits", "Avvisi e limiti") + (f" · {len(warnings)}" if warnings else "")
    with st.sidebar.expander(label, icon=":material/warning:"):
        if warnings:
            st.markdown(tx("**On your data**", "**Sui tuoi dati**"))
            st.markdown("\n".join(f"- {w}" for w in warnings))
        st.markdown(tx("**Limits of the software**", "**Limiti del software**"))
        for en_t, it_t, en, it in LIMITS:
            st.caption(f"**{tx(en_t, it_t)}** — {tx(en, it)}")
        st.caption(tx("Full list: README §12 and docs/08_methodology.md §7.",
                      "Elenco completo: README §12 e docs/08_methodology.md §7."))
