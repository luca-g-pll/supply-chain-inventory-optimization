"""English / Italian texts for the dashboard.

Every visible string goes through ``tx(english, italian)``: the language is the
one chosen with the switch at the top of the sidebar (``st.session_state["lang"]``).
Messages produced by the analysis package (data checks, period notes) are in
English and are translated here with ``translate_message``.
"""

from __future__ import annotations

import re

import streamlit as st

LANGS = {"en": "🇬🇧 English", "it": "🇮🇹 Italiano"}


def lang() -> str:
    return st.session_state.get("lang", "en") or "en"


def is_it() -> bool:
    return lang() == "it"


def tx(en: str, it: str) -> str:
    """The text in the current language."""
    return it if is_it() else en


def language_switch() -> None:
    """EN / IT switch, at the top of the sidebar on every page."""
    st.session_state.setdefault("lang", "en")
    st.session_state.setdefault("_lang_switch", st.session_state["lang"])

    def _changed() -> None:
        # clicking the active option deselects it: keep the current language instead
        if st.session_state["_lang_switch"] is None:
            st.session_state["_lang_switch"] = st.session_state["lang"]
        st.session_state["lang"] = st.session_state["_lang_switch"]

    with st.sidebar:
        st.segmented_control("Language · Lingua", list(LANGS), format_func=LANGS.get, key="_lang_switch",
                             on_change=_changed, label_visibility="collapsed", width="stretch")


# --------------------------------------------------------------------------- shared vocabulary
def policy_name(p: str, short: bool = False) -> str:
    names = {
        "P0_naive": ("P0 · naive weekly", "P0 · ordine fisso settimanale"),
        "P1_static_sq": ("P1 · static (s, Q)", "P1 · statica (s, Q)"),
        "P2_forecast_sq": ("P2 · forecast-driven", "P2 · guidata dalla previsione"),
    }
    short_names = {
        "P0_naive": ("P0 naive weekly", "P0 ordine fisso"),
        "P1_static_sq": ("P1 static (s, Q)", "P1 statica (s, Q)"),
        "P2_forecast_sq": ("P2 forecast-driven", "P2 da previsione"),
    }
    en, it = (short_names if short else names).get(p, (p, p))
    return tx(en, it)


def model_name(m: str) -> str:
    names = {"seasonal_naive": ("Seasonal naive", "Naive stagionale"),
             "moving_average": ("Moving average", "Media mobile"),
             "holt_winters": ("Holt-Winters", "Holt-Winters"), "lightgbm": ("LightGBM", "LightGBM")}
    en, it = names.get(m, (m, m))
    return tx(en, it)


def channel_name(c: str) -> str:
    return {"physical": tx("physical", "fisico"), "online": tx("online", "online")}.get(c, c)


def scenario_label(scenario) -> str:
    s, i, o = len(scenario.stores), len(scenario.items), len(scenario.online_stores)
    if is_it():
        text = f"{s} negozi{'o' if s == 1 else ''} · {i} prodott{'o' if i == 1 else 'i'}"
        return text + (f" · {o} online" if o else "")
    text = f"{s} store{'s' if s != 1 else ''} · {i} product{'s' if i != 1 else ''}"
    return text + (f" · {o} online" if o else "")


# --------------------------------------------------------------------------- messages from src/
_PATTERNS = [
    (r"No store column: all rows are treated as one location \(store = ALL\)\.",
     "Nessuna colonna negozio: tutte le righe sono trattate come un unico punto vendita (store = ALL)."),
    (r"Selling price missing for (\d+) item\(s\): synthetic prices \(€(\d+)–€(\d+), seed (\d+)\) were assumed\.",
     r"Prezzo di vendita mancante per \1 articolo/i: sono stati ipotizzati prezzi sintetici (€\2–€\3, seed \4)."),
    (r"Unit cost missing for (\d+) item\(s\): cost = (\d+)% of the selling price was assumed\.",
     r"Costo unitario mancante per \1 articolo/i: ipotizzato costo = \2% del prezzo di vendita."),
    (r"Lead time missing for (\d+) item\(s\): the default of (\d+) days is used\.",
     r"Tempo di consegna mancante per \1 articolo/i: si usa il valore predefinito di \2 giorni."),
    (r"([\d,]+) rows dropped because of unreadable dates\.", r"\1 righe scartate per date illeggibili."),
    (r"([\d,]+) rows dropped because of an empty item code\.", r"\1 righe scartate per codice articolo vuoto."),
    (r"([\d,]+) rows dropped because of an unreadable quantity\.", r"\1 righe scartate per quantità illeggibile."),
    (r"([\d,]+) negative quantities \(returns\) were set to 0\.", r"\1 quantità negative (resi) portate a 0."),
    (r"([\d,]+) rows with the same day, store and item were summed\.",
     r"\1 righe con stesso giorno, negozio e articolo sono state sommate."),
    (r"(\d+) store\(s\) have more than one channel: the most frequent was used\.",
     r"\1 negozio/i con più di un canale: è stato usato il più frequente."),
    (r"([\d,]+) rows have an unknown channel label and were read as physical\.",
     r"\1 righe con canale sconosciuto, lette come fisico."),
    (r"Dates like 03/04/2024 are ambiguous: day-first was assumed \(set the date format to change it\)\.",
     "Date come 03/04/2024 sono ambigue: si è assunto giorno/mese (cambia il formato data per modificarlo)."),
    (r"Only (\d+) days of history: at least (\d+) days \(8 weeks\) are needed\.",
     r"Solo \1 giorni di storico: ne servono almeno \2 (8 settimane)."),
    (r"No series left after reading the file\.", "Nessuna serie rimasta dopo la lettura del file."),
    (r"([\d,]+) rows is above the online limit of ([\d,]+): run the analysis locally \(see README\)\.",
     r"\1 righe superano il limite online di \2: esegui l'analisi in locale (vedi README)."),
    (r"([\d,]+) store-item series is above the online limit of ([\d,]+): run the analysis locally \(see README\)\.",
     r"\1 serie negozio-articolo superano il limite online di \2: esegui l'analisi in locale (vedi README)."),
    (r"(\d+)% of the days have zero sales \(intermittent demand\): forecasts will be less reliable\.",
     r"Il \1% dei giorni ha vendite zero (domanda intermittente): le previsioni saranno meno affidabili."),
    (r"History: ([\d,]+) days \((.+) → (.+)\)\.", r"Storico: \1 giorni (\2 → \3)."),
    (r"Train (\S+) → (\S+) \(([\d,]+) days\) · validation (\S+) → (\S+) \(([\d,]+) days\) · test (\S+) → (\S+) "
     r"\(([\d,]+) days\)\.",
     r"Allenamento \1 → \2 (\3 giorni) · validazione \4 → \5 (\6 giorni) · test \7 → \8 (\9 giorni)."),
    (r"Yearly seasonality is modelled \(lag 364, day of year\)\.",
     "La stagionalità annuale è modellata (lag 364, giorno dell'anno)."),
    (r"Less than (\d+) training days: yearly seasonality is not modelled\.",
     r"Meno di \1 giorni di allenamento: la stagionalità annuale non è modellata."),
    (r"(\d+) series excluded for short history\.", r"\1 serie escluse per storico troppo corto."),
    (r"LightGBM skipped: only ([\d,]+) training rows \(minimum ([\d,]+)\)\.",
     r"LightGBM escluso: solo \1 righe di allenamento (minimo \2)."),
    (r"Holt-Winters skipped in fast mode: ([\d,]+) series \(limit ([\d,]+)\)\.",
     r"Holt-Winters escluso in modalità veloce: \1 serie (limite \2)."),
    (r"Short history: validation or training period is very short, results have low confidence\.",
     "Storico corto: periodo di validazione o di allenamento molto breve, risultati poco affidabili."),
    (r"Fast mode: models are fitted once per period, fewer LightGBM trees\.",
     "Modalità veloce: modelli stimati una volta per periodo, meno alberi LightGBM."),
    (r"Map the required column\(s\): (.+)\.", r"Associa le colonne obbligatorie: \1."),
    (r"Promotions read from the file: ([\d,]+) promotion days on (\d+) item\(s\)\.",
     r"Promozioni lette dal file: \1 giorni di promozione su \2 articolo/i."),
    (r"Promotions found from prices: ([\d,]+) days at least (\d+)% below the regular price, on (\d+) item\(s\)\.",
     r"Promozioni trovate dai prezzi: \1 giorni con sconto ≥ \2% rispetto al prezzo normale, su \3 articolo/i."),
    (r"The promo column marks no promotion day\.", "La colonna promozioni non segna nessun giorno di promozione."),
    (r"Could not read the file: (.+)", r"Impossibile leggere il file: \1"),
    (r"The file has no rows\.", "Il file non contiene righe."),
    (r"Old \.xls files are not supported: save the file as \.xlsx or \.csv\.",
     "I vecchi file .xls non sono supportati: salva il file come .xlsx o .csv."),
    (r"No date in column '(.+)' could be read\.", r"Nessuna data leggibile nella colonna '\1'."),
    (r"No number in column '(.+)' could be read\.", r"Nessun numero leggibile nella colonna '\1'."),
    (r"([\d,]+) rows have a negative quantity \(returns\)\.", r"\1 righe hanno quantità negativa (resi)."),
    (r"No usable rows after reading dates, items and quantities\.",
     "Nessuna riga utilizzabile dopo la lettura di date, articoli e quantità."),
]


def translate_message(msg: str) -> str:
    """Italian version of a message produced by src/ (unchanged in English or if unknown)."""
    if not is_it():
        return msg
    for pattern, repl in _PATTERNS:
        if re.fullmatch(pattern, msg):
            return re.sub(pattern, repl, msg)
    return msg
