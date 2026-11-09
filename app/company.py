"""Company folder on the Data page: the files found, the company's settings and the analysis of the folder."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from common import analyse_company, config, csv_download
from i18n import translate_message, tx

from src.readers import SUPPORTED, InputError
from src.workspace import DATA, RESULTS, fingerprint, load_settings, reorder_parameters, save_settings, scan

NUM = st.column_config.NumberColumn


def role_name(role: str) -> str:
    return {
        "sales": tx("Sales", "Vendite"), "purchases": tx("Purchases", "Acquisti"),
        "item_master": tx("Item list", "Anagrafica articoli"), "stock": tx("Stock", "Giacenze"),
        "supplier_terms": tx("Supplier terms", "Condizioni fornitori"),
        "promo_calendar": tx("Promotion calendar", "Calendario promozioni"),
        "unknown": tx("Not recognised", "Non riconosciuto"), "error": tx("Cannot be read", "Illeggibile"),
        "skipped": tx("Too large", "Troppo grande"),
    }.get(role, role)


def open_folder(path: Path) -> None:
    """Show the folder in the file manager of this computer (the app runs on the user's own machine)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # noqa: S606 - a folder of the user, opened in Explorer
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        st.toast(tx("Could not open the folder: open it by hand.", "Impossibile aprire la cartella: aprila a mano."))


@st.cache_data(show_spinner=False)
def _files(folder: str, stamp: str) -> pd.DataFrame:
    rows = [{"name": f.name, "role": f.role, "rows": f.rows, "message": f.message, "notes": list(f.notes)}
            for f in scan(Path(folder), config())]
    return pd.DataFrame(rows, columns=["name", "role", "rows", "message", "notes"])


def _settings_form(folder: Path) -> None:
    cfg = config()
    try:
        settings = load_settings(folder)
    except InputError as exc:
        st.error(translate_message(str(exc)))
        return
    with st.expander(tx("Company settings: costs and lead time", "Impostazioni dell'azienda: costi e tempi di consegna"),
                     icon=":material/tune:"):
        st.caption(tx("Used wherever your files have no value of their own. They are saved in `settings.yaml` in the "
                      "company folder and stay there for the next times.",
                      "Usati dove i tuoi file non hanno un valore proprio. Sono salvati in `settings.yaml` nella "
                      "cartella aziendale e restano per le prossime volte."))
        with st.form("ws_settings"):
            name = st.text_input(tx("Company name", "Nome dell'azienda"), settings["company"].get("name", ""))
            a1, a2, a3, a4, a5 = st.columns(5)
            lead = a1.number_input(tx("Lead time (days)", "Tempo di consegna (giorni)"), 1, 60,
                                   cfg.inventory.lead_time_days)
            order_cost = a2.number_input(tx("Ordering cost (€/order)", "Costo per ordine (€)"), 1.0, 5000.0,
                                         cfg.inventory.order_cost, 5.0)
            holding = a3.number_input(tx("Holding rate (%/year)", "Costo di mantenimento (%/anno)"), 1.0, 100.0,
                                      cfg.inventory.holding_rate * 100, 1.0)
            ratio = a4.number_input(tx("Unit cost (% of price)", "Costo unitario (% del prezzo)"), 5.0, 100.0,
                                    cfg.inventory.cost_to_price_ratio * 100, 5.0)
            target = a5.number_input(tx("Target fill rate (%)", "Fill rate obiettivo (%)"), 80.0, 99.5,
                                     cfg.simulation.target_fill_rate * 100, 0.5)
            grains = {"auto": tx("Detect automatically", "Riconosci automaticamente"), "daily": tx("Daily", "Giornaliere"),
                      "weekly": tx("Weekly totals", "Totali settimanali"), "monthly": tx("Monthly totals", "Totali mensili")}
            grain = st.selectbox(tx("Sales in the files are", "Le vendite nei file sono"), list(grains),
                                 index=list(grains).index(cfg.input.granularity), format_func=grains.get)
            if st.form_submit_button(tx("Save the settings", "Salva le impostazioni"), icon=":material/save:"):
                save_settings(folder, {
                    "inventory": {"lead_time_days": int(lead), "order_cost": float(order_cost),
                                  "holding_rate": round(holding / 100, 4), "cost_to_price_ratio": round(ratio / 100, 4)},
                    "simulation": {"target_fill_rate": round(target / 100, 4)},
                    "input": {"granularity": grain}}, {"name": name.strip()})
                config.clear()
                st.toast(tx("Settings saved: run the analysis again to apply them.",
                            "Impostazioni salvate: rifai l'analisi per applicarle."))
                st.rerun()


def company_section(folder: Path) -> None:
    st.subheader(tx("Company folder", "Cartella aziendale"))
    st.markdown(tx(
        "Put your files in the **data** folder: sales, purchases, item list, stock, supplier terms, promotions. "
        "They are recognised by their columns, in any of these formats: " + ", ".join(SUPPORTED) + ". "
        "Nothing leaves this computer.",
        "Metti i tuoi file nella cartella **data**: vendite, acquisti, anagrafica articoli, giacenze, condizioni dei "
        "fornitori, promozioni. Vengono riconosciuti dalle colonne, in uno di questi formati: " + ", ".join(SUPPORTED)
        + ". Niente lascia questo computer."))
    c1, c2 = st.columns([4, 1.3])
    c1.code(str(folder / DATA), language=None)
    if c2.button(tx("Open the folder", "Apri la cartella"), icon=":material/folder_open:", width="stretch"):
        open_folder(folder / DATA)

    stamp = fingerprint(folder)
    files = _files(str(folder), stamp)
    if files.empty:
        st.info(tx("The folder is empty. Copy at least your sales file into it (date, item, quantity sold), then come "
                   "back here: this page updates by itself.",
                   "La cartella è vuota. Copiaci almeno il file delle vendite (data, articolo, quantità venduta), poi "
                   "torna qui: la pagina si aggiorna da sola."), icon=":material/folder_open:")
        if st.button(tx("Check again", "Controlla di nuovo"), icon=":material/refresh:"):
            st.rerun()
        _settings_form(folder)
        return
    shown = files.assign(
        role=files["role"].map(role_name),
        note=[" ".join(translate_message(m) for m in ([row.message] if row.message else []) + list(row.notes))
              for row in files.itertuples()])
    st.dataframe(shown[["name", "role", "rows", "note"]], hide_index=True, column_config={
        "name": tx("File", "File"), "role": tx("Read as", "Letto come"), "rows": NUM(tx("Rows", "Righe"), format="%,d"),
        "note": tx("Notes", "Note")})
    n_sales = int((files["role"] == "sales").sum())
    if not n_sales:
        st.warning(tx("No sales file was recognised: it needs a date, an item code and the quantity sold.",
                      "Nessun file di vendite riconosciuto: servono una data, un codice articolo e la quantità venduta."))
    _settings_form(folder)

    saved = st.session_state.get("upload_results")
    saved_ws = (saved.meta.get("workspace") or {}) if saved is not None else {}
    if saved_ws and saved_ws.get("fingerprint") == stamp:
        st.success(tx(f"The analysis of {str(saved_ws.get('analysed_at', ''))[:16].replace('T', ' ')} is up to date with "
                      "the files in the folder.",
                      f"L'analisi del {str(saved_ws.get('analysed_at', ''))[:16].replace('T', ' ')} è aggiornata con i "
                      "file della cartella."), icon=":material/check_circle:")
    elif saved_ws:
        st.warning(tx("The files or the settings changed after the last analysis: run it again.",
                      "I file o le impostazioni sono cambiati dopo l'ultima analisi: rifalla."), icon=":material/sync:")
    b1, b2 = st.columns([2, 3])
    if b1.button(tx("Analyse the folder", "Analizza i dati della cartella"), type="primary", disabled=not n_sales,
                 icon=":material/play_arrow:", width="stretch"):
        if analyse_company():
            st.rerun()
    if saved_ws:
        for note in saved_ws.get("notes", []):
            st.caption(translate_message(note))
        d1, d2 = st.columns(2)
        with d1:
            csv_download(reorder_parameters(saved.tables), tx("Reorder parameters per item (CSV)",
                                                              "Parametri di riordino per articolo (CSV)"),
                         "reorder_parameters.csv", "ws_dl_reorder")
        if d2.button(tx("Go to the overview", "Vai alla panoramica"), icon=":material/insights:", width="stretch"):
            st.switch_page("views/overview.py")
        st.caption(tx(f"The analysis is saved in `{folder / RESULTS}` and reopens by itself at the next start.",
                      f"L'analisi è salvata in `{folder / RESULTS}` e si riapre da sola al prossimo avvio."))
