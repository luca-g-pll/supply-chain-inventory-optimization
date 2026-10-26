"""New products on the Orders page: the suggested similar product, confirmed or changed by the user."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from i18n import tx

from src.cold_start import AVERAGE, blend_forecasts, item_cv, new_series_params
from src.series import natural_sorted

NUM = st.column_config.NumberColumn


def reason_text(codes: str, corr) -> str:
    labels = {
        "same_item": tx("same product in other stores", "stesso prodotto negli altri negozi"),
        "early_sales": tx("similar first weeks", "prime settimane simili") + (f" ({corr:.2f})" if pd.notna(corr) else ""),
        "name": tx("similar name", "nome simile"),
        "category": tx("same category", "stessa categoria"),
        "supplier": tx("same supplier", "stesso fornitore"),
        "price": tx("close price", "prezzo vicino"),
        "average": tx("no signal: average product of the store", "nessun segnale: prodotto medio del negozio"),
        "weak": tx("weak signals only", "solo segnali deboli"),
    }
    return ", ".join(labels.get(c, c) for c in str(codes).split(",") if c)


def confidence_text(c: str) -> str:
    return {"high": tx("high", "alta"), "medium": tx("medium", "media"), "low": tx("low", "bassa")}.get(c, c)


def choices_key(data) -> str:
    return f"new_choices:{data.source}:{data.meta.get('file_name', '')}"


def new_products_section(data, sc, cfg, channels: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Show the new products and return ``(params, forecast)`` of the new series of the scenario.

    ``params`` has the columns of the P2 rows the Orders page reads (plus estimated stock);
    ``forecast`` is ``[store, item, h, forecast, weight]``.
    """
    series = data.t("new_series")
    empty = pd.DataFrame(), pd.DataFrame()
    if series.empty:
        return empty
    series = series[series["store"].isin(sc.stores) & series["item"].isin(sc.items)]
    if series.empty:
        return empty
    items = data.t("new_items")
    master = data.t("item_master")
    names = master.set_index("item")["item_name"] if "item_name" in master else pd.Series(dtype=str)
    est = data.t("plan_forecast")
    options = [AVERAGE] + natural_sorted(est["item"].unique())

    key = choices_key(data)
    if key not in st.session_state and len(items):
        st.session_state[key] = pd.DataFrame({
            "item": items["item"], "name": items["item"].map(names).fillna(""), "age_days": items["age_days"],
            "units": items["units"], "analog": items["analog"].fillna(AVERAGE),
            "scale_pct": (items["scale"] * 100).round(1),
            "growth_pct": ((items.get("growth", 1.0) - 1) * 100).round(0),
            "confidence": items["confidence"].map(confidence_text),
            "why": [reason_text(r, c) for r, c in zip(items["reasons"], items["correlation"])],
            "confirmed": False})
    choices = st.session_state.get(key, pd.DataFrame())
    choices = choices[choices["item"].isin(series["item"])] if len(choices) else choices
    same_item = series[series["rule"] == "same_item"]
    to_confirm = int((~choices["confirmed"]).sum()) if len(choices) else 0

    label = tx("New products", "Prodotti nuovi") + f" · {series['item'].nunique()}"
    if to_confirm:
        label += tx(f" ({to_confirm} to confirm)", f" ({to_confirm} da confermare)")
    with st.expander(label, icon=":material/new_releases:", expanded=to_confirm > 0):
        st.caption(tx(
            f"Products with fewer than {cfg.input.min_history_days} days of sales: too little history to test them on "
            "the past, so they are planned from a **similar product** that is still selling. The forecast moves to the "
            f"product's own sales as they come in (50% after {cfg.cold_start.half_life_days:.0f} days), the safety stock "
            f"is higher and the lot covers at most {cfg.cold_start.first_lot_days} days.",
            f"Prodotti con meno di {cfg.input.min_history_days} giorni di vendite: troppo poco storico per provarli sul "
            "passato, quindi sono pianificati da un **prodotto simile** che vende ancora. La previsione passa alle "
            f"vendite del prodotto man mano che arrivano (50% dopo {cfg.cold_start.half_life_days:.0f} giorni), la scorta "
            f"di sicurezza è più alta e il lotto copre al massimo {cfg.cold_start.first_lot_days} giorni."))
        if len(choices):
            edited = st.data_editor(
                choices, hide_index=True, width="stretch", num_rows="fixed",
                disabled=["item", "name", "age_days", "units", "growth_pct", "confidence", "why"],
                column_order=["item", "analog", "scale_pct", "confirmed", "confidence", "why", "name", "age_days",
                              "units", "growth_pct"],
                key=f"{key}:editor", column_config={
                    "item": tx("New product", "Prodotto nuovo"), "name": tx("Description", "Descrizione"),
                    "age_days": NUM(tx("Days of sales", "Giorni di vendita"), format="%.0f"),
                    "units": NUM(tx("Units sold", "Pezzi venduti"), format="%.0f"),
                    "analog": st.column_config.SelectboxColumn(
                        tx("Similar product", "Prodotto simile"), options=options, required=False,
                        format_func=lambda i: (f"{i} · {names.get(i, '')}".rstrip(" ·") if i else
                                               tx("average product of the store", "prodotto medio del negozio"))),
                    "scale_pct": NUM(tx("Sells % of it", "Vende il % di quello"), min_value=1.0, max_value=5000.0,
                                     step=5.0, format="%.0f%%",
                                     help=tx("Own sales ÷ similar product's sales, last 14 days, same stores",
                                             "Vendite proprie ÷ vendite del prodotto simile, ultimi 14 giorni, stessi "
                                             "negozi")),
                    "growth_pct": NUM(tx("Weekly growth", "Crescita settimanale"), format="%+.0f%%",
                                      help=tx("Launch still growing: the forecast is raised by this (first 3 months, "
                                              "at most +50%)", "Lancio ancora in crescita: la previsione è alzata di "
                                              "tanto (primi 3 mesi, al massimo +50%)")),
                    "confidence": tx("Confidence", "Affidabilità"), "why": tx("Why this one", "Perché questo"),
                    "confirmed": st.column_config.CheckboxColumn(tx("Confirmed", "Confermato"))})
            st.session_state[key] = edited
            choices = edited
        if len(same_item):
            st.caption(tx(f"{len(same_item)} series are new only in some stores: they use the same product in the "
                          "other stores, scaled to the store.",
                          f"{len(same_item)} serie sono nuove solo in alcuni negozi: usano lo stesso prodotto negli altri "
                          "negozi, in proporzione al negozio."))

    # the user's choices replace the suggestion for the products new everywhere
    if len(choices):
        ch = choices.set_index("item")
        pick = series["rule"] != "same_item"
        series = series.copy()
        series.loc[pick, "analog"] = series.loc[pick, "item"].map(ch["analog"]).fillna(AVERAGE)
        series.loc[pick, "scale"] = series.loc[pick, "item"].map(ch["scale_pct"]).fillna(100) / 100
    forecast = blend_forecasts(series, data.t("new_forecast_own"), est, cfg.cold_start.half_life_days)
    params = new_series_params(forecast, series, master, item_cv(data.t("policy_summary")),
                               {c: cfg.channel(c) for c in channels}, cfg.cold_start)
    return params, forecast


def unconfirmed_count(data) -> tuple[int, int]:
    """(new products, of which not confirmed yet) for the warnings panel."""
    items = data.t("new_items")
    if items.empty:
        return 0, 0
    choices = st.session_state.get(choices_key(data))
    if choices is None or choices.empty:
        return len(items), len(items)
    return len(items), int((~choices["confirmed"].astype(bool)).sum())

