"""Website essentials around the dashboard: titles, descriptions, alt text, footer, 404, analytics.

Streamlit renders every page from one HTML shell in the browser, so the ``<head>`` tags
(description, Open Graph, canonical link, robots) are written by a small script when the
page loads: search engines that run JavaScript read them, simple link previews may not.
The browser-tab title is set by Streamlit itself for every page.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

import streamlit as st
from i18n import lang, tx

from src import analytics
from src.site_files import PUBLIC_PAGES

# url path -> (description EN, description IT); "" = home (Overview)
DESCRIPTIONS = {
    "": ("Forecast demand per store and product, compare reorder policies at the same service level and see the "
         "stock and cost you save.",
         "Prevedi la domanda per negozio e prodotto, confronta le politiche di riordino a parità di servizio e vedi "
         "quanta scorta e quanti costi risparmi."),
    "demand": ("Sales trend, weekly and yearly seasonality, ABC / XYZ classification of the products.",
               "Andamento delle vendite, stagionalità settimanale e annuale, classificazione ABC / XYZ dei prodotti."),
    "forecast": ("Forecast accuracy of four models (WAPE, bias) by horizon, class and store.",
                 "Precisione delle previsioni di quattro modelli (WAPE, bias) per orizzonte, classe e negozio."),
    "promotions": ("Effect of promotions on sales, forecast error on promotion days and a promotion planner.",
                   "Effetto delle promozioni sulle vendite, errore di previsione nei giorni di promozione e "
                   "pianificatore di promozioni."),
    "inventory": ("Stock versus service level for three reorder policies, reorder point and lot of one product, "
                  "what-if.", "Scorta contro livello di servizio per tre politiche di riordino, punto di riordino e "
                  "lotto di un prodotto, what-if."),
    "orders": ("What to order today by supplier and a 4-week order calendar, with supplier terms and limits.",
               "Cosa ordinare oggi per fornitore e calendario ordini di 4 settimane, con vincoli dei fornitori."),
    "data": ("Upload your sales history or try the sample datasets; column mapping and data checks.",
             "Carica il tuo storico vendite o prova i dataset di esempio; associazione colonne e controlli."),
    "summary": ("Every formula, chart and number of the software explained with small examples.",
                "Ogni formula, grafico e numero del software spiegato con piccoli esempi."),
    "faq": ("Frequently asked questions: data needed, accuracy, privacy, costs, large datasets.",
            "Domande frequenti: dati necessari, precisione, privacy, costi, dataset grandi."),
    "privacy": ("How the app treats your data and the anonymous usage statistics.",
                "Come l'app tratta i tuoi dati e le statistiche anonime di utilizzo."),
    "terms": ("Terms of use of the demand forecasting and inventory optimization app.",
              "Condizioni d'uso dell'app di previsione della domanda e ottimizzazione delle scorte."),
    "404": ("Page not found.", "Pagina non trovata."),
}


def cfg():
    from common import config

    return config()


# --------------------------------------------------------------------------- 404
def requested_path() -> str:
    """Path the browser asked for on the first load of the session ("" = home)."""
    try:
        return urlparse(st.context.url or "").path.strip("/")
    except Exception:
        return ""


def is_unknown_path(known: set[str]) -> bool:
    """True only on the first run of a session that opened an address no page has.

    The check runs once: after that the visitor moves with the menu and the URL in
    ``st.context`` (taken when the session started) no longer describes the page shown.
    """
    if st.session_state.get("_site_first_run_done"):
        return False
    st.session_state["_site_first_run_done"] = True
    path = requested_path()
    return bool(path) and path.split("/")[0] not in known


# --------------------------------------------------------------------------- head tags
def head_tags(url_path: str, title: str) -> None:
    """Page title, description, Open Graph, canonical, robots, html lang and logo alt text."""
    c = cfg()
    desc_en, desc_it = DESCRIPTIONS.get(url_path, DESCRIPTIONS[""])
    description = tx(desc_en, desc_it)
    base = c.site.base_url.rstrip("/")
    canonical = f"{base}/{url_path}" if base and url_path in PUBLIC_PAGES else ""
    indexable = c.site.indexable and url_path in PUBLIC_PAGES
    image = f"{base}/app/static/logo.png" if base else ""
    data = {"title": f"{title} · {c.site.name}", "description": description, "canonical": canonical,
            "robots": "index, follow" if indexable else "noindex, nofollow", "lang": lang(), "image": image,
            "logoAlt": tx(f"{c.site.name} logo: a parcel with a forecast arrow",
                          f"Logo di {c.site.name}: un pacco con una freccia di previsione"),
            "plausible": c.analytics.plausible_domain if tracking_on() else ""}
    st.html(f"""<script>
(() => {{
  const d = {json.dumps(data)};
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  doc.title = d.title;
  doc.documentElement.lang = d.lang;
  const meta = (attr, key, value) => {{
    let el = doc.head.querySelector(`meta[${{attr}}="${{key}}"]`);
    if (!value) {{ if (el) el.remove(); return; }}
    if (!el) {{ el = doc.createElement("meta"); el.setAttribute(attr, key); doc.head.appendChild(el); }}
    el.setAttribute("content", value);
  }};
  meta("name", "description", d.description);
  meta("name", "robots", d.robots);
  meta("property", "og:title", d.title);
  meta("property", "og:description", d.description);
  meta("property", "og:type", "website");
  meta("property", "og:image", d.image);
  meta("name", "twitter:card", d.image ? "summary" : "");
  let link = doc.head.querySelector('link[rel="canonical"]');
  if (d.canonical) {{
    if (!link) {{ link = doc.createElement("link"); link.rel = "canonical"; doc.head.appendChild(link); }}
    link.href = d.canonical;
  }} else if (link) {{ link.remove(); }}
  // the logo may be drawn after this script runs: retry for a few seconds
  let tries = 0;
  const setAlt = () => {{
    const imgs = [...doc.querySelectorAll('[data-testid="stSidebarLogo"], img.stLogo, [data-testid="stHeaderLogo"]')]
      .flatMap(el => el.tagName === "IMG" ? [el] : [...el.querySelectorAll("img")]);
    imgs.forEach(img => img.alt = d.logoAlt);
    if ((!imgs.length || imgs.some(i => i.alt !== d.logoAlt)) && tries++ < 20) setTimeout(setAlt, 250);
  }};
  setAlt();
  if (d.plausible && !doc.head.querySelector('script[data-domain]')) {{
    const s = doc.createElement("script");
    s.defer = true; s.dataset.domain = d.plausible; s.src = "https://plausible.io/js/script.js";
    doc.head.appendChild(s);
  }}
}})();
</script>""", unsafe_allow_javascript=True, width="content")


# --------------------------------------------------------------------------- analytics
def tracking_on() -> bool:
    c = cfg()
    if not c.analytics.enabled:
        return False
    try:
        headers = dict(st.context.headers)
    except Exception:
        headers = {}
    return analytics.tracking_allowed(headers, st.session_state.get("analytics_opt_out", False))


def track(event: str, page: str | None = None, **detail) -> None:
    """Record one anonymous event (no-op when analytics is off or the visitor opted out)."""
    if not tracking_on():
        return
    sid = st.session_state.setdefault("analytics_session", analytics.new_session_id())
    data = None
    try:
        from common import active_data

        data = active_data()
    except Exception:
        pass
    source = "none" if data is None else ("demo" if data.source == "demo" else
                                          ("sample" if data.meta.get("sample") else "upload"))
    path = cfg().root / cfg().analytics.path
    analytics.log_event(path, event, sid, page=page if page is not None else st.session_state.get("_site_page", ""),
                        lang=lang(), source=source, **detail)


def page_view(url_path: str) -> None:
    """One page view per page change (not on every rerun of the same page)."""
    if st.session_state.get("_site_page") != url_path:
        st.session_state["_site_page"] = url_path
        track("page_view", page=url_path)


# --------------------------------------------------------------------------- footer
def footer(pages: dict) -> None:
    """Links to FAQ, privacy and terms at the bottom of the sidebar."""
    with st.sidebar:
        st.divider()
        c1, c2 = st.columns(2)
        c1.page_link(pages["privacy"], label="Privacy")
        c2.page_link(pages["terms"], label=tx("Terms of use", "Condizioni"))
        st.caption(tx(f"{cfg().site.name} · costs, prices and lead times are assumptions unless your data provides "
                      f"them · [source code]({cfg().site.repository})",
                      f"{cfg().site.name} · costi, prezzi e tempi di consegna sono ipotesi se i tuoi dati non li "
                      f"contengono · [codice sorgente]({cfg().site.repository})"))
