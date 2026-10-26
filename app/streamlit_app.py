"""Streamlit entry point: navigation, page titles, the shared sidebar and the website essentials.

Run locally:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="Demand & Inventory Planner", page_icon=str(ASSETS / "favicon.png"), layout="wide",
                   initial_sidebar_state="expanded")
# fixed at the top-left of the sidebar (and shown alone when the sidebar is collapsed)
st.logo(str(ASSETS / "logo.png"), size="large", icon_image=str(ASSETS / "logo.png"))
st.html("<style>img.stLogo{height:3.5rem;max-width:none}"
        "[data-testid='stSidebarHeader']{padding-top:1rem;padding-bottom:1rem}"
        # text for screen readers only (chart descriptions)
        ".sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;"
        "clip:rect(0,0,0,0);white-space:nowrap;border:0}</style>")

import website  # noqa: E402
from common import active_data, config, scenario_sidebar  # noqa: E402
from i18n import language_switch, tx  # noqa: E402
from limits import limits_panel  # noqa: E402

language_switch()

PAGES = {
    "overview": st.Page("views/overview.py", title=tx("Overview", "Panoramica"), icon=":material/insights:",
                        default=True),
    "demand": st.Page("views/demand.py", title=tx("Demand", "Domanda"), icon=":material/bar_chart:"),
    "forecast": st.Page("views/forecast.py", title=tx("Forecast", "Previsioni"), icon=":material/timeline:"),
    "promotions": st.Page("views/promotions.py", title=tx("Promotions", "Promozioni"), icon=":material/sell:"),
    "inventory": st.Page("views/inventory.py", title=tx("Inventory", "Scorte"), icon=":material/inventory_2:"),
    "orders": st.Page("views/orders.py", title=tx("Orders", "Ordini"), icon=":material/shopping_cart:"),
    "data": st.Page("views/data.py", title=tx("Data", "Dati"), icon=":material/upload_file:"),
    "summary": st.Page("views/summary.py", title=tx("Summary", "Riepilogo"), icon=":material/menu_book:"),
    "faq": st.Page("views/faq.py", title="FAQ", icon=":material/help:"),
    # reachable from the footer links, not listed in the menu
    "privacy": st.Page("views/privacy.py", title="Privacy", icon=":material/shield:", visibility="hidden"),
    "terms": st.Page("views/terms.py", title=tx("Terms", "Termini"), icon=":material/gavel:", visibility="hidden"),
    "404": st.Page("views/not_found.py", title=tx("Page not found", "Pagina non trovata"), icon=":material/error:",
                   url_path="404", visibility="hidden"),
}
page = st.navigation(list(PAGES.values()))
url_path = "" if page.url_path in ("", "overview") else page.url_path
if website.is_unknown_path({p.url_path for p in PAGES.values()} | {"app"}):
    st.session_state["_site_missing_path"] = website.requested_path()
    website.track("not_found", page="404")
    st.switch_page(PAGES["404"])

st.set_page_config(page_title=f"{page.title} · {config().site.name}")
website.page_view(url_path)

data = active_data()
limits_panel(data)
if data is not None and url_path not in ("data", "summary", "faq", "privacy", "terms", "404"):
    scenario_sidebar(data)
website.footer(PAGES)
website.head_tags(url_path, page.title)     # before the page: a page can stop the script (st.stop)

page.run()
