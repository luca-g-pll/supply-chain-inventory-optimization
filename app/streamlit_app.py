"""Streamlit entry point: navigation + the scenario sidebar shared by every page.

Run locally:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="Inventory Optimization", page_icon=str(ASSETS / "favicon.png"), layout="wide",
                   initial_sidebar_state="expanded")
# fixed at the top-left of the sidebar (and shown alone when the sidebar is collapsed)
st.logo(str(ASSETS / "logo.png"), size="large", icon_image=str(ASSETS / "logo.png"))
st.html("<style>img.stLogo{height:3.5rem;max-width:none}"
        "[data-testid='stSidebarHeader']{padding-top:1rem;padding-bottom:1rem}</style>")

from common import active_data, scenario_sidebar  # noqa: E402
from i18n import language_switch, tx  # noqa: E402

language_switch()

pages = [
    st.Page("views/overview.py", title=tx("Overview", "Panoramica"), icon=":material/insights:", default=True),
    st.Page("views/demand.py", title=tx("Demand", "Domanda"), icon=":material/bar_chart:"),
    st.Page("views/forecast.py", title=tx("Forecast", "Previsioni"), icon=":material/timeline:"),
    st.Page("views/promotions.py", title=tx("Promotions", "Promozioni"), icon=":material/sell:"),
    st.Page("views/inventory.py", title=tx("Inventory", "Scorte"), icon=":material/inventory_2:"),
    st.Page("views/orders.py", title=tx("Orders", "Ordini"), icon=":material/shopping_cart:"),
    st.Page("views/data.py", title=tx("Data", "Dati"), icon=":material/upload_file:"),
    st.Page("views/summary.py", title=tx("Summary", "Riepilogo"), icon=":material/menu_book:"),
]
page = st.navigation(pages)

data = active_data()
if data is not None and page.url_path not in ("data", "summary"):
    scenario_sidebar(data)

with st.sidebar:
    st.divider()
    st.caption(tx("Demand forecasting & inventory optimization · costs, prices and lead times are "
                  "assumptions unless your data provides them (see the README).",
                  "Previsione della domanda e ottimizzazione delle scorte · costi, prezzi e tempi di consegna sono "
                  "ipotesi, se i tuoi dati non li contengono (vedi il README)."))

page.run()
