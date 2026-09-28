"""Streamlit entry point: navigation + the scenario sidebar shared by every page.

Run locally:  streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Inventory Optimization", page_icon="📦", layout="wide",
                   initial_sidebar_state="expanded")

from common import active_data, scenario_sidebar  # noqa: E402

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/insights:", default=True),
    st.Page("views/demand.py", title="Demand", icon=":material/bar_chart:"),
    st.Page("views/forecast.py", title="Forecast", icon=":material/timeline:"),
    st.Page("views/inventory.py", title="Inventory", icon=":material/inventory_2:"),
    st.Page("views/data.py", title="Data", icon=":material/upload_file:"),
]
page = st.navigation(pages)

data = active_data()
if data is not None and page.title != "Data":
    scenario_sidebar(data)

with st.sidebar:
    st.divider()
    st.caption("Demand forecasting & inventory optimization · costs, prices and lead times are "
               "assumptions unless your data provides them (see the README).")

page.run()
