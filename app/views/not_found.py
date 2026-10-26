"""Custom 404: the address does not match any page."""

import streamlit as st
from i18n import tx
from website import requested_path

# Streamlit opens its own "page not found" dialog before redirecting here: this page replaces it
st.html("<style>[data-testid='stDialog']{display:none !important}</style>")
st.title(tx("Page not found", "Pagina non trovata"))
path = st.session_state.get("_site_missing_path") or requested_path()
st.markdown(tx(f"There is no page at **/{path}**. The link may be old or mistyped.",
               f"Non esiste una pagina all'indirizzo **/{path}**. Il link potrebbe essere vecchio o scritto male."))
st.markdown(tx("Where do you want to go?", "Dove vuoi andare?"))
c1, c2, c3 = st.columns(3)
c1.page_link("views/overview.py", label=tx("Home", "Home"), icon=":material/home:")
c2.page_link("views/data.py", label=tx("Use your data", "Usa i tuoi dati"), icon=":material/upload_file:")
c3.page_link("views/faq.py", label=tx("Questions", "Domande frequenti"), icon=":material/help:")
