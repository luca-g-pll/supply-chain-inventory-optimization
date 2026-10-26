"""Terms of use."""

import streamlit as st
from common import config
from i18n import tx

cfg = config()
UPDATED = "2026-10-26"
owner = cfg.site.owner or tx("the operator of this app", "chi gestisce questa app")

st.title(tx("Terms of use", "Condizioni d'uso"))
st.caption(tx(f"Last updated: {UPDATED}", f"Ultimo aggiornamento: {UPDATED}"))
if not cfg.site.owner:
    st.warning(tx("For the operator: set `site.owner` and `site.contact_email` in `config/params.yaml`, and have these "
                  "terms reviewed by a lawyer before any commercial use.",
                  "Per chi pubblica l'app: imposta `site.owner` e `site.contact_email` in `config/params.yaml`, e fai "
                  "rivedere queste condizioni da un legale prima di qualsiasi uso commerciale."), icon=":material/build:")

st.markdown(tx(f"""
**1. What the app is.** A decision-support tool by {owner}: it analyses the sales history you provide, forecasts
demand and proposes reorder parameters and orders. Using it means accepting these terms.

**2. Results are estimates, not instructions.** Forecasts, savings, reorder points and order proposals come from
statistical models and from assumptions shown in the **Warnings and limits** panel (for example the cost of an order).
They can be wrong. Check them before ordering, buying or making any business decision: the decisions and their
consequences remain yours. The app is not financial, legal or accounting advice.

**3. Your data.** Upload only data you are entitled to use, and no personal data the analysis does not need.
The files are processed as described in the Privacy policy.

**4. No warranty.** The app is provided "as is", without guarantees of accuracy, availability or fitness for a
particular purpose. To the extent allowed by law, {owner} is not liable for losses arising from its use or from
decisions based on its results, except in cases of wilful misconduct or gross negligence.

**5. Fair use.** Do not try to disrupt the service, overload it on purpose, or access data that are not yours.

**6. Demo data.** The demo uses the Kaggle "Store Item Demand Forecasting" dataset under its own licence; the sample
datasets are invented. Neither represents a real company.

**7. Source code.** The code is published at {cfg.site.repository}; its reuse follows the licence stated in the
repository.

**8. Changes.** These terms can change; the date above shows the last version. Continuing to use the app means
accepting the current version.
""", f"""
**1. Cos'è l'app.** Uno strumento di supporto alle decisioni di {owner}: analizza lo storico vendite che fornisci,
prevede la domanda e propone parametri di riordino e ordini. Usarla significa accettare queste condizioni.

**2. I risultati sono stime, non istruzioni.** Previsioni, risparmi, punti di riordino e ordini proposti derivano da
modelli statistici e da ipotesi mostrate nel pannello **Avvisi e limiti** (per esempio il costo di un ordine).
Possono essere sbagliati. Verificali prima di ordinare, acquistare o prendere decisioni aziendali: le decisioni e le
loro conseguenze restano tue. L'app non è una consulenza finanziaria, legale o contabile.

**3. I tuoi dati.** Carica solo dati che hai diritto di usare, e nessun dato personale di cui l'analisi non ha
bisogno. I file sono trattati come descritto nell'Informativa sulla privacy.

**4. Nessuna garanzia.** L'app è fornita "così com'è", senza garanzie di esattezza, disponibilità o idoneità a uno
scopo particolare. Nei limiti consentiti dalla legge, {owner} non risponde di perdite derivanti dal suo uso o da
decisioni basate sui suoi risultati, salvo dolo o colpa grave.

**5. Uso corretto.** Non tentare di interrompere il servizio, sovraccaricarlo di proposito o accedere a dati non tuoi.

**6. Dati demo.** La demo usa il dataset Kaggle "Store Item Demand Forecasting" secondo la sua licenza; i dataset di
esempio sono inventati. Nessuno dei due rappresenta un'azienda reale.

**7. Codice sorgente.** Il codice è pubblicato su {cfg.site.repository}; il suo riuso segue la licenza indicata nel
repository.

**8. Modifiche.** Queste condizioni possono cambiare; la data qui sopra indica l'ultima versione. Continuare a usare
l'app significa accettare la versione in vigore.
"""))
