"""Privacy policy: what the app does with uploaded files and with the anonymous usage counts."""

import streamlit as st
from common import config
from i18n import tx
from website import tracking_on

cfg = config()
UPDATED = "2026-10-26"
owner = cfg.site.owner or tx("the person or company that publishes this app", "la persona o l'azienda che pubblica "
                                                                              "questa app")
contact = cfg.site.contact_email

st.title(tx("Privacy policy", "Informativa sulla privacy"))
st.caption(tx(f"Last updated: {UPDATED}", f"Ultimo aggiornamento: {UPDATED}"))
if not cfg.site.owner or not contact:
    st.warning(tx("For the operator: set `site.owner` and `site.contact_email` in `config/params.yaml` before "
                  "publishing the app; this page shows them.",
                  "Per chi pubblica l'app: imposta `site.owner` e `site.contact_email` in `config/params.yaml` prima "
                  "di pubblicarla; questa pagina li mostra."), icon=":material/build:")

st.markdown(tx(f"""
### In short
- **Your files stay yours.** Sales, stock and supplier files are analysed in memory for your browser session only;
  they are never written to disk, never sent to third parties and disappear when you close or reload the page.
- **No profiling, no advertising, no tracking cookies.**
- The app counts **anonymous** page views to understand which pages are useful. You can switch this off below.

### Who is responsible
The data controller is **{owner}**{f", contact: {contact}" if contact else ""}.

### What happens to the files you upload
| Data | Use | Kept |
|---|---|---|
| Sales history, item master, stock, supplier terms, promotion calendar | Analysis shown to you only | In memory, for your session; deleted when you close or reload the page |
| Results (forecasts, orders, KPIs) | Shown to you, downloadable | In memory, for your session |

Nothing you upload is used to train models for anyone else, and nobody else can see it. If your files contain
personal data (for example customer names in order lines), remove them before uploading: the software needs none.

### Anonymous usage statistics
When enabled, every page change and a few actions (analysis started, file downloaded) are recorded as one line with:
time (to the minute, UTC), a random code that lives only in this browser tab, the page, the language and the kind of
data in use (demo, sample or your file). **Not** recorded: IP address, browser, device, cookies, file names, file
contents or anything you type. The lines cannot identify you. They are kept by the operator for at most 13 months.

Browsers that send *Do Not Track* or *Global Privacy Control* are never recorded.
""", f"""
### In breve
- **I tuoi file restano tuoi.** Vendite, giacenze e condizioni dei fornitori sono analizzati in memoria solo per la tua
  sessione del browser; non vengono mai scritti su disco né inviati a terzi e spariscono quando chiudi o ricarichi la
  pagina.
- **Nessuna profilazione, nessuna pubblicità, nessun cookie di tracciamento.**
- L'app conta le visite **anonime** alle pagine per capire quali sono utili. Puoi disattivarlo qui sotto.

### Chi è responsabile
Il titolare del trattamento è **{owner}**{f", contatto: {contact}" if contact else ""}.

### Cosa succede ai file che carichi
| Dati | Uso | Conservazione |
|---|---|---|
| Storico vendite, anagrafica articoli, giacenze, condizioni fornitori, calendario promozioni | Analisi mostrata solo a te | In memoria, per la tua sessione; cancellati quando chiudi o ricarichi la pagina |
| Risultati (previsioni, ordini, KPI) | Mostrati a te, scaricabili | In memoria, per la tua sessione |

Nulla di ciò che carichi viene usato per addestrare modelli per altri, e nessun altro può vederlo. Se i tuoi file
contengono dati personali (per esempio nomi di clienti nelle righe d'ordine), toglili prima di caricarli: il software
non ne ha bisogno.

### Statistiche anonime di utilizzo
Se attive, ogni cambio di pagina e alcune azioni (analisi avviata, file scaricato) sono registrati come una riga con:
ora (al minuto, UTC), un codice casuale che vive solo in questa scheda del browser, la pagina, la lingua e il tipo di
dati in uso (demo, esempio o tuo file). **Non** vengono registrati: indirizzo IP, browser, dispositivo, cookie, nomi o
contenuti dei file, né nulla di ciò che scrivi. Le righe non permettono di identificarti. Sono conservate da chi gestisce
l'app per al massimo 13 mesi.

I browser che inviano *Do Not Track* o *Global Privacy Control* non vengono mai registrati.
"""))

if cfg.analytics.enabled:
    off = st.toggle(tx("Do not record my usage in this session", "Non registrare il mio utilizzo in questa sessione"),
                    value=st.session_state.get("analytics_opt_out", False), key="privacy_opt_out")
    st.session_state["analytics_opt_out"] = off
    st.caption(tx("Usage statistics for you right now: ", "Statistiche di utilizzo per te in questo momento: ")
               + (tx("**on** (anonymous)", "**attive** (anonime)") if tracking_on() else tx("**off**", "**disattivate**")))
else:
    st.caption(tx("Usage statistics are switched off in this installation.",
                  "Le statistiche di utilizzo sono disattivate in questa installazione."))

st.markdown(tx(f"""
### Cookies
The app sets no advertising or tracking cookies. The web framework (Streamlit) may set one technical cookie that
protects uploads from cross-site attacks; it is strictly necessary and needs no consent.
{"The cookie-free Plausible script counts page views for the operator (no personal data)." if cfg.analytics.plausible_domain else ""}

### Hosting
The company that hosts the app may keep technical server logs (for example IP addresses) for security, under its own
policy. Running the app on your own computer avoids this entirely.

### Your rights
Under the GDPR you can ask for access, correction or deletion of personal data and complain to your data protection
authority (in Italy the Garante per la protezione dei dati personali). Uploaded files are not kept and the usage
lines are anonymous, so in practice there is nothing that refers to you; write to the contact above for any question.

### Changes
This page states the date of the last change. Important changes are announced in the app.
""", f"""
### Cookie
L'app non imposta cookie pubblicitari o di tracciamento. Il framework web (Streamlit) può impostare un cookie tecnico
che protegge i caricamenti da attacchi tra siti; è strettamente necessario e non richiede consenso.
{"Lo script Plausible, senza cookie, conta le visite per chi gestisce l'app (nessun dato personale)." if cfg.analytics.plausible_domain else ""}

### Hosting
L'azienda che ospita l'app può tenere log tecnici del server (per esempio indirizzi IP) per sicurezza, secondo la sua
informativa. Eseguire l'app sul tuo computer lo evita del tutto.

### I tuoi diritti
In base al GDPR puoi chiedere accesso, rettifica o cancellazione dei dati personali e presentare reclamo all'autorità
di controllo (in Italia il Garante per la protezione dei dati personali). I file caricati non vengono conservati e le
righe di utilizzo sono anonime, quindi in pratica non c'è nulla che si riferisca a te; scrivi al contatto qui sopra per
qualsiasi domanda.

### Modifiche
Questa pagina riporta la data dell'ultima modifica. Le modifiche importanti sono segnalate nell'app.
"""))
