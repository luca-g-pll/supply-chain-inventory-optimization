"""FAQ: the questions a buyer, a store manager or an analyst asks before trusting the numbers."""

import streamlit as st
from common import config
from i18n import tx

cfg = config()
st.title(tx("Frequently asked questions", "Domande frequenti"))
st.caption(tx("Short answers; the details are in the Summary page and in the documentation.",
              "Risposte brevi; i dettagli sono nella pagina Riepilogo e nella documentazione."))

FAQ = [
    ("What does the software do, in one sentence?", "Cosa fa il software, in una frase?",
     "It reads your daily sales, forecasts the next weeks for every product in every store, and tells you when and "
     "how much to reorder — after proving on your own past data how much stock and cost the rule would have saved.",
     "Legge le tue vendite giornaliere, prevede le prossime settimane per ogni prodotto in ogni negozio e ti dice "
     "quando e quanto riordinare — dopo aver dimostrato sul tuo passato quanta scorta e quanti costi la regola avrebbe "
     "fatto risparmiare."),
    ("What data do I need?", "Di quali dati ho bisogno?",
     "A file with one row per day (or per sale) with **date, product and quantity**; at least 8 weeks, better one "
     "year or more. Store, price, cost, lead time, supplier and promotions are optional and make the results more "
     "precise. CSV, Excel or Parquet; Italian or English column names. Template: "
     "`data/templates/sales_history_template.csv`.",
     "Un file con una riga per giorno (o per vendita) con **data, prodotto e quantità**; almeno 8 settimane, meglio un "
     "anno o più. Negozio, prezzo, costo, tempo di consegna, fornitore e promozioni sono facoltativi e rendono i "
     "risultati più precisi. CSV, Excel o Parquet; nomi delle colonne in italiano o in inglese. Modello: "
     "`data/templates/sales_history_template.csv`."),
    ("How accurate are the forecasts?", "Quanto sono precise le previsioni?",
     "It depends on your products. The Forecast page measures it on a period the models never saw (WAPE = total "
     "error ÷ total sales). On the sample retail chain the best model misses 34% of the units day by day, against 50% "
     "for \"same as last week\": single days are noisy, but over a lead time errors largely cancel out, and the "
     "safety stock covers what is left.",
     "Dipende dai tuoi prodotti. La pagina Previsioni la misura su un periodo che i modelli non hanno mai visto (WAPE = "
     "errore totale ÷ vendite totali). Sulla catena di esempio il modello migliore sbaglia il 34% dei pezzi giorno per "
     "giorno, contro il 50% di \"come la settimana scorsa\": il singolo giorno è rumoroso, ma sul tempo di consegna gli "
     "errori in gran parte si compensano e la scorta di sicurezza copre il resto."),
    ("Are the euro savings real?", "I risparmi in euro sono reali?",
     "They are **estimates**. Stock and units are measured on your data; euros also depend on the cost of an order "
     "and of holding stock, which never come from the sales file. The **Warnings and limits** panel shows the values "
     "used and how much the saving moves if they are off by ±50%. Type your real costs on the Data page before "
     "showing the numbers to anyone.",
     "Sono **stime**. Scorta e pezzi sono misurati sui tuoi dati; gli euro dipendono anche dal costo di un ordine e "
     "di mantenimento, che non arrivano mai dal file vendite. Il pannello **Avvisi e limiti** mostra i valori usati e "
     "quanto si sposta il risparmio se sono sbagliati del ±50%. Inserisci i tuoi costi reali nella pagina Dati prima "
     "di mostrare i numeri a qualcuno."),
    ("Is my data saved or shared?", "I miei dati vengono salvati o condivisi?",
     "No. An uploaded file is analysed in memory for your browser session and disappears when you close or reload "
     "the page; it is never written to disk or sent anywhere. The app only counts anonymous page views (see "
     "Privacy). For confidential data run the app on your own computer.",
     "No. Un file caricato viene analizzato in memoria per la tua sessione del browser e sparisce quando chiudi o "
     "ricarichi la pagina; non viene mai scritto su disco né inviato altrove. L'app conta solo visite anonime alle "
     "pagine (vedi Privacy). Per dati riservati esegui l'app sul tuo computer."),
    ("How big can my data be?", "Quanto possono essere grandi i miei dati?",
     "In the dashboard up to 10,000 store-product combinations (about a minute). Larger files — up to millions of "
     "combinations — run in batch mode from the command line (`python -m src.partition`), which exports the reorder "
     "parameters for every store and product.",
     "Nella dashboard fino a 10.000 combinazioni negozio-prodotto (circa un minuto). File più grandi — fino a milioni "
     "di combinazioni — si elaborano a lotti da riga di comando (`python -m src.partition`), che esporta i parametri "
     "di riordino per ogni negozio e prodotto."),
    ("Does it place the orders for me?", "Fa gli ordini al posto mio?",
     "No: the Orders page proposes today's orders by supplier and a 4-week calendar, respecting order days, minimum "
     "orders, packs, discounts, budget and space. You download them (CSV) and send them from your usual system.",
     "No: la pagina Ordini propone gli ordini di oggi per fornitore e un calendario di 4 settimane, rispettando giorni "
     "d'ordine, minimi, confezioni, sconti, budget e spazio. Li scarichi (CSV) e li invii dal tuo sistema abituale."),
    ("My shelves were sometimes empty: is that a problem?", "A volte i miei scaffali erano vuoti: è un problema?",
     "Yes, partly: a day with no stock looks like a day with no customers, so the forecast of that product comes out "
     "a little low. If it happened often, keep it in mind for those products or remove those days from the file.",
     "Sì, in parte: un giorno senza merce sembra un giorno senza clienti, quindi la previsione di quel prodotto esce un "
     "po' bassa. Se è successo spesso, tienine conto per quei prodotti o togli quei giorni dal file."),
    ("What about products launched a few weeks ago?", "E i prodotti lanciati da poche settimane?",
     "With fewer than 8 weeks of sales they cannot be tested on the past, so the software plans them from a **similar "
     "product** it suggests (first weeks of sales, name, category, supplier, price — with the reasons shown) and you "
     "confirm on the Orders page. The forecast moves to the product's own sales as they come in, with more safety "
     "stock and short lots. Expect a large error in the first weeks: launches are hard for anyone.",
     "Con meno di 8 settimane di vendite non si possono provare sul passato, quindi il software li pianifica da un "
     "**prodotto simile** che propone lui (prime settimane di vendita, nome, categoria, fornitore, prezzo — con i motivi "
     "indicati) e tu confermi nella pagina Ordini. La previsione passa alle vendite del prodotto man mano che arrivano, "
     "con più scorta di sicurezza e lotti brevi. Aspettati un errore alto nelle prime settimane: i lanci sono difficili "
     "per chiunque."),
    ("How are promotions handled?", "Come vengono gestite le promozioni?",
     "They are found from a promo column or from discounted prices; the software measures for each product how much "
     "it sells more during a promotion and less in the week after, uses the promotion calendar in the forecasts and "
     "lets you plan future promotions with their extra stock and margin.",
     "Vengono trovate da una colonna promozione o dai prezzi scontati; il software misura per ogni prodotto quanto vende "
     "in più durante una promozione e in meno la settimana dopo, usa il calendario delle promozioni nelle previsioni e "
     "ti fa pianificare le promozioni future con la loro scorta in più e il margine."),
    ("Which reorder policy should I use?", "Quale politica di riordino dovrei usare?",
     "The software compares three at the same service level: the same order every week (P0), a fixed reorder point "
     "(P1) and a reorder point that follows the forecast (P2). Use the one with less stock and cost on your data — "
     "usually P2, especially for seasonal and promoted products.",
     "Il software ne confronta tre allo stesso livello di servizio: lo stesso ordine ogni settimana (P0), un punto di "
     "riordino fisso (P1) e un punto di riordino che segue la previsione (P2). Usa quella con meno scorta e costi sui "
     "tuoi dati — di solito P2, soprattutto per prodotti stagionali e in promozione."),
    ("What is the fill rate?", "Cos'è il fill rate?",
     "The share of the units customers asked for that you could sell from stock: 95% means 5 units in 100 were "
     "missing. Policies are compared at the same fill rate, so less stock really means a better policy.",
     "La quota dei pezzi richiesti dai clienti che hai potuto vendere dal magazzino: 95% significa che mancavano 5 pezzi "
     "su 100. Le politiche sono confrontate allo stesso fill rate, quindi meno scorta significa davvero una politica "
     "migliore."),
    ("Can I run it on my computer?", "Posso usarlo sul mio computer?",
     "Yes, on Windows and macOS: the README explains it step by step (Python, one install command, one command to "
     "start). Your data then never leave your computer.",
     "Sì, su Windows e macOS: il README lo spiega passo per passo (Python, un comando di installazione, un comando per "
     "avviare). I tuoi dati così non lasciano mai il tuo computer."),
    ("Who made it and how do I report a problem?", "Chi l'ha realizzato e come segnalo un problema?",
     f"It is an open project: the code, the documentation and the tests are on GitHub ({cfg.site.repository}). "
     "Open an issue there to report a problem or ask for a feature.",
     f"È un progetto aperto: codice, documentazione e test sono su GitHub ({cfg.site.repository}). Apri una "
     "segnalazione (issue) lì per un problema o una richiesta."),
]

for q_en, q_it, a_en, a_it in FAQ:
    with st.expander(tx(q_en, q_it)):
        st.markdown(tx(a_en, a_it))

st.divider()
c1, c2 = st.columns(2)
c1.page_link("views/data.py", label=tx("Try it with your data", "Provalo con i tuoi dati"), icon=":material/upload_file:")
c2.page_link("views/summary.py", label=tx("How every number is computed", "Come è calcolato ogni numero"),
             icon=":material/menu_book:")
