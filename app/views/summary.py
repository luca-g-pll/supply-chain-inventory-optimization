"""Summary: plain-language guide to everything the software uses, with worked examples."""

import numpy as np
import pandas as pd
import streamlit as st
from i18n import tx

from src.inventory import eoq, normal_loss, reorder_point, safety_factor, safety_stock, sigma_lead_time
from src.metrics import bias, mape, rmse, wape

st.title(tx("Summary", "Riepilogo"))
st.markdown(tx(
    "Everything the software does, explained simply and with small examples. The numbers in the examples are "
    "computed live with the same functions the analysis uses.",
    "Tutto quello che fa il software, spiegato in modo semplice e con piccoli esempi. I numeri degli esempi sono "
    "calcolati al momento con le stesse funzioni usate dall'analisi."))

(t_data, t_class, t_fc, t_inv, t_sim, t_kpi, t_charts, t_glossary) = st.tabs([
    tx("1 · Data", "1 · Dati"), "2 · ABC / XYZ", tx("3 · Forecasts", "3 · Previsioni"),
    tx("4 · Reorder rules", "4 · Regole di riordino"), tx("5 · Simulation", "5 · Simulazione"),
    tx("6 · KPIs & costs", "6 · KPI e costi"), tx("7 · Reading the charts", "7 · Leggere i grafici"),
    tx("8 · Glossary", "8 · Glossario")])

# --------------------------------------------------------------------------- 1 data
with t_data:
    st.header(tx("What goes in", "Cosa entra nel software"))
    st.markdown(tx("""
The software needs **daily sales**: for each day, which **item** was sold and **how many units**.
Store, price, cost, channel (physical / online) and supplier lead time are optional.

**Example of input rows** (one shop, two days):
""", """
Il software ha bisogno delle **vendite giornaliere**: per ogni giorno, quale **articolo** è stato venduto e **quanti
pezzi**. Negozio, prezzo, costo, canale (fisico / online) e tempo di consegna del fornitore sono facoltativi.

**Esempio di righe in ingresso** (un negozio, due giorni):
"""))
    st.dataframe(pd.DataFrame({
        "date": ["2024-03-01", "2024-03-01", "2024-03-01", "2024-03-02"],
        "store": ["Milano", "Milano", "Milano", "Milano"],
        "item": [tx("Pasta 500g", "Pasta 500g"), tx("Pasta 500g", "Pasta 500g"), tx("Coffee 250g", "Caffè 250g"),
                 tx("Coffee 250g", "Caffè 250g")],
        "quantity": [3, 2, 4, 6], "unit_price": [0.99, 0.99, 3.29, 3.29]}), hide_index=True)
    st.markdown(tx("""
What the software does with them, step by step:

1. **Sums rows of the same day** → Pasta on 1 March = 3 + 2 = **5 units**.
2. **Fills missing days with zero**: if Pasta has no row on 2 March, it sold **0** that day.
3. **Sets returns to zero**: a quantity of −2 (a return) is not demand.
4. **Keeps a price per item**: the median price of the last 90 days, so a promotion at the very end of the
   history does not become the normal price. Missing prices or costs are *assumed* and clearly flagged.
5. **Splits time into three periods** (like studying, a mock exam and the real exam):
""", """
Cosa ne fa il software, passo per passo:

1. **Somma le righe dello stesso giorno** → Pasta il 1° marzo = 3 + 2 = **5 pezzi**.
2. **Riempie i giorni mancanti con zero**: se la pasta non ha righe il 2 marzo, quel giorno ha venduto **0**.
3. **Porta i resi a zero**: una quantità di −2 (un reso) non è domanda.
4. **Tiene un prezzo per articolo**: la mediana dei prezzi degli ultimi 90 giorni, così una promozione proprio alla
   fine dello storico non diventa il prezzo normale. Prezzi o costi mancanti sono *ipotizzati* e segnalati.
5. **Divide il tempo in tre periodi** (come lo studio, la simulazione d'esame e l'esame vero):
"""))
    st.markdown(tx("""
| Period | Used for | Example with 5 years of data |
|---|---|---|
| **Train** | models learn the patterns | 2013 – 2015 |
| **Validation** | choose the best model, set the reorder parameters | 2016 |
| **Test** | final, honest check: models never saw it | 2017 |

The test period is about **a quarter of the history** (at most one year, at least 4 weeks). With 2 years of data:
6 months test, 6 months validation, 1 year training. Yearly seasonality is used only with more than ~400 days of
training history.
""", """
| Periodo | Serve a | Esempio con 5 anni di dati |
|---|---|---|
| **Allenamento** | i modelli imparano gli andamenti | 2013 – 2015 |
| **Validazione** | scegliere il modello migliore e fissare i parametri di riordino | 2016 |
| **Test** | verifica finale e onesta: i modelli non l'hanno mai visto | 2017 |

Il periodo di test è circa **un quarto dello storico** (al massimo un anno, almeno 4 settimane). Con 2 anni di dati:
6 mesi di test, 6 di validazione, 1 anno di allenamento. La stagionalità annuale si usa solo con più di ~400 giorni
di allenamento.
"""))

# --------------------------------------------------------------------------- 2 classes
with t_class:
    st.header(tx("Which products matter, and how predictable they are",
                 "Quali prodotti contano e quanto sono prevedibili"))
    st.markdown(tx("""
**ABC — by money.** Sort items by revenue, from highest to lowest, and add up their shares:

- **A** = the items that together make the first **80%** of revenue (few, very important);
- **B** = the next ones, up to **95%**;
- **C** = the rest (many items, little revenue).
""", """
**ABC — per soldi.** Ordina gli articoli per fatturato, dal più alto al più basso, e somma le loro quote:

- **A** = gli articoli che insieme fanno il primo **80%** del fatturato (pochi, molto importanti);
- **B** = i successivi, fino al **95%**;
- **C** = il resto (tanti articoli, poco fatturato).
"""))
    ex = pd.DataFrame({"item": [tx("Olive oil", "Olio d'oliva"), tx("Coffee", "Caffè"), "Pasta",
                                tx("Sugar", "Zucchero"), tx("Kettle", "Bollitore")],
                       "revenue €": [5000, 3000, 1200, 500, 300]})
    ex["share"] = ex["revenue €"] / ex["revenue €"].sum()
    ex["cumulative"] = ex["share"].cumsum()
    ex["class"] = np.where(ex["cumulative"] <= 0.80 + 1e-9, "A", np.where(ex["cumulative"] <= 0.95 + 1e-9, "B", "C"))
    ex.loc[0, "class"] = "A"
    ex.columns = [tx("item", "articolo"), tx("revenue €", "fatturato €"), tx("share", "quota"),
                  tx("cumulative", "cumulato"), tx("class", "classe")]
    st.dataframe(ex.style.format({ex.columns[2]: "{:.0%}", ex.columns[3]: "{:.0%}", ex.columns[1]: "€{:,.0f}"}),
                 hide_index=True)
    st.markdown(tx("""
**XYZ — by variability.** Take weekly sales and compute the **coefficient of variation**
CV = standard deviation ÷ average:

- **X**: CV < 0.5 → steady (e.g. 100, 95, 105, 98 units a week);
- **Y**: 0.5 – 1.0 → variable;
- **Z**: CV ≥ 1.0 → erratic (e.g. panettone: 0, 0, 0, 300 units).

Weekly totals are used because the weekday pattern (more sales on Saturday) is predictable, not "noise".
""", """
**XYZ — per variabilità.** Prendi le vendite settimanali e calcola il **coefficiente di variazione**
CV = deviazione standard ÷ media:

- **X**: CV < 0,5 → stabile (es. 100, 95, 105, 98 pezzi a settimana);
- **Y**: 0,5 – 1,0 → variabile;
- **Z**: CV ≥ 1,0 → irregolare (es. panettone: 0, 0, 0, 300 pezzi).

Si usano i totali settimanali perché l'andamento nei giorni della settimana (più vendite il sabato) è prevedibile,
non è "rumore".
"""))
    weeks = st.text_input(tx("Try it: weekly sales separated by commas", "Prova: vendite settimanali separate da virgole"),
                          "100, 95, 105, 98, 110, 92")
    try:
        w = np.array([float(x) for x in weeks.split(",") if x.strip()])
        cv = w.std(ddof=1) / w.mean()
        cls = "X" if cv < 0.5 else ("Y" if cv < 1 else "Z")
        st.success(tx(f"Average {w.mean():.1f} · standard deviation {w.std(ddof=1):.1f} · CV = {cv:.2f} → class **{cls}**",
                      f"Media {w.mean():.1f} · deviazione standard {w.std(ddof=1):.1f} · CV = {cv:.2f} → classe **{cls}**"))
    except (ValueError, ZeroDivisionError):
        st.warning(tx("Write at least two numbers separated by commas.", "Scrivi almeno due numeri separati da virgole."))

# --------------------------------------------------------------------------- 3 forecasts
with t_fc:
    st.header(tx("Forecasting: guessing next weeks' sales", "Previsione: stimare le vendite delle prossime settimane"))
    st.markdown(tx("""
Four methods, from the simplest to the smartest. Every week (the *origin*) each method forecasts the next
**1 to 14 days**, using only data up to that day.

| Model | Idea in one line | Example |
|---|---|---|
| **Seasonal naive** (baseline) | "Like last week": next Monday = last Monday | last Monday 40 → forecast 40 |
| **Moving average** | the average of the last 28 days, flat | average 35/day → forecast 35 every day |
| **Holt-Winters** | keeps track of the level, the trend and the weekly pattern, and updates them every day | level 35, growing, Saturday +30% → Saturday 46 |
| **LightGBM** | a machine-learning model that learns from all products and stores together, using calendar and past sales (14+ days old) | learns "Saturday in July, item usually 40 two weeks ago" → 52 |

The model with the lowest error on the **validation** period is chosen; its score on the **test** period is the one
reported.
""", """
Quattro metodi, dal più semplice al più intelligente. Ogni settimana (l'*origine*) ogni metodo prevede i successivi
**da 1 a 14 giorni**, usando solo i dati fino a quel giorno.

| Modello | L'idea in una riga | Esempio |
|---|---|---|
| **Naive stagionale** (base) | "Come la settimana scorsa": il prossimo lunedì = l'ultimo lunedì | ultimo lunedì 40 → previsione 40 |
| **Media mobile** | la media degli ultimi 28 giorni, piatta | media 35/giorno → previsione 35 ogni giorno |
| **Holt-Winters** | tiene traccia del livello, della tendenza e dell'andamento settimanale, e li aggiorna ogni giorno | livello 35, in crescita, sabato +30% → sabato 46 |
| **LightGBM** | un modello di machine learning che impara da tutti i prodotti e negozi insieme, usando calendario e vendite passate (di almeno 14 giorni prima) | impara "sabato di luglio, articolo che vendeva 40 due settimane fa" → 52 |

Si sceglie il modello con l'errore più basso sul periodo di **validazione**; il risultato riportato è quello sul
periodo di **test**.
"""))
    st.subheader(tx("Measuring the error", "Misurare l'errore"))
    st.latex(r"\text{WAPE}=\frac{\sum |A-F|}{\sum A}\qquad \text{Bias}=\frac{\sum (F-A)}{\sum A}\qquad"
             r"\text{MAPE}=\text{mean}\frac{|A-F|}{A}")
    st.markdown(tx("A = actual sales, F = forecast. Edit the example:",
                   "A = vendite reali, F = previsione. Modifica l'esempio:"))
    c1, c2 = st.columns(2)
    a_txt = c1.text_input(tx("Actual (A)", "Reale (A)"), "10, 20, 30, 40")
    f_txt = c2.text_input(tx("Forecast (F)", "Previsto (F)"), "12, 18, 33, 35")
    try:
        a = np.array([float(x) for x in a_txt.split(",")])
        f = np.array([float(x) for x in f_txt.split(",")])
        if len(a) != len(f):
            raise ValueError
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("WAPE", f"{wape(a, f):.1%}", help=tx("Main metric: total error / total sales",
                                                     "Metrica principale: errore totale / vendite totali"))
        m2.metric("MAPE", f"{mape(a, f):.1%}", help=tx("Average % error per day (days with sales only)",
                                                     "Errore % medio per giorno (solo giorni con vendite)"))
        m3.metric("Bias", f"{bias(a, f):+.1%}", help=tx("Positive = forecasts too high, negative = too low",
                                                      "Positivo = previsioni troppo alte, negativo = troppo basse"))
        m4.metric("RMSE", f"{rmse(a, f):.2f}", help=tx("Typical error in units, big errors weigh more",
                                                     "Errore tipico in pezzi, gli errori grandi pesano di più"))
        st.caption(tx(
            f"Here: total error {np.abs(a - f).sum():.0f} units on {a.sum():.0f} sold → WAPE {wape(a, f):.1%}. "
            "\"27% less error than the baseline\" means 1 − WAPE(model) / WAPE(baseline).",
            f"Qui: errore totale {np.abs(a - f).sum():.0f} pezzi su {a.sum():.0f} venduti → WAPE {wape(a, f):.1%}. "
            "\"27% di errore in meno della base\" significa 1 − WAPE(modello) / WAPE(base)."))
    except ValueError:
        st.warning(tx("Write the same number of values in both boxes.", "Scrivi lo stesso numero di valori nei due campi."))

# --------------------------------------------------------------------------- 4 reorder rules
with t_inv:
    st.header(tx("Reorder rules: when and how much to order", "Regole di riordino: quando e quanto ordinare"))
    st.markdown(tx("""
Three ideas cover everything:

- **Reorder point (s)** → *when*: order as soon as the stock (on hand + already ordered) drops to *s*.
- **Lot size (Q)** → *how much* to order each time.
- **Safety stock (SS)** → the extra cushion inside *s* for when demand is higher than expected.
""", """
Tre idee coprono tutto:

- **Punto di riordino (s)** → *quando*: ordina appena la scorta (a magazzino + già ordinata) scende a *s*.
- **Lotto (Q)** → *quanto* ordinare ogni volta.
- **Scorta di sicurezza (SS)** → il cuscinetto in più dentro *s* per quando la domanda supera le attese.
"""))
    st.latex(r"SS = z\cdot\sigma_L \qquad s = \bar d \cdot L + SS \qquad Q^* = \sqrt{\frac{2\,D\,S}{h}}")
    st.caption(tx("D = yearly demand, S = cost of one order, h = yearly cost of holding one unit. "
                  "The lot Q* is capped at 28 days of demand.",
                  "D = domanda annua, S = costo di un ordine, h = costo annuo per tenere un pezzo a magazzino. "
                  "Il lotto Q* ha un tetto di 28 giorni di domanda."))
    st.markdown(tx("Try it with your numbers:", "Prova con i tuoi numeri:"))
    c1, c2, c3 = st.columns(3)
    d = c1.number_input(tx("Average daily demand d̄ (units)", "Domanda media giornaliera d̄ (pezzi)"), 0.1, 10_000.0, 20.0)
    sd = c1.number_input(tx("Daily standard deviation σ (units)", "Deviazione standard giornaliera σ (pezzi)"),
                         0.0, 10_000.0, 6.0)
    L = c2.number_input(tx("Lead time L (days)", "Tempo di consegna L (giorni)"), 1, 60, 7)
    sl = c2.slider(tx("Service target", "Livello di servizio"), 0.80, 0.995, 0.95, 0.005)
    S = c3.number_input(tx("Cost per order S (€)", "Costo per ordine S (€)"), 1.0, 5_000.0, 50.0)
    cost = c3.number_input(tx("Unit cost c (€)", "Costo unitario c (€)"), 0.01, 10_000.0, 6.0)
    i = c3.number_input(tx("Holding rate i (per year)", "Tasso di mantenimento i (all'anno)"), 0.01, 1.0, 0.25)
    z = safety_factor(sl)
    sig_L = sigma_lead_time(sd, L)
    ss = safety_stock(z, sig_L)
    s = reorder_point(d, L, ss)
    q_raw = eoq(365 * d, S, i * cost)
    q = min(np.ceil(q_raw), np.ceil(28 * d))
    st.markdown(tx(f"""
| Step | Formula | Result |
|---|---|---|
| Safety factor | z = value of the normal curve for {sl:.1%} | **{z:.2f}** |
| Uncertainty over the lead time | σ_L = σ · √L = {sd:g} · √{L} | **{sig_L:.1f}** units |
| Safety stock | SS = z · σ_L | **{ss:.0f}** units |
| Reorder point | s = d̄ · L + SS = {d:g} · {L} + {ss:.0f} | **{s:.0f}** units |
| Economic lot (EOQ) | √(2 · D · S / h), D = {365 * d:,.0f}/year, h = i · c = €{i * cost:.2f} | {q_raw:,.0f} units ({q_raw / d:.0f} days) |
| Lot used | at most 28 days of demand ({28 * d:,.0f}) | **{q:,.0f}** units |

**In words:** when stock falls to **{s:.0f}** units, order **{q:,.0f}**. The order arrives in {L} days; meanwhile
about {d * L:.0f} units are sold on average, and the {ss:.0f} extra units cover a bad week.
""", f"""
| Passo | Formula | Risultato |
|---|---|---|
| Fattore di sicurezza | z = valore della curva normale per il {sl:.1%} | **{z:.2f}** |
| Incertezza nel tempo di consegna | σ_L = σ · √L = {sd:g} · √{L} | **{sig_L:.1f}** pezzi |
| Scorta di sicurezza | SS = z · σ_L | **{ss:.0f}** pezzi |
| Punto di riordino | s = d̄ · L + SS = {d:g} · {L} + {ss:.0f} | **{s:.0f}** pezzi |
| Lotto economico (EOQ) | √(2 · D · S / h), D = {365 * d:,.0f}/anno, h = i · c = €{i * cost:.2f} | {q_raw:,.0f} pezzi ({q_raw / d:.0f} giorni) |
| Lotto usato | al massimo 28 giorni di domanda ({28 * d:,.0f}) | **{q:,.0f}** pezzi |

**In parole:** quando la scorta scende a **{s:.0f}** pezzi, ordina **{q:,.0f}** pezzi. L'ordine arriva in {L} giorni;
nel frattempo se ne vendono in media circa {d * L:.0f}, e i {ss:.0f} pezzi in più coprono una settimana storta.
"""))
    st.info(tx("**EOQ** balances two costs: big lots mean few orders (low ordering cost) but more stock on the shelf "
               "(high holding cost). The cap stops cheap fast movers from getting lots of several months.",
               "L'**EOQ** bilancia due costi: lotti grandi significano pochi ordini (basso costo d'ordine) ma più merce "
               "a scaffale (alto costo di mantenimento). Il tetto evita che gli articoli economici e molto venduti "
               "abbiano lotti di diversi mesi."))
    st.subheader(tx("The three policies compared", "Le tre politiche a confronto"))
    st.markdown(tx("""
| Policy | Rule | Weak point |
|---|---|---|
| **P0 · naive weekly** | every Monday order last year's average weekly demand | ignores stock, trend and seasons: too much in winter, too little in summer |
| **P1 · static (s, Q)** | reorder point from last year's average and variability | same *s* all year: follows neither growth nor seasons |
| **P2 · forecast-driven (s, Q)** | *s* changes every day = forecast of the next L days + safety stock on the forecast error | needs a good forecast (that is why we measure it) |

Example — panettone: in July P1 still keeps its reorder point (stock waits on the shelf for months), while P2's
forecast is near zero so it holds almost nothing; in December P2's forecast rises and it orders ahead of the peak.
""", """
| Politica | Regola | Punto debole |
|---|---|---|
| **P0 · ordine fisso settimanale** | ogni lunedì ordina la domanda settimanale media dell'anno prima | ignora scorta, tendenza e stagioni: troppo in inverno, troppo poco in estate |
| **P1 · statica (s, Q)** | punto di riordino dalla media e dalla variabilità dell'anno prima | stesso *s* tutto l'anno: non segue né la crescita né le stagioni |
| **P2 · guidata dalla previsione (s, Q)** | *s* cambia ogni giorno = previsione dei prossimi L giorni + scorta di sicurezza sull'errore di previsione | serve una buona previsione (per questo la misuriamo) |

Esempio — panettone: a luglio P1 mantiene il suo punto di riordino (la merce resta a scaffale per mesi), mentre la
previsione di P2 è quasi zero e quindi tiene pochissimo; a dicembre la previsione di P2 sale e ordina in anticipo
sul picco.
"""))

# --------------------------------------------------------------------------- 5 simulation
with t_sim:
    st.header(tx("The simulation: replaying the test period day by day",
                 "La simulazione: rigiocare il periodo di test giorno per giorno"))
    st.markdown(tx("""
For every store-item and every policy, the software replays each day of the test period with the **real sales**:

1. **Morning** — orders placed L days ago arrive and go on the shelf (online: waiting customers are served first).
2. **During the day** — customers buy. If the shelf is empty, the sale is **lost** (online: 80% of customers
   wait for the next delivery, 20% give up).
3. **Evening** — the policy looks at *stock on hand + stock already ordered − customers waiting* and decides
   whether to order. The order will arrive in L days.
4. The software records the stock left, the sales, the lost sales and the orders.
""", """
Per ogni negozio-articolo e ogni politica, il software rigioca ogni giorno del periodo di test con le **vendite
reali**:

1. **Mattina** — arrivano gli ordini fatti L giorni prima e vanno a scaffale (online: si servono prima i clienti in
   attesa).
2. **Durante il giorno** — i clienti comprano. Se lo scaffale è vuoto, la vendita è **persa** (online: l'80% dei
   clienti aspetta la consegna successiva, il 20% rinuncia).
3. **Sera** — la politica guarda *scorta a magazzino + scorta già ordinata − clienti in attesa* e decide se ordinare.
   L'ordine arriverà dopo L giorni.
4. Il software registra la scorta rimasta, le vendite, le vendite perse e gli ordini.
"""))
    st.markdown(tx("**Small example** (reorder point 30, lot 50, lead time 2 days, start with 40 units):",
                   "**Piccolo esempio** (punto di riordino 30, lotto 50, consegna in 2 giorni, si parte con 40 pezzi):"))
    rows, on_hand, pipeline = [], 40, {}
    for day, demand in enumerate([12, 15, 10, 18, 14, 9], start=1):
        arriving = pipeline.pop(day, 0)
        on_hand += arriving
        sold = min(on_hand, demand)
        on_hand -= sold
        position = on_hand + sum(pipeline.values())
        order = 50 if position <= 30 else 0
        if order:
            pipeline[day + 2] = pipeline.get(day + 2, 0) + order
        rows.append({tx("day", "giorno"): day, tx("arrives", "arriva"): arriving, tx("demand", "domanda"): demand,
                     tx("sold", "venduto"): sold, tx("lost", "perso"): demand - sold,
                     tx("stock in the evening", "scorta la sera"): on_hand,
                     tx("stock + on order", "scorta + in arrivo"): position,
                     tx("order placed", "ordine fatto"): order})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown(tx("""
**Fair comparison.** The three policies would naturally give different service levels, so comparing their stock
would be unfair. For every product the software finds the **smallest safety stock** (or, for P0, the smallest weekly
quantity) that still serves **95% of demand**, and only then compares the stock. This is the *calibrated* view;
the *fixed* view uses the textbook z = 1.645 instead.
""", """
**Confronto equo.** Le tre politiche darebbero livelli di servizio diversi, quindi confrontarne la scorta sarebbe
ingiusto. Per ogni prodotto il software trova la **scorta di sicurezza più piccola** (o, per P0, la quantità
settimanale più piccola) che serve ancora il **95% della domanda**, e solo dopo confronta la scorta. Questa è la
vista *calibrata*; la vista *fissa* usa invece la z da manuale = 1,645.
"""))
    st.markdown(tx("**Why 95% \"cycle service\" is not 95% \"fill rate\":**",
                   "**Perché il 95% di \"servizio per ciclo\" non è il 95% di \"fill rate\":**"))
    q_ex, sig_ex = 400.0, 60.0
    short = sig_ex * normal_loss(1.645)
    st.markdown(tx(f"""
- *Cycle service level* = share of replenishment cycles without any stock-out (z = 1.645 → 95%).
- *Fill rate* = share of **units** sold from stock. With a lot of {q_ex:.0f} units and σ_L = {sig_ex:.0f}, the
  expected shortage per cycle is σ_L · G(z) = {sig_ex:.0f} · {normal_loss(1.645):.3f} = {short:.1f} units, so the
  fill rate is 1 − {short:.1f}/{q_ex:.0f} = **{1 - short / q_ex:.1%}**, far above 95%. Big lots already protect most
  demand.
""", f"""
- *Servizio per ciclo* = quota dei cicli di riordino senza nessuna rottura di stock (z = 1,645 → 95%).
- *Fill rate* = quota di **pezzi** venduti dal magazzino. Con un lotto di {q_ex:.0f} pezzi e σ_L = {sig_ex:.0f}, la
  mancanza attesa per ciclo è σ_L · G(z) = {sig_ex:.0f} · {normal_loss(1.645):.3f} = {short:.1f} pezzi, quindi il
  fill rate è 1 − {short:.1f}/{q_ex:.0f} = **{1 - short / q_ex:.1%}**, ben sopra il 95%. I lotti grandi proteggono
  già gran parte della domanda.
"""))

# --------------------------------------------------------------------------- 6 KPIs
with t_kpi:
    st.header(tx("KPIs and costs", "KPI e costi"))
    st.markdown(tx("""
| KPI | Meaning | Example |
|---|---|---|
| **Fill rate** | units sold from stock ÷ units demanded | 950 sold of 1,000 asked → **95%** |
| Cycle service level | replenishment cycles without stock-outs | 19 of 20 cycles fine → 95% |
| **Average inventory (€)** | average value of the stock on the shelf (unit cost × units) | 200 units × €6 → **€1,200** |
| Days of supply | how many days the average stock would last | 200 units ÷ 20 per day → **10 days** |
| Inventory turns | how many times a year the stock is renewed | 20 units/day × 365 × €6 = €43,800 cost of sales ÷ €1,200 stock → **36** |
| Stock-out days | days with at least one lost sale | 4 days in December |
| Orders | number of orders placed | 13 a year (every 4 weeks) |

**Costs** (over the test period):

| Cost | Formula | Example |
|---|---|---|
| Ordering | orders × cost per order | 13 × €50 = **€650** |
| Holding | average stock value × holding rate | €1,200 × 25% = **€300** a year |
| Shortage | lost units × lost margin (price − cost) + €2 per online customer kept waiting | 50 × (€10 − €6) = **€200** |
| **Total** | sum of the three | **€1,150** |

When several products or stores are combined, the software **adds up the totals first** and then computes the
ratios: the fill rate of two products is (sold₁ + sold₂) ÷ (demand₁ + demand₂), never the average of the two
percentages.
""", """
| KPI | Significato | Esempio |
|---|---|---|
| **Fill rate** | pezzi venduti dal magazzino ÷ pezzi richiesti | 950 venduti su 1.000 richiesti → **95%** |
| Servizio per ciclo | cicli di riordino senza rotture di stock | 19 cicli su 20 senza problemi → 95% |
| **Scorta media (€)** | valore medio della merce a scaffale (costo unitario × pezzi) | 200 pezzi × €6 → **€1.200** |
| Giorni di copertura | quanti giorni durerebbe la scorta media | 200 pezzi ÷ 20 al giorno → **10 giorni** |
| Rotazione | quante volte all'anno si rinnova la scorta | 20 pezzi/giorno × 365 × €6 = €43.800 di costo del venduto ÷ €1.200 di scorta → **36** |
| Giorni di rottura | giorni con almeno una vendita persa | 4 giorni a dicembre |
| Ordini | numero di ordini fatti | 13 all'anno (ogni 4 settimane) |

**Costi** (sul periodo di test):

| Costo | Formula | Esempio |
|---|---|---|
| Ordini | ordini × costo per ordine | 13 × €50 = **€650** |
| Mantenimento | valore medio della scorta × tasso di mantenimento | €1.200 × 25% = **€300** all'anno |
| Rotture | pezzi persi × margine perso (prezzo − costo) + €2 per cliente online fatto aspettare | 50 × (€10 − €6) = **€200** |
| **Totale** | somma dei tre | **€1.150** |

Quando si combinano più prodotti o negozi, il software **somma prima i totali** e poi calcola i rapporti: il fill rate
di due prodotti è (venduto₁ + venduto₂) ÷ (domanda₁ + domanda₂), mai la media delle due percentuali.
"""))

# --------------------------------------------------------------------------- 7 charts
with t_charts:
    st.header(tx("How to read the charts", "Come leggere i grafici"))
    st.markdown(tx("""
| Page | Chart | How to read it |
|---|---|---|
| Overview | **Average inventory at the target fill rate** | shorter bar = less money tied up in stock for the same service; the blue bar is the forecast-driven policy |
| Overview | **Total cost by component** | blue = ordering, orange = holding, green = lost sales; the shorter the bar, the cheaper the policy |
| Demand | **Units per day, by month** | the line shows trend (going up?) and seasons (summer peaks?) |
| Demand | **Day-of-week / month index** | 1.0 = an average day; 1.2 = 20% busier, 0.8 = 20% quieter |
| Demand | **Pareto curve** | each dot is a product, from the best-selling; when the curve reaches 80%, you have found the A items |
| Demand | **ABC × XYZ matrix** | top-left (AX) = important and predictable; bottom-right (CZ) = small and erratic |
| Forecast | **Leaderboard** | WAPE of each model; lower is better, the blue bar is the chosen model |
| Forecast | **Error by days ahead** | how the error grows when forecasting further into the future |
| Forecast | **Actual vs. forecast** | gray = what really sold, blue = the chosen model's forecast |
| Inventory | **Inventory vs. fill rate** | each line is a policy with more or less safety stock; **the lower the line, the better** (less stock for the same service); big dots = calibrated to the target |
| Inventory | **Total cost vs. fill rate** | where the cheapest service level lies with the current cost assumptions |
| Inventory | **Units on hand, day by day** | the saw-tooth: a delivery makes it jump up, sales bring it down; red ticks = days with lost sales |
| Inventory | **Sensitivity** | how much stock you need for longer lead times (darker line) or higher targets |
""", """
| Pagina | Grafico | Come leggerlo |
|---|---|---|
| Panoramica | **Scorta media al fill rate obiettivo** | barra più corta = meno soldi fermi in magazzino a parità di servizio; la barra blu è la politica guidata dalla previsione |
| Panoramica | **Costo totale per componente** | blu = ordini, arancione = mantenimento, verde = vendite perse; più corta la barra, più economica la politica |
| Domanda | **Pezzi al giorno, per mese** | la linea mostra la tendenza (in crescita?) e le stagioni (picchi d'estate?) |
| Domanda | **Indice per giorno / per mese** | 1,0 = un giorno medio; 1,2 = 20% più vendite, 0,8 = 20% in meno |
| Domanda | **Curva di Pareto** | ogni punto è un prodotto, dal più venduto; quando la curva arriva all'80% hai trovato gli articoli A |
| Domanda | **Matrice ABC × XYZ** | in alto a sinistra (AX) = importanti e prevedibili; in basso a destra (CZ) = piccoli e irregolari |
| Previsioni | **Classifica** | WAPE di ogni modello; più basso è meglio, la barra blu è il modello scelto |
| Previsioni | **Errore per giorni di anticipo** | come cresce l'errore prevedendo più lontano nel futuro |
| Previsioni | **Reale vs. previsto** | grigio = quanto si è venduto davvero, blu = la previsione del modello scelto |
| Scorte | **Scorta vs. fill rate** | ogni linea è una politica con più o meno scorta di sicurezza; **più bassa è la linea, meglio è** (meno scorta a parità di servizio); i pallini grandi = calibrati sull'obiettivo |
| Scorte | **Costo totale vs. fill rate** | dove si trova il livello di servizio più economico con le ipotesi di costo attuali |
| Scorte | **Pezzi a magazzino, giorno per giorno** | il dente di sega: una consegna lo fa salire, le vendite lo fanno scendere; tacche rosse = giorni con vendite perse |
| Scorte | **Sensibilità** | quanta scorta serve con tempi di consegna più lunghi (linea più scura) o obiettivi più alti |
"""))

# --------------------------------------------------------------------------- 8 glossary
with t_glossary:
    st.header(tx("Glossary", "Glossario"))
    st.markdown(tx("""
| Term | Meaning |
|---|---|
| **Series** | one product in one store (8 stores × 32 products = 256 series) |
| **Lead time (L)** | days between placing an order and receiving it |
| **Inventory position** | stock on hand + stock already ordered − customers waiting |
| **Back-test** | testing a model on past data as if it were the future |
| **Origin** | the day a forecast is made (one every 7 days) |
| **Horizon** | how many days ahead a forecast looks (1 to 14) |
| **Leakage** | accidentally using future information; the software is tested to prevent it |
| **Calibration** | tuning each policy so that all reach the same fill rate before comparing them |
| **Scenario** | the stores, products and channels selected in the sidebar |
| **Physical / online channel** | in a physical store an unmet customer leaves; online, most customers wait for the delivery |
| **Fast mode** | quicker settings used for files uploaded in the dashboard (models fitted once) |
| **Assumed value** | a price, cost or lead time not present in the data and replaced by a default |
""", """
| Termine | Significato |
|---|---|
| **Serie** | un prodotto in un negozio (8 negozi × 32 prodotti = 256 serie) |
| **Tempo di consegna (L)** | giorni tra l'invio di un ordine e il suo arrivo |
| **Posizione di magazzino** | scorta a magazzino + scorta già ordinata − clienti in attesa |
| **Back-test** | provare un modello su dati passati come se fossero il futuro |
| **Origine** | il giorno in cui si fa una previsione (una ogni 7 giorni) |
| **Orizzonte** | quanti giorni avanti guarda una previsione (da 1 a 14) |
| **Leakage** | usare per errore informazioni dal futuro; il software è testato per evitarlo |
| **Calibrazione** | regolare ogni politica perché tutte raggiungano lo stesso fill rate prima di confrontarle |
| **Scenario** | i negozi, i prodotti e i canali scelti nella barra laterale |
| **Canale fisico / online** | in un negozio fisico il cliente non servito se ne va; online, la maggior parte aspetta la consegna |
| **Modalità veloce** | impostazioni più rapide usate per i file caricati nella dashboard (modelli stimati una volta) |
| **Valore ipotizzato** | un prezzo, costo o tempo di consegna assente nei dati e sostituito da un valore predefinito |
"""))
    st.caption(tx("More detail: docs/08_methodology.md and docs/09_kpi_definitions.md in the project folder.",
                  "Più dettagli: docs/08_methodology.md e docs/09_kpi_definitions.md nella cartella del progetto."))
