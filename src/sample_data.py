"""Hypothetical sales data of a fictional Italian retailer, for trying the software.

Unlike the Kaggle demo (very regular: every item is class X), these files contain
what real sales histories contain: seasonal products, promotions, Italian
holidays and closures, an online store, a store that opens later and one that
closes for renovation, a product launch, a discontinued product, intermittent
and lumpy demand, price changes and transaction-level exports with returns.

    python -m src.sample_data --out data/samples        # same seed -> same files

Files (see data/samples/README.md for the story behind each one):

* ``retail_chain_sales.csv``   8 stores (7 physical + 1 online) x 32 products x 3 years
* ``retail_chain_items.csv``   item master: name, category, supplier, price, cost, lead time
* ``retail_chain_stores.csv``  store list (for reference)
* ``single_store_it.csv``      one neighbourhood shop, Italian headers, ``;``, dd/mm/yyyy, decimal commas
* ``ecommerce_order_lines.xlsx``  one year of online order lines with returns (negative quantities)

All of it is invented: no real company, store or sales figure is represented.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter

SUPPLIERS = {
    "BEV": ("Beverage distributor (regional)", 2),
    "GRO": ("Grocery wholesaler", 4),
    "HOM": ("Household goods supplier", 6),
    "SEA": ("Confectionery (seasonal)", 12),
    "APP": ("Small appliances importer", 21),
}
CATEGORIES = {"BEV": "Beverages", "GRO": "Grocery", "HOM": "Household", "SEA": "Seasonal confectionery",
              "APP": "Small appliances"}
PHYSICAL_WEEK = (0.85, 0.85, 0.90, 0.95, 1.15, 1.45, 0.85)   # Mon .. Sun
ONLINE_WEEK = (1.25, 1.15, 1.00, 0.95, 0.90, 0.80, 0.95)
ONLINE_CATEGORY_MIX = {"BEV": 1.3, "GRO": 0.7, "HOM": 1.2, "SEA": 0.8, "APP": 3.0}   # heavy / bulky sells online
LIST_PRICE_INDEX = {2022: 1.00, 2023: 1.05, 2024: 1.08}                              # yearly price increases


@dataclass(frozen=True)
class StoreSpec:
    code: str
    name: str
    city: str
    channel: str                       # physical | online
    size: float                        # traffic multiplier
    weekly: tuple[float, ...]          # Mon .. Sun
    closed_weekdays: tuple[int, ...] = ()
    open_from: str | None = None       # store opening date
    closures: tuple[tuple[str, str], ...] = ()
    august_dip: float = 0.0            # share of customers away in mid-August
    growth: float = 0.0                # yearly traffic growth
    note: str = ""


@dataclass(frozen=True)
class ProductSpec:
    code: str
    name: str
    price: float                       # 2022 list price, EUR
    cost_ratio: float                  # unit cost / price
    base: float                        # units per day in a size-1 store (at the seasonal peak for seasonal items)
    season: str = "flat"               # flat | summer | winter | christmas | easter
    amplitude: float = 0.03
    dispersion: float = 20.0           # negative-binomial k (lower = noisier)
    promos_per_year: float = 0.0
    weather: float = 0.0               # sensitivity to heat waves
    trend: float = 0.0                 # yearly growth
    launch: str | None = None
    discontinued: str | None = None
    lumpy: bool = False                # occasional large B2B orders on top of retail demand

    @property
    def family(self) -> str:
        return self.code.split("-")[0]


CHAIN_STORES = (
    StoreSpec("MI01", "Milano Centro", "Milano", "physical", 1.40, PHYSICAL_WEEK, august_dip=0.35),
    StoreSpec("MI02", "Milano Città Studi", "Milano", "physical", 1.00, PHYSICAL_WEEK, closed_weekdays=(6,),
              august_dip=0.35, note="closed on Sundays"),
    StoreSpec("TO01", "Torino Lingotto", "Torino", "physical", 1.10, PHYSICAL_WEEK,
              closures=(("2023-08-07", "2023-08-20"),), august_dip=0.30, note="closed for renovation 7–20 Aug 2023"),
    StoreSpec("BO01", "Bologna Centro", "Bologna", "physical", 0.90, PHYSICAL_WEEK, closed_weekdays=(6,),
              august_dip=0.25, note="closed on Sundays"),
    StoreSpec("FI01", "Firenze Campo di Marte", "Firenze", "physical", 0.80, PHYSICAL_WEEK, august_dip=0.10),
    StoreSpec("RM01", "Roma Prati", "Roma", "physical", 1.30, PHYSICAL_WEEK, august_dip=0.20),
    StoreSpec("NA01", "Napoli Vomero", "Napoli", "physical", 1.00, PHYSICAL_WEEK, open_from="2022-06-01",
              august_dip=0.10, note="opened on 1 Jun 2022 (ramp-up over 6 months)"),
    StoreSpec("WEB", "Online shop", "Italy", "online", 0.55, ONLINE_WEEK, growth=0.25, august_dip=0.10,
              note="e-commerce, +25% per year, Black Friday peak"),
)

CHAIN_PRODUCTS = (
    # beverages: summer peak, heat waves, frequent promotions
    ProductSpec("BEV-001", "Acqua naturale 6x1.5L", 2.49, 0.55, 22, "summer", 0.35, 25, 5, weather=1.0),
    ProductSpec("BEV-002", "Acqua frizzante 6x1.5L", 2.69, 0.55, 14, "summer", 0.30, 25, 5, weather=0.8),
    ProductSpec("BEV-003", "Tè freddo pesca 1.5L", 1.39, 0.60, 9, "summer", 0.60, 12, 6, weather=1.3),
    ProductSpec("BEV-004", "Birra lager 66cl", 1.29, 0.62, 12, "summer", 0.45, 10, 6, weather=1.1, trend=0.03),
    ProductSpec("BEV-005", "Succo d'arancia 1L", 1.79, 0.60, 6, "winter", 0.15, 15, 4),
    ProductSpec("BEV-006", "Ghiaccioli al limone x10", 2.99, 0.58, 4, "summer", 1.6, 6, 2, weather=1.6),
    ProductSpec("BEV-007", "Bevanda energetica zero 33cl", 1.49, 0.55, 7, "summer", 0.20, 8, 4, trend=0.15,
                launch="2023-03-01"),
    ProductSpec("BEV-008", "Acqua naturale cassa 24 bottiglie", 5.99, 0.55, 0.3, "summer", 0.30, 5, 0,
                lumpy=True),
    # grocery staples: stable, promotions drive the variability
    ProductSpec("GRO-001", "Spaghetti 500g", 0.99, 0.65, 26, "flat", 0.03, 30, 6),
    ProductSpec("GRO-002", "Penne rigate 500g", 0.99, 0.65, 20, "flat", 0.03, 30, 6),
    ProductSpec("GRO-003", "Riso Carnaroli 1kg", 3.49, 0.62, 5, "winter", 0.15, 20, 3),
    ProductSpec("GRO-004", "Olio extravergine di oliva 1L", 8.99, 0.72, 6, "flat", 0.05, 18, 4, trend=-0.06),
    ProductSpec("GRO-005", "Passata di pomodoro 700g", 1.19, 0.60, 15, "summer", 0.12, 25, 5),
    ProductSpec("GRO-006", "Caffè macinato 250g", 3.29, 0.66, 9, "winter", 0.10, 25, 5),
    ProductSpec("GRO-007", "Biscotti frollini 700g", 2.79, 0.60, 8, "winter", 0.15, 15, 6),
    ProductSpec("GRO-008", "Zucchero 1kg", 1.15, 0.70, 5, "flat", 0.03, 30, 2, trend=-0.04),
    ProductSpec("GRO-009", "Farina 00 1kg", 0.89, 0.68, 6, "winter", 0.25, 20, 2),
    ProductSpec("GRO-010", "Cioccolata calda in polvere", 2.49, 0.58, 3, "winter", 0.90, 10, 2),
    # household
    ProductSpec("HOM-001", "Carta igienica 8 rotoli", 3.99, 0.60, 8, "flat", 0.03, 20, 6),
    ProductSpec("HOM-002", "Carta cucina 4 rotoli", 2.99, 0.60, 5, "flat", 0.03, 20, 5),
    ProductSpec("HOM-003", "Detersivo piatti 1L", 1.99, 0.55, 4, "flat", 0.03, 18, 4),
    ProductSpec("HOM-004", "Sapone mani 500ml", 1.79, 0.55, 3, "winter", 0.20, 15, 3),
    ProductSpec("HOM-005", "Detersivo lavatrice 30 lavaggi", 7.99, 0.58, 4, "flat", 0.03, 15, 4,
                discontinued="2023-09-30"),
    ProductSpec("HOM-006", "Detersivo lavatrice EcoPlus 30 lavaggi", 8.49, 0.56, 4.4, "flat", 0.03, 15, 4,
                trend=0.08, launch="2023-10-01"),
    ProductSpec("HOM-007", "Crema solare SPF30 200ml", 12.90, 0.50, 2.2, "summer", 1.8, 5, 2, weather=1.2),
    # seasonal confectionery: sells only around Christmas / Easter
    ProductSpec("SEA-001", "Panettone classico 1kg", 9.90, 0.55, 6, "christmas", 0.0, 6),
    ProductSpec("SEA-002", "Pandoro 1kg", 9.50, 0.55, 5, "christmas", 0.0, 6),
    ProductSpec("SEA-003", "Colomba pasquale 750g", 8.90, 0.55, 4, "easter", 0.0, 6),
    # small appliances: slow, intermittent demand, long lead time
    ProductSpec("APP-001", "Bollitore elettrico 1.7L", 24.90, 0.62, 0.16, "flat", 0.03, 50, 1),
    ProductSpec("APP-002", "Tostapane 2 fette", 29.90, 0.62, 0.10, "winter", 0.20, 50, 1),
    ProductSpec("APP-003", "Lampadine LED E27 x3", 9.90, 0.55, 0.9, "winter", 0.30, 10, 2),
    ProductSpec("APP-004", "Ventilatore da tavolo", 34.90, 0.60, 0.25, "summer", 2.0, 3, 1, weather=1.5),
)

SHOP = StoreSpec("BTG", "Bottega di quartiere", "Torino", "physical", 0.45, PHYSICAL_WEEK, closed_weekdays=(6,),
                 closures=(("2023-08-07", "2023-08-20"), ("2024-08-12", "2024-08-25")),
                 note="closed on Sundays and for two weeks around Ferragosto")
SHOP_PRODUCTS = ("BEV-001", "BEV-002", "BEV-004", "GRO-001", "GRO-002", "GRO-004", "GRO-005", "GRO-006",
                 "GRO-007", "GRO-008", "HOM-001", "HOM-003", "HOM-004", "SEA-001", "APP-003")
ECOMMERCE_PRODUCTS = ("BEV-001", "BEV-002", "BEV-007", "GRO-004", "GRO-006", "HOM-001", "HOM-002", "HOM-006",
                      "HOM-007", "SEA-001", "SEA-002", "APP-001", "APP-002", "APP-003", "APP-004")


# --------------------------------------------------------------------------- calendar
def italian_holidays(years) -> dict[pd.Timestamp, str]:
    """Italian public holidays plus Easter Sunday."""
    out: dict[pd.Timestamp, str] = {}
    for y in years:
        e = pd.Timestamp(easter(y))
        for d, name in ((f"{y}-01-01", "Capodanno"), (f"{y}-01-06", "Epifania"), (e, "Pasqua"),
                        (e + pd.Timedelta(days=1), "Pasquetta"), (f"{y}-04-25", "Liberazione"),
                        (f"{y}-05-01", "Festa del lavoro"), (f"{y}-06-02", "Festa della Repubblica"),
                        (f"{y}-08-15", "Ferragosto"), (f"{y}-11-01", "Ognissanti"), (f"{y}-12-08", "Immacolata"),
                        (f"{y}-12-25", "Natale"), (f"{y}-12-26", "Santo Stefano")):
            out[pd.Timestamp(d)] = name
    return out


def _ramp(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0) ** 2


def seasonal_profile(dates: pd.DatetimeIndex, season: str, amplitude: float) -> np.ndarray:
    """Multiplier of the base demand on each day."""
    doy = dates.dayofyear.to_numpy()
    if season == "summer":
        return np.clip(1 + amplitude * np.cos(2 * np.pi * (doy - 196) / 365.25), 0.03, None)
    if season == "winter":
        return np.clip(1 + amplitude * np.cos(2 * np.pi * (doy - 15) / 365.25), 0.03, None)
    if season == "flat":
        return 1 + amplitude * np.cos(2 * np.pi * (doy - 15) / 365.25)
    out = np.zeros(len(dates))
    if season == "christmas":
        for y in sorted(set(dates.year)):
            start, peak = pd.Timestamp(f"{y}-11-01"), pd.Timestamp(f"{y}-12-24")
            span = (peak - start).days
            m = (dates >= start) & (dates <= peak)
            out[m] = 0.05 + 0.95 * _ramp((dates[m] - start).days.to_numpy() / span)
            out[(dates >= pd.Timestamp(f"{y}-12-26")) & (dates <= pd.Timestamp(f"{y}-12-31"))] = 0.18
            out[(dates >= pd.Timestamp(f"{y}-01-01")) & (dates <= pd.Timestamp(f"{y}-01-06"))] = 0.05
        return out
    if season == "easter":
        for y in sorted(set(dates.year)):
            e = pd.Timestamp(easter(y))
            start = e - pd.Timedelta(days=35)
            m = (dates >= start) & (dates < e)
            out[m] = 0.06 + 0.94 * _ramp((dates[m] - start).days.to_numpy() / 34)
            after = (dates > e) & (dates <= e + pd.Timedelta(days=7))
            out[after] = 0.12
        return out
    raise ValueError(f"Unknown season {season!r}")


def clearance_discount(dates: pd.DatetimeIndex, season: str) -> np.ndarray:
    """Price cut on seasonal leftovers (after Christmas / Easter)."""
    disc = np.zeros(len(dates))
    if season == "christmas":
        m = (dates.month == 12) & (dates.day >= 26) | (dates.month == 1) & (dates.day <= 6)
        disc[m] = 0.40
    if season == "easter":
        for y in sorted(set(dates.year)):
            e = pd.Timestamp(easter(y))
            disc[(dates > e) & (dates <= e + pd.Timedelta(days=7))] = 0.40
    return disc


def promo_calendar(dates: pd.DatetimeIndex, per_year: float, rng: np.random.Generator):
    """Chain-wide flyer promotions: (uplift, discount) per day, with a post-promo dip."""
    n = len(dates)
    uplift = np.ones(n)
    discount = np.zeros(n)
    if per_year <= 0:
        return uplift, discount
    expected = per_year * n / 365.25
    day = int(rng.integers(5, 40))
    while day < n:
        length = int(rng.choice([7, 7, 14]))
        lift = float(rng.uniform(1.4, 2.1))
        cut = float(rng.choice([0.20, 0.25, 0.30]))
        uplift[day: day + length] = lift
        discount[day: day + length] = cut
        uplift[day + length: day + length + 7] = np.minimum(uplift[day + length: day + length + 7], 0.85)
        day += length + int(rng.exponential(n / max(expected, 1e-9)) + 21)
    return uplift, discount


def weather_index(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Heat-wave index: an AR(1) process that only matters in the warm months."""
    w = np.zeros(len(dates))
    eps = rng.normal(0, 0.09, len(dates))
    for t in range(1, len(dates)):
        w[t] = 0.85 * w[t - 1] + eps[t]
    warm = np.clip(np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 196) / 365.25), 0, None)
    return w * warm


def store_factor(dates: pd.DatetimeIndex, store: StoreSpec, holidays: dict) -> np.ndarray:
    """Traffic of a store on each day: weekday, holidays, August, closures, opening, growth."""
    dow = dates.dayofweek.to_numpy()
    f = np.asarray(store.weekly, dtype=float)[dow] / np.mean(store.weekly)
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    f = f * store.size * (1 + store.growth) ** t_years
    if store.closed_weekdays:
        f[np.isin(dow, store.closed_weekdays)] = 0.0
    names = pd.Series([holidays.get(d, "") for d in dates])
    month_day = dates.strftime("%m-%d")
    if store.channel == "physical":
        f[names.isin(["Capodanno", "Pasqua", "Natale"]).to_numpy()] = 0.0
        reduced = names.isin(["Epifania", "Pasquetta", "Liberazione", "Festa del lavoro", "Festa della Repubblica",
                              "Ferragosto", "Ognissanti", "Immacolata", "Santo Stefano"]).to_numpy()
        f[reduced] *= 0.6
        for md, boost in (("12-23", 1.35), ("12-24", 1.6), ("12-31", 1.4), ("08-14", 1.25)):
            f[month_day == md] *= boost
        easter_sat = np.array([d == pd.Timestamp(easter(d.year)) - pd.Timedelta(days=1) for d in dates])
        f[easter_sat] *= 1.4
    else:
        f[(names != "").to_numpy()] *= 0.8
        for y in sorted(set(dates.year)):
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            black_friday = nov[nov.dayofweek == 4][3]
            f[dates == black_friday] *= 1.8
            f[dates == black_friday + pd.Timedelta(days=3)] *= 1.5   # Cyber Monday
    if store.august_dip:
        doy = dates.dayofyear.to_numpy()
        f *= 1 - store.august_dip * np.exp(-0.5 * ((doy - 227) / 9.0) ** 2)
    for a, b in store.closures:
        f[(dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))] = 0.0
    if store.open_from:
        start = pd.Timestamp(store.open_from)
        f[dates < start] = 0.0
        since = (dates - start).days.to_numpy()
        ramp = np.clip(0.6 + 0.4 * since / 180, 0.6, 1.0)
        f[dates >= start] *= ramp[dates >= start]
    return f


def lifecycle(dates: pd.DatetimeIndex, p: ProductSpec) -> np.ndarray:
    out = np.ones(len(dates))
    if p.launch:
        start = pd.Timestamp(p.launch)
        since = (dates - start).days.to_numpy()
        out = np.where(dates < start, 0.0, np.clip(0.15 + 0.85 * since / 90, 0.15, 1.0))
    if p.discontinued:
        end = pd.Timestamp(p.discontinued)
        out = np.where(dates > end, 0.0, out)
        last = (dates > end - pd.Timedelta(days=14)) & (dates <= end)
        out = np.where(last, out * 1.8, out)     # clearance
    return out


def list_price(dates: pd.DatetimeIndex, price: float) -> np.ndarray:
    idx = np.array([LIST_PRICE_INDEX.get(y, LIST_PRICE_INDEX[max(LIST_PRICE_INDEX)]) for y in dates.year])
    return np.round(np.ceil(price * idx * 10) / 10 - 0.01, 2)


def _negative_binomial(rng: np.random.Generator, mu: np.ndarray, k: float) -> np.ndarray:
    mu = np.maximum(mu, 0.0)
    return rng.negative_binomial(k, k / (k + mu))


# --------------------------------------------------------------------------- generators
def generate_chain(start: str = "2022-01-01", end: str = "2024-12-31", stores=CHAIN_STORES,
                   products=CHAIN_PRODUCTS, seed: int = 2024) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily sales of the fictional chain: (sales, items, stores).

    ``sales`` has one row per day x store x item with a positive quantity (days
    without sales are simply missing, as in most ERP exports), the price
    actually charged that day (promotions included) and the store's channel.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    holidays = italian_holidays(sorted(set(dates.year)))
    weather = weather_index(dates, rng)
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    store_f = {s.code: store_factor(dates, s, holidays) for s in stores}
    frames = []
    for p in products:
        uplift, discount = promo_calendar(dates, p.promos_per_year, rng)
        discount = np.maximum(discount, clearance_discount(dates, p.season))
        profile = (p.base * (1 + p.trend) ** t_years * seasonal_profile(dates, p.season, p.amplitude)
                   * uplift * np.exp(p.weather * weather) * lifecycle(dates, p))
        price = np.round(list_price(dates, p.price) * (1 - discount), 2)
        for s in stores:
            affinity = float(rng.lognormal(0, 0.15)) * (ONLINE_CATEGORY_MIX[p.family] if s.channel == "online" else 1)
            mu = profile * store_f[s.code] * affinity
            qty = _negative_binomial(rng, mu, p.dispersion)
            if p.lumpy:
                events = rng.random(len(dates)) < 0.06 * (store_f[s.code] > 0)
                qty = qty + events * (6 + rng.geometric(1 / 8, len(dates)))
            keep = qty > 0
            frames.append(pd.DataFrame({
                "date": dates[keep], "store": s.code, "item": p.code, "quantity": qty[keep].astype(int),
                "unit_price": price[keep], "channel": s.channel,
            }))
    sales = pd.concat(frames, ignore_index=True).sort_values(["date", "store", "item"], kind="mergesort")
    sales["date"] = sales["date"].dt.strftime("%Y-%m-%d")
    last_year = max(LIST_PRICE_INDEX)
    items = pd.DataFrame([{
        "item": p.code, "name": p.name, "category": CATEGORIES[p.family], "supplier": SUPPLIERS[p.family][0],
        "lead_time_days": SUPPLIERS[p.family][1],
        "unit_price": float(list_price(pd.DatetimeIndex([f"{last_year}-06-30"]), p.price)[0]),
        "unit_cost": round(float(list_price(pd.DatetimeIndex([f"{last_year}-06-30"]), p.price)[0]) * p.cost_ratio, 2),
        "status": ("discontinued " + p.discontinued) if p.discontinued else
                  (("launched " + p.launch) if p.launch else "active"),
    } for p in products])
    store_table = pd.DataFrame([{
        "store": s.code, "name": s.name, "city": s.city, "channel": s.channel,
        "opened": s.open_from or "before " + start, "note": s.note,
    } for s in stores])
    return sales.reset_index(drop=True), items, store_table


def single_store_italian(seed: int = 7) -> pd.DataFrame:
    """One small shop, 18 months, exported the Italian way (``;``, dd/mm/yyyy, decimal comma)."""
    products = [p for p in CHAIN_PRODUCTS if p.code in SHOP_PRODUCTS]
    sales, items, _ = generate_chain("2023-07-01", "2024-12-31", stores=(SHOP,), products=products, seed=seed)
    names = items.set_index("item")["name"]
    out = pd.DataFrame({
        "Data": pd.to_datetime(sales["date"]).dt.strftime("%d/%m/%Y"),
        "Codice": sales["item"],
        "Descrizione": sales["item"].map(names),
        "Quantità": sales["quantity"],
        "Prezzo": sales["unit_price"].map(lambda v: f"{v:.2f}".replace(".", ",")),
    })
    return out


def ecommerce_order_lines(seed: int = 11) -> pd.DataFrame:
    """One year of online order lines: several lines per order, timestamps, returns as negative quantities."""
    rng = np.random.default_rng(seed)
    web = next(s for s in CHAIN_STORES if s.code == "WEB")
    products = [p for p in CHAIN_PRODUCTS if p.code in ECOMMERCE_PRODUCTS]
    sales, items, _ = generate_chain("2024-01-01", "2024-12-31", stores=(web,), products=products, seed=seed)
    names = items.set_index("item")["name"]
    lines = []
    for row in sales.itertuples(index=False):
        q = int(row.quantity)
        while q > 0:
            c = int(min(q, rng.choice([1, 1, 1, 2, 2, 3])))
            lines.append((row.date, row.item, c, row.unit_price))
            q -= c
    df = pd.DataFrame(lines, columns=["date", "item", "qty", "price"])
    df = df.sample(frac=1.0, random_state=seed).sort_values("date", kind="mergesort").reset_index(drop=True)
    # several lines per order: consecutive lines of the same day share an order id
    new_order = (df["date"] != df["date"].shift()) | (rng.random(len(df)) < 0.45)
    order_no = np.cumsum(new_order)
    seconds = rng.integers(7 * 3600, 24 * 3600, size=order_no.max() + 1)[order_no]
    ts = pd.to_datetime(df["date"]) + pd.to_timedelta(seconds, unit="s")
    out = pd.DataFrame({
        "ID ordine": [f"WEB-2024-{n:06d}" for n in order_no],
        "Data ordine": ts,
        "Codice articolo": df["item"],
        "Descrizione": df["item"].map(names),
        "Qtà": df["qty"],
        "Prezzo unitario": df["price"],
        "Canale": "E-commerce",
        "Stato": "Consegnato",
    })
    returns = out.sample(frac=0.025, random_state=seed).copy()
    returns["Data ordine"] = returns["Data ordine"] + pd.to_timedelta(rng.integers(3, 15, len(returns)), unit="D")
    returns["Qtà"] = -returns["Qtà"]
    returns["Stato"] = "Reso"
    returns = returns[returns["Data ordine"] < pd.Timestamp("2025-01-01")]
    out = pd.concat([out, returns]).sort_values("Data ordine", kind="mergesort").reset_index(drop=True)
    out["Data ordine"] = out["Data ordine"].dt.floor("min")
    return out


def write_samples(out_dir: str | Path, seed: int = 2024) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sales, items, stores = generate_chain(seed=seed)
    paths = [out / "retail_chain_sales.csv", out / "retail_chain_items.csv", out / "retail_chain_stores.csv",
             out / "single_store_it.csv", out / "ecommerce_order_lines.xlsx"]
    sales.to_csv(paths[0], index=False)
    items.to_csv(paths[1], index=False)
    stores.to_csv(paths[2], index=False)
    single_store_italian().to_csv(paths[3], sep=";", index=False, encoding="utf-8-sig")
    lines = ecommerce_order_lines()
    notes = pd.DataFrame({"Note": [
        "Righe d'ordine del negozio online, anno 2024 (dati inventati).",
        "Una riga per articolo ordinato; i resi sono righe con quantità negativa e stato 'Reso'.",
        "Foglio da caricare: 'Ordini'.",
    ]})
    with pd.ExcelWriter(paths[4], engine="openpyxl") as xw:
        notes.to_excel(xw, sheet_name="Note", index=False)
        lines.to_excel(xw, sheet_name="Ordini", index=False)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the hypothetical sample datasets.")
    parser.add_argument("--out", default="data/samples", help="output folder")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(argv)
    for p in write_samples(args.out, args.seed):
        print(f"{p}  ({p.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
