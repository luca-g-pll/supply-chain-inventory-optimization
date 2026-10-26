"""New products: plan them from a similar product until their own sales take over.

A series with fewer than ``input.min_history_days`` of sales before the test period
cannot be tested on the past, so it stays out of the policy comparison; here it gets
a forecast, reorder parameters and orders (docs/08_methodology.md §6.11).

1. **Similar product** (*analog*). A product new in one store but sold elsewhere uses
   itself in the other stores. A product new everywhere gets the most similar product
   that is still selling, scored on the signals available — the first weeks of sales
   compared day by day (correlation), a similar name, same category, same supplier,
   close price —
   with the reasons shown; without any signal, the average product of the store. The
   suggestion is confirmed or changed by the user.
2. **Scale**: own sales ÷ the similar product's sales on the same stores over the last
   14 days (a launch is still ramping up in its first days).
3. **Handover**: ``forecast = w · own + (1 − w) · scale · similar``, with
   ``w = 1 − 0.5^(age / half_life)`` (28 days → 50%, 84 days → 88%).
4. **Caution**: safety stock × ``(1 + boost · (1 − w))``, lot at most
   ``first_lot_days`` of demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ColdStartConfig
from src.inventory import lot_size, safety_factor
from src.series import DemandMatrix, natural_sorted

AVERAGE = ""                     # "analog" meaning: the average product of the store
WEIGHTS = {"early_sales": 0.35, "name": 0.25, "category": 0.15, "supplier": 0.10, "price": 0.15}
EARLY_WINDOW = 56
SCALE_WINDOW = 14                # the scale uses the most recent days: a launch is still ramping up at first


def _trigrams(text: str) -> set[str]:
    t = " " + "".join(ch if ch.isalnum() else " " for ch in str(text).lower()) + " "
    t = " ".join(t.split())
    return {t[k: k + 3] for k in range(max(len(t) - 2, 0))}


def name_similarity(a: str, b: str) -> float:
    """Share of common 3-letter groups (Jaccard): "Penne integrali" vs "Penne rigate" ≈ 0.3, unrelated ≈ 0."""
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def age_days(dm: DemandMatrix) -> np.ndarray:
    """Days from the first sale of each series to the end of the data (inclusive)."""
    return dm.n_days - dm.start


def new_series_rows(dm: DemandMatrix, excluded: list | tuple) -> np.ndarray:
    """Rows of ``dm`` that the profile excluded for short history."""
    ex = {(str(s), str(i)) for s, i, *_ in excluded}
    return np.array([k for k, (s, i) in enumerate(zip(dm.keys["store"], dm.keys["item"])) if (s, i) in ex],
                    dtype=int)


def _attr(master: pd.DataFrame, col: str) -> pd.Series:
    return master[col].astype("string").fillna("") if col in master else pd.Series("", index=master.index)


def suggest_analogs(dm: DemandMatrix, new_rows: np.ndarray, master: pd.DataFrame,
                    cfg: ColdStartConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Similar product for every new item, and analog + scale for every new series.

    Returns ``(items, series)``: ``items`` has one row per product new everywhere
    (analog, score, confidence, reasons, scale, correlation); ``series`` one row per new
    store-item series (age, analog, scale, reason).
    """
    n = dm.n_series
    is_new = np.zeros(n, dtype=bool)
    is_new[new_rows] = True
    keys = dm.keys.astype(str)
    stores, items = keys["store"].to_numpy(), keys["item"].to_numpy()
    vals = np.nan_to_num(dm.values)
    T = dm.n_days
    m = master.set_index("item")
    price = pd.to_numeric(m["unit_price"], errors="coerce") if "unit_price" in m else pd.Series(dtype=float)
    supplier, category, name = _attr(m, "supplier"), _attr(m, "category"), _attr(m, "item_name")

    established_items = sorted(set(items[~is_new]))
    active = {it for it in established_items
              if vals[(items == it) & ~is_new, max(0, T - cfg.active_days):].sum() > 0}
    candidates = natural_sorted(active)
    item_rows = {it: np.flatnonzero((items == it) & ~is_new) for it in candidates}

    def sales(rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        return vals[np.ix_(rows, days)].sum(axis=0) if len(rows) else np.zeros(len(days))

    item_out, series_out = [], []
    new_items = natural_sorted(set(items[is_new]))
    for it in new_items:
        rows_i = np.flatnonzero((items == it) & is_new)
        first = int(dm.start[rows_i].min())
        days = np.arange(first, T)[-EARLY_WINDOW:]
        x = sales(rows_i, days)
        st_i = set(stores[rows_i])
        if it in established_items:          # new only in some stores: the product itself elsewhere
            for r in rows_i:
                d = np.arange(int(dm.start[r]), T)[-EARLY_WINDOW:]
                other = np.flatnonzero((items == it) & ~is_new)
                y = sales(other, d) / max(len(other), 1)
                xs = vals[r, d].sum()
                series_out.append((stores[r], it, int(T - dm.start[r]), it,
                                   float(np.clip(xs / y.sum(), 0.02, 50)) if y.sum() > 0 else 1.0, "same_item", 1.0))
            continue
        best = None
        for c in candidates:
            rows_c = np.array([r for r in item_rows[c] if stores[r] in st_i], dtype=int)
            rows_c = rows_c if len(rows_c) else item_rows[c]
            y = sales(rows_c, days)
            signals, reasons = {}, []
            corr = np.nan
            if len(days) >= cfg.min_early_days and x.std() > 0 and y.std() > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
                signals["early_sales"] = max(corr, 0.0)
                if corr >= 0.5:
                    reasons.append("early_sales")
            if name.get(it, "") and name.get(c, ""):
                sim = name_similarity(name[it], name[c])
                signals["name"] = min(1.0, sim / 0.4)        # 0.4 of common letter groups = as similar as it gets
                if sim >= 0.25:
                    reasons.append("name")
            if category.get(it, "") and category.get(c, ""):
                same = category[it] == category[c]
                signals["category"] = float(same)
                if same:
                    reasons.append("category")
            if supplier.get(it, "") and supplier.get(c, ""):
                same = supplier[it] == supplier[c]
                signals["supplier"] = float(same)
                if same:
                    reasons.append("supplier")
            p_i, p_c = price.get(it, np.nan), price.get(c, np.nan)
            if np.isfinite(p_i) and np.isfinite(p_c) and p_i > 0 and p_c > 0:
                sim = float(np.exp(-abs(np.log(p_i / p_c)) / 0.35))
                signals["price"] = sim
                if sim >= 0.7:
                    reasons.append("price")
            if not signals:
                continue
            score = sum(WEIGHTS[k] * v for k, v in signals.items()) / sum(WEIGHTS[k] for k in signals)
            rank = (score, 0.0 if np.isnan(corr) else corr, y.sum())
            if best is None or rank > best[0]:
                best = (rank, c, score, corr, reasons, y)
        if best is None or best[2] <= 0:
            analog, score, corr, reasons = AVERAGE, 0.0, np.nan, ["average"]
            y = np.mean([sales(item_rows[c], days) for c in candidates], axis=0) if candidates else np.zeros(len(days))
        else:
            _, analog, score, corr, reasons, y = best
        k = min(SCALE_WINDOW, len(days))
        scale = float(np.clip(x[-k:].sum() / y[-k:].sum(), 0.02, 50)) if y[-k:].sum() > 0 else 1.0
        growth = launch_growth(x, y, int(T - first), cfg)
        has_early = not np.isnan(corr)
        strong_attr = ("category" in reasons) or ("supplier" in reasons)
        confidence = ("high" if has_early and corr >= 0.6 and strong_attr else
                      "medium" if (has_early and corr >= 0.4) or strong_attr else "low")
        item_out.append({"item": it, "age_days": int(T - first), "first_sale": dm.dates[first],
                         "n_stores": len(st_i), "units": float(x.sum()), "analog": analog,
                         "score": round(float(score), 3), "correlation": None if np.isnan(corr) else round(corr, 3),
                         "confidence": confidence, "reasons": ",".join(reasons) or "weak", "scale": round(scale, 3),
                         "growth": round(growth, 3)})
        for r in rows_i:
            series_out.append((stores[r], it, int(T - dm.start[r]), analog, round(scale, 3), "suggested",
                               round(growth, 3)))
    items_df = pd.DataFrame(item_out, columns=["item", "age_days", "first_sale", "n_stores", "units", "analog",
                                               "score", "correlation", "confidence", "reasons", "scale", "growth"])
    series_df = pd.DataFrame(series_out, columns=["store", "item", "age_days", "analog", "scale", "rule", "growth"])
    return items_df, series_df


def launch_growth(x: np.ndarray, y: np.ndarray, age: int, cfg: ColdStartConfig) -> float:
    """Week-on-week growth of a launch, net of the similar product's own movement (season, promotions).

    A launch keeps growing for weeks: without this the forecast lags behind. Only upward, only in the
    first ``launch_days``, at most ``launch_growth_cap``; 1.0 when the last two weeks are not available.
    """
    if age > cfg.launch_days or len(x) < 14:
        return 1.0
    x1, x0, y1, y0 = x[-7:].sum(), x[-14:-7].sum(), y[-7:].sum(), y[-14:-7].sum()
    if min(x0, y0, y1) <= 0:
        return 1.0
    return float(np.clip((x1 / x0) / (y1 / y0), 1.0, cfg.launch_growth_cap))


# --------------------------------------------------------------------------- forecast
def handover_weight(age, half_life: float) -> np.ndarray:
    """Weight of the product's own forecast: 0 at launch, 0.5 after ``half_life`` days."""
    return 1 - 0.5 ** (np.asarray(age, dtype="float64") / half_life)


def blend_forecasts(new_series: pd.DataFrame, own: pd.DataFrame, established: pd.DataFrame,
                    half_life: float) -> pd.DataFrame:
    """Forecast ``[store, item, h, forecast, weight]`` of the new series.

    ``own``: the model's own forecast of the new series; ``established``: forecasts of
    the established series (the similar products), both ``[store, item, h, forecast]``.
    """
    if new_series.empty:
        return pd.DataFrame(columns=["store", "item", "h", "forecast", "weight"])
    est = established.pivot_table(index=["store", "item"], columns="h", values="forecast")
    H = est.shape[1]
    by_item = est.groupby(level="item").mean()
    by_store = est.groupby(level="store").mean()
    own_p = own.pivot_table(index=["store", "item"], columns="h", values="forecast").reindex(columns=est.columns)
    rows = []
    for r in new_series.itertuples(index=False):
        if r.analog and (r.store, r.analog) in est.index and r.analog != r.item:
            ref = est.loc[(r.store, r.analog)].to_numpy()
        elif r.analog and r.analog in by_item.index:
            ref = by_item.loc[r.analog].to_numpy()
        elif r.store in by_store.index:
            ref = by_store.loc[r.store].to_numpy()
        else:
            ref = by_item.mean().to_numpy()
        mine = own_p.loc[(r.store, r.item)].to_numpy() if (r.store, r.item) in own_p.index else np.zeros(H)
        w = float(handover_weight(r.age_days, half_life))
        growth = float(getattr(r, "growth", 1.0) or 1.0)
        f = (w * np.nan_to_num(mine) + (1 - w) * float(r.scale) * np.nan_to_num(ref)) * growth
        rows.append(pd.DataFrame({"store": r.store, "item": r.item, "h": est.columns.to_numpy(),
                                  "forecast": np.clip(f, 0, None), "weight": w}))
    return pd.concat(rows, ignore_index=True)


def item_cv(summary: pd.DataFrame) -> pd.Series:
    """Daily coefficient of variation per item (from the static policy rows: σ_d = σ_L / √L)."""
    s = summary[(summary["policy"] == "P1_static_sq") & (summary["variant"] == "fixed")]
    if s.empty:
        return pd.Series(dtype=float)
    sd = s["sigma_L"] / np.sqrt(s["lead_time"].clip(lower=1))
    cv = (sd / s["mean_daily"].where(s["mean_daily"] > 0)).replace([np.inf, -np.inf], np.nan)
    return cv.groupby(s["item"]).median().dropna()


def new_series_params(forecast: pd.DataFrame, new_series: pd.DataFrame, master: pd.DataFrame, cv: pd.Series,
                      channels: dict, cfg: ColdStartConfig) -> pd.DataFrame:
    """Reorder parameters of the new series for every channel, in the columns the Orders page reads.

    ``channels``: ``{name: ChannelParams}``. The safety stock uses the similar product's
    variability, inflated while the forecast still relies on it; the lot is short.
    """
    cols = ["store", "item", "channel", "knob", "order_qty", "sigma_L", "lead_time", "unit_cost", "mean_daily",
            "weight", "on_hand", "backlog"]
    if forecast.empty:
        return pd.DataFrame(columns=cols)
    m = master.set_index("item")
    agg = forecast.groupby(["store", "item"]).agg(mean_daily=("forecast", "mean"), weight=("weight", "first"))
    analog = new_series.set_index(["store", "item"])["analog"]
    default_cv = float(cv.median()) if len(cv) else 1.0
    out = []
    for (store, item), r in agg.iterrows():
        ref = analog.get((store, item), "")
        c = float(cv.get(ref if ref else item, cv.get(item, default_cv)))
        c = c if np.isfinite(c) and c > 0 else default_cv
        mean = max(float(r["mean_daily"]), 0.0)
        cost = float(m.at[item, "unit_cost"]) if item in m.index else np.nan
        lt_item = pd.to_numeric(m.at[item, "lead_time_days"], errors="coerce") if item in m.index else np.nan
        for name, ch in channels.items():
            inv = ch.inventory
            L = int(lt_item) if np.isfinite(lt_item) else int(inv.lead_time_days)
            z = float(safety_factor(inv.service_level))
            sigma_L = c * mean * (1 + cfg.uncertainty_boost * (1 - float(r["weight"]))) * np.sqrt(L)
            q = float(lot_size(mean * inv.days_per_year, inv.order_cost, inv.holding_rate * np.nan_to_num(cost, nan=1.0)))
            q = max(1.0, min(q, float(np.ceil(mean * cfg.first_lot_days - 1e-9))))
            out.append((store, item, name, z, q, sigma_L, L, cost, mean, float(r["weight"]),
                        float(np.ceil(mean * L + z * sigma_L)), 0.0))
    return pd.DataFrame(out, columns=cols)
