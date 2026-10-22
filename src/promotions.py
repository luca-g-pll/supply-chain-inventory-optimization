"""Promotions: find them in the data, measure their effect, use them in the forecasts.

Where promotions come from (``promo_from_rows``):

* a **promo column** in the file (``promo``, ``promozione``, ``sconto`` …) holding a
  flag (1 / yes / sì) or a discount (``20``, ``20%``, ``0.2``);
* otherwise the **price**: a day is a promotion when the price charged is at least
  ``min_discount`` below the item's *regular price* (75th percentile of the prices
  in a centred 9-week window, so list-price changes are followed and promotions
  are not taken for the regular price).

Effect of a promotion (``estimate_effects``), per item, pooled over its stores:

* **lift** — ``sales / baseline`` on promotion days, where the baseline is the
  average of the previous 28 normal days. It is modelled as ``lift(d) = exp(β·d)``
  for a discount ``d``, so a promotion with a different discount can be planned;
* **post-promo dip** — the same ratio in the ``post_promo_days`` after a promotion
  (customers stocked up).

Items with few promotion days borrow the average effect of all items (shrinkage).
The planned promotion calendar is known in advance, so using the promotion of the
*target* day in a forecast is not leakage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PromotionsConfig
from src.features import rolling_stats, shift
from src.series import DemandMatrix

# models whose forecasts are corrected for promotions (the seasonal-naive baseline is
# left as it is: it is the "same as last week" rule used without any software)
PROMO_ADJUSTED = ("moving_average", "holt_winters")
BETA_BOUNDS = (0.0, 10.0)
DIP_BOUNDS = (0.5, 1.2)
BASELINE_WINDOW = 28

_YES = {"1", "y", "yes", "si", "sì", "sí", "s", "true", "x", "promo", "promotion", "promozione", "offerta", "volantino",
        "on", "vero", "ok"}
_NO = {"", "0", "n", "no", "false", "falso", "off", "none", "nan", "-"}


# --------------------------------------------------------------------------- reading
def parse_promo_values(values: pd.Series) -> pd.Series:
    """Promo column -> discount in [0, 1); ``NaN`` = "promotion, depth unknown".

    ``1`` / yes / sì / x -> NaN (flag) · ``0`` / no / empty -> 0 · ``20`` or ``20%``
    -> 0.20 · ``0.2`` -> 0.20. A column of only 0 and 1 is read as flags.
    """
    if pd.api.types.is_bool_dtype(values):
        return pd.Series(np.where(values.fillna(False), np.nan, 0.0), index=values.index)
    text = values.astype("string").str.strip().str.lower().fillna("")
    is_yes = text.isin(_YES - {"1"})
    is_no = text.isin(_NO - {"0"})
    num = pd.to_numeric(text.str.replace("%", "", regex=False).str.replace("-", "", regex=False)
                        .str.replace(",", ".", regex=False), errors="coerce").abs()
    flags_only = num.dropna().isin([0.0, 1.0]).all()
    out = pd.Series(0.0, index=values.index)
    if flags_only:
        out[num == 1] = np.nan
    else:
        frac = np.where(num > 1, num / 100.0, num)
        out[:] = np.where(num.notna(), frac, 0.0)
        out[(num >= 100)] = 0.0
    out[is_yes] = np.nan
    out[is_no] = 0.0
    return out.clip(upper=0.95)


def regular_price(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Regular price per (date, series): 75th percentile of a centred rolling window.

    ``prices`` is wide: one column per series, one row per calendar day, NaN on
    days without a price.
    """
    return prices.rolling(window, center=True, min_periods=3).quantile(0.75)


def _fill_from_item(wide: pd.DataFrame) -> pd.DataFrame:
    """Days without a value for a store take the median over the item's other stores."""
    items = wide.columns.get_level_values("item")
    by_item = wide.T.groupby(items).median().T
    filled = wide.copy()
    for item in by_item.columns:
        cols = [c for c in wide.columns if c[1] == item]
        filled[cols] = wide[cols].apply(lambda col, ref=by_item[item]: col.fillna(ref))
    return filled


def promo_from_rows(rows: pd.DataFrame, cfg: PromotionsConfig, has_flag: bool, has_price: bool
                    ) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Daily discounts per series from the file rows.

    ``rows``: ``[date, store, item, quantity, unit_price, promo]`` (promo = output of
    :func:`parse_promo_values`). Returns ``(promo, source, notes)`` where ``promo`` is
    ``[date, store, item, discount]`` with only promotion days, and ``source`` is
    ``"file"``, ``"price"`` or ``None``.
    """
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "store": pd.Series(dtype=str),
                          "item": pd.Series(dtype=str), "discount": pd.Series(dtype="float64")})
    if not cfg.enabled or not (has_flag or has_price):
        return empty, None, []
    keys = ["store", "item"]
    discount_price = None
    if has_price:
        p = rows.dropna(subset=["unit_price"])
        p = p[p["unit_price"] > 0]
        if len(p):
            w = p.assign(v=p["unit_price"] * p["quantity"].clip(lower=0))
            g = w.groupby(["date", *keys]).agg(v=("v", "sum"), q=("quantity", "sum"), m=("unit_price", "mean"))
            daily = (g["v"] / g["q"]).where(g["q"] > 0, g["m"])
            wide = daily.unstack(keys)
            full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
            wide = wide.reindex(full)
            reg = regular_price(wide, cfg.regular_price_window)
            disc = (1 - wide / reg).clip(lower=0)
            disc = _fill_from_item(disc).fillna(0.0)
            disc = disc.where(disc >= cfg.min_discount, 0.0)
            disc.index.name = "date"
            discount_price = disc.stack(keys, future_stack=True).rename("discount").reset_index()
    if has_flag:
        f = rows.groupby(["date", *keys])["promo"].agg(lambda s: np.nan if s.isna().any() else s.max())
        f = f.rename("discount").reset_index()
        if discount_price is not None:
            f = f.merge(discount_price.rename(columns={"discount": "d_price"}), on=["date", *keys], how="left")
            depth = f["d_price"].where(f["d_price"] > 0, cfg.default_discount)
        else:
            depth = pd.Series(cfg.default_discount, index=f.index)
        f["discount"] = f["discount"].fillna(depth)
        promo, source = f[["date", *keys, "discount"]], "file"
    else:
        promo, source = discount_price, "price"
    promo = promo[promo["discount"] > 0].reset_index(drop=True)
    promo["store"] = promo["store"].astype(str)
    promo["item"] = promo["item"].astype(str)
    if promo.empty:
        return empty, None, ([] if source == "price" else ["The promo column marks no promotion day."])
    return promo, source, []


def promo_note(promo: pd.DataFrame, source: str, cfg: PromotionsConfig) -> str:
    """One line for the validation report."""
    n_items = promo["item"].nunique()
    if source == "file":
        return f"Promotions read from the file: {len(promo):,} promotion days on {n_items} item(s)."
    return (f"Promotions found from prices: {len(promo):,} days at least {cfg.min_discount:.0%} below the "
            f"regular price, on {n_items} item(s).")


def promo_matrix(promo: pd.DataFrame | None, keys: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray | None:
    """Dense ``(n_series, n_days)`` discount matrix (0 = no promotion)."""
    if promo is None or promo.empty:
        return None
    out = np.zeros((len(keys), len(dates)))
    index = pd.MultiIndex.from_frame(keys[["store", "item"]].astype(str))
    row = index.get_indexer(pd.MultiIndex.from_frame(promo[["store", "item"]].astype(str)))
    col = (pd.to_datetime(promo["date"]) - dates[0]).dt.days.to_numpy()
    ok = (row >= 0) & (col >= 0) & (col < len(dates))
    out[row[ok], col[ok]] = promo["discount"].to_numpy(dtype="float64")[ok]
    return out


# --------------------------------------------------------------------------- calendar helpers
def post_promo_mask(promo: np.ndarray, days: int) -> np.ndarray:
    """True on non-promotion days that follow a promotion within ``days`` days."""
    on = promo > 0
    if days <= 0:
        return np.zeros_like(on)
    cs = np.concatenate([np.zeros((on.shape[0], 1)), np.cumsum(on, axis=1)], axis=1)
    t = np.arange(on.shape[1])
    recent = (cs[:, t] - cs[:, np.clip(t - days, 0, None)]) > 0
    return recent & ~on


def promo_day_counter(promo: np.ndarray, cap: int = 28) -> np.ndarray:
    """1, 2, 3 … on consecutive promotion days, 0 elsewhere (capped)."""
    on = promo > 0
    out = np.zeros(on.shape)
    run = np.zeros(on.shape[0])
    for t in range(on.shape[1]):
        run = (run + 1) * on[:, t]
        out[:, t] = np.minimum(run, cap)
    return out


def baseline_level(values: np.ndarray, promo: np.ndarray, post: np.ndarray,
                   window: int = BASELINE_WINDOW) -> np.ndarray:
    """Average of the previous ``window`` normal days (no promotion, no dip)."""
    normal = np.where((promo > 0) | post, np.nan, values)
    mean, _ = rolling_stats(shift(normal, 1), window)
    return mean


# --------------------------------------------------------------------------- effect
@dataclass
class PromoEffects:
    beta: np.ndarray          # (n_series,) lift(d) = exp(beta · d), shared by the stores of an item
    dip: np.ndarray           # (n_series,) demand factor in the days after a promotion
    beta_global: float
    dip_global: float
    item_table: pd.DataFrame  # one row per item

    def factor(self, promo: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Demand multiplier per (series, day): lift on promotion days, dip after them."""
        lift = np.exp(self.beta[:, None] * promo)
        return np.where(post, self.dip[:, None], lift)


def _ratio_parts(values, base, mask, weights=None):
    w = np.where(mask, 1.0, 0.0)
    y = np.nansum(np.where(mask, values, 0.0), axis=1)
    b = np.nansum(np.where(mask, base, 0.0), axis=1)
    bd = np.nansum(np.where(mask, base * (weights if weights is not None else 1.0), 0.0), axis=1)
    return y, b, bd, w.sum(axis=1)


def estimate_effects(dm: DemandMatrix, fit_end: int, cfg: PromotionsConfig) -> PromoEffects:
    """Lift and post-promotion dip per item, from days ``<= fit_end`` only."""
    promo = dm.promo if dm.promo is not None else np.zeros_like(dm.values)
    post = post_promo_mask(promo, cfg.post_promo_days)
    base = baseline_level(dm.values, promo, post)
    upto = np.arange(dm.n_days)[None, :] <= fit_end
    ok = upto & ~np.isnan(dm.values) & ~np.isnan(base) & (base > 0)
    item_cat = pd.Categorical(dm.keys["item"])
    codes, n_items = item_cat.codes, len(item_cat.categories)

    def per_item(arr):
        return np.bincount(codes, weights=arr, minlength=n_items)

    yp, bp, bdp, npd = (per_item(x) for x in _ratio_parts(dm.values, base, ok & (promo > 0), promo))
    yq, bq, _, nq = (per_item(x) for x in _ratio_parts(dm.values, base, ok & post))
    k = cfg.shrinkage_days

    def log_lift(y, b):
        return np.log(np.clip(y, 1e-9, None) / np.clip(b, 1e-9, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        if bp.sum() > 0:
            d_glob = bdp.sum() / bp.sum()
            beta_g = float(np.clip(log_lift(yp.sum(), bp.sum()) / d_glob, *BETA_BOUNDS))
        else:
            beta_g = 0.0
        d_item = np.where(bp > 0, bdp / np.where(bp > 0, bp, 1), np.nan)
        beta_raw = np.where(bp > 0, log_lift(yp, bp) / d_item, beta_g)
        dip_g = float(np.clip(yq.sum() / bq.sum(), *DIP_BOUNDS)) if bq.sum() > 0 else 1.0
        dip_raw = np.where(bq > 0, yq / np.where(bq > 0, bq, 1), dip_g)
    beta_i = np.clip((npd * np.clip(beta_raw, *BETA_BOUNDS) + k * beta_g) / (npd + k), *BETA_BOUNDS)
    dip_i = np.clip((nq * np.clip(dip_raw, *DIP_BOUNDS) + k * dip_g) / (nq + k), *DIP_BOUNDS)

    units_all = per_item(np.nansum(np.where(upto, dm.values, 0.0), axis=1))
    units_promo = per_item(np.nansum(np.where(upto & (promo > 0), dm.values, 0.0), axis=1))
    table = pd.DataFrame({
        "item": item_cat.categories.astype(str),
        "promo_days": npd.astype(int),
        "avg_discount": np.where(bp > 0, d_item, np.nan),
        "observed_lift": np.where(bp > 0, yp / np.where(bp > 0, bp, 1), np.nan),
        "beta": beta_i,
        "lift_at_avg": np.where(bp > 0, np.exp(beta_i * np.nan_to_num(d_item)), np.nan),
        "lift_20": np.exp(beta_i * 0.20),
        "post_dip": dip_i,
        "own_weight": npd / (npd + k),
        "units_share_promo": np.where(units_all > 0, units_promo / np.where(units_all > 0, units_all, 1), 0.0),
    })
    return PromoEffects(beta=beta_i[codes], dip=dip_i[codes], beta_global=beta_g, dip_global=dip_g,
                        item_table=table)


def price_elasticity(beta: float, change: float) -> float:
    """Demand factor for a permanent price change (``+0.05`` = 5% dearer), from the promo response."""
    return float(np.exp(-beta * change))


# --------------------------------------------------------------------------- forecasting
class PromoAware:
    """Wrap a statistical model: fit it on promotion-free demand, then add the promotions back.

    1. estimate lift and dip on the training days;
    2. divide past demand by them (a promotion week looks like a normal week);
    3. fit / forecast the wrapped model on that clean demand;
    4. multiply the forecast by the lift of the promotions planned on the target days.
    """

    def __init__(self, base, cfg: PromotionsConfig):
        self.base = base
        self.cfg = cfg
        self.name = base.name
        self.effects_: PromoEffects | None = None
        self._clean: DemandMatrix | None = None
        self._factor: np.ndarray | None = None

    @property
    def n_fallback(self) -> int:
        return getattr(self.base, "n_fallback", 0)

    def fit(self, dm: DemandMatrix, fit_end: int) -> PromoAware:
        promo = dm.promo if dm.promo is not None else np.zeros_like(dm.values)
        post = post_promo_mask(promo, self.cfg.post_promo_days)
        self.effects_ = estimate_effects(dm, fit_end, self.cfg)
        self._factor = self.effects_.factor(promo, post)
        self._clean = DemandMatrix(dm.keys, dm.dates, dm.values / self._factor, dm.start, dm.promo)
        self.base.fit(self._clean, fit_end)
        return self

    def predict(self, dm: DemandMatrix, origins: np.ndarray, horizon: int) -> np.ndarray:
        if self._clean is None:
            raise RuntimeError("fit() must be called before predict()")
        origins = np.asarray(origins)
        F = self.base.predict(self._clean, origins, horizon)
        pad = np.concatenate([self._factor, np.ones((dm.n_series, horizon))], axis=1)
        tgt = origins[:, None] + np.arange(1, horizon + 1)[None, :]
        return F * pad[:, tgt]


def has_promotions(dm: DemandMatrix, cfg: PromotionsConfig) -> bool:
    return cfg.enabled and dm.promo is not None and bool((dm.promo > 0).any())


# --------------------------------------------------------------------------- dashboard tables
def promo_events(dm: DemandMatrix, cfg: PromotionsConfig) -> pd.DataFrame:
    """One row per promotion of each series: dates, discount, units sold vs the normal level."""
    cols = ["store", "item", "start", "end", "days", "discount", "units", "baseline"]
    if dm.promo is None:
        return pd.DataFrame(columns=cols)
    promo = dm.promo
    post = post_promo_mask(promo, cfg.post_promo_days)
    base = baseline_level(dm.values, promo, post)
    on = promo > 0
    edges = np.diff(np.concatenate([np.zeros((dm.n_series, 1)), on, np.zeros((dm.n_series, 1))], axis=1), axis=1)
    rows = []
    for i in range(dm.n_series):
        starts = np.flatnonzero(edges[i] == 1)
        ends = np.flatnonzero(edges[i] == -1) - 1
        for a, b in zip(starts, ends):
            y = np.nan_to_num(dm.values[i, a: b + 1])
            # normal level frozen at the start of the promotion (the days inside are excluded anyway)
            lvl = base[i, a]
            rows.append((dm.keys.at[i, "store"], dm.keys.at[i, "item"], dm.dates[a], dm.dates[b], b - a + 1,
                         float(promo[i, a: b + 1].mean()), float(y.sum()),
                         float(lvl * (b - a + 1)) if np.isfinite(lvl) else np.nan))
    return pd.DataFrame(rows, columns=cols)


def promo_day_types(dm: DemandMatrix, cfg: PromotionsConfig) -> np.ndarray:
    """0 = normal day, 1 = promotion, 2 = after a promotion; ``(n_series, n_days)``."""
    if dm.promo is None:
        return np.zeros(dm.values.shape, dtype=int)
    post = post_promo_mask(dm.promo, cfg.post_promo_days)
    return np.where(dm.promo > 0, 1, np.where(post, 2, 0))


def promo_accuracy(bt, dm: DemandMatrix, cfg: PromotionsConfig) -> pd.DataFrame:
    """Error sums per model x series x day type (normal / promo / post_promo) over a back-test."""
    types = promo_day_types(dm, cfg)
    tgt = np.clip(bt.targets(), 0, dm.n_days - 1)
    kinds = types[:, tgt]                                      # (n, J, H)
    names = np.array(["normal", "promo", "post_promo"])
    frames = []
    ok = ~np.isnan(bt.actual)
    for model, F in bt.forecasts.items():
        for k, label in enumerate(names):
            m = ok & (kinds == k)
            A = np.where(m, bt.actual, 0.0)
            E = np.where(m, F - np.nan_to_num(bt.actual), 0.0)
            frames.append(pd.DataFrame({
                "period": bt.period, "model": model, "store": bt.keys["store"].to_numpy(),
                "item": bt.keys["item"].to_numpy(), "day_type": label,
                "actual": A.sum(axis=(1, 2)), "abs_error": np.abs(E).sum(axis=(1, 2)), "error": E.sum(axis=(1, 2)),
                "n": m.sum(axis=(1, 2)),
            }))
    return pd.concat(frames, ignore_index=True)


def promo_baseline(dm: DemandMatrix, cfg: PromotionsConfig, weeks: int = 8) -> pd.DataFrame:
    """Normal daily demand by weekday over the last ``weeks`` weeks (planner input)."""
    promo = dm.promo if dm.promo is not None else np.zeros_like(dm.values)
    post = post_promo_mask(promo, cfg.post_promo_days)
    a = max(0, dm.n_days - 7 * weeks)
    vals = np.where((promo > 0) | post, np.nan, dm.values)[:, a:]
    dow = dm.dates[a:].dayofweek.to_numpy()
    frames = []
    for d in range(7):
        cols = vals[:, dow == d]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(cols, axis=1) if cols.shape[1] else np.full(dm.n_series, np.nan)
        frames.append(pd.DataFrame({"store": dm.keys["store"].to_numpy(), "item": dm.keys["item"].to_numpy(),
                                    "dow": d, "units": np.nan_to_num(mean)}))
    out = pd.concat(frames, ignore_index=True)
    out.attrs["last_date"] = str(dm.dates[-1].date())
    return out


# --------------------------------------------------------------------------- planner
_CALENDAR = {
    "start": {"start", "inizio", "dal", "from", "data_inizio", "start_date", "begin", "da"},
    "end": {"end", "fine", "al", "to", "data_fine", "end_date", "until"},
}


def read_promo_calendar(raw: pd.DataFrame, default_discount: float) -> pd.DataFrame:
    """Planned promotions ``[item, store, start, end, discount]`` from a user file.

    Columns are recognised in English or Italian; ``store`` is optional (empty = every
    store), ``discount`` as in :func:`parse_promo_values` (a flag gets ``default_discount``).
    Raises ``InputError`` when item, start or end are missing.
    """
    from src.ingest import SYNONYMS, InputError, normalise_name, parse_dates

    names = {c: normalise_name(c) for c in raw.columns}
    want = {"item": set(SYNONYMS["item"]), "store": set(SYNONYMS["store"]), **_CALENDAR,
            "discount": set(SYNONYMS["promo"]) | {"percentuale", "pct", "sconto_percentuale"}}
    cols: dict[str, str | None] = {}
    for fld, syns in want.items():
        cols[fld] = next((c for c, n in names.items() if n in syns and c not in cols.values()), None)
    missing = [f for f in ("item", "start", "end") if cols[f] is None]
    if missing:
        raise InputError("Missing column(s): " + ", ".join(missing))
    start, _ = parse_dates(raw[cols["start"]])
    end, _ = parse_dates(raw[cols["end"]])
    disc = (parse_promo_values(raw[cols["discount"]]).fillna(default_discount) if cols["discount"]
            else pd.Series(default_discount, index=raw.index))
    store = (raw[cols["store"]].astype("string").str.strip().fillna("") if cols["store"]
             else pd.Series("", index=raw.index))
    out = pd.DataFrame({"item": raw[cols["item"]].astype("string").str.strip(), "store": store,
                        "start": start, "end": end, "discount": disc})
    return out.dropna(subset=["item", "start", "end"]).reset_index(drop=True)

def plan_promotion(baseline_dow: pd.Series, start: pd.Timestamp, end: pd.Timestamp, discount: float,
                   beta: float, dip: float, post_days: int, price: float, cost: float) -> dict:
    """Expected effect of one planned promotion for a group of stores.

    ``baseline_dow``: normal units per weekday (index 0..6) summed over the stores.
    """
    days = pd.date_range(start, end, freq="D")
    after = pd.date_range(end + pd.Timedelta(days=1), periods=post_days, freq="D")
    base_units = float(sum(baseline_dow.get(d, 0.0) for d in days.dayofweek))
    base_after = float(sum(baseline_dow.get(d, 0.0) for d in after.dayofweek))
    lift = float(np.exp(beta * discount))
    promo_units = base_units * lift
    lost_after = base_after * (1 - dip)
    promo_price = price * (1 - discount)
    margin_normal = base_units * (price - cost) + base_after * (price - cost)
    margin_promo = promo_units * (promo_price - cost) + (base_after - lost_after) * (price - cost)
    return {
        "days": len(days), "lift": lift, "baseline_units": base_units, "promo_units": promo_units,
        "extra_units": promo_units - base_units, "dip_units": lost_after,
        "revenue_normal": base_units * price, "revenue_promo": promo_units * promo_price,
        "margin_normal": margin_normal, "margin_promo": margin_promo,
        "margin_change": margin_promo - margin_normal,
    }

