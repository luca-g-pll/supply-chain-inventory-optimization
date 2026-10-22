"""Purchase plan on the current calendar: what to order today and in the coming weeks.

The analysis compares policies on the past; this module turns the chosen policy
(P2, forecast-driven (s, Q), calibrated to the target fill rate) into **orders**:

1. **Forward forecast** — the best model is refitted on the whole history and
   forecasts the ``horizon`` days after the last day of data; longer plans reuse
   the same weekday of the last forecast week.
2. **Starting stock** — stock on hand, back-orders and orders on the way at the
   end of the data. By default they are *estimated*: the state reached by the P2
   simulation of the test period; the user can replace them with real stock.
3. **Bringing the data to today** — when the history ends before today, the
   calendar is moved forward by whole weeks (weekdays are kept) and the days
   between the end of the data and today are projected with the forecast.
4. **Projection** — every day: deliveries arrive, forecast demand is served,
   P2 orders ``n·Q`` when the inventory position falls to ``s_t = forecast over
   the lead time + z·σ_E``. The orders of each day form the purchase calendar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.inventory import ForecastSQPolicy
from src.simulation import BatchResult, simulate_batch

PLAN_DAYS = 28


# --------------------------------------------------------------------------- analysis side
def future_forecast(dm, best_model: str, cfg, profile, fast: bool = False, n_jobs: int = -1) -> np.ndarray:
    """Forecast of the ``cfg.forecast.horizon`` days after the last day of data, ``(n, H)``."""
    from src.forecast import get_model
    from src.promotions import has_promotions

    model = get_model(best_model, cfg, profile, fast, n_jobs, promotions=has_promotions(dm, cfg.promotions))
    last = dm.n_days - 1
    model.fit(dm, last)
    return np.clip(model.predict(dm, np.array([last]), cfg.forecast.horizon)[:, 0, :], 0.0, None)


def end_state(inp, cfg, summary: pd.DataFrame, channel_name: str) -> BatchResult:
    """Stock reached at the end of the test period by the calibrated P2 policy (one channel).

    The safety factor is floored at zero, as in the order plan (see :func:`project`).
    """
    from src.policy import default_knob, make_policy, policy_parameters

    channel = cfg.channel(channel_name)
    p = policy_parameters(inp, channel, cfg)
    s = summary[(summary["policy"] == "P2_forecast_sq") & (summary["variant"] == "calibrated")
                & (summary["channel"] == channel_name)]
    knob = inp.keys.merge(s[["store", "item", "knob"]], on=["store", "item"], how="left")["knob"].to_numpy()
    knob = np.where(np.isnan(knob), default_knob("P2_forecast_sq", p), knob)
    knob = np.maximum(knob, 0.0)    # same rule as the order plan: never a negative safety stock
    return simulate_batch(inp.demand, inp.test_dates, make_policy("P2_forecast_sq", p, inp, knob),
                          lead_time=p.lead_time, initial_on_hand=p.initial_on_hand,
                          backorder_share=channel.backorder_share)


def plan_tables(dm, inp, cfg, summary: pd.DataFrame, best_model: str, profile, fast: bool,
                n_jobs: int = -1) -> dict[str, pd.DataFrame]:
    """Tables the Orders page needs: forward forecast, estimated stock, orders on the way."""
    F = future_forecast(dm, best_model, cfg, profile, fast, n_jobs)
    n, H = F.shape
    keys = dm.keys
    forecast = pd.DataFrame({"store": np.repeat(keys["store"].to_numpy(), H),
                             "item": np.repeat(keys["item"].to_numpy(), H),
                             "h": np.tile(np.arange(1, H + 1), n), "forecast": F.ravel().astype("float32")})
    states, pipes = [], []
    for ch in cfg.simulation.channels:
        res = end_state(inp, cfg, summary, ch)
        states.append(inp.keys.assign(channel=ch, on_hand=res.end_on_hand, backlog=res.end_backlog))
        r, d = np.nonzero(res.pipeline > 0)
        pipes.append(pd.DataFrame({"store": inp.keys["store"].to_numpy()[r], "item": inp.keys["item"].to_numpy()[r],
                                   "channel": ch, "day": d, "qty": res.pipeline[r, d]}))
    return {"plan_forecast": forecast, "plan_state": pd.concat(states, ignore_index=True),
            "plan_pipeline": pd.concat(pipes, ignore_index=True)}


# --------------------------------------------------------------------------- calendar
@dataclass(frozen=True)
class PlanCalendar:
    data_end: pd.Timestamp     # last day of data (real date)
    shift_days: int            # days added to every date to bring the data to today (multiple of 7)
    start: pd.Timestamp        # first projected day, on the shown calendar
    today: pd.Timestamp

    @property
    def gap_days(self) -> int:
        """Days between the end of the (shifted) data and today, projected with the forecast."""
        return int((self.today - self.start).days)


def plan_calendar(data_end, today, bring_to_today: bool = True) -> PlanCalendar:
    """Shown calendar: real dates, or the data moved forward by whole weeks so they end just before today.

    Without the shift, ``today`` cannot be before the day after the data.
    """
    data_end, today = pd.Timestamp(data_end).normalize(), pd.Timestamp(today).normalize()
    first = data_end + pd.Timedelta(days=1)
    gap = (today - first).days
    shift = 7 * (gap // 7) if bring_to_today and gap >= 7 else 0
    start = first + pd.Timedelta(days=shift)
    return PlanCalendar(data_end, shift, start, max(today, start))


def extend_weekly(F: np.ndarray, days: int) -> np.ndarray:
    """Columns 1..days of a forecast with H columns; beyond H the same weekday of the last week is reused."""
    n, H = F.shape
    if H < 7:
        return np.repeat(F.mean(axis=1, keepdims=True), days, axis=1)
    h = np.arange(1, days + 1)
    src = np.where(h > H, h - 7 * np.ceil((h - H) / 7).astype(int), h)
    return F[:, src - 1]


def apply_planned_promotions(F: np.ndarray, keys: pd.DataFrame, dates: pd.DatetimeIndex, plan: pd.DataFrame,
                             beta: dict, dip: dict, post_days: int, all_stores: str = "*") -> np.ndarray:
    """Multiply the forecast by the effect of the planned promotions (shown-calendar dates)."""
    out = F.copy()
    if plan is None or plan.empty:
        return out
    for _, r in plan.iterrows():
        if pd.isna(r.get("start")) or pd.isna(r.get("end")):
            continue
        item, start, end = str(r["item"]), pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        rows = (keys["item"] == item).to_numpy().copy()
        if str(r.get("store", all_stores)) != all_stores:
            rows &= (keys["store"] == str(r["store"])).to_numpy()
        b, d = float(beta.get(item, 0.0)), float(dip.get(item, 1.0))
        on = (dates >= start) & (dates <= end)
        after = (dates > end) & (dates <= end + pd.Timedelta(days=post_days))
        disc = float(r["discount"])
        disc = disc / 100 if disc > 1 else disc
        out[np.ix_(rows, on)] *= np.exp(b * disc)
        out[np.ix_(rows, after)] *= d
    return out


# --------------------------------------------------------------------------- projection
def lead_time_sums(F_ext: np.ndarray, lead_time: np.ndarray, days: int) -> np.ndarray:
    """``Σ F[t+1 .. t+L]`` for every decision day t < days (``F_ext`` must cover days + max L)."""
    n = F_ext.shape[0]
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(F_ext, axis=1)], axis=1)
    t = np.arange(days)
    rows = np.arange(n)[:, None]
    return cs[rows, np.minimum(t[None, :] + 1 + lead_time[:, None], cs.shape[1] - 1)] - cs[:, t + 1]


def project(F_ext: np.ndarray, days: int, dates: pd.DatetimeIndex, lead_time: np.ndarray, Q: np.ndarray,
            safety_stock: np.ndarray, on_hand: np.ndarray, backlog: np.ndarray, arrivals: np.ndarray,
            backorder_share: float = 0.0) -> BatchResult:
    """Run the P2 rule on the forecast for ``days`` days from the given stock (daily detail recorded)."""
    lead_time = np.asarray(lead_time, dtype=int)
    ltd = lead_time_sums(F_ext, lead_time, days)
    policy = ForecastSQPolicy(ltd=ltd, ss=np.asarray(safety_stock, dtype="float64"), Q=np.asarray(Q, dtype="float64"))
    return simulate_batch(F_ext[:, :days], dates[:days], policy, lead_time=lead_time, initial_on_hand=on_hand,
                          backorder_share=backorder_share, record_daily=True,
                          initial_arrivals=arrivals, initial_backlog=backlog)


def orders_frame(res: BatchResult, keys: pd.DataFrame, dates: pd.DatetimeIndex, lead_time: np.ndarray,
                 F: np.ndarray) -> pd.DataFrame:
    """One row per projected order: date, series, quantity, arrival, stock at the moment of the order."""
    q = res.daily["order_qty"]
    r, t = np.nonzero(q > 0)
    cover = F[:, :PLAN_DAYS].mean(axis=1)
    return pd.DataFrame({
        "date": dates[t], "store": keys["store"].to_numpy()[r], "item": keys["item"].to_numpy()[r],
        "qty": q[r, t], "arrival": dates[t] + pd.to_timedelta(lead_time[r], unit="D"),
        "on_hand": res.daily["on_hand_end"][r, t], "on_order": res.daily["on_order"][r, t] - q[r, t],
        "cover_days": np.where(cover[r] > 0, res.daily["on_hand_end"][r, t] / np.where(cover[r] > 0, cover[r], 1), np.inf),
        "row": r,
    })


def stockout_risks(res: BatchResult, keys: pd.DataFrame, dates: pd.DatetimeIndex,
                   min_units: float = 1.0) -> pd.DataFrame:
    """Series whose projected stock runs out: first day, units short (expected demand not covered).

    Shortfalls below ``min_units`` in total (fractions of a unit of expected demand) are ignored.
    """
    return risks_from_short(res.daily["lost"] + res.daily["backordered"], keys, dates, min_units)


def risks_from_short(short: np.ndarray, keys: pd.DataFrame, dates: pd.DatetimeIndex,
                     min_units: float = 1.0) -> pd.DataFrame:
    """Series with expected demand not covered (``short``, ``(n, days)``): first day and units short."""
    hit = (short > 1e-6) & (short.sum(axis=1, keepdims=True) >= min_units)
    rows = np.flatnonzero(hit.any(axis=1))
    first = hit[rows].argmax(axis=1)
    return pd.DataFrame({"store": keys["store"].to_numpy()[rows], "item": keys["item"].to_numpy()[rows],
                         "first_day": dates[first], "units_short": short[rows].sum(axis=1), "row": rows})


# --------------------------------------------------------------------------- user stock file
STOCK_SYNONYMS = {
    "on_hand": {"on_hand", "stock", "giacenza", "giacenze", "disponibile", "esistenza", "qty_on_hand", "inventory",
                "stock_on_hand", "quantita_disponibile", "magazzino_qta", "giacenza_attuale", "scorta"},
    "on_order": {"on_order", "ordinato", "in_arrivo", "qty_on_order", "ordered", "open_orders", "ordini_aperti"},
    "arrival": {"arrival", "arrival_date", "data_arrivo", "consegna", "data_consegna", "delivery_date", "eta"},
}


def read_stock_file(raw: pd.DataFrame, default_store: str | None = None) -> pd.DataFrame:
    """Current stock ``[store, item, on_hand, on_order, arrival]`` from a user file (EN / IT column names).

    ``store`` may be missing when the data has one location (``default_store``).
    Raises ``InputError`` when item or on-hand quantity are missing.
    """
    from src.ingest import SYNONYMS, InputError, normalise_name, parse_dates, parse_numbers

    names = {c: normalise_name(c) for c in raw.columns}
    want = {"item": set(SYNONYMS["item"]), "store": set(SYNONYMS["store"]) - {"magazzino", "warehouse"},
            **STOCK_SYNONYMS}
    cols: dict[str, str | None] = {}
    for fld, syns in want.items():
        cols[fld] = next((c for c, n in names.items() if n in syns and c not in cols.values()), None)
    missing = [f for f in ("item", "on_hand") if cols[f] is None]
    if missing:
        raise InputError("Missing column(s): " + ", ".join(missing))
    if cols["store"] is None and default_store is None:
        raise InputError("Missing column(s): store")
    out = pd.DataFrame({
        "store": raw[cols["store"]].astype("string").str.strip() if cols["store"] else default_store,
        "item": raw[cols["item"]].astype("string").str.strip(),
        "on_hand": parse_numbers(raw[cols["on_hand"]]).clip(lower=0),
        "on_order": parse_numbers(raw[cols["on_order"]]).clip(lower=0) if cols["on_order"] else 0.0,
        "arrival": parse_dates(raw[cols["arrival"]])[0] if cols["arrival"] else pd.NaT,
    })
    return out.dropna(subset=["item", "on_hand"]).reset_index(drop=True)
