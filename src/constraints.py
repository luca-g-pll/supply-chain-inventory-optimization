"""Supplier terms and business limits for the purchase plan.

Rules applied every day, in this order (docs/08_methodology.md §6.10):

1. **Order and delivery days** — a series can order only on its supplier's order days,
   and only if the delivery (order day + lead time) falls on a delivery day. Its
   reorder point then covers the lead time **plus the days until the next chance to
   order**, so the stock lasts until that order arrives.
2. **Line rules** — at least the *minimum order quantity*, rounded **up to whole packs**;
   if a **quantity discount** is a few days of demand away (``break_extra_days``), the
   line is raised to the discount threshold and bought at the discounted cost.
3. **Warehouse space** — total units per store (stock + on order) never above the
   store's capacity; the most urgent lines are served first.
4. **Minimum order value per supplier and store** — a smaller order is postponed to the
   next order day, unless a line is *urgent* (its stock would run out before the next
   chance arrives): then the order is topped up with whole packs of the items with the
   fewest days of cover until it reaches the minimum.
5. **Weekly budget** — purchases of a calendar week (Monday-Sunday) at most the budget;
   the most urgent lines first, the others are deferred.

Every adjustment is recorded on the order line, every postponed line in a list of
deferred lines, so the user sees what each constraint changed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALL_DAYS = tuple(range(7))
_DAY_NAMES = {
    0: ("mon", "monday", "lun", "lunedi", "lu"), 1: ("tue", "tuesday", "mar", "martedi", "ma"),
    2: ("wed", "wednesday", "mer", "mercoledi", "me"), 3: ("thu", "thursday", "gio", "giovedi", "gi"),
    4: ("fri", "friday", "ven", "venerdi", "ve"), 5: ("sat", "saturday", "sab", "sabato", "sa"),
    6: ("sun", "sunday", "dom", "domenica", "do"),
}
_NAME_TO_DAY = {name: d for d, names in _DAY_NAMES.items() for name in names}


def parse_weekdays(text) -> tuple[int, ...]:
    """``"lun mer ven"``, ``"Mon, Thu"``, ``"1 4"`` (1 = Monday) or a list -> weekday numbers (0 = Monday).

    Empty -> every day.
    """
    if isinstance(text, (list, tuple, np.ndarray)):
        days = sorted({int(x) for x in text if str(x).strip() != ""})
        return tuple(days) or ALL_DAYS
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        return ALL_DAYS
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    out = set()
    for tok in re.split(r"[^a-z0-9]+", plain):
        if not tok:
            continue
        if tok.isdigit() and 1 <= int(tok) <= 7:
            out.add(int(tok) - 1)
        elif tok in _NAME_TO_DAY:
            out.add(_NAME_TO_DAY[tok])
    return tuple(sorted(out)) or ALL_DAYS


@dataclass(frozen=True)
class SupplierTerms:
    order_days: tuple[int, ...] = ALL_DAYS       # weekdays on which orders can be placed
    delivery_days: tuple[int, ...] = ALL_DAYS    # weekdays on which deliveries can arrive
    min_order_value: float = 0.0                 # EUR per order (one supplier, one store, one day)
    discount_qty: float = 0.0                    # units per line from which the discount applies
    discount_pct: float = 0.0                    # e.g. 0.03 = 3%


@dataclass
class Constraints:
    suppliers: dict[str, SupplierTerms] = field(default_factory=dict)
    pack_size: dict[str, float] = field(default_factory=dict)        # item -> units per pack
    min_order_qty: dict[str, float] = field(default_factory=dict)    # item -> minimum units per line
    store_capacity: dict[str, float] = field(default_factory=dict)   # store -> max units (stock + on order)
    weekly_budget: float | None = None                               # EUR per calendar week, all suppliers
    break_extra_days: float = 7.0       # raise a line to the discount threshold if <= this many days of demand away

    @property
    def active(self) -> bool:
        return bool(self.suppliers or self.pack_size or self.min_order_qty or self.store_capacity
                    or self.weekly_budget)


# --------------------------------------------------------------------------- reading
_TERM_COLUMNS = {
    "supplier": {"supplier", "fornitore", "vendor", "supplier_name", "nome_fornitore"},
    "order_days": {"order_days", "giorni_ordine", "giorni_d_ordine", "giorni_di_ordine", "order_weekdays",
                   "giorni_ordini"},
    "delivery_days": {"delivery_days", "giorni_consegna", "giorni_di_consegna", "delivery_weekdays"},
    "min_order_value": {"min_order_value", "ordine_minimo", "ordine_minimo_eur", "minimum_order_value",
                        "min_order", "importo_minimo", "minimo_ordine_eur"},
    "discount_qty": {"discount_qty", "sconto_da_pezzi", "sconto_da", "discount_from_qty", "soglia_sconto",
                     "discount_from"},
    "discount_pct": {"discount_pct", "sconto", "sconto_pct", "sconto_percentuale", "discount", "discount_percent"},
}


def read_supplier_terms(raw: pd.DataFrame) -> pd.DataFrame:
    """Supplier terms ``[supplier, order_days, delivery_days, min_order_value, discount_qty, discount_pct]``.

    Column names in English or Italian; days as in :func:`parse_weekdays`; the discount
    as ``3``, ``3%`` or ``0.03``. Raises ``InputError`` without a supplier column.
    """
    from src.ingest import InputError, normalise_name, parse_numbers

    names = {c: normalise_name(c) for c in raw.columns}
    cols = {fld: next((c for c, n in names.items() if n in syns), None) for fld, syns in _TERM_COLUMNS.items()}
    if cols["supplier"] is None:
        raise InputError("Missing column(s): supplier")
    out = pd.DataFrame({"supplier": raw[cols["supplier"]].astype("string").str.strip()})
    for fld in ("order_days", "delivery_days"):
        out[fld] = [list(map(str, parse_weekdays(v))) for v in raw[cols[fld]]] if cols[fld] else \
            [list(map(str, ALL_DAYS))] * len(raw)
    for fld in ("min_order_value", "discount_qty"):
        out[fld] = parse_numbers(raw[cols[fld]]).fillna(0.0) if cols[fld] else 0.0
    if cols["discount_pct"]:
        pct = parse_numbers(raw[cols["discount_pct"]].astype("string").str.replace("%", "", regex=False)).fillna(0.0)
        out["discount_pct"] = np.where(pct > 1, pct / 100, pct)
    else:
        out["discount_pct"] = 0.0
    return out.dropna(subset=["supplier"]).drop_duplicates("supplier", keep="last").reset_index(drop=True)


def terms_from_frame(df: pd.DataFrame) -> dict[str, SupplierTerms]:
    """``read_supplier_terms`` / editor table -> ``{supplier: SupplierTerms}``."""
    out = {}
    for r in df.itertuples(index=False):
        out[str(r.supplier)] = SupplierTerms(
            order_days=parse_weekdays(r.order_days), delivery_days=parse_weekdays(r.delivery_days),
            min_order_value=float(r.min_order_value or 0), discount_qty=float(r.discount_qty or 0),
            discount_pct=float(r.discount_pct or 0) / (100 if (r.discount_pct or 0) > 1 else 1))
    return out


# --------------------------------------------------------------------------- engine
@dataclass
class PlanResult:
    daily: dict[str, np.ndarray]     # (n, days): demand, on_hand_end, order_qty, receipt, short
    orders: pd.DataFrame             # one row per order line
    deferred: pd.DataFrame           # lines wanted but not ordered (reason)


def _order_masks(keys: pd.DataFrame, lead_time: np.ndarray, cons: Constraints) -> np.ndarray:
    """(n, 7) True where series i may order on weekday w (order day and delivery day both allowed)."""
    n = len(keys)
    mask = np.ones((n, 7), dtype=bool)
    for i, sup in enumerate(keys["supplier"].to_numpy()):
        t = cons.suppliers.get(str(sup))
        if t is None:
            continue
        for w in range(7):
            mask[i, w] = (w in t.order_days) and ((w + int(lead_time[i])) % 7 in t.delivery_days)
    none = ~mask.any(axis=1)
    mask[none] = True        # inconsistent terms (no valid day): fall back to every day rather than never order
    return mask


def _days_to_next_chance(mask: np.ndarray) -> np.ndarray:
    """(n, 7): days from weekday w to the next weekday (strictly after w) on which the series may order."""
    n = mask.shape[0]
    out = np.full((n, 7), 7, dtype=int)
    for w in range(7):
        for k in range(7, 0, -1):
            out[mask[:, (w + k) % 7], w] = k
    return out


def project_constrained(F_ext: np.ndarray, days: int, dates: pd.DatetimeIndex, keys: pd.DataFrame,
                        lead_time: np.ndarray, Q: np.ndarray, safety_stock: np.ndarray, on_hand: np.ndarray,
                        backlog: np.ndarray, arrivals: np.ndarray, backorder_share: np.ndarray,
                        unit_cost: np.ndarray, cons: Constraints | None = None) -> PlanResult:
    """Project the P2 rule for ``days`` days under the supplier terms and limits.

    ``keys`` has ``store, item, supplier``; ``F_ext`` must cover ``days`` + the longest
    lead time + 7 days; ``arrivals`` ``(n, k)`` are the orders already on the way.
    """
    cons = cons or Constraints()
    n = len(keys)
    L = np.asarray(lead_time, dtype=int)
    Q = np.asarray(Q, dtype="float64")
    ss = np.asarray(safety_stock, dtype="float64")
    share = np.broadcast_to(np.asarray(backorder_share, dtype="float64"), (n,))
    cost = np.nan_to_num(np.asarray(unit_cost, dtype="float64"))
    stores = keys["store"].astype(str).to_numpy()
    items = keys["item"].astype(str).to_numpy()
    sups = keys["supplier"].astype(str).to_numpy()
    pack = np.array([max(float(cons.pack_size.get(i, 1) or 1), 1.0) for i in items])
    moq = np.array([float(cons.min_order_qty.get(i, 0) or 0) for i in items])
    terms = [cons.suppliers.get(s, SupplierTerms()) for s in sups]
    disc_qty = np.array([t.discount_qty for t in terms])
    disc_pct = np.array([t.discount_pct for t in terms])
    min_value = {s: cons.suppliers[s].min_order_value for s in cons.suppliers}
    mask = _order_masks(keys, L, cons)
    next_gap = _days_to_next_chance(mask)
    mean_d = F_ext[:, :min(28, F_ext.shape[1])].mean(axis=1)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(F_ext, axis=1)], axis=1)
    rows = np.arange(n)

    oh = np.asarray(on_hand, dtype="float64").copy()
    bl = np.asarray(backlog, dtype="float64").copy()
    k0 = arrivals.shape[1]
    arr = np.zeros((n, max(days, k0) + int(L.max()) + 2))
    arr[:, :k0] = arrivals
    on_order = arr.sum(axis=1)
    daily = {f: np.zeros((n, days)) for f in ("demand", "on_hand_end", "order_qty", "receipt", "short")}
    lines, deferred = [], []
    spent_week, week = 0.0, None

    for t in range(days):
        rec = arr[:, t]
        oh += rec
        on_order -= rec
        served = np.minimum(oh, bl)
        oh -= served
        bl -= served
        d = F_ext[:, t]
        sold = np.minimum(oh, d)
        oh -= sold
        unmet = d - sold
        bl += unmet * share
        daily["demand"][:, t], daily["receipt"][:, t], daily["short"][:, t] = d, rec, unmet

        w = int(dates[t].dayofweek)
        wk = dates[t].isocalendar()[:2]
        if wk != week:
            week, spent_week = wk, 0.0
        ip = oh + on_order - bl
        gap = next_gap[:, w]
        cover_until = np.minimum(t + 1 + L + gap - 1, cs.shape[1] - 1)
        expected = cs[rows, cover_until] - cs[:, t + 1]      # demand until the next chance to order arrives
        # the safety stock covers the forecast error over L days; with fewer order days it must cover L + gap − 1
        s = expected + ss * np.sqrt((L + gap - 1) / L)
        need = mask[:, w] & (ip <= s)
        urgent = need & (ip < expected)
        q = np.where(need, (np.floor((s - ip) / Q) + 1) * Q, 0.0)
        notes = [[] for _ in range(n)]
        price = cost.copy()
        for i in np.flatnonzero(q > 0):
            qi = q[i]
            if qi < moq[i]:
                qi = moq[i]
                notes[i].append("moq")
            packed = np.ceil(qi / pack[i] - 1e-9) * pack[i]
            if packed != qi:
                notes[i].append("pack")
            qi = packed
            if disc_qty[i] > 0 and qi < disc_qty[i] and disc_qty[i] - qi <= cons.break_extra_days * mean_d[i]:
                qi = np.ceil(disc_qty[i] / pack[i] - 1e-9) * pack[i]
                notes[i].append("discount_up")
            if disc_qty[i] > 0 and qi >= disc_qty[i]:
                price[i] = cost[i] * (1 - disc_pct[i])
                notes[i].append("discount")
            q[i] = qi
        cover = np.where(mean_d > 0, ip / np.where(mean_d > 0, mean_d, 1), np.inf)

        # warehouse space per store
        for store, cap in cons.store_capacity.items():
            if not cap or cap <= 0:
                continue
            in_store = stores == str(store)
            room = cap - float((oh + on_order)[in_store].sum())
            for i in sorted(np.flatnonzero(in_store & (q > 0)), key=lambda j: cover[j]):
                fit = np.floor(max(room, 0.0) / pack[i]) * pack[i]
                if fit >= q[i]:
                    room -= q[i]
                    continue
                if fit < max(moq[i], pack[i]) or fit <= 0:
                    deferred.append((dates[t], i, q[i], "space"))
                    q[i] = 0.0
                else:
                    notes[i].append("space")
                    q[i] = fit
                    room -= fit

        # minimum order value per supplier and store
        groups: dict[tuple[str, str], list[int]] = {}
        for i in np.flatnonzero(q > 0):
            groups.setdefault((sups[i], stores[i]), []).append(int(i))
        for (sup, store), idx in groups.items():
            minimum = min_value.get(sup, 0.0)
            idx = np.array(idx)
            value = float((q[idx] * price[idx]).sum())
            if minimum <= 0 or value >= minimum:
                continue
            if not urgent[idx].any():
                for i in idx:
                    deferred.append((dates[t], i, q[i], "min_value"))
                    q[i] = 0.0
                continue
            # top up with whole packs of the items of this supplier and store with the fewest days of cover
            pool = np.flatnonzero((sups == sup) & (stores == store) & mask[:, w] & (mean_d > 0))
            for _ in range(1000):
                if value >= minimum or len(pool) == 0:
                    break
                proj = (ip[pool] + q[pool]) / mean_d[pool]
                i = pool[int(np.argmin(proj))]
                add = max(pack[i], moq[i] if q[i] == 0 else 0.0)
                q[i] += add
                value += add * price[i]
                if "topup" not in notes[i]:
                    notes[i].append("topup")

        # weekly budget
        if cons.weekly_budget and cons.weekly_budget > 0 and (q > 0).any():
            for i in sorted(np.flatnonzero(q > 0), key=lambda j: (not urgent[j], cover[j])):
                v = q[i] * price[i]
                if spent_week + v <= cons.weekly_budget:
                    spent_week += v
                else:
                    deferred.append((dates[t], i, q[i], "budget"))
                    q[i] = 0.0
        elif (q > 0).any():
            spent_week += float((q * price).sum())

        for i in np.flatnonzero(q > 0):
            arr[i, t + L[i]] += q[i]
            lines.append((dates[t], i, q[i], price[i], dates[t] + pd.Timedelta(days=int(L[i])), oh[i],
                          on_order[i], cover[i], bool(urgent[i]), ",".join(dict.fromkeys(notes[i]))))
        on_order += q
        daily["order_qty"][:, t] = q
        daily["on_hand_end"][:, t] = oh

    orders = pd.DataFrame(lines, columns=["date", "row", "qty", "unit_cost", "arrival", "on_hand", "on_order",
                                          "cover_days", "urgent", "notes"])
    orders["value"] = orders["qty"] * orders["unit_cost"]
    orders["saving"] = orders["qty"] * (cost[orders["row"].to_numpy(dtype=int)] - orders["unit_cost"]) if len(orders) else []
    deferred = pd.DataFrame(deferred, columns=["date", "row", "qty", "reason"])
    return PlanResult(daily=daily, orders=orders, deferred=deferred)
