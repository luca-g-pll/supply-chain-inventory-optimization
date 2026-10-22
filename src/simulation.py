"""Day-by-day inventory simulation, vectorised across series.

Event order within day ``t`` (docs/08_methodology.md §6.2):

1. receive the orders due on ``t``; serve the open backlog first (online channel);
2. serve demand from stock; the unmet part is split into back-ordered
   (``backorder_share``) and lost units;
3. inventory position ``IP = on hand + on order − backlog``;
4. the policy decides the order, which arrives at the start of day ``t + L``;
5. record the end-of-day state (holding cost is charged on it).

The engine loops over days only; every series is updated at once with NumPy,
so hundreds of series x one year take a fraction of a second.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

SUM_FIELDS = (
    "demand", "sales", "lost", "backordered", "bo_served", "inv_unit_days", "backlog_unit_days",
    "n_orders", "units_ordered", "stockout_days", "stockout_events", "cycles", "cycles_with_stockout",
)
DAILY_FIELDS = ("demand", "sales", "lost", "backordered", "backlog", "on_hand_end", "on_order",
                "order_qty", "receipt")


@dataclass
class BatchResult:
    sums: dict[str, np.ndarray]                   # additive counters per series
    n_days: int
    daily: dict[str, np.ndarray] | None = None    # (n_series, n_days) arrays if recorded
    end_on_hand: np.ndarray | None = None         # (n_series,) stock at the end of the last day
    end_backlog: np.ndarray | None = None
    pipeline: np.ndarray | None = None            # (n_series, k) orders arriving k days after the last day

    @property
    def fill_rate(self) -> np.ndarray:
        d = self.sums["demand"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, self.sums["sales"] / d, 1.0)

    def sums_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.sums)
        df["n_days"] = self.n_days
        return df


def simulate_batch(demand: np.ndarray, dates: pd.DatetimeIndex, policy, lead_time,
                   initial_on_hand, backorder_share: float = 0.0,
                   record_daily: bool = False, initial_arrivals: np.ndarray | None = None,
                   initial_backlog=None) -> BatchResult:
    """Simulate every series (rows of ``demand``) over the days of ``dates``.

    ``initial_arrivals`` ``(n, k)``: orders already placed, arriving on days 0..k-1
    (they count as *on order* from the start).
    """
    demand = np.asarray(demand, dtype="float64")
    n, T = demand.shape
    if len(dates) != T:
        raise ValueError("demand and dates have different lengths")
    if not 0 <= backorder_share <= 1:
        raise ValueError("backorder_share must be in [0, 1]")
    L = np.broadcast_to(np.asarray(lead_time, dtype=int), (n,)).copy()
    if (L < 1).any():
        raise ValueError("lead_time must be >= 1")
    on_hand = np.broadcast_to(np.asarray(initial_on_hand, dtype="float64"), (n,)).copy()
    on_order = np.zeros(n)
    backlog = np.zeros(n) if initial_backlog is None else \
        np.broadcast_to(np.asarray(initial_backlog, dtype="float64"), (n,)).copy()
    k0 = 0 if initial_arrivals is None else np.asarray(initial_arrivals).shape[1]
    arrivals = np.zeros((n, max(T, k0) + int(L.max()) + 1))
    if k0:
        arrivals[:, :k0] = initial_arrivals
        on_order += arrivals[:, :k0].sum(axis=1)
    rows = np.arange(n)
    weekday = dates.dayofweek.to_numpy()

    s = {f: np.zeros(n) for f in SUM_FIELDS}
    prev_so = np.zeros(n, dtype=bool)
    so_since_receipt = np.zeros(n, dtype=bool)
    daily = {f: np.zeros((n, T)) for f in DAILY_FIELDS} if record_daily else None

    for t in range(T):
        rec = arrivals[:, t]
        received = rec > 0
        s["cycles"] += received
        s["cycles_with_stockout"] += received & so_since_receipt
        so_since_receipt &= ~received
        on_hand += rec
        on_order -= rec

        if backorder_share > 0:
            served = np.minimum(on_hand, backlog)
            on_hand -= served
            backlog -= served
            s["bo_served"] += served

        d = demand[:, t]
        sold = np.minimum(on_hand, d)
        on_hand -= sold
        unmet = d - sold
        bo = unmet * backorder_share
        lost = unmet - bo
        backlog += bo
        so = unmet > 1e-9
        s["demand"] += d
        s["sales"] += sold
        s["lost"] += lost
        s["backordered"] += bo
        s["stockout_days"] += so
        s["stockout_events"] += so & ~prev_so
        prev_so = so
        so_since_receipt |= so

        ip = on_hand + on_order - backlog
        q = policy.order(t, int(weekday[t]), ip)
        has = q > 0
        if has.any():
            idx = rows[has]
            arrivals[idx, t + L[idx]] += q[has]
            on_order += q
            s["n_orders"] += has
            s["units_ordered"] += q
        s["inv_unit_days"] += on_hand
        s["backlog_unit_days"] += backlog

        if daily is not None:
            daily["demand"][:, t] = d
            daily["sales"][:, t] = sold
            daily["lost"][:, t] = lost
            daily["backordered"][:, t] = bo
            daily["backlog"][:, t] = backlog
            daily["on_hand_end"][:, t] = on_hand
            daily["on_order"][:, t] = on_order
            daily["order_qty"][:, t] = q
            daily["receipt"][:, t] = rec

    s["cycles"] += 1  # the cycle still open at the end
    s["cycles_with_stockout"] += so_since_receipt
    return BatchResult(sums=s, n_days=T, daily=daily, end_on_hand=on_hand, end_backlog=backlog,
                       pipeline=arrivals[:, T:].copy())


def simulate(demand, dates: pd.DatetimeIndex, policy, lead_time: int, initial_on_hand: float,
             backorder_share: float = 0.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """One series: returns the daily table and the summary counters."""
    res = simulate_batch(np.asarray(demand, dtype="float64")[None, :], dates, policy, [lead_time],
                         [initial_on_hand], backorder_share, record_daily=True)
    daily = pd.DataFrame({f: v[0] for f, v in res.daily.items()})
    daily.insert(0, "date", dates)
    summary = {k: float(v[0]) for k, v in res.sums.items()}
    summary["n_days"] = res.n_days
    summary["fill_rate"] = float(res.fill_rate[0])
    return daily, summary


def calibrate_to_fill_rate(policy_for: Callable[[np.ndarray], object], demand: np.ndarray,
                           dates: pd.DatetimeIndex, lead_time, initial_on_hand, backorder_share: float,
                           target: float, lo: float, hi: float, max_iter: int = 40,
                           knob_tol: float = 1e-3) -> tuple[np.ndarray, BatchResult, np.ndarray]:
    """Find, per series, the smallest knob whose simulated fill rate reaches ``target``.

    ``policy_for(knob)`` builds the policy for an array of knob values (``z`` for
    P1/P2, a quantity multiplier for P0). Vectorised bisection: every iteration
    simulates all series at once. Returns ``(knob, final_result, feasible)``; where
    even ``hi`` misses the target, ``knob = hi`` and ``feasible = False``.
    """
    n = demand.shape[0]
    a = np.full(n, float(lo))
    b = np.full(n, float(hi))
    kwargs = dict(lead_time=lead_time, initial_on_hand=initial_on_hand, backorder_share=backorder_share)
    feasible = simulate_batch(demand, dates, policy_for(b), **kwargs).fill_rate >= target
    for _ in range(max_iter):
        mid = (a + b) / 2
        ok = simulate_batch(demand, dates, policy_for(mid), **kwargs).fill_rate >= target
        b = np.where(ok, mid, b)
        a = np.where(ok, a, mid)
        if np.max(b - a) < knob_tol:
            break
    b = np.where(feasible, b, float(hi))
    final = simulate_batch(demand, dates, policy_for(b), **kwargs)
    return b, final, feasible
