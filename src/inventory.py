"""Inventory theory: safety stock, reorder point, EOQ and the three policies.

Formulas (docs/08_methodology.md §5):

* ``σ_L = sqrt(L·σ_d² + d̄²·σ_LT²)`` (fixed lead time: ``σ_d·√L``)
* ``SS = z·σ_L`` · ``s = d̄·L + SS`` · ``Q* = sqrt(2·D·S / h)``
* Type-2 (fill rate β): solve ``G(k) = (1 − β)·Q / σ_L`` with the standard
  normal loss function ``G(k) = φ(k) − k·(1 − Φ(k))``.

The functions accept scalars or NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- formulas
def safety_factor(service_level):
    """z = Φ⁻¹(SL). Raises ``ValueError`` unless 0.5 < SL < 1."""
    sl = np.asarray(service_level, dtype="float64")
    if np.any((sl <= 0.5) | (sl >= 1)):
        raise ValueError("service_level must be in (0.5, 1)")
    z = norm.ppf(sl)
    return float(z) if np.ndim(z) == 0 else z


def sigma_lead_time(sigma_daily, lead_time, mean_daily=0.0, sigma_lead_time_days=0.0):
    """Std of demand over the lead time: sqrt(L·σ_d² + d̄²·σ_LT²)."""
    lt = np.asarray(lead_time, dtype="float64")
    if np.any(lt < 0):
        raise ValueError("lead_time must be >= 0")
    out = np.sqrt(lt * np.asarray(sigma_daily, dtype="float64") ** 2
                  + np.asarray(mean_daily, dtype="float64") ** 2 * np.asarray(sigma_lead_time_days) ** 2)
    return float(out) if np.ndim(out) == 0 else out


def safety_stock(z, sigma_L):
    out = np.asarray(z, dtype="float64") * np.asarray(sigma_L, dtype="float64")
    return float(out) if np.ndim(out) == 0 else out


def reorder_point(mean_daily, lead_time, ss):
    out = np.asarray(mean_daily, dtype="float64") * np.asarray(lead_time, dtype="float64") + np.asarray(ss)
    return float(out) if np.ndim(out) == 0 else out


def eoq(annual_demand, order_cost, holding_cost_unit_year):
    """Economic order quantity sqrt(2·D·S / h). Raises ``ValueError`` if h <= 0 or S <= 0."""
    h = np.asarray(holding_cost_unit_year, dtype="float64")
    s = np.asarray(order_cost, dtype="float64")
    d = np.asarray(annual_demand, dtype="float64")
    if np.any(h <= 0):
        raise ValueError("holding cost must be > 0")
    if np.any(s <= 0):
        raise ValueError("order cost must be > 0")
    if np.any(d < 0):
        raise ValueError("annual demand must be >= 0")
    out = np.sqrt(2.0 * d * s / h)
    return float(out) if np.ndim(out) == 0 else out


def normal_loss(k):
    """Standard normal loss function G(k) = φ(k) − k·(1 − Φ(k))."""
    k = np.asarray(k, dtype="float64")
    out = norm.pdf(k) - k * norm.sf(k)
    return float(out) if np.ndim(out) == 0 else out


def safety_factor_fill_rate(fill_rate, Q, sigma_L, lo: float = -4.0, hi: float = 8.0):
    """Safety factor k for a Type-2 (fill-rate) target: G(k) = (1 − β)·Q / σ_L.

    Solved by vectorised bisection (G is decreasing). Where σ_L = 0 no safety
    stock is needed and k = 0 is returned; results are clipped to [lo, hi].
    """
    beta = np.asarray(fill_rate, dtype="float64")
    if np.any((beta <= 0) | (beta >= 1)):
        raise ValueError("fill_rate must be in (0, 1)")
    Q = np.asarray(Q, dtype="float64")
    sig = np.asarray(sigma_L, dtype="float64")
    shape = np.broadcast(beta, Q, sig).shape
    beta, Q, sig = (np.broadcast_to(x, shape).astype("float64") for x in (beta, Q, sig))
    with np.errstate(divide="ignore", invalid="ignore"):
        target = np.where(sig > 0, (1 - beta) * Q / sig, np.inf)
    a = np.full(shape, lo)
    b = np.full(shape, hi)
    for _ in range(80):
        mid = (a + b) / 2
        too_low = normal_loss(mid) > target   # loss still too high -> need larger k
        a = np.where(too_low, mid, a)
        b = np.where(too_low, b, mid)
    k = np.where(sig > 0, b, 0.0)
    return float(k) if k.ndim == 0 else k


# --------------------------------------------------------------------------- policies
@dataclass
class NaiveWeeklyPolicy:
    """P0: order ``qty`` units every ``order_weekday`` (0 = Monday), whatever the stock."""

    qty: np.ndarray
    order_weekday: int = 0
    name: str = "P0_naive"

    def order(self, t: int, weekday: int, inventory_position: np.ndarray) -> np.ndarray:
        if weekday != self.order_weekday:
            return np.zeros_like(inventory_position)
        return np.ceil(np.maximum(self.qty, 0.0) - 1e-9)


def _sq_order(s: np.ndarray, Q: np.ndarray, ip: np.ndarray) -> np.ndarray:
    """(s, Q) rule: while IP <= s order Q -> n·Q with n = floor((s − IP)/Q) + 1."""
    need = ip <= s
    n = np.where(need, np.floor((s - ip) / Q) + 1, 0)
    return n * Q


@dataclass
class StaticSQPolicy:
    """P1: continuous review, fixed reorder point ``s`` and fixed lot ``Q``."""

    s: np.ndarray
    Q: np.ndarray
    name: str = "P1_static_sq"

    def order(self, t: int, weekday: int, inventory_position: np.ndarray) -> np.ndarray:
        return _sq_order(self.s, self.Q, inventory_position)


@dataclass
class ForecastSQPolicy:
    """P2: continuous review, reorder point ``s_t = forecast lead-time demand + SS``."""

    ltd: np.ndarray       # (n_series, n_days) forecast demand over the lead time, per decision day
    ss: np.ndarray        # (n_series,)
    Q: np.ndarray
    name: str = "P2_forecast_sq"

    def order(self, t: int, weekday: int, inventory_position: np.ndarray) -> np.ndarray:
        return _sq_order(self.ltd[:, t] + self.ss, self.Q, inventory_position)


def lot_size(annual_demand, order_cost, holding_cost_unit_year) -> np.ndarray:
    """EOQ rounded up to whole units, at least 1."""
    q = np.asarray(eoq(annual_demand, order_cost, holding_cost_unit_year), dtype="float64")
    return np.maximum(np.ceil(q - 1e-9), 1.0)
