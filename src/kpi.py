"""KPIs from the simulation sums (docs/09_kpi_definitions.md).

Every KPI is derived from **additive** sums, so any group of series (a
dashboard scenario, an ABC class, a channel) is aggregated exactly: sum the
rows first, then compute the ratios. A fill rate is never an average of ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUM_COLUMNS = (
    "demand", "sales", "lost", "backordered", "bo_served", "inv_unit_days", "backlog_unit_days",
    "n_orders", "units_ordered", "stockout_days", "stockout_events", "cycles", "cycles_with_stockout",
    "inv_value_days", "cogs", "revenue", "lost_revenue", "ordering_cost", "holding_cost",
    "shortage_cost", "total_cost", "n_series",
)
KPI_LABELS = {
    "fill_rate": "Fill rate",
    "cycle_service_level": "Cycle service level",
    "avg_inventory_units": "Average inventory (units)",
    "avg_inventory_value": "Average inventory (€)",
    "stockout_days": "Stock-out days (series-days)",
    "stockout_events": "Stock-out events",
    "turns": "Inventory turns (per year)",
    "days_of_supply": "Days of supply",
    "n_orders": "Orders",
    "total_cost": "Total cost (€)",
}


def add_costs(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cost and value columns to a per-series simulation frame.

    Needs the columns written by ``policy.run_frame`` (unit cost / price, holding
    rate, order cost, backorder cost, days per year) and the simulation sums.
    """
    df = frame.copy()
    c, p = df["unit_cost"], df["unit_price"]
    df["inv_value_days"] = df["inv_unit_days"] * c
    df["cogs"] = (df["sales"] + df["bo_served"]) * c
    df["revenue"] = (df["sales"] + df["bo_served"]) * p
    df["lost_revenue"] = df["lost"] * p
    df["ordering_cost"] = df["n_orders"] * df["order_cost"]
    df["holding_cost"] = df["inv_value_days"] * df["holding_rate"] / df["days_per_year"]
    df["shortage_cost"] = df["lost"] * (p - c) + df["backordered"] * df["backorder_cost"]
    df["total_cost"] = df["ordering_cost"] + df["holding_cost"] + df["shortage_cost"]
    df["n_series"] = 1
    return derive_kpis(df)


def derive_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios from the sums (works on per-series rows or on aggregated rows)."""
    out = df.copy()
    n_days = out["n_days"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fill_rate"] = np.where(out["demand"] > 0, out["sales"] / out["demand"], 1.0)
        out["cycle_service_level"] = np.where(out["cycles"] > 0, 1 - out["cycles_with_stockout"] / out["cycles"], 1.0)
        out["avg_inventory_units"] = out["inv_unit_days"] / n_days
        out["avg_inventory_value"] = out["inv_value_days"] / n_days
        out["avg_backlog_units"] = out["backlog_unit_days"] / n_days
        annual_cogs = out["cogs"] * 365.0 / n_days
        out["turns"] = np.where(out["avg_inventory_value"] > 0, annual_cogs / out["avg_inventory_value"], np.nan)
        daily_sales = (out["sales"] + out["bo_served"]) / n_days
        out["days_of_supply"] = np.where(daily_sales > 0, out["avg_inventory_units"] / daily_sales, np.nan)
    return out


def aggregate_kpis(frame: pd.DataFrame, by) -> pd.DataFrame:
    """Sum the additive columns by ``by`` and derive the KPIs."""
    by = [by] if isinstance(by, str) else list(by)
    cols = [c for c in SUM_COLUMNS if c in frame.columns]
    if by:
        agg = frame.groupby(by, as_index=False, observed=True, sort=False)[cols].sum()
        agg["n_days"] = frame.groupby(by, observed=True, sort=False)["n_days"].max().to_numpy()
    else:
        agg = frame[cols].sum().to_frame().T
        agg["n_days"] = frame["n_days"].max()
    return derive_kpis(agg)


def headline(summary: pd.DataFrame, variant: str = "calibrated", baseline: str = "P0_naive",
             candidate: str = "P2_forecast_sq", classic: str = "P1_static_sq") -> dict:
    """Headline numbers for README / dashboard from a (scenario-filtered) summary."""
    sub = summary[summary["variant"] == variant]
    if sub.empty:
        return {}
    k = aggregate_kpis(sub, ["policy"]).set_index("policy")

    def _get(policy: str, col: str) -> float:
        return float(k.at[policy, col]) if policy in k.index else float("nan")

    def _red(a: float, b: float) -> float:
        return 1 - b / a if a and np.isfinite(a) and a > 0 else float("nan")

    inv_b, inv_c, inv_p1 = (_get(p, "avg_inventory_value") for p in (baseline, candidate, classic))
    cost_b, cost_c, cost_p1 = (_get(p, "total_cost") for p in (baseline, candidate, classic))
    return {
        "variant": variant,
        "fill_rate_baseline": _get(baseline, "fill_rate"),
        "fill_rate_classic": _get(classic, "fill_rate"),
        "fill_rate_candidate": _get(candidate, "fill_rate"),
        "avg_inventory_value_baseline": inv_b,
        "avg_inventory_value_classic": inv_p1,
        "avg_inventory_value_candidate": inv_c,
        "inventory_reduction_pct": _red(inv_b, inv_c),
        "inventory_reduction_vs_classic_pct": _red(inv_p1, inv_c),
        "total_cost_baseline": cost_b,
        "total_cost_classic": cost_p1,
        "total_cost_candidate": cost_c,
        "total_cost_reduction_pct": _red(cost_b, cost_c),
        "total_cost_reduction_eur": cost_b - cost_c,
        "total_cost_reduction_vs_classic_pct": _red(cost_p1, cost_c),
        "stockout_days_baseline": _get(baseline, "stockout_days"),
        "stockout_days_candidate": _get(candidate, "stockout_days"),
        "n_series": int(_get(candidate, "n_series")) if candidate in k.index else 0,
    }
