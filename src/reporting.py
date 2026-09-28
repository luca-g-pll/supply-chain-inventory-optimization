"""Static figures for the README (matplotlib) and artefact files for the dashboard.

Colour roles follow the project's validated palette (see ``PALETTE``): policies
are categorical and keep the same colour in every chart (P2 blue, P1 orange,
P0 aqua); single-measure comparisons use *emphasis* (one accent, the rest
gray); ordered classes use one-hue ordinal steps. Marks are thin, gridlines are
solid hairlines and the series that matter carry direct labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

PALETTE = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "context": "#c3c2b7", "context_dark": "#898781",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948",
    "critical": "#d03b3b",
}
POLICY_ORDER = ["P2_forecast_sq", "P1_static_sq", "P0_naive"]
POLICY_COLORS = {"P2_forecast_sq": PALETTE["blue"], "P1_static_sq": PALETTE["orange"], "P0_naive": PALETTE["aqua"]}
POLICY_SHORT = {"P2_forecast_sq": "P2 forecast-driven", "P1_static_sq": "P1 static (s, Q)",
                "P0_naive": "P0 naive weekly"}
CHANNEL_COLORS = {"physical": PALETTE["blue"], "online": PALETTE["orange"]}
COST_COLORS = {"ordering_cost": PALETTE["blue"], "holding_cost": PALETTE["orange"],
               "shortage_cost": PALETTE["aqua"]}
COST_LABELS = {"ordering_cost": "Ordering", "holding_cost": "Holding", "shortage_cost": "Shortage (lost margin)"}
ORDINAL_BLUES = ["#86b6ef", "#2a78d6", "#104281"]
SEQUENTIAL_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    p = PALETTE
    plt.rcParams.update({
        "figure.facecolor": p["surface"], "axes.facecolor": p["surface"], "savefig.facecolor": p["surface"],
        "axes.edgecolor": p["axis"], "axes.labelcolor": p["ink2"], "axes.titlecolor": p["ink"],
        "text.color": p["ink"], "xtick.color": p["muted"], "ytick.color": p["muted"],
        "xtick.labelcolor": p["ink2"], "ytick.labelcolor": p["ink2"],
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": p["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9.5, "figure.dpi": 100,
    })


def _title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=26 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=PALETTE["ink2"], va="bottom")


def _fig_title(fig, title: str, subtitle: str | None = None) -> None:
    """Title + subtitle at fixed distances from the top, then lay out the axes below them."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.18 / h, ha="left", va="top", fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.01, 1 - 0.48 / h, subtitle, ha="left", va="top", fontsize=9.5, color=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.8 if subtitle else 0.5) / h))


def _save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def _pct_axis(axis, vmax: float, decimals: int = 0) -> None:
    """Percent ticks on clean steps (1, 2, 5, 10, 20 % ...), about 4-7 ticks."""
    from matplotlib.ticker import MultipleLocator

    for step in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25):
        if vmax / step <= 7:
            break
    axis.set_major_locator(MultipleLocator(step))
    axis.set_major_formatter(PercentFormatter(1.0, decimals=decimals if step >= 0.01 else 1))


def _euro(x, _pos=None) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"€{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:.0f}K"
    return f"€{x:.0f}"


def _thousands(x, _pos=None) -> str:
    return f"{x:,.0f}"


# --------------------------------------------------------------------------- demand
def fig_demand_trend(daily: pd.DataFrame, by_year: pd.DataFrame, folder: Path) -> Path:
    apply_style()
    d = daily.sort_values("date").copy()
    d["date"] = pd.to_datetime(d["date"])
    d["ma28"] = d["units"].rolling(28, min_periods=7).mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d["date"], d["units"], color=PALETTE["context"], linewidth=0.8)
    ax.plot(d["date"], d["ma28"], color=PALETTE["blue"], linewidth=2)
    last = d.dropna(subset=["ma28"]).iloc[-1]
    ax.annotate("28-day average", (last["date"], last["ma28"]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9.5, color=PALETTE["ink2"])
    growth = by_year.dropna(subset=["yoy_growth_pct"])
    sub = "Daily units, all stores and items"
    if len(growth):
        sub += f" · average growth {growth['yoy_growth_pct'].mean():+.1f}% per year"
    _title(ax, "Demand grows every year and peaks every summer", sub)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    return _save(fig, folder, "demand_trend.png")


def fig_seasonality(dow: pd.DataFrame, month: pd.DataFrame, folder: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [7, 12]}, sharey=True)
    lo = min(dow["dow_index"].min(), month["month_index"].min())
    hi = max(dow["dow_index"].max(), month["month_index"].max())
    for ax, df, col, labels, title in (
        (axes[0], dow, "dow_index", [x[:3] for x in dow["day_name"]], "Day of week"),
        (axes[1], month, "month_index", [x[:3] for x in month["month_name"]], "Month of year"),
    ):
        vals = df[col].to_numpy() - 1.0
        colors = np.where(vals >= 0, PALETTE["blue"], PALETTE["red"])
        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.55, color=colors, bottom=1.0)
        ax.axhline(1.0, color=PALETTE["axis"], linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}"))
        k_max, k_min = int(np.argmax(vals)), int(np.argmin(vals))
        for k in (k_max, k_min):
            ax.annotate(f"{df[col].iloc[k]:.2f}", (k, 1 + vals[k]), xytext=(0, 4 if vals[k] >= 0 else -4),
                        textcoords="offset points", ha="center", va="bottom" if vals[k] >= 0 else "top",
                        fontsize=9, color=PALETTE["ink2"])
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(lo - 0.08, hi + 0.08)
    _fig_title(fig, "Seasonality index (1.0 = average day)",
               "Blue = busier than average, red = quieter · whole history, all stores")
    return _save(fig, folder, "seasonality.png")


def fig_abc_pareto(classification: pd.DataFrame, thresholds, folder: Path) -> Path:
    apply_style()
    c = classification.sort_values("rank")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(1, len(c) + 1)
    class_color = {"A": ORDINAL_BLUES[2], "B": ORDINAL_BLUES[1], "C": ORDINAL_BLUES[0]}
    ax.plot(np.r_[0, x], np.r_[0, c["cum_share"]], color=PALETTE["axis"], linewidth=1.5, zorder=1)
    counts = c["abc"].value_counts()
    for cls in "ABC":
        m = (c["abc"] == cls).to_numpy()
        ax.scatter(x[m], c["cum_share"].to_numpy()[m], s=34, color=class_color[cls], edgecolor=PALETTE["surface"],
                   linewidth=1.5, zorder=3, label=f"{cls} · {int(counts.get(cls, 0))} items")
    for thr, cls in zip(thresholds, "AB"):
        ax.axhline(thr, color=PALETTE["axis"], linewidth=0.8, zorder=0)
        ax.text(0.6, thr + 0.01, f"{cls} up to {thr:.0%}", va="bottom", ha="left", fontsize=9, color=PALETTE["ink2"])
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, len(c) + 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Items ranked by revenue (validation period)")
    ax.set_ylabel("Cumulative share of revenue")
    ax.legend(loc="lower right")
    n_a = int(counts.get("A", 0))
    _title(ax, f"ABC classification: {n_a} of {len(c)} items make {thresholds[0]:.0%} of revenue",
           f"A = first {thresholds[0]:.0%} of revenue, B = up to {thresholds[1]:.0%}, C = the rest")
    return _save(fig, folder, "abc_pareto.png")


def fig_abc_xyz_matrix(matrix: pd.DataFrame, folder: Path) -> Path:
    apply_style()
    pivot_share = matrix.pivot(index="abc", columns="xyz", values="revenue_share").reindex(index=list("ABC"), columns=list("XYZ"))
    pivot_n = matrix.pivot(index="abc", columns="xyz", values="items").reindex(index=list("ABC"), columns=list("XYZ"))
    cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES)
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    vmax = max(float(np.nanmax(pivot_share.to_numpy())), 1e-9)
    ax.imshow(pivot_share.to_numpy(), cmap=cmap, vmin=0, vmax=vmax)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            share, n = pivot_share.iat[i, j], int(pivot_n.iat[i, j])
            dark = share / vmax > 0.55
            ax.text(j, i, f"{n} items\n{share:.0%} of revenue", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if dark else PALETTE["ink"])
    ax.set_xticks(range(3), ["X · stable", "Y · variable", "Z · erratic"])
    ax.set_yticks(range(3), ["A", "B", "C"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _title(ax, "ABC × XYZ matrix", "Value class (rows) × demand variability (columns)")
    return _save(fig, folder, "abc_xyz_matrix.png")


# --------------------------------------------------------------------------- forecasting
MODEL_SHORT = {"seasonal_naive": "Seasonal naive (baseline)", "moving_average": "Moving average 28 d",
               "holt_winters": "Holt-Winters", "lightgbm": "LightGBM"}


def fig_wape_by_model(leaderboard: pd.DataFrame, best: str, baseline: str, folder: Path) -> Path:
    apply_style()
    lb = leaderboard.sort_values("wape", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(lb))
    colors = [PALETTE["blue"] if m == best else PALETTE["context"] for m in lb["model"]]
    ax.barh(y, lb["wape"], color=colors, height=0.5)
    ax.set_yticks(y, [MODEL_SHORT.get(m, m) for m in lb["model"]])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _pct_axis(ax.xaxis, float(lb["wape"].max()) * 1.18)
    for k, (_, r) in enumerate(lb.iterrows()):
        ax.text(r["wape"], k, f"  {r['wape']:.1%}", va="center", fontsize=9.5, color=PALETTE["ink2"])
    base = float(lb.set_index("model").at[baseline, "wape"])
    best_w = float(lb.set_index("model").at[best, "wape"])
    _title(ax, f"{MODEL_SHORT.get(best, best)} cuts forecast error by {1 - best_w / base:.0%} vs. the baseline",
           "WAPE on the test-period back-test (weekly origins, 1–14 days ahead) · lower is better")
    ax.set_xlim(0, lb["wape"].max() * 1.18)
    return _save(fig, folder, "forecast_wape_by_model.png")


def fig_wape_by_horizon(by_h: pd.DataFrame, best: str, baseline: str, folder: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    order = [m for m in by_h["model"].unique() if m not in (best, baseline)] + [baseline, best]
    for m in order:
        d = by_h[by_h["model"] == m].sort_values("h")
        color = PALETTE["blue"] if m == best else (PALETTE["context_dark"] if m == baseline else PALETTE["context"])
        ax.plot(d["h"], d["wape"], color=color, linewidth=2 if m in (best, baseline) else 1.5)
        ax.annotate(MODEL_SHORT.get(m, m), (d["h"].iloc[-1], d["wape"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=PALETTE["ink2"])
    _pct_axis(ax.yaxis, float(by_h["wape"].max()) * 1.1)
    ax.set_ylim(0, float(by_h["wape"].max()) * 1.1)
    ax.set_xlabel("Days ahead")
    ax.set_xticks(range(1, int(by_h["h"].max()) + 1))
    ax.set_xlim(0.8, by_h["h"].max() + 3.2)
    _title(ax, "Forecast error by horizon", "WAPE on the test period · the gap to the baseline holds at every horizon")
    return _save(fig, folder, "forecast_wape_by_horizon.png")


def fig_forecast_example(view: pd.DataFrame, best: str, baseline: str, label: str, folder: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    a = view[view["model"] == best].sort_values("date")
    ax.plot(a["date"], a["actual"], color=PALETTE["ink2"], linewidth=1.2, label="Actual")
    b = view[view["model"] == baseline].sort_values("date")
    ax.plot(b["date"], b["forecast"], color=PALETTE["context"], linewidth=1.5,
            label=MODEL_SHORT.get(baseline, baseline))
    ax.plot(a["date"], a["forecast"], color=PALETTE["blue"], linewidth=2, label=MODEL_SHORT.get(best, best))
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Units per day")
    _title(ax, f"Actual vs. forecast · {label}", "Forecast made up to 7 days ahead from the latest weekly origin")
    return _save(fig, folder, "forecast_example.png")


def fig_feature_importance(importance: pd.Series, folder: Path) -> Path:
    apply_style()
    imp = (importance / importance.sum()).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(imp))
    ax.barh(y, imp.to_numpy(), color=PALETTE["blue"], height=0.5)
    ax.set_yticks(y, imp.index)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _pct_axis(ax.xaxis, float(imp.max()))
    _title(ax, "What LightGBM relies on", "Share of total gain, top 12 features")
    return _save(fig, folder, "lightgbm_feature_importance.png")


# --------------------------------------------------------------------------- inventory
def fig_policy_tradeoff(curves: pd.DataFrame, calibrated: pd.DataFrame, target: float, folder: Path,
                        metric: str = "avg_inventory_value", name: str = "policy_tradeoff.png",
                        title: str | None = None, subtitle: str | None = None, ylabel: str = "Average inventory value") -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5))
    cal = calibrated.set_index("policy")
    y_cap = float(cal[metric].max()) * 1.7 if len(cal) else float(curves[metric].max())
    x_min = max(0.8, float(curves["fill_rate"].min()) - 0.005)
    clipped = []
    for pol in POLICY_ORDER:
        c = curves[curves["policy"] == pol].sort_values("fill_rate")
        if c.empty:
            continue
        label = POLICY_SHORT[pol]
        if pol in cal.index:
            label += f" · {_euro(cal.at[pol, metric])} at {cal.at[pol, 'fill_rate']:.1%}"
        ax.plot(c["fill_rate"], c[metric], color=POLICY_COLORS[pol], linewidth=2, label=label, zorder=2)
        inside = c[(c[metric] <= y_cap) & (c["fill_rate"] >= x_min)]
        ax.scatter(inside["fill_rate"], inside[metric], s=16, color=POLICY_COLORS[pol],
                   edgecolor=PALETTE["surface"], linewidth=1.2, zorder=3)
        if (c[metric] > y_cap).any():
            clipped.append(POLICY_SHORT[pol])
        if pol in cal.index:
            ax.scatter([cal.at[pol, "fill_rate"]], [cal.at[pol, metric]], s=80, color=POLICY_COLORS[pol],
                       edgecolor=PALETTE["surface"], linewidth=2, zorder=4)
    ax.set_ylim(0, y_cap)
    ax.set_xlim(x_min, 1.002)
    ax.axvline(target, color=PALETTE["axis"], linewidth=1, zorder=1)
    ax.text(target, y_cap * 0.02, f" target {target:.0%}", va="bottom", fontsize=9, color=PALETTE["ink2"])
    if clipped:
        ax.text(0.99, 0.98, f"{', '.join(clipped)}: keeps rising off the chart", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=PALETTE["ink2"])
    _pct_axis(ax.xaxis, 1.0 - x_min)
    ax.yaxis.set_major_formatter(FuncFormatter(_euro))
    ax.set_xlabel("Fill rate (share of demand served from stock)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x")
    ax.legend(loc="upper left")
    _title(ax, title or "Less stock for the same service: the forecast-driven curve sits lowest",
           subtitle or "Each line = one policy with more or less safety stock (same knob for every series) · "
                       "big dots = calibrated per series to the target")
    return _save(fig, folder, name)


def fig_cost_vs_service(curves: pd.DataFrame, calibrated: pd.DataFrame, target: float, folder: Path) -> Path:
    """Total cost against fill rate: where the cost-optimal service level lies."""
    best = float(curves.loc[curves["total_cost"].idxmin(), "fill_rate"])
    shown = np.floor(best * 1000) / 1000
    return fig_policy_tradeoff(
        curves, calibrated, target, folder, metric="total_cost", name="cost_vs_service.png",
        title=f"With these costs, total cost keeps falling up to a {shown:.1%} fill rate",
        subtitle="Total cost (ordering + holding + lost margin) for each policy as safety stock grows · "
                 "lost sales cost far more than holding stock",
        ylabel="Total cost over the test period")


def fig_inventory_at_target(kpis: pd.DataFrame, target: float, folder: Path) -> Path:
    apply_style()
    k = kpis.set_index("policy").reindex([p for p in reversed(POLICY_ORDER) if p in set(kpis["policy"])])
    fig, ax = plt.subplots(figsize=(9, 3.3))
    y = np.arange(len(k))
    colors = [PALETTE["blue"] if p == "P2_forecast_sq" else PALETTE["context"] for p in k.index]
    ax.barh(y, k["avg_inventory_value"], color=colors, height=0.5)
    ax.set_yticks(y, [POLICY_SHORT[p] for p in k.index])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(_euro))
    base = float(k.at["P0_naive", "avg_inventory_value"]) if "P0_naive" in k.index else np.nan
    for i, (pol, r) in enumerate(k.iterrows()):
        txt = f"  {_euro(r['avg_inventory_value'])} · fill rate {r['fill_rate']:.1%}"
        if pol != "P0_naive" and np.isfinite(base):
            txt += f" · {r['avg_inventory_value'] / base - 1:+.0%} vs P0"
        ax.text(r["avg_inventory_value"], i, txt, va="center", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_xlim(0, k["avg_inventory_value"].max() * 1.55)
    _title(ax, f"Average inventory needed for a {target:.0%} fill rate",
           "Each policy calibrated per series to the same fill rate · physical stores, test period")
    return _save(fig, folder, "policy_inventory_at_target.png")


def fig_cost_breakdown(kpis: pd.DataFrame, folder: Path, name: str = "cost_breakdown.png",
                       title: str = "Where the money goes") -> Path:
    apply_style()
    k = kpis.set_index("policy").reindex([p for p in reversed(POLICY_ORDER) if p in set(kpis["policy"])])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(k))
    left = np.zeros(len(k))
    gap = k["total_cost"].max() * 0.004
    for comp in ("ordering_cost", "holding_cost", "shortage_cost"):
        vals = k[comp].to_numpy()
        ax.barh(y, np.maximum(vals - gap, 0), left=left, color=COST_COLORS[comp], height=0.5, label=COST_LABELS[comp])
        left += vals
    for i, tot in enumerate(k["total_cost"]):
        ax.text(tot, i, f"  {_euro(tot)}", va="center", fontsize=9.5, color=PALETTE["ink2"])
    ax.set_yticks(y, [POLICY_SHORT[p] for p in k.index])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(_euro))
    ax.set_xlim(0, k["total_cost"].max() * 1.2)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.14))
    _title(ax, title, "Total cost over the test period by component · policies calibrated to the same fill rate")
    return _save(fig, folder, name)


def fig_inventory_trajectory(daily: pd.DataFrame, label: str, folder: Path) -> Path:
    apply_style()
    pols = [p for p in POLICY_ORDER if p in set(daily["policy"])]
    fig, axes = plt.subplots(len(pols), 1, figsize=(10, 2.1 * len(pols) + 0.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    y_max = float(daily["on_hand_end"].max()) * 1.05
    for ax, pol in zip(axes, pols):
        d = daily[daily["policy"] == pol].sort_values("date")
        ax.fill_between(d["date"], d["on_hand_end"], color=POLICY_COLORS[pol], alpha=0.10, linewidth=0)
        ax.plot(d["date"], d["on_hand_end"], color=POLICY_COLORS[pol], linewidth=1.5)
        so = d[(d["lost"] + d["backordered"]) > 0]
        if len(so):
            ax.scatter(so["date"], np.zeros(len(so)), marker="|", s=60, color=PALETTE["critical"], zorder=3,
                       clip_on=False)
        ax.set_title(f"{POLICY_SHORT[pol]} · average {d['on_hand_end'].mean():,.0f} units · "
                     f"{len(so)} stock-out days", loc="left", fontsize=10, fontweight="normal", color=PALETTE["ink"])
        ax.set_ylim(0, y_max)
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    _fig_title(fig, f"Units on hand, day by day · {label}",
               "Red ticks = days with unmet demand · all three policies reach the same fill rate")
    return _save(fig, folder, "inventory_trajectory.png")


def fig_sensitivity(sens: pd.DataFrame, folder: Path) -> Path:
    apply_style()
    pols = [p for p in ("P1_static_sq", "P2_forecast_sq") if p in set(sens["policy"])]
    fig, axes = plt.subplots(1, len(pols), figsize=(11, 4), sharey=True)
    axes = np.atleast_1d(axes)
    lts = sorted(sens["lead_time"].unique())
    colors = dict(zip(lts, ORDINAL_BLUES[-len(lts):] if len(lts) <= 3 else SEQUENTIAL_BLUES[2:]))
    for ax, pol in zip(axes, pols):
        for lt in lts:
            d = sens[(sens["policy"] == pol) & (sens["lead_time"] == lt)].sort_values("service_level")
            ax.plot(d["service_level"], d["avg_inventory_value"], color=colors[lt], marker="o", markersize=5,
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.5)
            ax.annotate(f"L = {lt} d", (d["service_level"].iloc[-1], d["avg_inventory_value"].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=PALETTE["ink2"])
        ax.set_title(POLICY_SHORT[pol], loc="left", fontsize=11)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Cycle service target (sets z)")
        ax.yaxis.set_major_formatter(FuncFormatter(_euro))
        ax.margins(x=0.12)
    axes[0].set_ylabel("Average inventory value")
    _fig_title(fig, "Longer lead times and higher targets cost more stock",
               "Average inventory value in the test period, physical stores, z set from the target (not calibrated)")
    return _save(fig, folder, "sensitivity.png")


def fig_channel_comparison(kpis: pd.DataFrame, folder: Path) -> Path:
    apply_style()
    pols = [p for p in reversed(POLICY_ORDER) if p in set(kpis["policy"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.34
    x = np.arange(len(pols))
    for ax, col, title in ((axes[0], "avg_inventory_value", "Average inventory"),
                           (axes[1], "total_cost", "Total cost")):
        for k, ch in enumerate(("physical", "online")):
            d = kpis[kpis["channel"] == ch].set_index("policy").reindex(pols)
            ax.bar(x + (k - 0.5) * (width + 0.02), d[col], width=width, color=CHANNEL_COLORS[ch],
                   label="Physical stores" if ch == "physical" else "Online (80% of unmet demand waits)")
        ax.set_xticks(x, [POLICY_SHORT[p] for p in pols])
        ax.yaxis.set_major_formatter(FuncFormatter(_euro))
        ax.set_title(title, loc="left", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    _fig_title(fig, "Same stores run as physical or online",
               "Each policy calibrated to the same fill rate · online: unmet demand mostly waits instead of being lost")
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.05 / fig.get_figheight()))
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.62 / fig.get_figheight()), ncol=2)
    return _save(fig, folder, "channel_comparison.png")


# --------------------------------------------------------------------------- artefacts
def write_artefact(df: pd.DataFrame, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.parquet"
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    out.to_parquet(path, index=False, compression="zstd")
    return path


def write_json(obj, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


# --------------------------------------------------------------------------- README results block
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def results_markdown(headline: dict, n_series: int) -> str:
    """The README 'Key results' block, generated from the run's headline (never typed by hand)."""
    f = headline["forecast"]
    inv = headline["inventory"]
    fixed = headline.get("inventory_fixed_z", {})
    target = headline["target_fill_rate"]
    t0, t1 = headline["test_period"]
    lines = [
        README_START,
        f"*Test period {t0} → {t1} · {headline['scenario']} ({n_series} store-item series) · "
        "generated by `python -m src.pipeline --all`*",
        "",
        "| Question | Answer |",
        "|---|---|",
        f"| How much better is the best forecast than \"like last week\"? | **{f['best_model_label']}** cuts forecast "
        f"error by **{f['wape_improvement_pct']:.0%}**: WAPE {f['wape_best']:.1%} vs {f['wape_baseline']:.1%} for the "
        f"seasonal-naive baseline (bias {f['bias_best']:+.1%}) |",
        f"| How much less stock does the optimised policy need? | At the same **{target:.0%} fill rate**, the "
        f"forecast-driven policy holds **{_change(inv['inventory_reduction_pct'], 'less', 'more')} inventory** than "
        f"the naive weekly order ({_euro(inv['avg_inventory_value_candidate'])} vs "
        f"{_euro(inv['avg_inventory_value_baseline'])} on average) and "
        f"**{_change(inv['inventory_reduction_vs_classic_pct'], 'less', 'more')}** than a classic static (s, Q) policy |",
        f"| What does it mean in money? | Total cost (ordering + holding + lost margin) over the test period is "
        f"**{_change(inv['total_cost_reduction_pct'], 'lower', 'higher')}** "
        f"({_euro(abs(inv['total_cost_reduction_eur']))}) than with the naive order |",
    ]
    if fixed:
        lines.append(
            f"| And with the textbook z = 1.645? | The (s, Q) policies over-deliver: fill rate "
            f"{fixed['fill_rate_candidate']:.1%} (P2) — a 95% *cycle* service level is not a 95% *fill rate*, "
            "which is why the comparison above calibrates every policy to the same fill rate |")
    if inv["inventory_reduction_pct"] >= 0:
        claim = (f"cuts average inventory by {inv['inventory_reduction_pct']:.0%} at a {target:.0%} fill rate "
                 "versus a naive weekly order")
    else:
        claim = (f"cuts total cost by {inv['total_cost_reduction_pct']:.0%} at a {target:.0%} fill rate "
                 "versus a naive weekly order")
    lines += [
        "",
        f"> 📌 **CV headline:** *\"Forecast-driven reorder policy ({f['best_model_label']} + safety stock + EOQ) that "
        f"{claim}, validated with a day-by-day simulation of {n_series} store-item series; forecast error "
        f"{f['wape_improvement_pct']:.0%} lower than the baseline.\"*",
        README_END,
    ]
    return "\n".join(lines)


def _change(frac: float, down: str, up: str) -> str:
    """'44% less' for a reduction of 0.44, '12% more' for -0.12."""
    return f"{abs(frac):.0%} {down if frac >= 0 else up}"


def update_readme(readme: Path, headline: dict, n_series: int) -> bool:
    """Replace the block between the RESULTS markers. Returns False if the markers are missing."""
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    a = text.index(README_START)
    b = text.index(README_END) + len(README_END)
    readme.write_text(text[:a] + results_markdown(headline, n_series) + text[b:], encoding="utf-8")
    return True
