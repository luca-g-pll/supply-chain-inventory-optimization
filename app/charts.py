"""Plotly figures for the dashboard, with light and dark palettes.

Same colour roles as the README figures (src/reporting.py): policies keep one
colour everywhere (P2 blue, P1 orange, P0 aqua), single-measure comparisons use
one accent + gray, gridlines are solid hairlines, and every chart has a hover
tooltip plus a table view elsewhere on the page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from i18n import model_name, policy_name, tx

LIGHT = {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
         "context": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948",
         "critical": "#d03b3b", "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
         "ordinal": ["#86b6ef", "#2a78d6", "#104281"]}
DARK = {"ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "context": "#5a5955", "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "red": "#e66767",
        "critical": "#d03b3b", "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "ordinal": ["#184f95", "#3987e5", "#9ec5f4"]}
POLICY_ORDER = ["P2_forecast_sq", "P1_static_sq", "P0_naive"]


class _Names:
    """Policy / model names in the current language (dict-like)."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, key):
        return self.fn(key)

    def get(self, key, default=None):
        return self.fn(key)


POLICY_NAMES = _Names(lambda p: policy_name(p, short=True))
MODEL_NAMES = _Names(model_name)


def pal() -> dict:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return DARK if theme == "dark" else LIGHT


def policy_color(p: str) -> str:
    c = pal()
    return {"P2_forecast_sq": c["blue"], "P1_static_sq": c["orange"], "P0_naive": c["aqua"]}[p]


def layout(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True,
           yfmt: str | None = None, xfmt: str | None = None) -> go.Figure:
    c = pal()
    top = (40 if title else 8) + (30 if legend else 0)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", yref="container", y=1 - 14 / height, yanchor="top",
                   font=dict(size=15, color=c["ink"])) if title else None,
        height=height, margin=dict(l=8, r=8, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=c["ink2"]),
        hoverlabel=dict(font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        bargap=0.35,
    )
    horizontal = any(getattr(tr, "orientation", None) == "h" for tr in fig.data)
    fig.update_xaxes(showgrid=horizontal, gridcolor=c["grid"], gridwidth=1, linecolor=c["axis"], linewidth=1,
                     showline=not horizontal, ticks="", zeroline=False, tickfont=dict(color=c["ink2"]),
                     automargin=True, **({"tickformat": xfmt} if xfmt else {}))
    fig.update_yaxes(showgrid=not horizontal, gridcolor=c["grid"], gridwidth=1, zeroline=False, showline=False,
                     tickfont=dict(color=c["ink2"]), automargin=True, **({"tickformat": yfmt} if yfmt else {}))
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})


# --------------------------------------------------------------------------- inventory
def inventory_at_target(kpis: pd.DataFrame, target: float) -> go.Figure:
    c = pal()
    k = kpis.set_index("policy").reindex([p for p in reversed(POLICY_ORDER) if p in set(kpis["policy"])])
    colors = [c["blue"] if p == "P2_forecast_sq" else c["context"] for p in k.index]
    base = k["avg_inventory_value"].get("P0_naive", np.nan)
    text = []
    for p, r in k.iterrows():
        t = f"{_money(r['avg_inventory_value'])} · {r['fill_rate']:.1%}"
        if p != "P0_naive" and np.isfinite(base) and base > 0:
            t += f" · {r['avg_inventory_value'] / base - 1:+.0%} " + tx("vs P0", "vs P0")
        text.append(t)
    fig = go.Figure(go.Bar(
        y=[POLICY_NAMES[p] for p in k.index], x=k["avg_inventory_value"], orientation="h", marker_color=colors,
        width=0.5, text=text, textposition="outside", cliponaxis=False, textfont=dict(color=c["ink2"]),
        hovertemplate="<b>€%{x:,.0f}</b><br>%{y}<extra></extra>"))
    fig.update_xaxes(range=[0, float(k["avg_inventory_value"].max()) * 1.75], tickprefix="€")
    return layout(fig, tx(f"Average inventory needed for a {target:.0%} fill rate",
                          f"Scorta media necessaria per un fill rate del {target:.0%}"), height=260, legend=False)


def cost_breakdown(kpis: pd.DataFrame) -> go.Figure:
    c = pal()
    k = kpis.set_index("policy").reindex([p for p in reversed(POLICY_ORDER) if p in set(kpis["policy"])])
    fig = go.Figure()
    for comp, name, color in (("ordering_cost", tx("Ordering", "Ordini"), c["blue"]),
                              ("holding_cost", tx("Holding", "Mantenimento"), c["orange"]),
                              ("shortage_cost", tx("Shortage (lost margin + backorders)",
                                                   "Rotture (margine perso + attese)"), c["aqua"])):
        fig.add_bar(y=[POLICY_NAMES[p] for p in k.index], x=k[comp], name=name, orientation="h", marker_color=color,
                    marker_line=dict(width=2, color="rgba(0,0,0,0)"), width=0.5,
                    hovertemplate=f"<b>€%{{x:,.0f}}</b><br>{name}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickprefix="€")
    layout(fig, tx("Total cost by component", "Costo totale per componente"), height=300)
    # three long legend entries: below the plot, so they never cover the title on narrow columns
    fig.update_layout(legend=dict(yanchor="top", y=-0.16), margin=dict(t=40, b=70))
    return fig


def tradeoff(curves: pd.DataFrame, calibrated: pd.DataFrame, target: float, metric: str = "avg_inventory_value",
             title: str | None = None) -> go.Figure:
    c = pal()
    fig = go.Figure()
    cal = calibrated.set_index("policy") if len(calibrated) else pd.DataFrame()
    title = title or tx("Inventory vs. fill rate for each policy", "Scorta vs. fill rate per ogni politica")
    y_cap = float(cal[metric].max()) * 1.7 if len(cal) else float(curves[metric].max())
    x_min = max(0.8, float(curves["fill_rate"].min()) - 0.005)
    for p in POLICY_ORDER:
        d = curves[curves["policy"] == p].sort_values("fill_rate")
        if d.empty:
            continue
        fig.add_scatter(x=d["fill_rate"], y=d[metric], mode="lines+markers", name=POLICY_NAMES[p],
                        line=dict(color=policy_color(p), width=2),
                        marker=dict(size=7, color=policy_color(p), line=dict(width=2, color="rgba(0,0,0,0)")),
                        customdata=d["knob"], hovertemplate="<b>€%{y:,.0f}</b> " + tx("at fill rate", "con fill rate")
                        + " %{x:.1%}<br>" + tx("knob", "parametro") + " %{customdata:.2f}<extra>"
                        + POLICY_NAMES[p] + "</extra>")
        if p in cal.index:
            fig.add_scatter(x=[cal.at[p, "fill_rate"]], y=[cal.at[p, metric]], mode="markers", showlegend=False,
                            marker=dict(size=14, color=policy_color(p), line=dict(width=2, color=c["ink2"])),
                            hovertemplate="<b>€%{y:,.0f}</b> " + tx("at", "al") + " %{x:.1%}<extra>" + POLICY_NAMES[p]
                            + tx(" · calibrated", " · calibrata") + "</extra>")
    fig.add_vline(x=target, line=dict(color=c["axis"], width=1))
    fig.update_xaxes(range=[x_min, 1.003], tickformat=".0%")
    fig.update_yaxes(range=[0, y_cap], tickprefix="€")
    return layout(fig, title, height=380)


def trajectory(daily: pd.DataFrame) -> go.Figure:
    c = pal()
    fig = go.Figure()
    for p in POLICY_ORDER:
        d = daily[daily["policy"] == p]
        if d.empty:
            continue
        fig.add_scatter(x=d["date"], y=d["on_hand_end"], mode="lines", name=POLICY_NAMES[p],
                        line=dict(color=policy_color(p), width=1.6),
                        hovertemplate="%{y:,.0f} " + tx("units", "pezzi") + "<extra>" + POLICY_NAMES[p] + "</extra>")
    so = daily[(daily["lost"] + daily["backordered"]) > 0]
    if len(so):
        fig.add_scatter(x=so["date"], y=np.zeros(len(so)), mode="markers", name=tx("Days with unmet demand", "Giorni con domanda non servita"),
                        marker=dict(symbol="line-ns-open", size=10, color=c["critical"], line=dict(width=2)),
                        customdata=so["policy"].map(POLICY_NAMES),
                        hovertemplate="%{x|%d %b %Y}<br>%{customdata}<extra>" + tx("unmet demand", "domanda non servita") + "</extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    return layout(fig, tx("Units on hand, day by day", "Pezzi a magazzino, giorno per giorno"), height=380)


def sensitivity(sens: pd.DataFrame) -> go.Figure:
    c = pal()
    fig = go.Figure()
    lts = sorted(sens["lead_time"].unique())
    shades = dict(zip(lts, c["ordinal"][-len(lts):] if len(lts) <= 3 else c["seq"][2:]))
    for p in ("P2_forecast_sq", "P1_static_sq"):
        for lt in lts:
            d = sens[(sens["policy"] == p) & (sens["lead_time"] == lt)].sort_values("service_level")
            fig.add_scatter(x=d["service_level"], y=d["avg_inventory_value"], mode="lines+markers",
                            name=f"{POLICY_NAMES[p]} · L = {lt} " + tx("d", "gg"), legendgroup=p,
                            line=dict(color=shades[lt], width=2 if p == "P2_forecast_sq" else 1.2),
                            marker=dict(size=7, symbol="circle" if p == "P2_forecast_sq" else "diamond"),
                            hovertemplate="<b>€%{y:,.0f}</b> " + tx("at target", "con obiettivo") + " %{x:.0%}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickprefix="€")
    return layout(fig, tx("Sensitivity: lead time × service target (z not calibrated)",
                          "Sensibilità: tempo di consegna × livello di servizio (z non calibrato)"), height=380)


# --------------------------------------------------------------------------- forecasting
def leaderboard_bars(lb: pd.DataFrame, best: str) -> go.Figure:
    c = pal()
    lb = lb.sort_values("wape", ascending=True)
    colors = [c["blue"] if m == best else c["context"] for m in lb["model"]]
    fig = go.Figure(go.Bar(
        y=[MODEL_NAMES.get(m, m) for m in lb["model"]], x=lb["wape"], orientation="h", marker_color=colors, width=0.5,
        text=[f"{w:.1%}" for w in lb["wape"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=c["ink2"]), hovertemplate="<b>%{x:.2%}</b> WAPE<br>%{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=[0, float(lb["wape"].max()) * 1.2])
    return layout(fig, tx("Forecast error (WAPE, lower is better)", "Errore di previsione (WAPE, più basso è meglio)"),
                  height=260, legend=False)


def wape_by(frame: pd.DataFrame, col: str, best: str, baseline: str, title: str, xtitle: str) -> go.Figure:
    c = pal()
    fig = go.Figure()
    for m in frame["model"].unique():
        d = frame[frame["model"] == m].sort_values(col)
        color = c["blue"] if m == best else (c["muted"] if m == baseline else c["context"])
        fig.add_scatter(x=d[col], y=d["wape"], mode="lines+markers", name=MODEL_NAMES.get(m, m),
                        line=dict(color=color, width=2 if m in (best, baseline) else 1.5), marker=dict(size=6),
                        hovertemplate="<b>%{y:.1%}</b><extra>%{fullData.name}</extra>")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(title=xtitle)
    fig.update_layout(hovermode="x unified")
    return layout(fig, title, height=330)


def forecast_series(view: pd.DataFrame, models: list[str], best: str) -> go.Figure:
    c = pal()
    fig = go.Figure()
    first = view[view["model"] == models[0]].sort_values("date")
    fig.add_scatter(x=first["date"], y=first["actual"], mode="lines", name=tx("Actual", "Venduto reale"),
                    line=dict(color=c["ink2"], width=1.2), hovertemplate="%{y:,.0f}<extra>" + tx("Actual", "Venduto reale") + "</extra>")
    others = [c["orange"], c["aqua"], c["muted"]]
    k = 0
    for m in models:
        d = view[view["model"] == m].sort_values("date")
        if m == best:
            color, width = c["blue"], 2.2
        else:
            color, width = others[k % len(others)], 1.5
            k += 1
        fig.add_scatter(x=d["date"], y=d["forecast"], mode="lines", name=MODEL_NAMES.get(m, m),
                        line=dict(color=color, width=width),
                        hovertemplate="%{y:,.1f}<extra>" + MODEL_NAMES.get(m, m) + "</extra>")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero", title=tx("Units per day", "Pezzi al giorno"))
    return layout(fig, None, height=360)


def importance(imp: pd.DataFrame) -> go.Figure:
    c = pal()
    d = imp.assign(share=imp["gain"] / imp["gain"].sum()).sort_values("share").tail(12)
    fig = go.Figure(go.Bar(y=d["feature"], x=d["share"], orientation="h", marker_color=c["blue"], width=0.5,
                           hovertemplate="<b>%{x:.1%}</b> " + tx("of total gain", "del guadagno totale") + "<extra>%{y}</extra>"))
    fig.update_xaxes(tickformat=".0%")
    return layout(fig, tx("What LightGBM relies on (share of gain)", "Su cosa si basa LightGBM (quota di guadagno)"),
                  height=360, legend=False)


# --------------------------------------------------------------------------- demand
def monthly_trend(monthly: pd.DataFrame) -> go.Figure:
    c = pal()
    m = monthly.groupby("month", as_index=False).agg(units=("units", "sum"), n_days=("n_days", "max"))
    m["per_day"] = m["units"] / m["n_days"]
    fig = go.Figure(go.Scatter(x=m["month"], y=m["per_day"], mode="lines+markers", line=dict(color=c["blue"], width=2),
                               marker=dict(size=6), fill="tozeroy", fillcolor=_alpha(c["blue"], 0.10),
                               hovertemplate="<b>%{y:,.0f}</b> " + tx("units per day", "pezzi al giorno") + "<br>%{x|%b %Y}<extra></extra>"))
    fig.update_yaxes(rangemode="tozero")
    return layout(fig, tx("Units sold per day, by month", "Pezzi venduti al giorno, per mese"), height=330, legend=False)


def index_bars(labels, index_values, title: str) -> go.Figure:
    c = pal()
    vals = np.asarray(index_values, dtype=float) - 1
    fig = go.Figure(go.Bar(x=list(labels), y=vals, base=1.0, width=0.55,
                           marker_color=[c["blue"] if v >= 0 else c["red"] for v in vals],
                           customdata=np.asarray(index_values, dtype=float),
                           hovertemplate="<b>%{customdata:.2f}</b><br>%{x}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=c["axis"], width=1))
    return layout(fig, title, height=300, legend=False)


def pareto(classification: pd.DataFrame) -> go.Figure:
    c = pal()
    d = classification.sort_values("rank").reset_index(drop=True)
    d["pos"] = np.arange(1, len(d) + 1)
    d["cum"] = d["revenue"].cumsum() / d["revenue"].sum()
    fig = go.Figure()
    fig.add_scatter(x=np.r_[0, d["pos"]], y=np.r_[0, d["cum"]], mode="lines", line=dict(color=c["axis"], width=1.5),
                    showlegend=False, hoverinfo="skip")
    for cls, color in zip("ABC", (c["ordinal"][2], c["ordinal"][1], c["ordinal"][0])):
        s = d[d["abc"] == cls]
        fig.add_scatter(x=s["pos"], y=s["cum"], mode="markers", name=f"{cls} · {len(s)} " + tx("items", "articoli"),
                        marker=dict(size=9, color=color, line=dict(width=2, color="rgba(0,0,0,0)")),
                        customdata=s[["item", "revenue_share"]],
                        hovertemplate=tx("Item", "Articolo") + " <b>%{customdata[0]}</b><br>%{customdata[1]:.1%} "
                                      + tx("of revenue", "del fatturato") + "<br>" + tx("cumulative", "cumulato")
                                      + " %{y:.1%}<extra>" + cls + "</extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.03])
    fig.update_xaxes(title=tx("Items ranked by revenue", "Articoli ordinati per fatturato"))
    return layout(fig, tx("ABC classification (Pareto curve)", "Classificazione ABC (curva di Pareto)"), height=340)


def abc_xyz_heatmap(matrix: pd.DataFrame) -> go.Figure:
    c = pal()
    share = matrix.pivot(index="abc", columns="xyz", values="revenue_share").reindex(index=list("ABC"), columns=list("XYZ"))
    n = matrix.pivot(index="abc", columns="xyz", values="items").reindex(index=list("ABC"), columns=list("XYZ"))
    text = [[f"{int(n.iat[i, j])} " + tx("items", "articoli") + f"<br>{share.iat[i, j]:.0%}" for j in range(3)]
            for i in range(3)]
    fig = go.Figure(go.Heatmap(z=share.to_numpy(), x=[tx("X · stable", "X · stabile"), tx("Y · variable", "Y · variabile"), tx("Z · erratic", "Z · irregolare")], y=["A", "B", "C"],
                               colorscale=[[k / (len(c["seq"]) - 1), col] for k, col in enumerate(LIGHT["seq"])],
                               text=text, texttemplate="%{text}", showscale=False, xgap=2, ygap=2,
                               hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return layout(fig, tx("ABC × XYZ matrix (share of revenue)", "Matrice ABC × XYZ (quota di fatturato)"), height=320,
                  legend=False)


def _money(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"€{x / 1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:.1f}K"
    return f"€{x:.0f}"


def _alpha(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i: i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"
