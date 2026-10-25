"""Shared state for every page: active dataset, scenario and small helpers.

The app works on one *active dataset*: the precomputed demo (Kaggle) or the
results of an uploaded file computed live in this session. Both expose the same
tables (the artefact schemas of docs/06_api_specification.md §3), so pages never
need to know which one is active.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from i18n import channel_name, policy_name, scenario_label, tx  # noqa: E402

from src.config import Config, load_config  # noqa: E402
from src.kpi import aggregate_kpis  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.scenario import Scenario, apply_scenario, build_scenario  # noqa: E402
from src.series import natural_sorted  # noqa: E402

ARTEFACTS = ROOT / "data" / "processed" / "artefacts"
MODEL_LABELS = {"seasonal_naive": "Seasonal naive", "moving_average": "Moving average",
                "holt_winters": "Holt-Winters", "lightgbm": "LightGBM"}


@dataclass
class ActiveData:
    source: str                       # "demo" | "upload"
    label: str
    tables: dict[str, pd.DataFrame]
    meta: dict
    detail_note: str = ""
    extra: dict = field(default_factory=dict)

    def t(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    @property
    def stores(self) -> list[str]:
        return natural_sorted(self.t("stores")["store"])

    @property
    def revenue_ref(self) -> pd.Series:
        return self.t("classification").set_index("item")["revenue"]

    @property
    def default_online(self) -> list[str]:
        s = self.t("stores")
        return s.loc[s["channel"] == "online", "store"].tolist()

    @property
    def target(self) -> float:
        return float(self.meta.get("headline", {}).get("target_fill_rate", 0.95))

    @property
    def best_model(self) -> str:
        return self.meta.get("best_model", "lightgbm")


@st.cache_resource
def config() -> Config:
    return load_config()


@st.cache_data(show_spinner=False)
def _load_demo(folder: str, stamp: float) -> tuple[dict[str, pd.DataFrame], dict]:
    path = Path(folder)
    tables = {p.stem: pd.read_parquet(p) for p in sorted(path.glob("*.parquet"))}
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return tables, meta


def demo_available() -> bool:
    return (ARTEFACTS / "meta.json").exists()


def demo_data() -> ActiveData | None:
    if not demo_available():
        return None
    stamp = (ARTEFACTS / "meta.json").stat().st_mtime
    tables, meta = _load_demo(str(ARTEFACTS), stamp)
    detail = ", ".join(json.loads((ARTEFACTS / "manifest.json").read_text()).get("detail_stores", []))
    return ActiveData("demo", "Demo · Kaggle", tables, meta, extra={"detail_stores": detail})


def data_label(data: ActiveData) -> str:
    if data.source == "demo":
        return tx("Demo · Kaggle store-item sales", "Demo · vendite negozio-articolo Kaggle")
    return tx("Your data · ", "I tuoi dati · ") + data.meta.get("file_name", "")


def detail_note(data: ActiveData) -> str:
    if data.source != "demo":
        d = data.meta.get("detail") or {}
        if d and d.get("n_detail", 0) < d.get("n_series", 0):
            return tx(f"Large dataset: daily detail is kept for the {d['n_detail']:,} top-selling series out of "
                      f"{d['n_series']:,}. Totals and KPIs cover every series.",
                      f"Dataset grande: il dettaglio giornaliero è tenuto per le {d['n_detail']:,} serie più vendute su "
                      f"{d['n_series']:,}. Totali e KPI coprono tutte le serie.")
        return ""
    d = data.extra.get("detail_stores", "")
    return tx(f"Daily detail is published for store {d} only (dataset licence). "
              "Run the project locally to explore every store.",
              f"Il dettaglio giornaliero è pubblicato solo per il negozio {d} (licenza del dataset). "
              "Esegui il progetto in locale per vedere tutti i negozi.")


def active_data() -> ActiveData | None:
    """The upload of this session if the user switched to it, otherwise the demo."""
    up = st.session_state.get("upload_results")
    if st.session_state.get("active_source") == "upload" and up is not None:
        return up
    return demo_data()


# --------------------------------------------------------------------------- scenario
def _set_online(stores: list[str]) -> None:
    st.session_state["sc_online"] = stores


def scenario_sidebar(data: ActiveData) -> Scenario:
    """Stores · products · channel selector shared by every page (UC-11)."""
    stores = data.stores
    items_all = natural_sorted(data.revenue_ref.index)
    key = f"{data.source}:{data.meta.get('file_name', '')}"
    if st.session_state.get("scenario_key") != key:
        st.session_state["scenario_key"] = key
        st.session_state["sc_n_stores"] = len(stores)
        st.session_state["sc_n_items"] = len(items_all)
        st.session_state["sc_rule"] = "top_revenue"
        st.session_state["sc_manual"] = []
        st.session_state["sc_online"] = [s for s in data.default_online if s in stores]

    with st.sidebar:
        st.markdown(tx("### Scenario", "### Scenario"))
        st.caption(data_label(data))
        if len(stores) > 1:
            n_stores = st.slider(tx("Number of stores", "Numero di negozi"), 1, len(stores), key="sc_n_stores")
        else:
            n_stores = 1
        rules = {"top_revenue": tx("Top revenue", "Più venduti"), "random": tx("Random", "Casuali"),
                 "manual": tx("Manual", "A scelta")}
        rule = st.segmented_control(tx("Products", "Prodotti"), list(rules), format_func=rules.get,
                                    key="sc_rule") or "top_revenue"
        manual: list[str] = []
        if rule == "manual":
            manual = st.multiselect(tx("Choose products", "Scegli i prodotti"), items_all, key="sc_manual")
            n_items = len(manual) or len(items_all)
        elif len(items_all) > 1:
            n_items = st.slider(tx("Number of products", "Numero di prodotti"), 1, len(items_all), key="sc_n_items")
        else:
            n_items = 1
        chosen_stores = natural_sorted(stores)[:n_stores]
        current_online = [s for s in st.session_state.get("sc_online", []) if s in chosen_stores]
        st.session_state["sc_online"] = current_online
        st.multiselect(tx("Stores sold online", "Negozi che vendono online"), chosen_stores, key="sc_online",
                       help=tx("Online: 80% of unmet demand waits for the next delivery instead of being lost; "
                               "warehouse holding cost 20% instead of 25%.",
                               "Online: l'80% della domanda non servita aspetta la consegna successiva invece di "
                               "andare persa; costo di magazzino 20% invece del 25%."))
        b1, b2 = st.columns(2)
        b1.button(tx("All physical", "Tutti fisici"), width="stretch", on_click=_set_online, args=([],))
        b2.button(tx("All online", "Tutti online"), width="stretch", on_click=_set_online,
                  args=(list(chosen_stores),))

    rule_key = rule
    try:
        scenario = build_scenario(stores, data.revenue_ref, n_stores=n_stores,
                                  n_items=None if rule_key == "manual" else n_items, item_rule=rule_key,
                                  online_stores=st.session_state.get("sc_online", []),
                                  manual_items=manual if manual else items_all)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        scenario = build_scenario(stores, data.revenue_ref)
    st.session_state["scenario"] = scenario
    return scenario


def scenario_badge(data: ActiveData, scenario: Scenario) -> None:
    h = data.meta.get("headline", {})
    period = h.get("test_period", ["", ""])
    st.caption(f"**{scenario_label(scenario)}** · {tx('test period', 'periodo di test')} {period[0]} → {period[1]}"
               f" · {data_label(data)}" + (tx(" · fast mode", " · modalità veloce") if data.meta.get("fast") else ""))


# --------------------------------------------------------------------------- tables for a scenario
def scenario_summary(data: ActiveData, scenario: Scenario) -> pd.DataFrame:
    return apply_scenario(data.t("policy_summary"), scenario)


def policy_kpis(summary: pd.DataFrame, by=("policy",), variant: str = "calibrated") -> pd.DataFrame:
    sub = summary[summary["variant"] == variant]
    if sub.empty:
        return pd.DataFrame()
    return aggregate_kpis(sub, list(by))


def scenario_leaderboard(data: ActiveData, scenario: Scenario, by=("model",), period: str = "test") -> pd.DataFrame:
    e = data.t("forecast_errors")
    e = e[e["period"] == period]
    e = apply_scenario(e, scenario)
    if "abc" in by:
        e = e.merge(data.t("classification")[["item", "abc"]], on="item", how="left")
    return evaluate(e, list(by)) if len(e) else pd.DataFrame()


# --------------------------------------------------------------------------- formatting
def euro(x: float) -> str:
    x = float(x)
    if abs(x) >= 1e6:
        return f"€{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"€{x / 1e3:,.1f}K"
    return f"€{x:,.0f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{float(x):.{digits}%}"


def policy_label(p: str) -> str:
    return policy_name(p)


def channel_label(c: str) -> str:
    return channel_name(c)


def no_data_message() -> None:
    st.warning(tx(
        "No results to show yet. Open the **Data** page and use a sample dataset or upload your own sales history, "
        "or build the demo results locally with `python -m src.pipeline --all` (see the README).",
        "Ancora nessun risultato. Apri la pagina **Dati** e usa un dataset di esempio o carica il tuo storico "
        "vendite, oppure crea i risultati demo in locale con `python -m src.pipeline --all` (vedi il README)."))


def plan_calendar_for(data: ActiveData):
    """Calendar shown on the Orders and Promotions pages (today and the shift that brings the data to today)."""
    from datetime import date

    from src.replenishment import plan_calendar

    meta = data.meta.get("plan") or {}
    end = meta.get("data_end") or (data.meta.get("promotions") or {}).get("last_date")
    if not end:
        return None
    today = st.session_state.get("plan_today") or date.today()
    return plan_calendar(end, today, st.session_state.get("plan_bring", True))


def csv_download(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv",
                       key=key, icon=":material/download:")
