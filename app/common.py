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

from src.config import Config, load_config  # noqa: E402
from src.kpi import aggregate_kpis  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.policy import POLICY_LABELS  # noqa: E402
from src.scenario import Scenario, apply_scenario, build_scenario  # noqa: E402
from src.series import natural_sorted  # noqa: E402

ARTEFACTS = ROOT / "data" / "processed" / "artefacts"
POLICY_SHORT = {"P0_naive": "P0 · naive weekly", "P1_static_sq": "P1 · static (s, Q)",
                "P2_forecast_sq": "P2 · forecast-driven"}
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


@st.cache_data(show_spinner="Loading the demo results…")
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
    return ActiveData("demo", "Demo · Kaggle store-item sales", tables, meta,
                      detail_note=f"Daily detail is published for store {detail} only (dataset licence). "
                                  "Run the project locally to explore every store.")


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
        st.session_state["sc_rule"] = "Top revenue"
        st.session_state["sc_manual"] = []
        st.session_state["sc_online"] = [s for s in data.default_online if s in stores]

    with st.sidebar:
        st.markdown("### Scenario")
        st.caption(data.label)
        if len(stores) > 1:
            n_stores = st.slider("Number of stores", 1, len(stores), key="sc_n_stores")
        else:
            n_stores = 1
        rule = st.segmented_control("Products", ["Top revenue", "Random", "Manual"], key="sc_rule") or "Top revenue"
        manual: list[str] = []
        if rule == "Manual":
            manual = st.multiselect("Choose products", items_all, key="sc_manual")
            n_items = len(manual) or len(items_all)
        elif len(items_all) > 1:
            n_items = st.slider("Number of products", 1, len(items_all), key="sc_n_items")
        else:
            n_items = 1
        chosen_stores = natural_sorted(stores)[:n_stores]
        current_online = [s for s in st.session_state.get("sc_online", []) if s in chosen_stores]
        st.session_state["sc_online"] = current_online
        st.multiselect("Stores sold online", chosen_stores, key="sc_online",
                       help="Online: 80% of unmet demand waits for the next delivery instead of being lost; "
                            "warehouse holding cost 20% instead of 25%.")
        b1, b2 = st.columns(2)
        b1.button("All physical", width="stretch", on_click=_set_online, args=([],))
        b2.button("All online", width="stretch", on_click=_set_online, args=(list(chosen_stores),))

    rule_key = {"Top revenue": "top_revenue", "Random": "random", "Manual": "manual"}[rule]
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
    st.caption(f"**{scenario.label}** · test period {period[0]} → {period[1]} · {data.label}"
               + (" · fast mode" if data.meta.get("fast") else ""))


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
    return POLICY_SHORT.get(p, POLICY_LABELS.get(p, p))


def no_data_message() -> None:
    st.warning(
        "No results to show yet. Either upload your own sales history on the **Data** page, "
        "or build the demo results locally with `python -m src.pipeline --all` (see the README).")


def csv_download(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv",
                       key=key, icon=":material/download:")
