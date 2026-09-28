"""Scenario = which stores and products to analyse, and which stores sell online.

The heavy work (forecasts, calibrated simulations) is done once for every series
under both channels. A scenario only **filters** those rows — keeping, for each
store, the rows of its channel — and **sums** them (ADR-0005). The KPIs of a
scenario are therefore exactly those of simulating it from scratch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.series import natural_sorted


@dataclass(frozen=True)
class Scenario:
    stores: tuple[str, ...]
    items: tuple[str, ...]
    online_stores: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if not self.stores:
            raise ValueError("Select at least one store.")
        if not self.items:
            raise ValueError("Select at least one product.")
        extra = set(self.online_stores) - set(self.stores)
        if extra:
            raise ValueError(f"Online stores must be selected stores: {sorted(extra)}")

    def channel_of(self, store: str) -> Literal["physical", "online"]:
        return "online" if str(store) in self.online_stores else "physical"

    @property
    def label(self) -> str:
        s, i, o = len(self.stores), len(self.items), len(self.online_stores)
        text = f"{s} store{'s' if s != 1 else ''} · {i} product{'s' if i != 1 else ''}"
        return text + (f" · {o} online" if o else "")

    def to_dict(self) -> dict:
        return {"stores": list(self.stores), "items": list(self.items),
                "online_stores": natural_sorted(self.online_stores)}


def select_items(n_items: int | None, rule: str, revenue_ref: pd.Series, seed: int = 42,
                 manual: Sequence[str] = ()) -> tuple[str, ...]:
    """Top-N by reference-period revenue (default), seeded random N, or a manual list."""
    all_items = natural_sorted(revenue_ref.index)
    n = len(all_items) if n_items is None else int(n_items)
    if not 1 <= n <= len(all_items) and rule != "manual":
        raise ValueError(f"Number of products must be between 1 and {len(all_items)}")
    if rule == "manual":
        chosen = [str(x) for x in manual]
        unknown = set(chosen) - set(all_items)
        if unknown:
            raise ValueError(f"Unknown item(s): {sorted(unknown)}")
        if not chosen:
            raise ValueError("The manual product list is empty.")
        return tuple(natural_sorted(set(chosen)))
    if rule == "random":
        rng = np.random.default_rng(seed)
        return tuple(natural_sorted(rng.choice(all_items, size=n, replace=False)))
    if rule == "top_revenue":
        ranked = revenue_ref.rename("r").reset_index()
        ranked.columns = ["item", "r"]
        ranked["item"] = ranked["item"].astype(str)
        ranked["_k"] = ranked["item"].map({it: k for k, it in enumerate(all_items)})
        ranked = ranked.sort_values(["r", "_k"], ascending=[False, True])
        return tuple(natural_sorted(ranked["item"].head(n)))
    raise ValueError(f"Unknown item selection rule {rule!r}")


def select_stores(all_stores: Sequence[str], n_stores: int | None = None,
                  ids: Sequence[str] | None = None) -> tuple[str, ...]:
    """First N store IDs (natural order) or the given IDs."""
    ordered = natural_sorted(all_stores)
    if ids:
        unknown = {str(x) for x in ids} - set(ordered)
        if unknown:
            raise ValueError(f"Unknown store(s): {sorted(unknown)}")
        return tuple(natural_sorted({str(x) for x in ids}))
    n = len(ordered) if n_stores is None else int(n_stores)
    if not 1 <= n <= len(ordered):
        raise ValueError(f"Number of stores must be between 1 and {len(ordered)}")
    return tuple(ordered[:n])


def build_scenario(all_stores: Sequence[str], revenue_ref: pd.Series, n_stores: int | None = None,
                   n_items: int | None = None, item_rule: str = "top_revenue",
                   online_stores: Sequence[str] = (), store_ids: Sequence[str] | None = None,
                   seed: int = 42, manual_items: Sequence[str] = ()) -> Scenario:
    stores = select_stores(all_stores, n_stores, store_ids)
    items = select_items(n_items, item_rule, revenue_ref, seed, manual_items)
    online = frozenset(str(s) for s in online_stores if str(s) in stores)
    return Scenario(stores, items, online)


def scenario_from_config(cfg, all_stores: Sequence[str], revenue_ref: pd.Series,
                         default_online: Sequence[str] = ()) -> Scenario:
    sc = cfg.scenario
    online = sc.online_stores or tuple(default_online)
    return build_scenario(all_stores, revenue_ref, sc.n_stores, sc.n_items, sc.item_selection, online,
                          seed=sc.random_seed, manual_items=sc.manual_items)


def apply_scenario(frame: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Rows of the selected (store, item) pairs; for each store only its own channel.

    Tables without a ``channel`` column (e.g. forecast errors) are only filtered.
    """
    mask = frame["store"].isin(list(scenario.stores)) & frame["item"].isin(list(scenario.items))
    if "channel" in frame.columns:
        wanted = np.where(frame["store"].isin(list(scenario.online_stores)), "online", "physical")
        mask &= frame["channel"].to_numpy() == wanted
    return frame[mask]
