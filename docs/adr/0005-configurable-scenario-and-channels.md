# ADR-0005 — Configurable scenario (stores, products, channel) via precomputed results

**Status:** Accepted, extended by [ADR-0006](0006-user-uploaded-sales-history.md) (the scenario also applies to uploaded data, computed live) · **Date:** 2026-09-28

## Context
The user wants to set the **number of stores**, the **number of products**, and whether each store is **online or physical**, and see the results change. Re-running the calibrated simulation for a scenario in the dashboard would take tens of seconds (bisection × hundreds of series) and could break Streamlit Cloud's limits. The dataset caps the scenario at 10 stores × 50 products.

## Decision
1. **Precompute every series under both channels**: 500 series × {physical, online} × {P0, P1, P2} = 3,000 summary rows, each holding **additive sums** (demand, sales, lost, back-ordered, inventory unit-days, costs, n_orders).
2. A scenario is a pure **filter + sum**: keep the selected (store, item) pairs, take the row of each store's channel, then aggregate. Ratios (fill rate, turns, …) are computed *after* summing.
3. Scenario limits are the limits of the active dataset (demo: 10 stores, 50 products; uploaded file: whatever it contains). Product selection defaults to **top-N by revenue in the reference period**.
4. "Online" changes only the operating rules (partial backorders, backorder cost, holding rate) and **keeps the store's demand history**.

## Alternatives considered
- *Re-simulate live for each scenario* — exact but slow; calibration makes it much worse.
- *Synthetic generator for any size* — flexible, but the results would no longer rest on real data; kept as future work (FR-44).
- *Model online demand separately* — the dataset has no online data, so it would be pure invention.

## Consequences
- ➕ A scenario change updates the dashboard in under 2 s, and the result is exact (series are independent).
- ➕ One simple artefact contract: a `channel` column is added to the summary tables.
- ➖ Twice the offline simulation work (still a few minutes).
- ➖ Cross-store effects such as pooling online stock cannot be precomputed per series → they are handled by a separate live computation (FR-43, Could).
