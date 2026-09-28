# 01 — Project Charter

| Field | Value |
|---|---|
| Project | Demand Forecasting & Inventory Optimization |
| Owner | Luca Pillitu |
| Start / target end | 2026-09-28 → 2026-10-18 (3 weeks) |
| Type | Portfolio project (data science / supply chain analytics) |
| Status | Design phase |

## 1. Background

A distributor runs **10 stores** that sell the same **50 items**. Today replenishment follows a simple habit: *order roughly the same quantity every week*. This rule ignores trend, seasonality and demand uncertainty. The result is too much stock for some items and repeated stock-outs for others.

## 2. Problem statement

> Decide **how much** and **when** to reorder each item in each store so that **total inventory cost** (ordering + holding + lost sales) is **as low as possible** while the **service level stays at or above 95%**.

## 3. Objectives (SMART)

| ID | Objective | Measure | Target |
|---|---|---|---|
| O1 | Forecast daily demand per store-item better than a naive baseline | WAPE on the 2017 back-test vs. seasonal naive | ≥ 15% relative reduction *(aim; the real number gets reported whatever it is)* |
| O2 | Keep forecasts unbiased | Bias = Σ(F − A) / ΣA | within ±2% |
| O3 | Design a reorder policy that meets the service target | Simulated fill rate over 2017 | ≥ 95% |
| O4 | Lower the stock needed to reach that service | Average on-hand inventory vs. naive policy **at equal fill rate** | reduction reported as `X%` |
| O5 | Make the results easy to explore | Public Streamlit dashboard | Live URL in the README |
| O6 | Show the SQL side | KPIs computed with SQL in DuckDB | ≥ 8 documented queries |
| O7 | Let the user define the business scenario | Number of stores, number of products, channel (physical/online) per store — selectable in the dashboard | KPIs update in < 2 s |
| O8 | Work on any sales history, not just the demo | Upload a CSV/Excel file → same analysis, live | Runs in < 60 s for 200 series × 3 years (fast mode) |

## 4. Scope

### In scope
- **Demo data:** Kaggle *Store Item Demand Forecasting Challenge* `train.csv` (2013–2017), precomputed; it produces the README/CV numbers.
- **User data:** upload of any daily sales history (CSV / Excel / Parquet) with column mapping, validation and adaptive periods (history ≥ 8 weeks).
- EDA: trend, weekly and yearly seasonality, ABC (revenue) and XYZ (variability) classification.
- 4 forecasting models: seasonal naive, moving average, Holt-Winters, LightGBM.
- Continuous-review `(s, Q)` policy per store-item: safety stock, reorder point, EOQ.
- Day-by-day simulation of the test period (2017 in the demo) comparing ≥ 2 policies.
- KPIs: fill rate, stock-out events/days, average inventory, inventory turns, total cost.
- Streamlit dashboard on Streamlit Community Cloud.
- **Configurable scenario:** any number of stores and products available in the data, each store set as *physical* (lost sales) or *online* (partial backorders, different costs).
- Documentation (this folder) and a results-first README.

### Out of scope
- Storing uploaded user data on a server, user accounts, multi-user workspaces.
- Connectors to ERP / POS systems (the input is a file export).
- Real prices, costs or supplier data (they are **assumed** and documented).
- Multi-echelon inventory (central warehouse → stores) and transfers between stores.
- Supplier capacity limits, minimum order quantities, quantity discounts.
- Promotions, prices and external variables (weather, holidays other than calendar effects).
- A production MLOps setup (scheduling, monitoring, retraining pipelines).

### Optional (if time allows)
- Power BI version of the dashboard (**needs Windows**; the owner works on macOS → see [R-07](11_risk_register.md)).
- Fill-rate-based (Type 2) safety stock, periodic-review `(R, S)` policy, stochastic lead time.
- Risk pooling: online stores served from one shared warehouse.
- Weekly/monthly input data; intermittent-demand models (Croston/SBA).

## 5. Deliverables

| ID | Deliverable | Location |
|---|---|---|
| D1 | Results-first README with charts | `README.md` |
| D2 | SQL KPI queries | `sql/kpi_queries.sql` |
| D3 | Three clean, commented notebooks | `notebooks/` |
| D4 | Reusable Python package | `src/` |
| D5 | Tests | `tests/` |
| D6 | Streamlit dashboard (public URL) | `app/` |
| D7 | Project documentation | `docs/` |

## 6. Stakeholders

| Stakeholder | Role in this (simulated) business case | Interest |
|---|---|---|
| Supply Chain Manager | Sponsor; owns the service-level target and inventory budget | Lower working capital, fewer stock-outs |
| Inventory Planner | Main user of the reorder policy and the dashboard | Clear order quantities and reorder points per item |
| Store Manager | Receives the stock | Availability on the shelf |
| Finance / Controller | Watches inventory value | Capital tied up, holding cost |
| **Recruiter / Hiring Manager** | *Real* audience of the portfolio | Clear business impact in the README, in under 2 minutes of reading |

## 7. Success criteria (Definition of Done for the project)

1. The README shows the headline sentence with **real numbers** and at least 4 charts.
2. `python -m src.pipeline --all` rebuilds every result from the raw CSV with no manual steps.
3. The dashboard is public and loads in under 10 s.
4. All tests pass; the notebooks run top-to-bottom with no errors.
5. Every assumption is listed in the README and in `config/params.yaml`.

## 8. Constraints & assumptions

- **Time:** 3 weeks, about 15–20 h/week.
- **Budget:** €0 (free tiers only).
- **Hardware:** a laptop (macOS). The full LightGBM back-test must finish in under 15 min.
- **Hosting:** Streamlit Community Cloud (≈1 GB RAM) → the app reads **precomputed artefacts**; no model training in the app ([ADR-0003](adr/0003-precomputed-artefacts-for-dashboard.md)).
- **Data:** the raw data is not committed to git (Kaggle terms, repo size).
