# 02 — Requirements

Priority uses **MoSCoW**: **M**ust, **S**hould, **C**ould, **W**on't (this time).

## 1. Functional requirements

### Data & SQL
| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Load `train.csv` into a DuckDB database file (`data/processed/inventory.duckdb`), table `sales`. | M |
| FR-02 | Validate the raw data: no missing dates per series, no duplicate keys, no negative sales; write a validation report. | M |
| FR-03 | Generate a seeded synthetic price table (`item_prices`) from `config/params.yaml`. | M |
| FR-04 | Compute the descriptive KPIs with SQL (`sql/kpi_queries.sql`): yearly/monthly volume, YoY growth, day-of-week index, revenue per item/store, top/bottom items, CV per series. | M |

### Exploration & classification
| ID | Requirement | Priority |
|---|---|---|
| FR-05 | Break down demand into trend, weekly and yearly seasonality (charts + indices). | M |
| FR-06 | ABC classification of items by cumulative revenue share (A ≤ 80%, B ≤ 95%, C rest). | M |
| FR-07 | XYZ classification by coefficient of variation of **weekly** demand (X < 0.5, 0.5 ≤ Y < 1.0, Z ≥ 1.0; thresholds configurable). | M |
| FR-08 | ABC × XYZ 9-cell matrix with counts and revenue share. | M |

### Forecasting
| ID | Requirement | Priority |
|---|---|---|
| FR-09 | Seasonal-naive model: `ŷ(t) = y(t − 7)` (same weekday last week). | M |
| FR-10 | Moving-average model with a configurable window (default 28 days). | M |
| FR-11 | Holt-Winters (additive trend, damped, weekly seasonality) per series, `statsmodels`. | M |
| FR-12 | One global LightGBM model across all 500 series, with calendar, ID and lagged features (lags ≥ forecast horizon). | M |
| FR-13 | Rolling-origin back-test with a common interface for all models; horizon 1–14 days; weekly origins. | M |
| FR-14 | Error metrics: WAPE, MAPE (on non-zero actuals), bias, RMSE — overall, per ABC class, per horizon. | M |
| FR-15 | Report the relative improvement of the best model vs. seasonal naive. | M |
| FR-16 | Quantile / probabilistic forecasts. | C |

### Inventory policy & simulation
| ID | Requirement | Priority |
|---|---|---|
| FR-17 | Compute, per series: mean daily demand, σ of demand (or forecast error) over the lead time, safety stock, reorder point, EOQ. | M |
| FR-18 | Simulate the test period day by day for each series and policy, with lead time, lost sales and on-order tracking. | M |
| FR-19 | Policies: **P0** naive fixed weekly order · **P1** static `(s, Q)` · **P2** forecast-driven `(s, Q)`. | M |
| FR-20 | Iso-service comparison: calibrate each policy to a 95% fill rate before comparing inventory (see [methodology §6.4](08_methodology.md#64-fair-comparison-iso-service)). | M |
| FR-21 | Sensitivity analysis on lead time (3/7/14 days) and service target (90/95/98/99%). | S |
| FR-22 | Fill-rate (Type 2) safety stock using the standard normal loss function. | S |
| FR-23 | Periodic-review `(R, S)` policy. | C |
| FR-24 | Stochastic lead time. | C |

### KPIs & reporting
| ID | Requirement | Priority |
|---|---|---|
| FR-25 | KPIs per series, per ABC class and in total: fill rate, cycle service level, stock-out days, stock-out events, average inventory (units, €), turns, days of supply, number of orders, ordering/holding/shortage/total cost. | M |
| FR-26 | Export the main charts to `reports/figures/*.png` for the README. | M |
| FR-27 | Export the results to `data/processed/artefacts/*.parquet` for the dashboard. | M |

### Dashboard
| ID | Requirement | Priority |
|---|---|---|
| FR-28 | Overview page: headline KPIs, policy comparison. | M |
| FR-29 | Demand page: seasonality charts, ABC/XYZ matrix. | M |
| FR-30 | Forecast page: model leaderboard; actual vs. forecast for a selected store-item. | M |
| FR-31 | Inventory page: inventory trajectory, orders and stock-outs for a selected store-item and policy. | M |
| FR-32 | What-if panel: change service level, lead time and ordering cost → recompute SS/ROP/EOQ and re-simulate **one** series live. | S |
| FR-33 | Download results as CSV. | S |
| FR-34 | Power BI report (screenshot in the README). | C |

### Scenario configuration (stores, products, channel)
| ID | Requirement | Priority |
|---|---|---|
| FR-35 | The user sets the **number of stores** (1 – number of stores in the data) or picks the store IDs. | M |
| FR-36 | The user sets the **number of products** (1 – number of items in the data) and how they are picked: top-N by revenue (default), random N (seeded) or a manual list. | M |
| FR-37 | The user sets the **channel of each store**: `physical` or `online`; shortcuts "all physical" / "all online". | M |
| FR-38 | Online channel model: a configurable share of unmet demand is **back-ordered** (served when stock arrives) and the rest is lost; own holding rate, backorder cost and service target (`channels.online` in `params.yaml`). | M |
| FR-39 | Every series is precomputed offline under **both** channels and all policies, so changing the scenario only filters and re-aggregates (no re-simulation). | M |
| FR-40 | All pages, KPIs and the headline sentence follow the active scenario; the scenario is shown in a header badge (e.g. "8 stores · 30 products · 2 online"). | M |
| FR-41 | KPIs can be split by channel (physical vs. online). | S |
| FR-42 | The scenario can be shared through the URL query string (`?stores=8&items=30&online=9,10`). | C |
| FR-43 | Risk pooling: online stores are served from one shared warehouse (their demand is summed per item and simulated live with fixed z). | C |
| FR-44 | ~~More than 10 stores / 50 items through a synthetic generator~~ → replaced by FR-45+ (upload your own sales history). | – |

### Data input — own sales history
| ID | Requirement | Priority |
|---|---|---|
| FR-45 | Two data sources: the built-in **demo** (Kaggle, precomputed) or a **user file** (upload in the app, `--input` in the CLI). | M |
| FR-46 | Accepted formats: `.csv` (any delimiter, auto-detected), `.xlsx` (first sheet or chosen sheet), `.parquet`. | M |
| FR-47 | **Column mapping:** the user maps their column names to `date`, `item`, `quantity` (required) and `store`, `unit_price`, `unit_cost`, `channel`, `lead_time_days` (optional). The system guesses the mapping from common names (EN/IT: `data`, `sku`, `articolo`, `qta`, `quantità`, `negozio`, `prezzo`, …). | M |
| FR-48 | **Standardisation:** parse dates (ISO or day-first), aggregate duplicate rows by summing per `(date, store, item)` (transaction-level files accepted), fill missing days with 0 between each series' first and last date, handle negative quantities as set in the config (`clip` to 0 or `reject`). | M |
| FR-49 | **Validation report** before running: rows, date range, # stores, # items, # series, % zero days, series excluded for short history, assumed values (e.g. prices missing). Hard errors block the run; warnings don't. | M |
| FR-50 | **Adaptive periods:** test/validation lengths derived from the history length (methodology §1); the user can override them. | M |
| FR-51 | **Adaptive features and models:** yearly lag and yearly seasonality only when history allows; models that cannot run on a series are skipped for it (and it is logged). | M |
| FR-52 | Missing optional fields fall back to `params.yaml` defaults: no `store` → one location; no price/cost → assumed prices (flagged "assumed" everywhere they are used); no `channel` → physical; no `lead_time_days` → global lead time. | M |
| FR-53 | Optional **item master** file (item, unit_price, unit_cost, lead_time_days) for when prices/lead times are not in the sales file. | S |
| FR-54 | Download a **template** CSV and an example file. | M |
| FR-55 | **Live pipeline** in the app for uploaded data, with a progress bar and a "fast mode" (methodology §4.4); results cached by file hash for the session. | M |
| FR-56 | Download the results for the uploaded data (reorder parameters per series, KPIs) as CSV/Excel. | S |
| FR-57 | Weekly or monthly input data (all logic in periods instead of days). | C |
| FR-58 | Intermittent-demand classification (ADI / CV², Syntetos-Boylan) and a Croston/SBA model for sparse series. | C |

### Sample data
| ID | Requirement | Priority |
|---|---|---|
| FR-59 | Invented but realistic sample datasets (chain with online shop, Italian-format CSV, Excel order lines with returns) with a seeded generator, selectable in the dashboard without uploading anything. | S |

## 2. Non-functional requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Reproducibility | A single command (`python -m src.pipeline --all`) rebuilds every artefact; fixed random seeds; dependency versions pinned. |
| NFR-02 | Performance | The full pipeline runs in under 20 min on a laptop; the one-series what-if simulation runs in under 1 s; a scenario change updates the dashboard in under 2 s. |
| NFR-03 | Dashboard performance | First load in under 10 s; artefacts under 50 MB; memory under 1 GB (Streamlit Cloud limit). |
| NFR-04 | Maintainability | Business logic lives only in `src/`; notebooks and app import from it (no copy-paste). Type hints + docstrings on the public API. |
| NFR-05 | Configurability | Every business parameter lives in `config/params.yaml`; nothing is hard-coded. |
| NFR-06 | Quality | `pytest` covers formulas and simulation invariants; `ruff` lint passes. |
| NFR-07 | Readability | Notebooks: a markdown intro, one idea per section, a conclusion cell; outputs kept for the charts. |
| NFR-08 | Documentation | README readable in under 2 minutes; results before method. |
| NFR-09 | Legal / data | Raw Kaggle data is not redistributed; `data/README.md` explains how to download it. |
| NFR-10 | Language | Everything in English. |
| NFR-11 | No leakage | No forecast or policy parameter uses information from after the forecast origin. |
| NFR-12 | Privacy | Uploaded files are kept **in memory only** for the session; never written to disk, logged or sent anywhere. The app shows a notice that the public instance should not be used for confidential data and explains how to run it locally. |
| NFR-13 | Upload limits | Dashboard ≤ 300 MB, ≤ 10M rows, ≤ 10,000 series (≈ 1 minute in fast mode on a laptop; lower for a 1 GB online host). Larger files → batch mode `python -m src.partition` (out of core, linear time: 60,000 series in ≈ 4.5 minutes with 2 workers). |
| NFR-14 | Robustness | Any file that breaks the contract gives a clear message naming the column/row issue, never a stack trace. |

## 3. Business rules

| ID | Rule |
|---|---|
| BR-01 | **Physical** stores: unmet demand is **lost**. **Online** stores: `backorder_share` of unmet demand is back-ordered and served first when stock arrives; the rest is lost. |
| BR-02 | Orders are placed at the end of day *t* and arrive at the start of day *t + L*. |
| BR-03 | Stock is reviewed every day (continuous review) for P1/P2; P0 orders every Monday. |
| BR-04 | Order quantities are whole units ≥ 1. |
| BR-05 | Holding cost is charged on end-of-day on-hand inventory. |
| BR-06 | `(s, Q)` policies trigger orders on **inventory position** (on-hand + on-order), not on-hand, so more than one order can be open at a time. |
| BR-07 | Initial on-hand stock on the first test day is the same for every policy: `ROP + Q/2` of the P1 policy. |
| BR-08 | Scenario limits come from the active dataset: 1 ≤ stores ≤ stores in the data, 1 ≤ products ≤ items in the data. The product selection uses revenue in the reference (validation) period (no leakage). |
| BR-09 | A store's channel changes only its inventory parameters and its unmet-demand rule; its demand history stays the same. |

## 4. Acceptance criteria (headline)

- AC-1: Leaderboard table with 4 models × {WAPE, MAPE, bias} exists and is reproducible.
- AC-2: At least one policy reaches a fill rate ≥ 95% in the test-period simulation.
- AC-3: The inventory reduction `X%` is computed at iso-service and shown in the README.
- AC-4: The public dashboard URL works from an incognito browser window.

## 5. Traceability matrix

| Requirement | Use case | WBS |
|---|---|---|
| FR-01 – FR-04 | UC-09 | 2.1, 2.2, 2.3 |
| FR-05 – FR-08 | UC-01, UC-02 | 2.4, 2.5 |
| FR-09 – FR-15 | UC-03, UC-04 | 3.1 – 3.5 |
| FR-17 – FR-22 | UC-05, UC-06, UC-07 | 4.1 – 4.4, 5.1 |
| FR-25 – FR-27 | UC-07, UC-08, UC-10 | 5.2, 5.3 |
| FR-28 – FR-33 | UC-01 – UC-08 | 6.1 – 6.4 |
| NFR-01, NFR-11 | UC-09 | 1.3, 3.2, 4.4 |
| NFR-08 | UC-10 | 7.1 |
| FR-35 – FR-42 | UC-11 | 1.3, 4.6, 6.6 |
| FR-43 | UC-11 | 4.6 |
| FR-45 – FR-56, NFR-12 – NFR-14 | UC-12, UC-09 | 2.7, 3.7, 6.7 |

## 6. Implementation status (v1.0)

| Status | Requirements |
|---|---|
| ✅ Done | FR-01 – FR-15, FR-17 – FR-22, FR-25 – FR-33, FR-35 – FR-41, FR-45 – FR-56, FR-59; NFR-01 – NFR-14 |
| ⏭ Not in v1 (Could) | FR-16 quantile forecasts, FR-23 periodic review (R, S), FR-24 stochastic lead time in the simulation (the formula is in `src.inventory.sigma_lead_time`), FR-34 Power BI, FR-42 scenario in the URL, FR-43 online risk pooling, FR-57 weekly/monthly input, FR-58 intermittent-demand models |
| ✖ Dropped | FR-44 (synthetic generator for bigger scenarios) — replaced by user uploads (ADR-0006); `src/synthetic.py` exists only for tests and CI |

Notes: FR-22 is available through `inventory.service_type: fill_rate`; FR-02 validation also covers uploaded files (FR-49).
