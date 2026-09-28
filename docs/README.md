# Project Documentation

Index of the design documents. Read them in this order; each one builds on the previous.

| # | Document | Purpose |
|---|---|---|
| 01 | [Project Charter](01_project_charter.md) | Why the project exists, objectives, scope, success criteria, stakeholders |
| 02 | [Requirements](02_requirements.md) | Functional and non-functional requirements (traceable IDs) |
| 03 | [Use Cases](03_use_cases.md) | Actors, use case diagram, detailed use case specifications |
| 04 | [WBS & Schedule](04_wbs.md) | Work Breakdown Structure, WBS dictionary, Gantt, milestones |
| 05 | [Architecture](05_architecture.md) | Components, data flow, folder responsibilities, artefacts |
| 06 | [API Specification](06_api_specification.md) | Public Python API of `src/`, CLI, data contracts, dashboard contract |
| 07 | [Data Dictionary](07_data_dictionary.md) | Raw dataset, DuckDB schema, derived tables, parameters |
| 08 | [Methodology](08_methodology.md) | Forecasting, error metrics, ABC/XYZ, safety stock, EOQ, simulation |
| 09 | [KPI Definitions](09_kpi_definitions.md) | Exact formula, unit and grain of every KPI |
| 10 | [Test Plan](10_test_plan.md) | Unit, invariant and regression tests; acceptance checks |
| 11 | [Risk Register](11_risk_register.md) | Risks, likelihood/impact, mitigations |
| ADR | [Architecture Decision Records](adr/) | Key decisions and their rationale |

## Conventions

- **Language:** English everywhere (code, comments, docs, commit messages).
- **Requirement IDs:** `FR-xx` (functional), `NFR-xx` (non-functional), `UC-xx` (use case), `WBS x.y` (work package), `R-xx` (risk).
- **Grain vocabulary:** a *series* or *SKU-location* = one `(store, item)` pair (demo: 10 × 50 = **500 series**).
- **Periods:** *train* · *validation* (= reference period) · *test / simulation*. In the **demo** they are 2013–2015 · 2016 · 2017; for an **uploaded file** they are derived from the length of its history ([methodology §1](08_methodology.md#1-time-split-and-leakage-rules)). "2016"/"2017" in these docs refer to the demo.
- **Traceability:** every requirement maps to at least one use case and one WBS work package (see the matrix in [02_requirements.md](02_requirements.md#5-traceability-matrix)).
