# 11 — Risk Register

Scale: Likelihood (L) and Impact (I) from 1 (low) to 3 (high). Score = L × I.

| ID | Risk | L | I | Score | Mitigation | Owner / trigger |
|---|---|---|---|---|---|---|
| R-01 | The headline sentence claims 95% but the method gives a Type-1 (cycle) 95%, not a 95% fill rate → a misleading claim | 3 | 3 | **9** | Keep the two apart ([methodology §5.3](08_methodology.md#53-two-meanings-of-95-service--stated-clearly)); calibrate to fill rate; say "fill rate" in the README | Check before writing the README |
| R-02 | The dataset is synthetic and very regular → every model does well, small gains; XYZ is almost all "X" | 3 | 2 | **6** | Report honestly; use residual-CV; the inventory gain comes mostly from trend/seasonality awareness, which is still a strong story | After EDA (M1) |
| R-03 | Unfair policy comparison (different service levels) inflates the gain | 2 | 3 | **6** | Iso-service calibration (FR-20); show the non-calibrated results too | WBS 4.4 |
| R-04 | Data leakage in the lag features or policy parameters → optimistic results | 2 | 3 | **6** | `min_lag` guard, leakage tests T-L1–T-L3, strict time split | WBS 3.2 |
| R-05 | Holt-Winters on 500 series × 52 origins is too slow | 2 | 2 | 4 | Refit every 4 origins; `joblib` parallel; `--quick` mode for development | WBS 3.4 |
| R-06 | Streamlit Cloud memory or timeout limits | 2 | 2 | 4 | Precomputed Parquet only, `@st.cache_data`, daily detail for one store only | WBS 6.4 |
| R-07 | Power BI Desktop is Windows-only; the owner uses macOS | 3 | 1 | 3 | Optional; use a Windows VM/PC or skip it; mention "Power BI" only if it is really done | WBS 6.5 |
| R-08 | Scope creep (extra models, multi-echelon, …) | 2 | 2 | 4 | MoSCoW priorities; "Could" items only after M3 | Weekly check |
| R-09 | Kaggle terms forbid redistribution | 1 | 2 | 2 | Don't commit raw data; download instructions only | — |
| R-10 | The assumed costs drive the results and look arbitrary | 2 | 2 | 4 | Parameters in one visible table; sensitivity analysis; the what-if lets readers change them | WBS 5.1 |
| R-12 | The "online" channel looks made up, because it reuses physical-store demand | 2 | 2 | 4 | Present it clearly as a what-if on operating rules (backorders, costs), not as real online demand; state it in the README and methodology §7 | WBS 4.6 |
| R-14 | Precomputing both channels doubles simulation time and artefact size | 2 | 1 | 2 | 3,000 summary rows is tiny; daily detail stays limited to one store | WBS 4.6 |
| R-15 | Users upload confidential sales data to the public app | 2 | 3 | **6** | In-memory only (NFR-12); visible notice; clear "run locally" instructions; no logging of content | WBS 6.7 |
| R-16 | Messy real files (odd formats, codes, dates, duplicates) break the pipeline | 3 | 2 | **6** | Mapping UI + synonym guessing; standardisation rules; tests on messy fixtures (T-G1–T-G9); readable error messages | WBS 2.7 |
| R-17 | Short or sparse histories give weak forecasts and misleading KPIs | 3 | 2 | **6** | Adaptive rules (methodology §1.1); *low confidence* flag; exclusions listed; intermittent models as a later step | WBS 3.7 |
| R-18 | Live pipeline too slow / memory-heavy on Streamlit Cloud | 2 | 2 | 4 | Fast mode; vectorised simulation; hard limits; CLI for big files | WBS 3.7, 6.7 |
| R-19 | Scope growth from "any data" pushes the deadline | 3 | 2 | **6** | +12 h planned; daily-only v1; weekly data, Croston and item master are Should/Could; Power BI is cut first | Weekly check |
| R-11 | Notebooks and `src` drift apart | 2 | 1 | 2 | Notebooks import `src`; CI runs notebooks with `--quick` | Ongoing |
