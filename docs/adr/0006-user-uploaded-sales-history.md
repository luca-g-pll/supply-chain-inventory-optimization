# ADR-0006 — User-uploaded sales history

**Status:** Accepted · **Date:** 2026-09-28 · Amends [ADR-0003](0003-precomputed-artefacts-for-dashboard.md) and [ADR-0005](0005-configurable-scenario-and-channels.md)

## Context
The user wants to run the analysis on **their own sales history** instead of the fixed 5-year Kaggle dataset. Uploaded files will differ in column names, formats, IDs, length of history, and in whether prices, stores, channels and lead times are present. The project must still produce reproducible headline numbers for the README/CV.

## Decision
1. **Two data modes.** The **demo** (Kaggle) stays precomputed and gives the README numbers. **User files** are analysed live in the app (or with `--input` in the CLI).
2. **One input adapter** (`src.ingest`) for both: read → map columns → standardise → validate. The demo goes through it too, so it is tested all the time.
3. **Minimal contract:** `date, item, quantity` are required. Everything else is optional, with documented fallbacks that are **flagged as assumed** wherever they are used.
4. **Adaptive analysis** (`src.profile`): periods, lags, yearly features and enabled models are derived from the history length; explicit dates override them.
5. **Fast mode** for live runs: refit once, fewer trees, no hyper-parameter search, simulation vectorised across series. The CLI can run the full mode.
6. **Privacy by design:** uploads are kept in memory only, keyed by file hash; nothing is written to disk.
7. **Same output schemas** for live results and demo artefacts, so the pages don't need to know which is active.

## Alternatives considered
- *Only the CLI accepts files* — simpler, but the dashboard would not be a usable tool; rejected.
- *A server-side store for uploads* — needs auth and storage and brings privacy duties; out of scope.
- *A synthetic generator for other sizes* (old FR-44) — replaced by real user data.

## Consequences
- ➕ The project becomes a tool, not just an analysis: stronger for the portfolio ("upload your sales, get reorder points").
- ➕ Scenario limits (stores, products) follow the data.
- ➖ About +12 h of work (WBS 2.7, 3.7, 6.7) and more edge cases → messy-file fixtures and clear error messages are required.
- ➖ Live numbers (fast mode) can differ slightly from a full CLI run → the app shows which mode was used.
- ➖ The public app has size limits; big files must run locally.
