# ADR-0003 — The dashboard reads precomputed artefacts

**Status:** Accepted for the **demo**; amended by [ADR-0006](0006-user-uploaded-sales-history.md) (uploaded data is computed live) · **Date:** 2026-09-28

## Context
Streamlit Community Cloud gives about 1 GB of RAM, has cold starts, and cannot download Kaggle data (that needs credentials). Training models and simulating 500 series × 3 policies in the app would be slow and fragile.

## Decision
The offline pipeline writes small **Parquet artefacts** to `data/processed/artefacts/`. They are **committed to git** (< 50 MB total). The app only reads them. The one live computation is the **one-series what-if simulation**, which reuses `src.inventory` and `src.simulation`.

## Consequences
- ➕ Fast, reliable dashboard; no secrets needed in the cloud.
- ➕ The app and the notebooks show the same numbers (one source).
- ➖ Artefacts must be regenerated and committed after any change to the pipeline → `manifest.json` stores the git SHA and config hash so the app can display the "data version".
- ➖ Daily simulation detail is stored for one store only, to keep the size small.
