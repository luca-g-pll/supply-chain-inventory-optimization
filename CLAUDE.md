# Demand & Inventory Planner

Demand forecasting and inventory optimization for retail: forecasts, reorder policies compared on the past, a
purchase plan on today's calendar, and a bilingual Streamlit dashboard.

Python 3.11+ · pandas / NumPy · LightGBM · statsmodels · DuckDB · Streamlit 1.64 · pytest + hypothesis · ruff

## Commands
- `make install` — virtual environment in `.venv`, dependencies, LightGBM fix for macOS
- `make test` — pytest (synthetic data, no download)
- `make lint` — ruff on `src tests app scripts`
- `make app` — dashboard on http://localhost:8501
- `python -m src.pipeline --all [--input file.csv]` — full analysis from the command line
- `python -m src.partition --input big.csv --workers 2` — large files in batches
- `python -m src.sample_data` — rebuild the invented sample datasets

## Layout
- `src/` analysis engine (ingest, forecast, policy, simulation, promotions, replenishment, constraints, cold start)
- `app/` dashboard: `streamlit_app.py`, pages in `views/`, texts in `i18n.py`, warnings panel in `limits.py`
- `config/params.yaml` every business assumption (single source) · `docs/` design documents 01–12 · `tests/`

## Conventions
- Code, docs and commit messages in English; every dashboard text bilingual via `tx(en, it)` in `app/i18n.py`
- All business assumptions in `config/params.yaml`; no hard-coded parameters
- Forecasts without leakage: features at least `horizon` days old, models chosen on validation, scored on test
- ruff with 120 columns; vectorised NumPy code may update arrays in place for speed
- Every feature change also updates the "Warnings and limits" panel (`app/limits.py`) and README §12
- The Kaggle daily data must not be redistributed; the sample data are invented
- Commit only when asked; the user pushes from PyCharm

## Rules
General rules from ECC v2.2.3 are in `.claude/rules/ecc/`; `.claude/rules/project-overrides.md` says which project
convention wins where they disagree.
