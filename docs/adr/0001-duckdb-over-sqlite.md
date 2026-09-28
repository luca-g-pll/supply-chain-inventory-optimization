# ADR-0001 — DuckDB instead of SQLite

**Status:** Accepted · **Date:** 2026-09-28

## Context
The brief asks for SQL KPIs in SQLite *or* DuckDB. The workload is analytical: aggregations and window functions over about 1M rows, read from CSV/Parquet, used from Python.

## Decision
Use **DuckDB** (file `data/processed/inventory.duckdb`).

## Consequences
- ➕ Columnar engine: aggregations over 1M rows take milliseconds; it reads CSV/Parquet directly (`read_csv_auto`).
- ➕ Full analytical SQL (window functions, `QUALIFY`, `PIVOT`), and zero-copy exchange with pandas.
- ➕ "DuckDB" is a modern keyword for data roles.
- ➖ Less well known than SQLite to some reviewers → the queries are kept close to standard ANSI SQL where possible.
- ➖ The `.duckdb` file is binary and not committed; it is rebuilt by `--step ingest`.
