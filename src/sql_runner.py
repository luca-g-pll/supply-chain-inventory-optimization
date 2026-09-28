"""Run the named queries of ``sql/kpi_queries.sql`` against DuckDB."""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

_NAME = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)
_PARAM = re.compile(r"\$(\w+)")


def parse_queries(sql_path: str | Path) -> dict[str, str]:
    """Split the file on ``-- name: <query_name>`` markers (file order is kept)."""
    text = Path(sql_path).read_text(encoding="utf-8")
    matches = list(_NAME.finditer(text))
    if not matches:
        raise ValueError(f"No '-- name:' markers in {sql_path}")
    queries: dict[str, str] = {}
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
        body = text[m.end():end].strip().rstrip(";").strip()
        if m.group(1) in queries:
            raise ValueError(f"Duplicate query name {m.group(1)!r}")
        queries[m.group(1)] = body
    return queries


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run one query, passing only the named parameters it actually uses."""
    needed = set(_PARAM.findall(sql))
    params = params or {}
    missing = needed - set(params)
    if missing:
        raise ValueError(f"Missing SQL parameter(s): {sorted(missing)}")
    return con.execute(sql, {k: params[k] for k in needed} if needed else None).df()


def run_all(con: duckdb.DuckDBPyConnection, sql_path: str | Path, params: dict | None = None) -> dict[str, pd.DataFrame]:
    return {name: run_query(con, sql, params) for name, sql in parse_queries(sql_path).items()}


def profile_params(profile, cfg) -> dict:
    """SQL parameters from the data profile and the classification thresholds."""
    return {
        "ref_start": profile.valid[0], "ref_end": profile.valid[1],
        "valid_start": profile.valid[0], "valid_end": profile.valid[1],
        "test_start": profile.test[0], "test_end": profile.test[1],
        "a_threshold": cfg.classification.abc_thresholds[0],
        "b_threshold": cfg.classification.abc_thresholds[1],
        "x_threshold": cfg.classification.xyz_thresholds[0],
        "y_threshold": cfg.classification.xyz_thresholds[1],
    }
