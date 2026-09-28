"""SQL KPIs vs. pandas (T-Q1 .. T-Q3)."""

import numpy as np
import pytest

from src.classification import classify_items
from src.data import build_duckdb
from src.profile import profile_history
from src.sql_runner import parse_queries, profile_params, run_all, run_query


@pytest.fixture(scope="module")
def sql_results(data, dm, cfg):
    con = build_duckdb(data)
    prof = profile_history(dm, cfg)
    return run_all(con, cfg.paths.sql, profile_params(prof, cfg)), prof


def test_every_named_query_runs(sql_results, cfg):  # T-Q1
    res, _ = sql_results
    names = list(parse_queries(cfg.paths.sql))
    assert list(res) == names and len(names) >= 8
    assert all(len(df) > 0 for df in res.values())


def test_abc_and_xyz_match_python(sql_results, data, cfg):  # T-Q2
    res, prof = sql_results
    py = classify_items(data.sales, data.item_master, *prof.valid).set_index("item")
    sql_abc = res["abc_classification"].set_index("item")
    sql_xyz = res["weekly_cv_by_item"].set_index("item")
    assert (sql_abc["abc"].reindex(py.index) == py["abc"]).all()
    assert (sql_xyz["xyz"].reindex(py.index) == py["xyz"]).all()
    np.testing.assert_allclose(sql_xyz["weekly_cv"].reindex(py.index), py["weekly_cv"], rtol=1e-9)
    np.testing.assert_allclose(sql_abc["cum_share"].reindex(py.index), py["cum_share"], rtol=1e-9)


def test_dow_index_weighted_mean_is_one(sql_results):  # T-Q3
    res, _ = sql_results
    d = res["dow_index"]
    assert (d["dow_index"] * d["n_days"]).sum() / d["n_days"].sum() == pytest.approx(1.0)


def test_totals_agree_with_pandas(sql_results, data):
    res, _ = sql_results
    assert res["kpi_overview"].at[0, "total_units"] == pytest.approx(data.sales["sales"].sum())
    assert res["daily_totals"]["units"].sum() == pytest.approx(data.sales["sales"].sum())


def test_missing_parameter_is_reported(data):
    con = build_duckdb(data)
    with pytest.raises(ValueError, match="ref_start"):
        run_query(con, "SELECT * FROM sales WHERE date >= $ref_start", {})
