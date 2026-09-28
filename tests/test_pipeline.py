"""End-to-end in-memory run (T-P5) and the CLI on a small file."""

import os

import pandas as pd
import pytest

from src.pipeline import main, run_in_memory
from src.synthetic import write_kaggle_like_csv


@pytest.fixture(scope="module")
def results(data, cfg):
    return run_in_memory(data, cfg, fast=True, n_jobs=1)


def test_live_results_have_the_artefact_tables(results):  # T-P5
    tables = results.tables()
    for name in ("classification", "forecast_errors", "leaderboard", "policy_summary", "tradeoff", "sensitivity",
                 "demand_monthly", "demand_dow", "forecasts", "detail_actuals", "detail_paths", "sql_kpi_overview"):
        assert name in tables and len(tables[name]) > 0, name
    s = tables["policy_summary"]
    assert {"store", "item", "channel", "policy", "variant", "fill_rate", "total_cost"} <= set(s.columns)
    assert set(s["channel"]) == {"physical", "online"}
    cal = s[s["variant"] == "calibrated"]
    assert (cal["fill_rate"] >= 0.95 - 1e-9).all()
    h = results.headline
    assert h["forecast"]["wape_best"] <= h["forecast"]["wape_baseline"]


def test_in_memory_run_writes_nothing(data, cfg, tmp_path):
    before = {p for p in os.listdir(cfg.root)}
    run_in_memory(data, cfg, fast=True, n_jobs=1)
    assert {p for p in os.listdir(cfg.root)} == before


def test_cli_on_a_user_file(tmp_path, monkeypatch):
    src = tmp_path / "sales.csv"
    write_kaggle_like_csv(src, n_stores=2, n_items=3, start="2015-01-01", end="2016-12-31", seed=4)
    df = pd.read_csv(src).rename(columns={"date": "Data", "item": "SKU", "sales": "Qta", "store": "Negozio"})
    df.to_csv(src, sep=";", index=False)
    work = tmp_path / "work"
    code = main(["--all", "--input", str(src), "--fast", "--no-figures", "--n-jobs", "1",
                 "--config", str(_config_copy(tmp_path, work))])
    assert code == 0
    assert (work / "results.joblib").exists()


def test_cli_exit_codes(tmp_path):
    cfg_path = str(_config_copy(tmp_path, tmp_path / "w"))
    assert main(["--all", "--input", str(tmp_path / "nope.csv"), "--config", cfg_path]) == 2   # missing input
    short = tmp_path / "short.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20).astype(str), "item": "A", "qty": 1}).to_csv(
        short, index=False)
    assert main(["--all", "--input", str(short), "--config", cfg_path]) == 3                  # validation failed
    assert main(["--all", "--config", str(tmp_path / "missing.yaml")]) == 1                   # config error


def _config_copy(tmp_path, work):
    import yaml

    from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["paths"]["work"] = str(work)
    raw["paths"]["figures"] = str(tmp_path / "figures")
    raw["paths"]["artefacts"] = str(tmp_path / "artefacts")
    raw["paths"]["validation_report"] = str(tmp_path / "validation.md")
    raw["paths"]["sql"] = str(PROJECT_ROOT / "sql" / "kpi_queries.sql")
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(exist_ok=True)
    path = cfg_dir / "params.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path
