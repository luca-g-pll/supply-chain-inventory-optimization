"""Batch (out-of-core) runs for large datasets (T-B1 .. T-B2)."""

import json

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import ColumnMapping, standardise
from src.partition import partitions, prepare_input, run_batches
from src.pipeline import run_analysis
from src.synthetic import write_kaggle_like_csv


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("batch")
    csv = folder / "sales.csv"
    raw = write_kaggle_like_csv(csv, n_stores=3, n_items=12, start="2023-01-01", end="2024-12-31", seed=5)
    cfg = load_config(overrides={"forecast": {"models": ["seasonal_naive", "moving_average", "lightgbm"]}})
    res = run_batches(csv, folder / "out", cfg, items_per_part=5, progress=lambda m: None)
    return folder, raw, cfg, res


def test_partitions_keep_items_whole_and_balanced(batch_run):  # T-B1
    folder, *_ = batch_run
    src = prepare_input(folder / "sales.csv", folder / "out")
    parts = partitions(src, "item", 5, "store")
    assert [len(p) for p in parts] == [4, 4, 4]                        # 12 items in 3 equal partitions
    assert sorted(i for p in parts for i in p) == sorted(str(i) for i in range(1, 13))


def test_batch_results_match_a_single_run(batch_run):  # T-B2
    folder, raw, cfg, res = batch_run
    out = folder / "out"
    assert res["n_series"] == 36 and len(res["partitions"]) == 3
    assert json.loads((out / "results.json").read_text())["n_items"] == 12
    params = pd.read_csv(out / "reorder_parameters.csv")
    assert len(params) == 36 * 2                                       # every series, both channels
    assert {"lot_size", "safety_stock", "reorder_point_avg"} <= set(params.columns)
    # ABC / XYZ recomputed on all items = the classes of one run on the whole file
    data = standardise(raw.rename(columns={"sales": "quantity"}), ColumnMapping("date", "item", "quantity", "store"),
                       cfg.input, cfg.prices, cfg.inventory)
    single = run_analysis(data, cfg, fast=True).classification.set_index("item")
    batch = pd.read_parquet(out / "classification.parquet").set_index("item")
    assert batch["abc"].sort_index().equals(single["abc"].sort_index())
    assert batch["xyz"].sort_index().equals(single["xyz"].sort_index())
    assert {r["model"] for r in res["leaderboard_test"]} == {"seasonal_naive", "moving_average", "lightgbm"}
