"""Demo data, synthetic prices and the DuckDB database."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pandas as pd

from src.config import PricesConfig
from src.series import natural_sorted

if TYPE_CHECKING:
    from src.ingest import ColumnMapping, StandardisedData

DEMO_COLUMNS = ("date", "store", "item", "sales")
DEMO_DOWNLOAD_HINT = (
    "Download train.csv from the Kaggle competition 'Store Item Demand Forecasting Challenge' "
    "(https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) and save it as "
    "data/raw/train.csv — see data/README.md."
)


def load_demo(path: str | Path) -> pd.DataFrame:
    """Read the Kaggle ``train.csv`` and rename ``sales`` -> ``quantity``.

    The result then goes through ``src.ingest.standardise`` like any user file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Demo dataset not found at {path}. {DEMO_DOWNLOAD_HINT}")
    df = pd.read_csv(path, parse_dates=["date"], dtype={"store": "string", "item": "string", "sales": "float64"})
    missing = set(DEMO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is not the Kaggle train.csv: missing columns {sorted(missing)}")
    return df.rename(columns={"sales": "quantity"})


def demo_mapping() -> ColumnMapping:
    from src.ingest import ColumnMapping

    return ColumnMapping(date="date", item="item", quantity="quantity", store="store")


def make_price_table(items, cfg: PricesConfig, cost_to_price_ratio: float = 0.60) -> pd.DataFrame:
    """Seeded synthetic prices -> ``[item, unit_price, unit_cost]``.

    Each price is Uniform(min, max), drawn from a hash of the seed and the item code,
    rounded to ``round_to``: an item always gets the same price, whatever other items
    are in the list (batch runs on parts of a file agree with a run on the whole file).
    """
    ordered = natural_sorted(items)
    u = np.array([int.from_bytes(hashlib.sha256(f"{cfg.seed}:{i}".encode()).digest()[:8], "big") / 2.0 ** 64
                  for i in ordered])
    raw = cfg.min + (cfg.max - cfg.min) * u
    price = np.round(np.round(raw / cfg.round_to) * cfg.round_to, 2)
    return pd.DataFrame({
        "item": ordered,
        "unit_price": price,
        "unit_cost": np.round(price * cost_to_price_ratio, 4),
    })


def build_duckdb(data: StandardisedData, db_path: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """(Re)create the tables ``sales``, ``item_master``, ``stores`` and the view ``sales_enriched``.

    Use ``:memory:`` for uploads (nothing touches the disk).
    """
    if str(db_path) != ":memory:":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", ".wal"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()
    con = duckdb.connect(str(db_path))
    sales = data.sales[["date", "store", "item", "sales"]]
    master = data.item_master.assign(lead_time_days=data.item_master["lead_time_days"].astype("float64"))
    stores = data.stores
    con.register("sales_df", sales)
    con.register("master_df", master)
    con.register("stores_df", stores)
    con.execute("""
        CREATE TABLE sales AS
        SELECT CAST(date AS DATE) AS date, CAST(store AS VARCHAR) AS store,
               CAST(item AS VARCHAR) AS item, CAST(sales AS DOUBLE) AS sales
        FROM sales_df ORDER BY store, item, date
    """)
    con.execute("""
        CREATE TABLE item_master AS
        SELECT CAST(item AS VARCHAR) AS item, unit_price, unit_cost,
               CAST(lead_time_days AS INTEGER) AS lead_time_days, price_source, cost_source
        FROM master_df
    """)
    con.execute("CREATE TABLE stores AS SELECT CAST(store AS VARCHAR) AS store, channel FROM stores_df")
    con.execute("""
        CREATE VIEW sales_enriched AS
        SELECT s.*, m.unit_price, m.unit_cost, s.sales * m.unit_price AS revenue
        FROM sales s JOIN item_master m USING (item)
    """)
    for name in ("sales_df", "master_df", "stores_df"):
        con.unregister(name)
    return con


def load_sales(con: duckdb.DuckDBPyConnection, start=None, end=None) -> pd.DataFrame:
    """Read ``sales`` back from DuckDB, optionally filtered by date."""
    query = "SELECT date, store, item, sales FROM sales"
    params: list = []
    clauses = []
    if start is not None:
        clauses.append("date >= ?")
        params.append(pd.Timestamp(start).date())
    if end is not None:
        clauses.append("date <= ?")
        params.append(pd.Timestamp(end).date())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    df = con.execute(query + " ORDER BY store, item, date", params).df()
    df["date"] = pd.to_datetime(df["date"])
    return df
