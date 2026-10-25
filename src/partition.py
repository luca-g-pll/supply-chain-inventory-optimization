"""Large datasets: run the analysis in batches of items, out of core.

``python -m src.partition --input sales.csv --out data/processed/large_run``

The in-memory pipeline holds every series at once, so its memory grows with the
number of store-item series. Here the sales file is read with **DuckDB** (it never
loads the whole file), the items are split into **partitions** (every store of an
item stays in the same partition, so the ABC / XYZ classes stay exact), each
partition runs the normal pipeline and writes its tables to Parquet, and the
totals are computed at the end from those files. Memory is bounded by the size of
one partition; time grows linearly with the number of series and partitions can run
in parallel processes (``--workers``).

Outputs in ``--out``:

* ``tables/<name>/part-NNNNN.parquet`` — every result table, one file per partition
  (readable as one table: ``SELECT * FROM 'tables/policy_summary/*.parquet'``);
* ``classification.parquet`` — ABC / XYZ recomputed on all items;
* ``reorder_parameters.csv`` — per store and item: lead time, lot, safety stock and
  reorder point of the forecast-driven policy, ready for an ERP import;
* ``results.json`` — headline, leaderboard, policy KPIs, partitions and timings.

Differences from a single run, stated in the results: LightGBM is one model per
partition (not one for all series) and the model chosen as "best" can differ
between partitions (the reported leaderboard sums every partition).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.classification import abc_classify
from src.config import Config, load_config
from src.ingest import ColumnMapping, InputError, guess_mapping, merge_item_master, read_any, standardise
from src.kpi import aggregate_kpis, headline
from src.metrics import SUM_COLUMNS as ERROR_SUMS
from src.metrics import evaluate
from src.series import natural_key

# tables written for every partition (the per-series daily detail is left out: it is for the dashboard)
TABLES = ("classification", "policy_summary", "forecast_errors", "item_master", "stores", "plan_forecast",
          "plan_state", "plan_pipeline", "promo_items", "promo_events")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def prepare_input(path: Path, out: Path) -> Path:
    """CSV -> Parquet once (all columns as text, parsed later by the normal adapter); Parquet is used as is."""
    import duckdb

    if path.suffix.lower() in (".parquet", ".pq"):
        return path
    if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise InputError("Batch mode reads .csv or .parquet files (save Excel files as CSV).")
    target = out / "input.parquet"
    con = duckdb.connect()
    con.execute(f"COPY (SELECT * FROM read_csv('{path}', all_varchar=true, header=true)) "
                f"TO '{target}' (FORMAT parquet)")
    con.close()
    return target


def partitions(source: Path, item_col: str, items_per_part: int | None, store_col: str | None,
               target_series: int = 20_000) -> list[list[str]]:
    """Items split into groups; by default about ``target_series`` store-item series per group."""
    import duckdb

    con = duckdb.connect()
    items = [r[0] for r in con.execute(f"SELECT DISTINCT {_q(item_col)} FROM read_parquet('{source}')").fetchall()
             if r[0] is not None]
    n_stores = 1 if store_col is None else con.execute(
        f"SELECT COUNT(DISTINCT {_q(store_col)}) FROM read_parquet('{source}')").fetchone()[0]
    con.close()
    items = sorted((str(i).strip() for i in items), key=natural_key)
    size = items_per_part or max(1, target_series // max(int(n_stores), 1))
    n_parts = int(np.ceil(len(items) / size))
    return [list(chunk) for chunk in np.array_split(np.array(items, dtype=object), n_parts)]   # equal sizes


def _run_part(k: int, items: list[str], source: str, item_col: str, mapping: dict, cfg: Config,
              master_path: str | None, out: str, fast: bool, n_jobs: int) -> dict:
    import duckdb

    from src.pipeline import run_analysis

    t0 = time.perf_counter()
    con = duckdb.connect()
    con.register("part_items", pd.DataFrame({"item": items}))
    raw = con.execute(f"SELECT t.* FROM read_parquet('{source}') t "
                      f"JOIN part_items p ON trim(CAST(t.{_q(item_col)} AS VARCHAR)) = p.item").df()
    con.close()
    data = standardise(raw, ColumnMapping.from_dict(mapping), cfg.input, cfg.prices, cfg.inventory,
                       source="file", file_name=Path(source).name, promotions=cfg.promotions)
    del raw
    if master_path:
        data = merge_item_master(data, read_any(master_path), cfg.prices, cfg.inventory)
    res = run_analysis(data, cfg, fast=fast, n_jobs=n_jobs)
    tables = res.tables()
    for name in TABLES:
        df = tables.get(name)
        if df is None or df.empty:
            continue
        folder = Path(out) / "tables" / name
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(folder / f"part-{k:05d}.parquet", index=False)
    return {"part": k, "items": len(items), "series": int(res.dm.n_series), "best_model": res.best_model,
            "seconds": round(time.perf_counter() - t0, 1), "test_period": [str(d) for d in res.profile.test],
            "notes": list(res.profile.notes)}


def finalize(out: Path, cfg: Config) -> dict:
    """Totals over every partition, read straight from the Parquet files with DuckDB."""
    import duckdb

    con = duckdb.connect()
    t = out / "tables"

    # classification: revenue and units are complete per item inside its partition -> rank again on all items
    cls = con.execute(f"SELECT * FROM read_parquet('{t}/classification/*.parquet')").df()
    abc = abc_classify(cls.set_index("item")["revenue"], cfg.classification.abc_thresholds)
    vol = abc_classify(cls.set_index("item")["units"], cfg.classification.abc_thresholds)["abc"]
    cls = cls.drop(columns=["revenue_share", "cum_share", "abc", "abc_volume", "rank"]).set_index("item")
    cls = abc.join(cls.drop(columns=["revenue"])).assign(abc_volume=vol)
    cls["abc_xyz"] = cls["abc"] + cls["xyz"]
    cls["rank"] = np.arange(1, len(cls) + 1)
    cls.index.name = "item"
    cls = cls.reset_index()
    cls.to_parquet(out / "classification.parquet", index=False)

    # leaderboard from the additive error sums, only for the models that ran on every series
    # (a model skipped in some partitions, e.g. Holt-Winters above its series limit, is listed apart)
    sums = ", ".join(f"SUM({c}) AS {c}" for c in ERROR_SUMS)
    err = con.execute(f"SELECT period, model, {sums}, COUNT(DISTINCT store || '|' || item) AS series "
                      f"FROM read_parquet('{t}/forecast_errors/*.parquet') GROUP BY period, model").df()
    full = err["series"] == err.groupby("period")["series"].transform("max")
    partial = err.loc[~full & (err["period"] == "test"), ["model", "series"]].to_dict(orient="records")
    board = evaluate(err[full].drop(columns="series"), ["period", "model"])

    # policy KPIs from the additive simulation sums
    pol = con.execute(f"SELECT * FROM read_parquet('{t}/policy_summary/*.parquet') LIMIT 0").df()
    from src.kpi import SUM_COLUMNS as POLICY_SUMS

    cols = [c for c in POLICY_SUMS if c in pol.columns]
    agg = con.execute(
        "SELECT channel, policy, variant, " + ", ".join(f"SUM({c}) AS {c}" for c in cols)
        + f", MAX(n_days) AS n_days FROM read_parquet('{t}/policy_summary/*.parquet') "
        "GROUP BY channel, policy, variant").df()
    kpis = aggregate_kpis(agg, ["channel", "policy", "variant"])
    phys = agg[agg["channel"] == "physical"]

    # reorder parameters of the forecast-driven policy, ready for an ERP
    con.execute(f"""
        COPY (
            SELECT store, item, channel, lead_time, order_qty AS lot_size, round(safety_stock, 1) AS safety_stock,
                   round(reorder_point, 1) AS reorder_point_avg, round(mean_daily, 3) AS mean_daily_demand,
                   round(knob, 3) AS safety_factor
            FROM read_parquet('{t}/policy_summary/*.parquet')
            WHERE policy = 'P2_forecast_sq' AND variant = 'calibrated'
            ORDER BY store, item, channel
        ) TO '{out / "reorder_parameters.csv"}' (HEADER, DELIMITER ',')""")
    n_series = con.execute(f"SELECT COUNT(*) FROM read_parquet('{t}/plan_state/*.parquet') "
                           "WHERE channel = 'physical'").fetchone()[0]
    con.close()

    test = board[board["period"] == "test"].sort_values("wape")
    base = test.set_index("model")["wape"].get(cfg.forecast.baseline, np.nan)
    return {
        "n_series": int(n_series),
        "n_items": int(len(cls)),
        "leaderboard_test": test[["model", "wape", "mape", "bias", "rmse"]].to_dict(orient="records"),
        "best_model_overall": str(test.iloc[0]["model"]) if len(test) else None,
        "models_on_part_of_the_series": partial,
        "wape_improvement_vs_baseline": float(1 - test.iloc[0]["wape"] / base) if len(test) and base else None,
        "inventory_calibrated_physical": headline(phys, "calibrated"),
        "policy_kpis": kpis[["channel", "policy", "variant", "fill_rate", "avg_inventory_value", "total_cost",
                             "n_series"]].to_dict(orient="records"),
        "abc_counts": cls["abc"].value_counts().to_dict(),
    }


def run_batches(input_path: str | Path, out: str | Path, cfg: Config | None = None, item_master: str | None = None,
                column_map: dict | None = None, items_per_part: int | None = None, workers: int = 1,
                fast: bool = True, progress=print) -> dict:
    """Whole batch run; returns the content of ``results.json``."""
    import duckdb

    cfg = cfg or load_config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    progress("Reading the file with DuckDB")
    source = prepare_input(Path(input_path), out)
    con = duckdb.connect()
    columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{source}')").fetchall()]
    con.close()
    mapping = guess_mapping(columns)
    mapping.update({k: v for k, v in (column_map or {}).items() if v})
    ColumnMapping.from_dict(mapping)                      # raises if a required column is missing
    parts = partitions(source, mapping["item"], items_per_part, mapping.get("store"))
    progress(f"{len(parts)} partition(s) of up to {max(len(p) for p in parts):,} items")
    workers = max(1, int(workers))
    n_jobs = max(1, (os.cpu_count() or 2) // workers)
    args = [(k, items, str(source), mapping["item"], mapping, cfg, item_master, str(out), fast, n_jobs)
            for k, items in enumerate(parts)]
    done = []
    if workers == 1:
        for a in args:
            done.append(_run_part(*a))
            progress(f"Partition {len(done)}/{len(parts)}: {done[-1]['series']:,} series in {done[-1]['seconds']}s")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_run_part, *zip(*args)):
                done.append(r)
                progress(f"Partition {len(done)}/{len(parts)}: {r['series']:,} series in {r['seconds']}s")
    progress("Totals over every partition")
    result = finalize(out, cfg)
    result.update({
        "input": str(input_path), "mode": "fast" if fast else "full", "workers": workers,
        "partitions": done, "seconds": round(time.perf_counter() - t0, 1),
        "best_model_by_partition": pd.Series([d["best_model"] for d in done]).value_counts().to_dict(),
    })
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.partition", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="sales file (.csv or .parquet)")
    p.add_argument("--out", default="data/processed/large_run", help="output folder")
    p.add_argument("--item-master", default=None, help="optional price / cost / lead-time / supplier file")
    p.add_argument("--map", nargs="*", metavar="FIELD=COLUMN", help="column mapping, e.g. item=SKU store=Negozio")
    p.add_argument("--items-per-part", type=int, default=None, help="items per partition (default ≈ 20,000 series)")
    p.add_argument("--workers", type=int, default=1, help="partitions run in parallel (each needs its own memory)")
    p.add_argument("--full", action="store_true", help="full mode (slower) instead of fast mode")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    column_map = dict(pair.split("=", 1) for pair in (args.map or []))
    try:
        res = run_batches(args.input, args.out, load_config(args.config), args.item_master, column_map,
                          args.items_per_part, args.workers, fast=not args.full,
                          progress=lambda m: print(m, flush=True))
    except (InputError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    inv = res["inventory_calibrated_physical"]
    print(f"\n{res['n_series']:,} series · {res['n_items']:,} items · {len(res['partitions'])} partitions · "
          f"{res['seconds']:.0f}s")
    print(f"Best model overall: {res['best_model_overall']} · "
          f"{res['wape_improvement_vs_baseline']:.1%} less error than the baseline")
    if inv:
        print(f"At the target fill rate, average stock P2 vs P0 {-inv['inventory_reduction_pct']:+.1%}, "
              f"vs P1 {-inv['inventory_reduction_vs_classic_pct']:+.1%}")
    print(f"Results in {args.out}: results.json, reorder_parameters.csv, tables/*")
    return 0


if __name__ == "__main__":
    from src.partition import main as _main

    raise SystemExit(_main())
