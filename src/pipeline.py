"""End-to-end pipeline: CLI (``python -m src.pipeline``) and in-memory run for the app.

Steps: ingest -> profile -> sql -> classify -> forecast -> policy/simulate -> kpi -> report.
``run_analysis`` does all of it in memory and returns :class:`AnalysisResults`;
the CLI then writes the figures (README) and the artefacts (dashboard), or
runs single steps with their inputs cached in ``data/processed/work``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.classification import abc_xyz_matrix, classify_items
from src.config import KNOWN_MODELS, KNOWN_POLICIES, Config, PromotionsConfig, load_config
from src.data import build_duckdb, demo_mapping, load_demo
from src.forecast import MODEL_LABELS, BacktestResult, get_model, refit_schedule, run_backtest
from src.ingest import (
    ColumnMapping,
    InputError,
    StandardisedData,
    ValidationReport,
    file_hash,
    guess_mapping,
    merge_item_master,
    read_any,
    standardise,
    validate,
)
from src.kpi import add_costs, aggregate_kpis, headline
from src.metrics import evaluate, improvement
from src.policy import (
    PolicyInputs,
    build_policy_inputs,
    default_knob,
    make_policy,
    policy_parameters,
    run_all_policies,
    sensitivity,
    tradeoff_curves,
)
from src.profile import DataProfile, apply_exclusions, profile_history
from src.promotions import (
    estimate_effects,
    has_promotions,
    promo_accuracy,
    promo_baseline,
    promo_day_types,
    promo_events,
)
from src.replenishment import plan_tables
from src.scenario import Scenario, apply_scenario, scenario_from_config
from src.series import DemandMatrix, natural_sorted
from src.simulation import simulate_batch
from src.sql_runner import profile_params, run_all

# ingest  = read + standardise the input (cached in work/data.joblib)
# analyse = profile -> SQL -> classes -> forecasts -> policies -> KPIs (cached in work/results.joblib)
# report  = README figures + dashboard artefacts (+ scenario / headline from the current config)
STEPS = ("ingest", "analyse", "report")
Progress = Callable[[str, float], None]


def _noop(_msg: str, _frac: float) -> None:
    return None


# --------------------------------------------------------------------------- input
def load_input(cfg: Config) -> StandardisedData:
    """Read the configured source (demo or file) and standardise it."""
    if cfg.input.source == "demo":
        raw = load_demo(cfg.paths.raw_train)
        mapping = demo_mapping()
        name = "train.csv (Kaggle demo)"
        digest = file_hash(cfg.paths.raw_train.read_bytes())
        source = "demo"
    else:
        path = cfg.input.path
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        raw = read_any(path, sheet=cfg.input.sheet)
        guessed = guess_mapping(raw.columns)
        guessed.update({k: v for k, v in cfg.input.column_map.items() if v})
        mapping = ColumnMapping.from_dict(guessed)
        name = Path(path).name
        digest = file_hash(Path(path).read_bytes())
        source = "file"
    data = standardise(raw, mapping, cfg.input, cfg.prices, cfg.inventory, source=source,
                       file_name=name, file_hash_value=digest, promotions=cfg.promotions)
    if cfg.input.item_master:
        data = merge_item_master(data, read_any(cfg.input.item_master), cfg.prices, cfg.inventory)
    return data


# --------------------------------------------------------------------------- results
@dataclass
class AnalysisResults:
    data: StandardisedData
    validation: ValidationReport
    profile: DataProfile
    dm: DemandMatrix
    sql: dict[str, pd.DataFrame]
    classification: pd.DataFrame
    backtests: dict[str, BacktestResult]
    errors: pd.DataFrame
    leaderboard: pd.DataFrame
    best_model: str
    policy_inputs: PolicyInputs
    summary: pd.DataFrame
    tradeoff: pd.DataFrame
    sensitivity: pd.DataFrame
    scenario: Scenario
    headline: dict
    importance: pd.Series | None = None
    timings: dict = field(default_factory=dict)
    fast: bool = False
    promotions_cfg: PromotionsConfig = field(default_factory=PromotionsConfig)
    plan: dict[str, pd.DataFrame] = field(default_factory=dict)
    detail_max_series: int = 400

    # ---- tables shared by the artefacts and the live dashboard (same schemas)
    def tables(self, detail_stores: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
        """Every table the dashboard reads.

        Daily detail tables (charts of one series) cover the ``detail_stores`` (demo) or,
        on large datasets, only the ``detail_max_series`` top-revenue series; every other
        table covers all series.
        """
        keys = self.dm.keys
        rows = np.arange(len(keys)) if detail_stores is None else \
            np.flatnonzero(keys["store"].isin(list(detail_stores)).to_numpy())
        if detail_stores is None and len(keys) > self.detail_max_series:
            rows = top_revenue_series(self.dm, self.data.item_master, self.detail_max_series)
        bt_v, bt_t = self.backtests["validation"], self.backtests["test"]
        out: dict[str, pd.DataFrame] = {
            "classification": self.classification,
            "forecast_errors": self.errors,
            "leaderboard": leaderboard_table(self.errors, self.classification),
            "policy_summary": self.summary,
            "tradeoff": self.tradeoff,
            "sensitivity": self.sensitivity,
            "item_master": self.data.item_master,
            "stores": self.data.stores,
            "demand_monthly": demand_monthly(self.data),
            "demand_dow": demand_dow(self.data),
            "forecasts": bt_t.rolling_view(rows=rows),
            "detail_actuals": detail_actuals(self.dm, self.profile, rows),
            "detail_paths": pd.concat([bt_v.paths_frame(self.best_model, rows).assign(period="validation"),
                                       bt_t.paths_frame(self.best_model, rows).assign(period="test")],
                                      ignore_index=True),
        }
        out["forecasts"] = with_promo_columns(out["forecasts"], self.dm, self.promotions_cfg)
        out["detail_actuals"] = with_promo_columns(out["detail_actuals"], self.dm, self.promotions_cfg)
        if self.dm.promo is not None:
            cfg_p = self.promotions_cfg
            effects = estimate_effects(self.dm, self.dm.n_days - 1, cfg_p)
            out["promo_items"] = effects.item_table
            out["promo_events"] = promo_events(self.dm, cfg_p)
            out["promo_accuracy"] = promo_accuracy(bt_t, self.dm, cfg_p)
        out["promo_baseline"] = promo_baseline(self.dm, self.promotions_cfg)
        out.update(self.plan)
        for name, df in self.sql.items():
            out[f"sql_{name}"] = df
        if self.importance is not None:
            out["feature_importance"] = self.importance.rename("gain").rename_axis("feature").reset_index()
        return out

    def meta(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "validation": self.validation.to_dict(),
            "headline": self.headline,
            "best_model": self.best_model,
            "source": self.data.source,
            "file_name": self.data.file_name,
            "fast": self.fast,
            "timings": self.timings,
            "default_scenario": self.scenario.to_dict(),
            "promotions": promo_meta(self),
            "detail": {"n_series": int(self.dm.n_series),
                       "n_detail": int(min(self.dm.n_series, self.detail_max_series))},
            "plan": {"data_end": str(self.dm.dates[-1].date()),
                     "test_end": str(self.profile.test[1]), "model": self.best_model},
        }


def top_revenue_series(dm: DemandMatrix, item_master: pd.DataFrame, k: int) -> np.ndarray:
    """Rows of the ``k`` series with the highest revenue over the last 365 days (sorted)."""
    price = item_master.set_index("item")["unit_price"].reindex(dm.keys["item"]).to_numpy(dtype="float64")
    recent = np.nansum(dm.values[:, -365:], axis=1) * np.nan_to_num(price)
    return np.sort(np.argsort(-recent, kind="stable")[:k])


def promo_meta(res: AnalysisResults) -> dict:
    """Where the promotions come from and their average effect (``{}`` without promotions)."""
    dm = res.dm
    if dm.promo is None:
        return {"source": None, "last_date": str(dm.dates[-1].date())}
    effects = estimate_effects(dm, dm.n_days - 1, res.promotions_cfg)
    on = dm.promo > 0
    return {
        "source": res.data.notes.get("promo_source"),
        "promo_days": int(on.sum()),
        "share_days": float(on.sum() / max(1, (~np.isnan(dm.values)).sum())),
        "beta_global": effects.beta_global,
        "lift_20_global": float(np.exp(effects.beta_global * 0.20)),
        "dip_global": effects.dip_global,
        "units_share_promo": float(np.nansum(np.where(on, dm.values, 0)) / max(1e-9, np.nansum(dm.values))),
        "post_promo_days": res.promotions_cfg.post_promo_days,
        "min_discount": res.promotions_cfg.min_discount,
        "last_date": str(dm.dates[-1].date()),
    }


def with_promo_columns(df: pd.DataFrame, dm: DemandMatrix, cfg: PromotionsConfig) -> pd.DataFrame:
    """Add ``discount`` and ``day_type`` (normal / promo / post_promo) to a daily detail table."""
    if dm.promo is None or df.empty:
        return df
    rows = pd.MultiIndex.from_frame(dm.keys).get_indexer(pd.MultiIndex.from_frame(df[["store", "item"]]))
    cols = (pd.to_datetime(df["date"]) - dm.dates[0]).dt.days.to_numpy()
    types = promo_day_types(dm, cfg)
    names = np.array(["normal", "promo", "post_promo"])
    return df.assign(discount=dm.promo[rows, cols].astype("float32"), day_type=names[types[rows, cols]])


def leaderboard_table(errors: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    """Metrics by period x model, overall and by ABC class / horizon / store."""
    e = errors.merge(classification[["item", "abc"]], on="item", how="left")
    frames = [evaluate(e, ["period", "model"]).assign(segment_type="overall", segment="all")]
    for col, seg in (("abc", "abc"), ("h", "horizon"), ("store", "store")):
        t = evaluate(e, ["period", "model", col]).rename(columns={col: "segment"})
        t["segment"] = t["segment"].astype(str)
        frames.append(t.assign(segment_type=seg))
    return pd.concat(frames, ignore_index=True)


def demand_monthly(data: StandardisedData) -> pd.DataFrame:
    s = data.sales.merge(data.item_master[["item", "unit_price"]], on="item", how="left")
    s["revenue"] = s["sales"] * s["unit_price"]
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    g = s.groupby(["store", "item", "month"], as_index=False, observed=True).agg(
        units=("sales", "sum"), revenue=("revenue", "sum"), n_days=("date", "nunique"))
    return g


def demand_dow(data: StandardisedData) -> pd.DataFrame:
    s = data.sales.assign(dow=data.sales["date"].dt.dayofweek)
    return s.groupby(["store", "item", "dow"], as_index=False).agg(units=("sales", "sum"), n_days=("date", "nunique"))


def detail_actuals(dm: DemandMatrix, profile: DataProfile, rows: np.ndarray) -> pd.DataFrame:
    a = dm.date_index(profile.valid[0])
    b = dm.date_index(profile.test[1])
    vals = dm.values[rows, a: b + 1]
    dates = dm.dates[a: b + 1]
    return pd.DataFrame({
        "store": np.repeat(dm.keys["store"].to_numpy()[rows], len(dates)),
        "item": np.repeat(dm.keys["item"].to_numpy()[rows], len(dates)),
        "date": np.tile(dates.to_numpy(), len(rows)),
        "sales": np.nan_to_num(vals, nan=0.0).ravel(),
        "period": np.tile(np.where(dates < pd.Timestamp(profile.test[0]), "validation", "test"), len(rows)),
    })


# --------------------------------------------------------------------------- stages
def stage_profile(data: StandardisedData, cfg: Config, fast: bool, enforce_limits: bool):
    report = validate(data, cfg.input, enforce_limits, cfg.prices, cfg.inventory)
    if not report.ok:
        raise InputError("Data validation failed:\n- " + "\n- ".join(report.errors))
    dm_full = data.matrix()
    profile = profile_history(dm_full, cfg, fast)
    report.excluded_series = [tuple(x) for x in profile.excluded_series]
    return report, profile, apply_exclusions(dm_full, profile)


def stage_forecast(dm: DemandMatrix, profile: DataProfile, cfg: Config, fast: bool, n_jobs: int,
                   progress: Progress) -> tuple[dict[str, BacktestResult], pd.Series | None]:
    backtests: dict[str, BacktestResult] = {}
    importance = None
    periods = {"validation": profile.valid, "test": profile.test}
    promos = has_promotions(dm, cfg.promotions)
    for k, (period, (a, b)) in enumerate(periods.items()):
        models = [get_model(m, cfg, profile, fast, n_jobs, promotions=promos) for m in profile.models]

        def _p(msg, frac, k=k):
            progress(f"Forecasting · {msg}", 0.30 + 0.45 * (k + frac) / 2)

        backtests[period] = run_backtest(models, dm, period, dm.date_index(a), dm.date_index(b),
                                         cfg.forecast.horizon, cfg.forecast.origin_step,
                                         refit_schedule(cfg, fast), _p)
        for m in models:
            if getattr(m, "importance_", None) is not None and period == "test":
                importance = m.importance_
    return backtests, importance


def pick_best(leaderboard: pd.DataFrame, baseline: str) -> str:
    v = leaderboard[(leaderboard["period"] == "validation")].sort_values("wape")
    return str(v.iloc[0]["model"]) if len(v) else baseline


def forecast_headline(leaderboard: pd.DataFrame, best: str, baseline: str) -> dict:
    t = leaderboard[leaderboard["period"] == "test"].set_index("model")
    v = leaderboard[leaderboard["period"] == "validation"].set_index("model")
    imp = improvement(t.reset_index(), baseline)
    return {
        "best_model": best,
        "best_model_label": MODEL_LABELS.get(best, best),
        "baseline": baseline,
        "wape_best": float(t.at[best, "wape"]),
        "wape_baseline": float(t.at[baseline, "wape"]),
        "wape_improvement_pct": float(imp[best]),
        "mape_best": float(t.at[best, "mape"]),
        "bias_best": float(t.at[best, "bias"]),
        "wape_best_validation": float(v.at[best, "wape"]) if best in v.index else None,
        "models": {m: {"wape": float(t.at[m, "wape"]), "mape": float(t.at[m, "mape"]),
                       "bias": float(t.at[m, "bias"]), "improvement_pct": float(imp[m])} for m in t.index},
    }


def run_analysis(data: StandardisedData, cfg: Config, fast: bool = False, n_jobs: int = -1,
                 progress: Progress | None = None, enforce_limits: bool = False,
                 db_path: str | Path = ":memory:") -> AnalysisResults:
    """Run every step in memory (used by the CLI and by the dashboard upload)."""
    progress = progress or _noop
    timings: dict[str, float] = {}
    t = time.perf_counter()

    progress("Validating and profiling the data", 0.02)
    report, profile, dm = stage_profile(data, cfg, fast, enforce_limits)
    timings["profile"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    progress("Running the SQL KPIs", 0.10)
    con = build_duckdb(data, db_path)
    sql = run_all(con, cfg.paths.sql, profile_params(profile, cfg))
    con.close()
    timings["sql"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    progress("Classifying items (ABC / XYZ)", 0.20)
    classification = classify_items(data.sales, data.item_master, *profile.valid,
                                    cfg.classification.abc_thresholds, cfg.classification.xyz_thresholds)
    timings["classify"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    backtests, importance = stage_forecast(dm, profile, cfg, fast, n_jobs, progress)
    errors = pd.concat([bt.error_table() for bt in backtests.values()], ignore_index=True)
    leaderboard = evaluate(errors, ["period", "model"])
    best = pick_best(leaderboard, cfg.forecast.baseline)
    timings["forecast"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    inp = build_policy_inputs(dm, backtests["validation"], backtests["test"], best, data.item_master)

    def _pp(msg, frac):
        progress(msg, 0.76 + 0.18 * frac)

    summary = add_costs(run_all_policies(inp, cfg, _pp))
    summary = summary.merge(classification[["item", "abc", "xyz"]], on="item", how="left")
    progress("Trade-off curves and sensitivity", 0.95)
    tradeoff = tradeoff_curves(inp, cfg)
    sens = sensitivity(inp, cfg)
    timings["policy"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    progress("Purchase plan: forecast from the last day of data", 0.97)
    plan = plan_tables(dm, inp, cfg, summary, best, profile, fast, n_jobs)
    timings["plan"] = round(time.perf_counter() - t, 2)

    stores = natural_sorted(data.stores["store"])
    online_default = data.stores.loc[data.stores["channel"] == "online", "store"].tolist()
    revenue_ref = classification.set_index("item")["revenue"]
    scenario = scenario_from_config(cfg, stores, revenue_ref, online_default)
    head = build_headline(summary, leaderboard, best, cfg, scenario, profile)
    progress("Done", 1.0)
    return AnalysisResults(data=data, validation=report, profile=profile, dm=dm, sql=sql,
                           classification=classification, backtests=backtests, errors=errors,
                           leaderboard=leaderboard, best_model=best, policy_inputs=inp, summary=summary,
                           tradeoff=tradeoff, sensitivity=sens, scenario=scenario, headline=head,
                           importance=importance, timings=timings, fast=fast, promotions_cfg=cfg.promotions,
                           plan=plan, detail_max_series=cfg.scale.detail_max_series)


def build_headline(summary: pd.DataFrame, leaderboard: pd.DataFrame, best: str, cfg: Config,
                   scenario: Scenario, profile: DataProfile) -> dict:
    sc = apply_scenario(summary, scenario)
    return {
        "scenario": scenario.label,
        "test_period": [str(profile.test[0]), str(profile.test[1])],
        "target_fill_rate": cfg.simulation.target_fill_rate,
        "forecast": forecast_headline(leaderboard, best, cfg.forecast.baseline),
        "inventory": headline(sc, "calibrated"),
        "inventory_fixed_z": headline(sc, "fixed"),
    }


def run_in_memory(data: StandardisedData, cfg: Config, fast: bool = True, progress: Progress | None = None,
                  n_jobs: int = -1) -> AnalysisResults:
    """Dashboard entry point for an uploaded file: fast mode, limits enforced, nothing on disk."""
    return run_analysis(data, cfg, fast=fast, n_jobs=n_jobs, progress=progress, enforce_limits=True)


# --------------------------------------------------------------------------- outputs
def example_series(res: AnalysisResults, stores: tuple[str, ...]) -> tuple[str, str]:
    """Top-revenue item of the first detail store (for the example charts)."""
    keys = res.dm.keys
    store = next((s for s in stores if s in set(keys["store"])), keys["store"].iloc[0])
    items = set(keys.loc[keys["store"] == store, "item"])
    ranked = res.classification.sort_values("rank")
    item = next(i for i in ranked["item"] if i in items)
    return store, item


def make_figures(res: AnalysisResults, cfg: Config, folder: Path | None = None) -> list[Path]:
    from src import reporting as rp

    folder = folder or cfg.paths.figures
    paths: list[Path] = []
    sql = res.sql
    paths.append(rp.fig_demand_trend(sql["daily_totals"], sql["sales_by_year"], folder))
    paths.append(rp.fig_seasonality(sql["dow_index"], sql["month_index"], folder))
    paths.append(rp.fig_abc_pareto(res.classification, cfg.classification.abc_thresholds, folder))
    paths.append(rp.fig_abc_xyz_matrix(abc_xyz_matrix(res.classification), folder))

    lb_t = res.leaderboard[res.leaderboard["period"] == "test"]
    base = cfg.forecast.baseline
    paths.append(rp.fig_wape_by_model(lb_t, res.best_model, base, folder))
    by_h = evaluate(res.errors[res.errors["period"] == "test"], ["model", "h"])
    paths.append(rp.fig_wape_by_horizon(by_h, res.best_model, base, folder))
    store, item = example_series(res, cfg.simulation.detail_stores)
    row = res.dm.series_index(store, item)
    view = res.backtests["test"].rolling_view(models=[res.best_model, base], rows=[row])
    start = view["date"].min()
    view = view[view["date"] < start + pd.Timedelta(days=91)]
    paths.append(rp.fig_forecast_example(view, res.best_model, base, f"store {store}, item {item}", folder))
    if res.importance is not None:
        paths.append(rp.fig_feature_importance(res.importance, folder))

    target = cfg.simulation.target_fill_rate
    phys = res.summary[res.summary["channel"] == "physical"]
    k_cal = aggregate_kpis(phys[phys["variant"] == "calibrated"], ["policy"])
    paths.append(rp.fig_policy_tradeoff(res.tradeoff, k_cal, target, folder))
    paths.append(rp.fig_cost_vs_service(res.tradeoff, k_cal, target, folder))
    paths.append(rp.fig_inventory_at_target(k_cal, target, folder))
    paths.append(rp.fig_cost_breakdown(k_cal, folder))
    daily = trajectory_frame(res, cfg, row)
    paths.append(rp.fig_inventory_trajectory(daily, f"store {store}, item {item} (calibrated policies)", folder))
    paths.append(rp.fig_sensitivity(res.sensitivity, folder))
    k_ch = aggregate_kpis(res.summary[res.summary["variant"] == "calibrated"], ["channel", "policy"])
    paths.append(rp.fig_channel_comparison(k_ch, folder))
    return paths


def trajectory_frame(res: AnalysisResults, cfg: Config, row: int, channel_name: str = "physical",
                     variant: str = "calibrated") -> pd.DataFrame:
    """Daily simulation of one series for every policy, with the knob found for it."""
    inp = res.policy_inputs.subset([row])
    channel = cfg.channel(channel_name)
    p = policy_parameters(inp, channel, cfg)
    s = res.summary
    store, item = inp.keys.at[0, "store"], inp.keys.at[0, "item"]
    frames = []
    for pol in cfg.simulation.policies:
        knob = s.loc[(s["store"] == store) & (s["item"] == item) & (s["channel"] == channel_name)
                     & (s["policy"] == pol) & (s["variant"] == variant), "knob"]
        k = np.array([float(knob.iloc[0])]) if len(knob) else default_knob(pol, p)
        sim = simulate_batch(inp.demand, inp.test_dates, make_policy(pol, p, inp, k, cfg.simulation.naive_order_weekday),
                             lead_time=p.lead_time, initial_on_hand=p.initial_on_hand,
                             backorder_share=channel.backorder_share, record_daily=True)
        d = pd.DataFrame({f: v[0] for f, v in sim.daily.items()})
        d.insert(0, "date", inp.test_dates)
        d.insert(0, "policy", pol)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_artefacts(res: AnalysisResults, cfg: Config, folder: Path | None = None) -> list[Path]:
    from src import reporting as rp

    folder = folder or cfg.paths.artefacts
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.parquet"):
        old.unlink()
    paths = [rp.write_artefact(df, name, folder) for name, df in res.tables(cfg.simulation.detail_stores).items()]
    paths.append(rp.write_json(res.meta(), "meta.json", folder))
    cfg_hash = hashlib.sha256(json.dumps(cfg.raw, sort_keys=True, default=str).encode()).hexdigest()[:12]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(cfg.root),
        "config_sha256": cfg_hash,
        "source": res.data.source,
        "input_sha256": res.data.file_hash[:16],
        "detail_stores": list(cfg.simulation.detail_stores),
        "files": [{"name": p.name, "bytes": p.stat().st_size,
                   "rows": int(pd.read_parquet(p).shape[0]) if p.suffix == ".parquet" else None} for p in paths],
    }
    paths.append(rp.write_json(manifest, "manifest.json", folder))
    return paths


# --------------------------------------------------------------------------- CLI
def _parse_map(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--map expects FIELD=COLUMN, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="run every step")
    g.add_argument("--step", action="append", choices=STEPS, help="run only these steps (repeatable)")
    p.add_argument("--config", default=None, help="path to params.yaml")
    p.add_argument("--input", default=None, help="sales file (.csv/.xlsx/.parquet) instead of the demo")
    p.add_argument("--map", nargs="*", metavar="FIELD=COLUMN", help="column mapping, e.g. date=Data item=SKU")
    p.add_argument("--item-master", default=None, help="optional price / cost / lead-time file")
    p.add_argument("--sheet", default=None, help="Excel sheet name")
    p.add_argument("--test-days", type=int, default=None, help="override the automatic test length")
    p.add_argument("--models", nargs="*", choices=KNOWN_MODELS, help="models to run (default: all)")
    p.add_argument("--policies", nargs="*", choices=KNOWN_POLICIES, help="policies to simulate (default: all)")
    p.add_argument("--stores", type=int, default=None, help="scenario: number of stores")
    p.add_argument("--items", type=int, default=None, help="scenario: number of products")
    p.add_argument("--item-rule", choices=("top_revenue", "random", "manual"), default=None)
    p.add_argument("--online", nargs="*", default=None, help="scenario: store IDs run as online")
    p.add_argument("--fast", action="store_true", help="fast mode (as in the dashboard)")
    p.add_argument("--quick", action="store_true", help="20 random series only (development / CI)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-figures", action="store_true", help="skip the README figures")
    return p


def config_from_args(args) -> Config:
    overrides: dict = {"input": {}, "scenario": {}, "periods": {}, "forecast": {}, "simulation": {}}
    if args.input:
        overrides["input"].update({"source": "file", "path": str(Path(args.input).resolve())})
    if args.map:
        overrides["input"]["column_map"] = _parse_map(args.map)
    if args.item_master:
        overrides["input"]["item_master"] = str(Path(args.item_master).resolve())
    if args.sheet:
        overrides["input"]["sheet"] = args.sheet
    if args.test_days:
        overrides["periods"]["test_days"] = args.test_days
    if args.models:
        overrides["forecast"]["models"] = list(dict.fromkeys(["seasonal_naive", *args.models]))
    if args.policies:
        overrides["simulation"]["policies"] = args.policies
    for key, val in (("n_stores", args.stores), ("n_items", args.items), ("item_selection", args.item_rule)):
        if val is not None:
            overrides["scenario"][key] = val
    if args.online is not None:
        overrides["scenario"]["online_stores"] = args.online
    return load_config(args.config, overrides)


def _quick_subset(data: StandardisedData, n: int = 20, seed: int = 0) -> StandardisedData:
    keys = data.sales[["store", "item"]].drop_duplicates()
    keys = keys.sample(min(n, len(keys)), random_state=seed)
    sales = data.sales.merge(keys, on=["store", "item"])
    promo = None if data.promo is None else data.promo.merge(keys, on=["store", "item"])
    return StandardisedData(sales, data.item_master[data.item_master["item"].isin(keys["item"])],
                            data.stores[data.stores["store"].isin(keys["store"])], data.source,
                            data.file_name, data.file_hash, data.notes, promo)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    steps = STEPS if args.all else tuple(s for s in STEPS if s in set(args.step))
    work = cfg.paths.work
    work.mkdir(parents=True, exist_ok=True)

    def say(msg: str, frac: float) -> None:
        print(f"[{frac:5.0%}] {msg}", flush=True)

    t0 = time.perf_counter()
    try:
        if args.all:
            data = load_input(cfg)
            if args.quick:
                data = _quick_subset(data)
            db = cfg.paths.duckdb if data.source == "demo" and not args.quick else ":memory:"
            res = run_analysis(data, cfg, fast=args.fast, n_jobs=args.n_jobs, progress=say, db_path=db)
            joblib.dump(res, work / "results.joblib", compress=3)
        else:
            res = _run_steps(steps, cfg, args, say)
            if res is None:
                return 0
    except FileNotFoundError as exc:
        print(f"Missing input: {exc}", file=sys.stderr)
        return 2
    except InputError as exc:
        print(f"Data validation failed: {exc}", file=sys.stderr)
        return 3

    if "report" in steps:
        cfg.paths.validation_report.parent.mkdir(parents=True, exist_ok=True)
        cfg.paths.validation_report.write_text(res.validation.to_markdown(), encoding="utf-8")
        if not args.no_figures:
            figs = make_figures(res, cfg)
            print(f"Wrote {len(figs)} figures to {cfg.paths.figures}")
        if res.data.source == "demo" and len(res.dm.keys) == res.data.n_series and not args.quick:
            from src.reporting import update_readme

            _write_demo_prices(res, cfg)
            arts = write_artefacts(res, cfg)
            print(f"Wrote {len(arts)} artefacts to {cfg.paths.artefacts}")
            if update_readme(cfg.root / "README.md", res.headline, len(res.dm.keys)):
                print("Updated the results block in README.md")
        else:
            print("Dashboard artefacts are only written for the full demo run; "
                  f"results are cached in {work / 'results.joblib'}")
    _print_summary(res)
    print(f"Total time: {time.perf_counter() - t0:.0f}s")
    return 0


def _write_demo_prices(res: AnalysisResults, cfg: Config) -> None:
    cfg.paths.prices.parent.mkdir(parents=True, exist_ok=True)
    res.data.item_master[["item", "unit_price", "unit_cost"]].to_csv(cfg.paths.prices, index=False)


def _run_steps(steps, cfg: Config, args, say) -> AnalysisResults | None:
    """Run ``ingest`` / ``analyse`` / ``report`` separately, with cached inputs in the work folder."""
    work = cfg.paths.work
    data_path, results_path = work / "data.joblib", work / "results.joblib"
    res = None
    if "ingest" in steps:
        data = load_input(cfg)
        if args.quick:
            data = _quick_subset(data)
        joblib.dump(data, data_path, compress=3)
        print(f"Standardised data cached in {data_path}")
    if "analyse" in steps:
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} is missing: run --step ingest first.")
        data = joblib.load(data_path)
        db = cfg.paths.duckdb if data.source == "demo" and not args.quick else ":memory:"
        res = run_analysis(data, cfg, fast=args.fast, n_jobs=args.n_jobs, progress=say, db_path=db)
        joblib.dump(res, results_path, compress=3)
    if "report" in steps:
        if res is None:
            if not results_path.exists():
                raise FileNotFoundError(f"{results_path} is missing: run --step analyse first.")
            res = joblib.load(results_path)
        # The scenario and the headline follow the current config without recomputing anything.
        stores = natural_sorted(res.data.stores["store"])
        online = res.data.stores.loc[res.data.stores["channel"] == "online", "store"].tolist()
        res.scenario = scenario_from_config(cfg, stores, res.classification.set_index("item")["revenue"], online)
        res.headline = build_headline(res.summary, res.leaderboard, res.best_model, cfg, res.scenario, res.profile)
    return res


def _print_summary(res: AnalysisResults) -> None:
    h = res.headline
    f = h["forecast"]
    inv = h["inventory"]
    print("\n=== Headline", f"({h['scenario']}, test {h['test_period'][0]} → {h['test_period'][1]})")
    print(f"Best model: {f['best_model_label']} · WAPE {f['wape_best']:.1%} vs {f['wape_baseline']:.1%} "
          f"(baseline) → {f['wape_improvement_pct']:.1%} less error · bias {f['bias_best']:+.1%}")
    if inv:
        print(f"At {h['target_fill_rate']:.0%} fill rate: average inventory P2 vs P0 "
              f"{-inv['inventory_reduction_pct']:+.1%}, vs P1 {-inv['inventory_reduction_vs_classic_pct']:+.1%} · "
              f"total cost P2 vs P0 {-inv['total_cost_reduction_pct']:+.1%}")
    print("Timings (s):", res.timings)


if __name__ == "__main__":
    # Run through the imported module so cached objects are pickled as src.pipeline.*, not __main__.*
    from src.pipeline import main as _main

    raise SystemExit(_main())
