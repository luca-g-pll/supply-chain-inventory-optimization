# 06 — API Specification

The project has no web API. Its "API" is made of four contracts:

1. **Python API** — the public functions and classes of the `src` package (used by the notebooks, the dashboard and the tests).
2. **CLI** — `python -m src.pipeline`.
3. **Data contracts** — the tables written by the pipeline (demo artefacts) or computed live (uploads), read by the dashboard.
4. **SQL contract** — the named queries in `sql/kpi_queries.sql`.

This document describes the **implemented** interfaces (v1.0). Signatures are simplified to the essentials; the
docstrings in `src/` are the reference.

Conventions:
- The canonical long format is `[date, store, item, sales]`: `date` datetime64, `store` / `item` **text** (uploaded files can use any codes, e.g. `SKU-0042`), `sales` float.
- A *series* is one `(store, item)` pair. Dense arrays are `(n_series, n_days)`.
- Functions are pure unless their name starts with `load_`, `write_`, `build_` or `save_`.
- A problem with the user's file raises `src.ingest.InputError` (a `ValueError` with a user-facing message); other invalid arguments raise `ValueError`; a missing file raises `FileNotFoundError`.

---

## 1. Python API

### 1.1 `src.config`

```python
PROJECT_ROOT: Path
def load_config(path=None, overrides: dict | None = None) -> Config
    """Read config/params.yaml, deep-merge the nested `overrides`, validate every range."""

@dataclass(frozen=True)
class Config:
    root: Path
    paths: PathsConfig            # raw_train, duckdb, prices, work, artefacts, figures, sql, validation_report
    input: InputConfig            # source, path, sheet, item_master, column_map, date_format, negative_quantity, limits, min_history_days
    periods: PeriodsConfig        # test_days, valid_days, test_start, test_end, valid_start (None = automatic)
    prices: PricesConfig          # synthetic prices: min, max, round_to, seed
    classification: ClassificationConfig
    forecast: ForecastConfig      # horizon, origin_step, baseline, models, ma_window, holt_winters, lightgbm
    fast_mode: FastModeConfig     # refit_every, lightgbm_num_boost_round, holt_winters_max_series
    inventory: InventoryParams    # lead_time_days, order_cost, holding_rate, cost_to_price_ratio, service_level, service_type, days_per_year
    channels: dict[str, ChannelParams]   # "physical" / "online": inventory overrides + backorder_share, backorder_cost
    scenario: ScenarioConfig
    simulation: SimulationConfig  # policies, channels, naive_order_weekday, calibrate_to_fill_rate, target_fill_rate, detail_stores
    sensitivity: SensitivityConfig
    def channel(self, name) -> ChannelParams
    def replace(self, **blocks) -> Config
```

### 1.2 `src.ingest` — input adapter (demo and user files)

```python
REQUIRED = ("date", "item", "quantity")
OPTIONAL = ("store", "unit_price", "unit_cost", "channel", "lead_time_days")

def read_any(source, filename=None, sheet=None) -> pd.DataFrame
    """.csv / .txt / .tsv (delimiter and encoding sniffed, read as text), .xlsx / .xlsm, .parquet."""
def list_sheets(source) -> list[str]
def guess_mapping(columns) -> dict[str, str | None]
    """Field -> column from EN/IT synonyms (exact name > spacing variant > token match)."""

@dataclass(frozen=True)
class ColumnMapping:          # date, item, quantity + optional fields
    @classmethod
    def from_dict(cls, mapping) -> ColumnMapping   # InputError if a required field is missing

def parse_dates(values, fmt="auto") -> (pd.Series, list[str])   # ISO, dd/mm/yyyy, mm/dd/yyyy, yyyymmdd, Excel serials
def parse_numbers(values) -> pd.Series                          # 12 · 12,5 · 1.234,5 · 1,234.5 · €3,50

def standardise(raw, mapping, cfg_input, prices, inventory, source="file", file_name="", file_hash_value="") -> StandardisedData
    """Parse, drop unreadable rows, clip/reject negatives, SUM per (date, store, item), zero-fill every
    series from its first day to the last day of the file, build the item master (last known price /
    cost / lead time; missing prices -> seeded synthetic, missing cost -> ratio x price) and the store
    table (channel)."""

@dataclass
class StandardisedData:
    sales: pd.DataFrame        # [date, store, item, sales]
    item_master: pd.DataFrame  # [item, unit_price, unit_cost, lead_time_days, price_source, cost_source]
    stores: pd.DataFrame       # [store, channel, channel_source]
    source: str; file_name: str; file_hash: str; notes: dict
    def matrix(self) -> DemandMatrix

def merge_item_master(data, master_raw, prices, inventory, mapping=None) -> StandardisedData
def validate(data, cfg_input, enforce_limits=False, prices=None, inventory=None) -> ValidationReport   # never raises

@dataclass
class ValidationReport:
    n_rows_raw, n_rows, n_stores, n_items, n_series, date_min, date_max, history_days, pct_zero_days
    excluded_series: list[(store, item, reason)]; assumptions, errors, warnings: list[str]
    ok: bool (property); def to_markdown() -> str; def to_dict() -> dict
```

### 1.3 `src.series`

```python
@dataclass
class DemandMatrix:
    keys: pd.DataFrame          # [store, item], natural order ("2" before "10")
    dates: pd.DatetimeIndex     # consecutive days
    values: np.ndarray          # (n_series, n_days), NaN before each series' first day
    start: np.ndarray           # index of each series' first day
    @classmethod from_long(sales) -> DemandMatrix
    def to_long() -> pd.DataFrame; subset(rows); date_index(day); series_index(store, item); history_before(day_index)
```

### 1.4 `src.profile`

```python
@dataclass(frozen=True)
class DataProfile:
    data_start, data_end, history_days, train, valid, test   # (start, end) date tuples
    lags, rolling_windows, yearly_features, models, excluded_series, low_confidence, fast, notes

def split_periods(data_start, data_end, cfg) -> (train, valid, test)
def profile_history(dm, cfg, fast=False) -> DataProfile
def apply_exclusions(dm, profile) -> DemandMatrix
```

### 1.5 `src.data` and `src.sql_runner`

```python
def load_demo(path) -> pd.DataFrame                    # Kaggle train.csv, `sales` renamed to `quantity`
def demo_mapping() -> ColumnMapping
def make_price_table(items, cfg_prices, cost_to_price_ratio=0.6) -> pd.DataFrame
def build_duckdb(data, db_path=":memory:") -> duckdb.DuckDBPyConnection   # tables sales, item_master, stores + view sales_enriched
def load_sales(con, start=None, end=None) -> pd.DataFrame

def parse_queries(sql_path) -> dict[str, str]           # split on "-- name: <query>"
def run_query(con, sql, params=None) -> pd.DataFrame   # passes only the $parameters the query uses
def run_all(con, sql_path, params) -> dict[str, pd.DataFrame]
def profile_params(profile, cfg) -> dict                # $ref_start, $valid_start, $test_end, thresholds, ...
```

### 1.6 `src.classification`

```python
def abc_classify(revenue: pd.Series, thresholds=(0.80, 0.95)) -> pd.DataFrame   # [revenue, revenue_share, cum_share, abc]
def weekly_buckets(sales, start) -> pd.DataFrame                                 # complete 7-day buckets per item
def xyz_classify(weekly, thresholds=(0.5, 1.0)) -> pd.DataFrame                  # [weekly_mean, weekly_std, weekly_cv, xyz]
def classify_items(sales, item_master, ref_start, ref_end, abc_thresholds, xyz_thresholds) -> pd.DataFrame
def abc_xyz_matrix(classification) -> pd.DataFrame                               # 9 cells: items, revenue_share
```

### 1.7 `src.features` and `src.forecast`

```python
def build_features(values, dates, store_codes, item_codes, lags, rolling_windows, min_lag, yearly=True) -> FeatureSet
    """Lags, rolling mean/std ending at t - min_lag, same-weekday mean, 28/91-day trend ratio, calendar,
    store/item codes. ValueError if a lag < min_lag (leakage guard)."""

class Forecaster(Protocol):
    name: str
    def fit(self, dm: DemandMatrix, fit_end: int) -> Forecaster          # uses days <= fit_end only
    def predict(self, dm, origins: np.ndarray, horizon: int) -> np.ndarray   # (n_series, n_origins, horizon)

class SeasonalNaive(season_length=7)
class MovingAverage(window=28)
class HoltWinters(seasonal_periods=7, n_jobs=-1)
    # statsmodels estimates the parameters at each refit; a vectorised filter (tested to reproduce
    # statsmodels exactly) updates the states with new actuals between refits.
class LightGBMForecaster(params, num_boost_round, horizon, lags, rolling_windows, yearly=True, n_jobs=-1)
    # one global model, direct approach: every feature is >= horizon days old

def get_model(name, cfg, profile, fast=False, n_jobs=-1) -> Forecaster
def refit_schedule(cfg, fast=False) -> dict[str, int]
def make_origins(period_start, period_end, step=7) -> np.ndarray        # day before the period, then every 7 days
def run_backtest(models, dm, period, period_start, period_end, horizon, step=7, refit_every=None, progress=None) -> BacktestResult

@dataclass
class BacktestResult:
    period, period_start, period_end, origins, horizon, dates, keys
    actual: np.ndarray                    # (n_series, n_origins, horizon), NaN outside the period
    forecasts: dict[str, np.ndarray]      # model -> same shape
    info: dict[str, dict]                 # seconds, n_fits, fallback_series
    def error_table() -> pd.DataFrame     # additive error sums per model x series x horizon
    def rolling_view(models=None, rows=None) -> pd.DataFrame   # each day from the latest origin (h = 1..7)
    def paths_frame(model, rows=None) -> pd.DataFrame          # [store, item, origin, h, forecast]

def lead_time_demand(forecast, origins, decision_days, lead_time) -> np.ndarray
    """Σ forecast over (t, t+L] from the latest origin; horizons beyond H reuse the same weekday."""
```

### 1.8 `src.metrics`

```python
def wape(a, f); mape(a, f); bias(a, f); rmse(a, f) -> float
def error_sums(actual, forecast, axis=None) -> dict       # abs_err, err, actual, sq_err, n, ape_sum, n_pos
def metrics_from_sums(df) -> pd.DataFrame                  # adds wape, mape, bias, rmse
def evaluate(errors, by) -> pd.DataFrame                   # group sums, then metrics (exact for any subset)
def improvement(leaderboard, baseline="seasonal_naive", metric="wape") -> pd.Series
```

### 1.9 `src.inventory`, `src.simulation`, `src.policy`

```python
# inventory.py — formulas (scalars or arrays)
safety_factor(service_level); sigma_lead_time(sigma_daily, lead_time, mean_daily=0, sigma_lead_time_days=0)
safety_stock(z, sigma_L); reorder_point(mean_daily, lead_time, ss); eoq(annual_demand, order_cost, holding_cost)
normal_loss(k); safety_factor_fill_rate(fill_rate, Q, sigma_L); lot_size(...)   # EOQ rounded up, >= 1
NaiveWeeklyPolicy(qty, order_weekday)      # P0
StaticSQPolicy(s, Q)                       # P1
ForecastSQPolicy(ltd, ss, Q)               # P2: s_t = ltd[:, t] + ss
    .order(t, weekday, inventory_position) -> order quantities (n_series,)

# simulation.py — day-by-day engine, vectorised across series
simulate_batch(demand, dates, policy, lead_time, initial_on_hand, backorder_share=0.0, record_daily=False) -> BatchResult
    # BatchResult.sums: demand, sales, lost, backordered, bo_served, inv_unit_days, backlog_unit_days,
    #   n_orders, units_ordered, stockout_days, stockout_events, cycles, cycles_with_stockout; .fill_rate
simulate(demand_1d, dates, policy, lead_time, initial_on_hand, backorder_share=0.0) -> (daily DataFrame, summary)
calibrate_to_fill_rate(policy_for, demand, dates, lead_time, initial_on_hand, backorder_share, target, lo, hi)
    -> (knob, BatchResult, feasible)       # vectorised bisection, per series

# policy.py — from forecasts to P0 / P1 / P2
build_policy_inputs(dm, bt_valid, bt_test, best_model, item_master) -> PolicyInputs
forecast_error_sigma(inputs, lead_time) -> np.ndarray       # RMSE of the L-day cumulative error (validation)
policy_parameters(inputs, channel, cfg, lead_time_override=None, service_level=None) -> PolicyParams
make_policy(name, params, inputs, knob, weekday=0)          # knob = multiplier (P0) or z (P1, P2)
run_policy(name, variant, inputs, channel, cfg, params=None, record_daily=False) -> PolicyRun   # variant fixed | calibrated
run_all_policies(inputs, cfg, progress=None) -> pd.DataFrame     # channels x policies x variants, one row per series
tradeoff_curves(inputs, cfg, channel_name="physical") -> pd.DataFrame
sensitivity(inputs, cfg, channel_name="physical") -> pd.DataFrame
series_inputs(...) -> PolicyInputs                          # one series rebuilt from dated arrays (dashboard what-if)
```

### 1.10 `src.kpi` and `src.scenario`

```python
add_costs(frame) -> pd.DataFrame        # ordering, holding, shortage, total cost, values; then derive_kpis
derive_kpis(df) -> pd.DataFrame         # fill_rate, cycle_service_level, avg_inventory_units/value, turns, days_of_supply
aggregate_kpis(frame, by) -> pd.DataFrame   # SUM additive columns first, then ratios
headline(summary, variant="calibrated", baseline="P0_naive", candidate="P2_forecast_sq", classic="P1_static_sq") -> dict

@dataclass(frozen=True)
class Scenario:
    stores: tuple[str, ...]; items: tuple[str, ...]; online_stores: frozenset[str]
    def channel_of(store) -> "physical" | "online"; label: str
select_items(n_items, rule, revenue_ref, seed=42, manual=())      # top_revenue | random | manual
select_stores(all_stores, n_stores=None, ids=None)
build_scenario(all_stores, revenue_ref, n_stores, n_items, item_rule, online_stores, store_ids, seed, manual_items)
scenario_from_config(cfg, all_stores, revenue_ref, default_online=())
apply_scenario(frame, scenario) -> pd.DataFrame   # selected pairs; each store in its own channel
```

### 1.11 `src.pipeline` and `src.reporting`

```python
load_input(cfg) -> StandardisedData                  # demo or cfg.input.path, then the item master
run_analysis(data, cfg, fast=False, n_jobs=-1, progress=None, enforce_limits=False, db_path=":memory:") -> AnalysisResults
run_in_memory(data, cfg, fast=True, progress=None) -> AnalysisResults     # dashboard uploads: limits on, nothing on disk

@dataclass
class AnalysisResults:
    data, validation, profile, dm, sql, classification, backtests, errors, leaderboard, best_model,
    policy_inputs, summary, tradeoff, sensitivity, scenario, headline, importance, timings, fast
    def tables(detail_stores=None) -> dict[str, pd.DataFrame]   # the data contract of §3
    def meta() -> dict

make_figures(res, cfg, folder=None) -> list[Path]     # README charts (src/reporting.py)
write_artefacts(res, cfg, folder=None) -> list[Path]  # parquet + meta.json + manifest.json
```

---

## 2. CLI

```text
python -m src.pipeline (--all | --step {ingest,analyse,report} ...) [--config PATH]
       [--input FILE] [--map FIELD=COLUMN ...] [--item-master FILE] [--sheet NAME] [--test-days N]
       [--models M ...] [--policies P ...] [--stores N] [--items N] [--item-rule RULE] [--online ID ...]
       [--fast] [--quick] [--n-jobs N] [--no-figures]
```

| Step | Does | Cached in |
|---|---|---|
| `ingest` | read + standardise the input | `data/processed/work/data.joblib` |
| `analyse` | profile → SQL → classes → back-tests → policies → KPIs | `data/processed/work/results.joblib` |
| `report` | validation report, README figures, dashboard artefacts (demo only); scenario and headline follow the current config without recomputing | `reports/`, `data/processed/artefacts/` |

`--all` = the three steps. `--quick` runs 20 random series (development). `--fast` = the dashboard's fast mode.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | Configuration error |
| 2 | Missing input (demo file, `--input` file, or cached step) |
| 3 | Data validation failed |

Example: `python -m src.pipeline --all --input sales.xlsx --map date=Data item=Articolo quantity=Qta store=Negozio`

---

## 3. Data contracts (dashboard tables)

The demo writes these tables to `data/processed/artefacts/` (Parquet, zstd); an upload keeps the same tables in the
session (`AnalysisResults.tables()`), so every page works on both.

| Table | Grain | Main columns |
|---|---|---|
| `classification` | item | rank, units, revenue, revenue_share, cum_share, abc, abc_volume, weekly_mean/std/cv, xyz |
| `forecast_errors` | period × model × store × item × h | additive sums: abs_err, err, actual, sq_err, n, ape_sum, n_pos |
| `leaderboard` | period × model × segment | wape, mape, bias, rmse; segment_type overall / abc / horizon / store |
| `policy_summary` | store × item × channel × policy × variant | knob, lead_time, reorder_point, order_qty, safety_stock, unit cost/price, sums, costs, KPIs, abc, xyz |
| `tradeoff` | policy × knob | fill_rate, avg_inventory_value, total_cost (all series, physical) |
| `sensitivity` | policy × lead_time × service_level | fill_rate, avg_inventory_value, total_cost |
| `demand_monthly` | store × item × month | units, revenue, n_days |
| `demand_dow` | store × item × weekday | units, n_days |
| `forecasts` | model × store × item × date (detail stores) | h, actual, forecast (latest weekly origin) |
| `detail_actuals` | store × item × date (detail stores) | sales, period (validation / test) |
| `detail_paths` | period × store × item × origin × h (detail stores, best model) | forecast |
| `item_master`, `stores` | item / store | prices, costs, lead times, sources / channel |
| `sql_<query>` | as in the SQL file | one table per named query |
| `feature_importance` | feature | gain (LightGBM) |
| `meta.json` | run | profile, validation, headline, best_model, timings, default scenario |
| `manifest.json` | run | created_at, git_sha, config hash, input hash, detail_stores, files with rows and bytes |

**Licence rule:** the demo publishes daily values (`forecasts`, `detail_*`) only for `simulation.detail_stores`
(store 1); everything else is aggregated. Uploads keep every series in memory.

---

## 4. SQL contract

`sql/kpi_queries.sql` holds 13 named queries (`-- name: <query>` markers, each with a purpose/grain header):
`kpi_overview`, `daily_totals`, `sales_by_year`, `sales_by_month`, `dow_index`, `month_index`, `revenue_by_item`,
`revenue_by_store`, `abc_classification`, `weekly_cv_by_item`, `store_item_growth`, `top_bottom_series`,
`channel_mix`. Parameters (`$ref_start`, `$valid_end`, `$a_threshold`, …) come from `sql_runner.profile_params`.
`tests/test_sql.py` checks that the SQL ABC/XYZ classes equal the pandas ones.

---

## 5. Dashboard contract

`streamlit run app/streamlit_app.py` · `app/common.py` (active dataset, scenario sidebar) · `app/charts.py`
(Plotly, light/dark palette) · `app/tools.py` (one-series what-if) · `app/views/*.py`.

| Page | Reads | Live computation |
|---|---|---|
| Overview | policy_summary, forecast_errors, meta | aggregate for the scenario |
| Demand | demand_monthly, demand_dow, classification, sql_* | scenario filter |
| Forecast | forecast_errors, forecasts, feature_importance | scenario leaderboard, series metrics |
| Inventory | policy_summary, tradeoff, sensitivity, detail_actuals, detail_paths, item_master | one-series simulation and what-if (ms) |
| Data | — | read → map → standardise → validate → `run_in_memory` (fast mode) |

The app never writes to disk; it trains models only for an uploaded file.
