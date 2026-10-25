# 07 — Data Dictionary

## 0. Input contract — your own sales history

Any daily sales history can be analysed if it can be mapped to the fields below. Column **names** are free (they are mapped in the app or with `--map`); the **content** must follow this contract.

| Field | Required | Type | Rules | If missing |
|---|---|---|---|---|
| `date` | ✅ | date | ISO `2024-03-15` or day-first `15/03/2024` (auto-detected; can be forced); Excel dates OK | — (error) |
| `item` | ✅ | text | Any code (`SKU-0042`, `12345`); stored as text | — (error) |
| `quantity` | ✅ | number | Units sold. Several rows for the same day/store/item are **summed** (transaction files OK). Negative values (returns) → `clip` to 0 (default) or `reject` | — (error) |
| `store` | ➖ | text | Location / shop / warehouse code | One location `ALL` |
| `unit_price` | ➖ | number > 0 | Selling price per unit, € | Synthetic price (flagged *assumed*) |
| `unit_cost` | ➖ | number > 0 | Purchase cost per unit, € | `cost_to_price_ratio × unit_price` |
| `channel` | ➖ | `physical` / `online` | Per store (first value wins if inconsistent → warning) | `physical` |
| `lead_time_days` | ➖ | integer ≥ 1 | Supplier lead time per item | `inventory.lead_time_days` |
| `supplier` | ➖ | text | Supplier of the item (most recent value); orders are grouped by it | "Supplier not set" |
| `item_name` | ➖ | text | Description shown next to the code | empty |
| `pack_size` | ➖ | integer ≥ 1 | Units per pack: orders are rounded up to whole packs | 1 |
| `min_order_qty` | ➖ | integer | Minimum units per order line | none |
| `promo` | ➖ | flag or % | Promotion of the row: `1` / yes / sì / x, or the discount (`20`, `20%`, `0.2`); `0` / no / empty = none | Found from `unit_price` (days ≥ 8% below the regular price); none without prices |

Price, cost and lead time may also come from a separate **item master** file (`item, unit_price, unit_cost, lead_time_days`). When a value changes over time, price and cost are the **median of the item's last 90 days** (a promotion at the end of the history does not become the list price) and the lead time is the most recent value.

**Standardisation rules**
1. A day with **no row** for a series, from its first sale to the last date in the file, = **0 units sold**.
2. Series with **< 56 days** of history (`input.min_history_days`) are excluded and listed in the report.
3. Frequency must be **daily** in v1 (weekly/monthly = FR-57, Could).

**Validation — errors (block the run)**
- A required field is not mapped, or no date can be parsed.
- Fewer than 56 days of history overall, or no series left after exclusions.
- File above the dashboard limits (300 MB, 10M rows, 10,000 series) → message pointing to the batch mode
  (`python -m src.partition`, no size limit).

**Validation — warnings (run continues)**
- Unparseable dates / quantities in some rows (rows dropped, count shown).
- Negative quantities clipped; share of zero days > 50% (intermittent demand → FR-58).
- Prices / costs / lead times assumed for N items.
- History < 1 year → no yearly seasonality.

**Promotions** (methodology §4.5): a `promo` column wins; a flag without a discount takes the discount implied by the
price of that day, or `promotions.default_discount` (20%) without a price. Without a `promo` column, the regular price
of each store-item is the 75th percentile of its prices in a centred 9-week window, and every day at least
`promotions.min_discount` (8%) below it is a promotion day. A day without a sale (no price) takes the median
discount of the item's other stores. A list-price increase is not a promotion.

**Promotion calendar** (Promotions page, optional): `item, start, end, discount[, store]` — e.g.
[`data/samples/promo_calendar_2025.csv`](../data/samples/promo_calendar_2025.csv). An empty store = every store
of the scenario.

**Current stock** (Orders page, optional): `store, item, on_hand[, on_order, arrival]` — Italian names work
(`Negozio, Articolo, Giacenza, Ordinato, Data arrivo`), e.g.
[`data/samples/stock_snapshot.csv`](../data/samples/stock_snapshot.csv). `store` may be omitted with one location.
On-order quantities without an arrival date are assumed to arrive in half the lead time. Series missing from the
file start from zero stock.

**Supplier terms** (Orders page, optional): `supplier, order days, delivery days, minimum order €, discount from
units, discount %` — Italian names work (`Fornitore; Giorni ordine; Giorni consegna; Ordine minimo €; Sconto da
pezzi; Sconto %`), days as `lun mer ven`, `Mon, Thu` or `1 4`; e.g.
[`data/samples/supplier_terms.csv`](../data/samples/supplier_terms.csv). Weekly budget and warehouse capacity per
store are set on the page.

Template: [`data/templates/sales_history_template.csv`](../data/templates/sales_history_template.csv).

## 1. Demo dataset

| Field | Value |
|---|---|
| Name | Store Item Demand Forecasting Challenge |
| Source | Kaggle competition `demand-forecasting-kernels-only` |
| Files used | `train.csv` only (`test.csv` has no sales → not used) |
| Period | 2013-01-01 → 2017-12-31 (1,826 days) |
| Size | 913,000 rows = 10 stores × 50 items × 1,826 days · ~17 MB |
| Licence | Kaggle competition rules — **do not redistribute**; download it yourself |

### `train.csv`

| Column | Type | Description | Domain |
|---|---|---|---|
| `date` | date (`YYYY-MM-DD`) | Calendar day | 2013-01-01 … 2017-12-31 |
| `store` | int | Store ID | 1 … 10 |
| `item` | int | Item ID | 1 … 50 |
| `sales` | int | Units sold that day | ≥ 0 |

The demo goes through the same input adapter: mapping `date → date, store → store, item → item, sales → quantity`; IDs become text.

**Important caveats**
- `sales` is **units sold**, which we treat as **demand**. In reality sales are *censored* by stock-outs; this dataset has no stock-out flag, so we assume sales = demand.
- There are **no prices** → a synthetic price table is generated (see §3).
- The data is believed to be **synthetic** and very regular (strong, stable seasonality). Expect high forecastability; see [R-02](11_risk_register.md).

## 2. DuckDB schema (`data/processed/inventory.duckdb` for the demo, `:memory:` for uploads)

```sql
CREATE TABLE sales (                -- canonical long format after standardisation
    date   DATE    NOT NULL,
    store  VARCHAR NOT NULL,
    item   VARCHAR NOT NULL,
    sales  DOUBLE  NOT NULL CHECK (sales >= 0),
    PRIMARY KEY (date, store, item)
);

CREATE TABLE item_master (
    item            VARCHAR PRIMARY KEY,
    unit_price      DOUBLE  NOT NULL,
    unit_cost       DOUBLE  NOT NULL,
    lead_time_days  INTEGER,          -- NULL → global default
    price_source    VARCHAR NOT NULL  -- 'file' | 'item_master' | 'assumed'
);

CREATE TABLE stores (
    store    VARCHAR PRIMARY KEY,
    channel  VARCHAR NOT NULL CHECK (channel IN ('physical', 'online'))
);

CREATE VIEW sales_enriched AS
SELECT s.*, m.unit_price, m.unit_cost, s.sales * m.unit_price AS revenue
FROM sales s JOIN item_master m USING (item);
```

## 3. Reference data — `data/reference/item_prices.csv`

Generated by `src.data.make_price_table` with `prices.seed = 42`:
`unit_price ~ Uniform(prices.min, prices.max)` rounded to €0.05, `unit_cost = 0.60 × unit_price`. Each draw comes
from a hash of the seed and the item code, so an item always gets the same price whatever other items are analysed
with it (a batch run on part of a file agrees with a run on the whole file). The file is written by the demo run.

> Why uniform and not tied to volume? So that ABC by revenue is not just a copy of ABC by volume. The price draw is random but fixed, and the ABC result is shown next to a volume-only ABC for transparency.

## 4. Parameters — `config/params.yaml`

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `input.source` | demo | – | `demo` or `file` |
| `input.path` / `sheet` / `item_master` | null | – | User file, Excel sheet, optional item master |
| `input.column_map` | {} | – | Field → user column; unmapped fields are guessed |
| `input.date_format` | auto | – | `auto` / `ISO` / `dayfirst` / `monthfirst` |
| `input.negative_quantity` | clip | – | `clip` (returns → 0) / `reject` |
| `input.limits` | 300 MB · 10M rows · 10,000 series | – | Dashboard upload limits (≈ 1 minute at the limit on a laptop; lower them for a 1 GB online host) |
| `input.min_history_days` | 56 | days | Series with less history before the test period are excluded |
| `periods.test_days` / `valid_days` | null | days | null = automatic (methodology §1) |
| `periods.test_start` / `test_end` / `valid_start` | null | date | Explicit periods override the automatic split |
| `prices.min` / `max` / `round_to` / `seed` | 5 / 50 / 0.05 / 42 | € | Synthetic prices for items without a price |
| `classification.abc_thresholds` | [0.80, 0.95] | share | Cumulative revenue cut-offs |
| `classification.xyz_thresholds` | [0.5, 1.0] | CV | Weekly CV cut-offs |
| `forecast.horizon` | 14 | days | ≥ review gap (6) + lead time (7) |
| `forecast.origin_step` | 7 | days | A new forecast every week |
| `forecast.ma_window` | 28 | days | Moving-average window |
| `forecast.holt_winters.refit_every` | 13 | origins | Re-estimate parameters every quarter |
| `forecast.lightgbm.refit_every` | 13 | origins | Retrain every quarter |
| `forecast.lightgbm.lags` / `rolling_windows` | [14, 21, 28, 35, 364] / [7, 28, 91] | days | Lags must be ≥ horizon (checked) |
| `forecast.lightgbm.params` / `num_boost_round` | tweedie, lr 0.05, 63 leaves / 600 | – | LightGBM settings |
| `fast_mode.*` | refit once · 300 trees · HW ≤ 300 series | – | Dashboard uploads |
| `scale.lightgbm_max_train_rows` | 1,500,000 | rows | Above this LightGBM trains on a sample of series (every item kept) |
| `scale.predict_chunk_series` | 2,000 | series | Features and forecasts computed in blocks of series |
| `scale.holt_winters_max_series` | 3,000 | series | Full mode: above this Holt-Winters is skipped |
| `scale.detail_max_series` | 400 | series | Daily detail tables only for the top-revenue series |
| `promotions.enabled` | true | – | Use promotions when the data has them |
| `promotions.min_discount` | 0.08 | share | A price this far below the regular price = promotion day |
| `promotions.regular_price_window` | 63 | days | Centred window of the regular price (75th percentile) |
| `promotions.default_discount` | 0.20 | share | Discount of a promo flag without a price |
| `promotions.post_promo_days` | 7 | days | Days after a promotion with a possible demand dip |
| `promotions.shrinkage_days` | 14 | days | Promo days needed before an item's own effect outweighs the average one |
| `inventory.lead_time_days` | 7 | days | `L` (per-item lead time from the data wins) |
| `inventory.order_cost` | 50 | € / order | `S` |
| `inventory.holding_rate` | 0.25 | 1 / year | `i` → `h = i · c` |
| `inventory.cost_to_price_ratio` | 0.60 | – | `c = 0.6 · p` when the cost is unknown |
| `inventory.max_cover_days` | 28 | days | The lot never exceeds this many days of demand (null = pure EOQ) |
| `inventory.service_level` | 0.95 | – | Target used to set `z` (fixed variant) |
| `inventory.service_type` | cycle | – | `cycle` (Type 1) or `fill_rate` (Type 2) |
| `scenario.n_stores` / `n_items` | null | – | null = all stores / items in the data |
| `scenario.item_selection` | top_revenue | – | `top_revenue` / `random` / `manual` |
| `scenario.online_stores` | [] | store IDs | Stores run as online channel |
| `channels.physical.backorder_share` | 0.0 | – | Unmet demand fully lost |
| `channels.online.backorder_share` | 0.80 | – | Share of unmet online demand that waits |
| `channels.online.backorder_cost` | 2.0 | € / unit | Goodwill / expedited shipping |
| `channels.online.holding_rate` | 0.20 | 1 / year | Warehouse instead of shop floor |
| `simulation.naive_order_weekday` | 0 | Monday | P0 order day |
| `simulation.target_fill_rate` | 0.95 | – | Target of the iso-service calibration |
| `simulation.detail_stores` | ["1"] | store IDs | Stores whose daily detail is published with the demo (licence) |
| `sensitivity.lead_times` | [3, 7, 14] | days | |
| `sensitivity.service_levels` | [0.90, 0.95, 0.98, 0.99] | – | |

## 5. Derived entities

| Entity | Grain | Built from | Used by |
|---|---|---|---|
| Demand stats | store × item | Validation-period daily sales (P1) / validation forecast residuals (P2) | Policy parameters |
| Weekly demand | item × ISO week | `sales` | XYZ |
| Back-test | model × origin × h × store × item | `sales` + models | Leaderboard |
| Simulation daily | store × item × policy × day | Policy + test-period actuals | KPIs |

Artefact schemas: see [06_api_specification.md §3](06_api_specification.md#3-data-contracts-artefacts).
