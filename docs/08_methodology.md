# 08 — Methodology

## 1. Time split and leakage rules

The split is **derived from the length of the history** `H` (days), so the same method works on the demo and on any uploaded file.

```
|──────────── TRAIN ────────────|── VALID (V) ──|── TEST / SIMULATION (T) ──|
                                  reference       back-test + inventory
                                  period          simulation
```

| Quantity | Rule | Demo (H = 1,826) |
|---|---|---|
| Test length `T` | `min(365, max(28, round(0.25 · H)))` | 365 → 2017 |
| Validation length `V` | `T` if `H − 2T ≥ 56`; else `max(14, H − T − 56)` | 365 → 2016 |
| Train | everything before validation (at least 14 days) | 2013–2015 |
| Minimum `H` | 56 days (8 weeks), otherwise the file is rejected | – |

A length of 365 days is taken as one **calendar year** (so leap years work: the demo gets exactly 2016 and
2017). Explicit dates in `config/params.yaml` override these rules. When the validation period is shorter than
28 days or training shorter than 56 days, the run is flagged **low confidence** in the app and in the profile.

| Use | Data allowed |
|---|---|
| Model selection, hyper-parameters | Train → evaluated on the validation back-test |
| Policy parameters (σ, SS, EOQ), ABC/XYZ classes, product selection | Data up to the end of validation only |
| Final forecast scores and inventory simulation | Test back-test, models refit on data ≤ each origin |

Rule **NFR-11**: nothing used at time *t* may depend on data after *t*. Tests check it directly: changing every
value after a forecast origin never changes that origin's forecast (all four models), and LightGBM features at
day *t* never change when data after *t − 14* changes.

### 1.1 What adapts to the history

| Feature / model | Enabled when | Otherwise |
|---|---|---|
| Lag 364, day of year, week of year | ≥ 400 training days | Dropped; note "yearly seasonality not modelled" |
| Rolling windows 7 / 28 / 91 | window + 14 ≤ training days | Longer windows dropped |
| Holt-Winters (weekly season) | ≥ 21 days of history with some variation | That series falls back to the seasonal naive (counted in `fallback_series`) |
| LightGBM | ≥ 2,000 training rows after the 14-day lag | Skipped for the dataset |
| Seasonal naive, moving average | always | – |
| Series | ≥ 56 days of history before the test period | Excluded and listed (e.g. items launched recently) |

## 2. Exploratory analysis

- **Trend:** yearly totals and YoY growth (SQL `sales_by_year`); 365-day rolling mean.
- **Weekly seasonality:** day-of-week index = mean sales on that weekday / overall mean.
- **Yearly seasonality:** month index; a heatmap of year × week-of-year.
- **Heterogeneity:** do stores and items share the same seasonal shape? (Normalised profiles; if the shapes match, a global model is justified — [ADR-0002](adr/0002-global-lightgbm-model.md).)

## 3. ABC / XYZ classification

**ABC (value):** items sorted by revenue in the reference (validation) period, `Σ sales × unit_price` (descending); cumulative share ≤ 80% → **A**, ≤ 95% → **B**, else **C**. If prices are **assumed** (not in the uploaded data), ABC by volume is shown next to it and the revenue view is labelled *based on assumed prices*.

**XYZ (variability):** coefficient of variation of **weekly** item demand in the reference period, `CV = σ_week / μ_week`. X < 0.5 ≤ Y < 1.0 ≤ Z.
Weekly (not daily) aggregation removes the day-of-week pattern, which is predictable and is not real uncertainty.

> ⚠️ In the demo dataset the CV is probably low for every item (almost all **X**). If so, that is a finding worth writing up: the result is reported honestly, and a second view uses **CV of forecast residuals** (unexplained variability), which usually separates the items better.

**Use in the project:** results and KPIs are broken down by class. Optional: a different service level per class (e.g. A = 98%, B = 95%, C = 90%) as an extension.

## 4. Forecasting

### 4.1 Models

| Model | Definition | Notes |
|---|---|---|
| **Seasonal naive** *(baseline)* | `ŷ(t+h) = y(t+h−7·⌈h/7⌉)` → the same weekday of the last observed week | The "like last week" baseline from the brief |
| **Moving average** | `ŷ(t+h) = mean(y(t−27 … t))` | Flat; no weekly pattern → expected to lose to the seasonal naive |
| **Holt-Winters** | `ExponentialSmoothing(trend="add", damped_trend=True, seasonal="add", seasonal_periods=7)` per series | Captures level + trend + weekly season. Yearly seasonality is **not** modelled (a 365-day season is impractical for HW); this limitation is stated |
| **LightGBM** | One global model on all 500 series; target = daily sales; objective `tweedie` (non-negative counts) | Features below |

**LightGBM features** (all shifted by at least `min_lag = horizon = 14` days so that one model serves every h = 1…14 — the *direct, single-model* approach):
- Calendar: day of week, day of month, month, weekend and month-end flags; day of year and week of year when the history allows (no `year` feature: trees cannot extrapolate it).
- IDs: store, item (categorical).
- Lags: 14, 21, 28, 35, 364 days; mean of the four same-weekday lags.
- Rolling mean of the last 7, 28, 91 days and rolling std of the last 28 / 91 days, ending at lag 14; trend ratio `mean_28 / mean_91`.
- Objective `tweedie` (non-negative, count-like demand), 600 trees, 63 leaves, learning rate 0.05.

**Promotion features** (only when the data has promotions): discount of the target day, day number inside the
promotion (1, 2, …), and a flag for the 7 days after a promotion. They describe the **target** day: a promotion
calendar is decided weeks in advance, so it is known at the forecast origin (it is not demand data, no leakage).

### 4.2 Back-test design (rolling origin)

- **Origins:** the day before the period starts, then every 7 days (53 origins in a 1-year period). At the end of day *o* the planner knows sales up to *o* and forecasts *o+1 … o+14*.
- **Horizon:** 1–14 days (covers the gap to the next origin, up to 6 days, plus L = 7 → 13 ≤ 14). Longer lead times reuse the same weekday of the last forecast week.
- **Refit:** naive / MA need no fitting; Holt-Winters and LightGBM are re-estimated every 13 origins (a quarter). Between refits the parameters stay fixed but the forecasts use every new actual: Holt-Winters states are updated by a vectorised filter that reproduces statsmodels' recursion exactly (tested to 1e-8), LightGBM lag features read the new data.
- **Model selection** uses the validation back-test. **Reported numbers** come from the test back-test.

### 4.3 Error metrics

| Metric | Formula | Why |
|---|---|---|
| **WAPE** *(main)* | `Σ|A − F| / Σ A` | Scale-free, robust to zeros, weights big sellers more — matches the business view |
| MAPE | `mean(|A − F| / A)` over `A > 0` | Asked for by the brief; unstable when A is small → secondary metric |
| **Bias** | `Σ(F − A) / Σ A` | > 0 = over-forecast (too much stock), < 0 = under-forecast (stock-outs) |
| RMSE | `√mean((A − F)²)` | Feeds σ of the forecast error |

**Headline:** `improvement = 1 − WAPE_best / WAPE_seasonal_naive` → *"the best model cuts forecast error by X% vs. the seasonal-naive baseline"*.

Scores are reported overall, by ABC class, by horizon h and by store.

### 4.4 Fast mode (uploads in the app)

To keep an upload within NFR-13 (< 60 s for 200 series × 3 years), the live pipeline uses:

| Setting | Full (demo, CLI) | Fast (app upload) |
|---|---|---|
| Holt-Winters / LightGBM refit | every 13 origins | **once** per period (then updated with actuals, not retrained) |
| LightGBM trees | 600 | 300 |
| Holt-Winters | all series | only if ≤ 300 series, otherwise skipped (noted) |
| Simulation & calibration | vectorised over series | same (it is already cheap) |

Measured: a 600-day file with 8 series runs end-to-end in the app in a few seconds; the full demo (500 series ×
5 years, full mode) takes about 90 s on a laptop.

The CLI can run an uploaded file in full mode (`--input … ` without `--fast`); the app states which mode produced the numbers.

### 4.5 Promotions

**Where they come from.** A `promo` column in the file (flag or discount) or, without one, the selling price: a day
is a promotion when the price is at least 8% below the store-item's *regular price* (75th percentile of the prices in
a centred 9-week window — it follows list-price changes and is not pulled down by the promotions themselves).

**Effect, per item** (stores pooled), estimated on training days only:

| Quantity | Estimate |
|---|---|
| Normal level `b(t)` | mean of the previous 28 days without promotion or post-promotion dip |
| Lift at discount `d` | `lift(d) = exp(β · d)`, `β = ln(Σ y / Σ b) / d̄` over the item's promotion days (`d̄` = b-weighted mean discount) |
| Post-promotion dip | `Σ y / Σ b` over the 7 days after a promotion (customers stocked up) |
| Shrinkage | `β_item = (n · β_own + 14 · β_all) / (n + 14)`, n = promo days; same for the dip. β ∈ [0, 10], dip ∈ [0.5, 1.2] |

The exponential form lets the planner evaluate a promotion with a discount the item never had. Clearance sales after
a season (discount but falling demand) give β ≈ 0: the discount follows the end of the season, it does not create
demand.

**Use in the models.**
- *Moving average* and *Holt-Winters* (`PromoAware` wrapper): past demand is divided by the effect (lift on
  promotion days, dip after them) so a promotion week looks like a normal week; the model is fitted and run on that
  clean demand; its forecast is multiplied back by the lift of the promotions planned on the target days.
- *LightGBM*: the promotion features above; trees learn the effect by item and store.
- *Seasonal naive* stays promotion-blind on purpose: it is the "like last week" rule used without software, the
  reference all improvements are measured against.
- The forecast-driven policy P2 therefore raises its reorder point before a promotion and lowers it in the week after.

**Measured on the sample chain** (`data/samples/retail_chain_sales.csv`, fast mode, test 2024): WAPE LightGBM
36.9% → 34.3%, Holt-Winters 39.3% → 35.2%, moving average 44.4% → 41.3% when promotions are used. Estimated lift at
the true discounts 1.6–1.9× and dip 0.84 (generated: 1.4–2.1× and 0.85).

**Planner** (Promotions page): for a planned promotion (item, stores, dates, discount) — normal units = weekday
profile of the last 8 normal weeks; expected units = normal × lift(d); extra units to order; order-by date = start −
lead time; margin vs no promotion = promo units × discounted margin + dip-reduced week after − normal units × full
margin. A permanent price change uses half the promotion response (`exp(−β/2 · Δp)`), a rough estimate stated as
such.

### 4.6 Large datasets

Time and memory of the in-memory pipeline grow with the number of store-item series. Measured on a 10-core laptop
(fast mode, 2 years of daily sales): 10,000 series ≈ 1 minute and 3.5 GB; 30,000 series ≈ 2 minutes and 5.5 GB.
What keeps it bounded (`scale` block of `config/params.yaml`):

| Part | Small data | Large data |
|---|---|---|
| LightGBM training | every training row | above 1.5M rows: a random sample of series that contains every item; rows trimmed to the cap. On the sample chain × 8 (2,048 series) a quarter of the rows changes WAPE from 34.0% to 34.3% and cuts the forecasting time by 40% |
| LightGBM features and forecasts | one feature set, cached | blocks of 2,000 series |
| Holt-Winters | every series | skipped above 3,000 series (one statsmodels fit per series) |
| Daily detail for the dashboard | every series | the 400 top-revenue series (totals and KPIs still cover every series) |

**Batch mode** (`python -m src.partition`) removes the memory limit: DuckDB reads the file out of core, the items are
split into partitions of about 20,000 series (every store of an item in the same partition), each partition runs this
pipeline and writes Parquet tables, and the totals are recomputed from them — ABC on the revenue of every item (exact,
because each item is whole inside its partition), leaderboard and policy KPIs from the additive sums, reorder
parameters exported to CSV. A model that did not run on every series (e.g. Holt-Winters above its limit in one
partition) is listed apart, not ranked. Differences from one single run: one LightGBM per partition, and the best
model can differ between partitions (reported). Synthetic prices are drawn per item code, so they do not depend on
the partition. Measured: 60,000 series in 4.5 minutes with 2 parallel workers, 6 GB in total; the time is linear in
the number of series (≈ 2.5 hours for 2 million series with 2 workers).

## 5. Inventory theory

### 5.1 Notation

| Symbol | Meaning |
|---|---|
| `d̄` | Mean daily demand |
| `σ_d` | Std of daily demand (P1) or of daily forecast error (P2) |
| `L` | Lead time (days) |
| `σ_L` | Std of demand (or forecast error) over the lead time |
| `z` | Safety factor = Φ⁻¹(SL) |
| `S` | Ordering cost per order (€) |
| `c`, `p` | Unit cost, unit price (€) |
| `h` | Holding cost per unit per year = `i · c` |
| `D` | Annual demand = 365 · d̄ |
| `IP` | Inventory position = on hand + on order |

### 5.2 Formulas

- **Std over lead time** (fixed lead time, independent days): `σ_L = σ_d · √L`.
  General case with a random lead time: `σ_L = √(L·σ_d² + d̄²·σ_LT²)`.
- **Safety stock:** `SS = z · σ_L`
- **Reorder point:** `s = d̄ · L + SS`
- **Economic order quantity:** `Q* = √(2 · D · S / h)`, **capped at `max_cover_days` (28) days of demand**:
  `Q = min(Q*, 28 · d̄)`. Without the cap, cheap fast movers (pasta at €0.99 with S = €50 per order) get EOQ lots of
  4–9 months of demand, which no store would hold; the sample retail data made this visible.
- **Expected annual cost** (analytic check): `S·D/Q + h·(Q/2 + SS)`

### 5.3 Two meanings of "95% service" — stated clearly

| Type | Definition | How z is set |
|---|---|---|
| **Type 1 — cycle service level (α)** | P(no stock-out during a replenishment cycle) | `z = Φ⁻¹(0.95) = 1.645` |
| **Type 2 — fill rate (β)** | Share of demand served from stock | Solve `G(k) = (1 − β) · Q / σ_L`, `G(k) = φ(k) − k·(1 − Φ(k))` |

The brief uses `z × σ`, which is **Type 1**. The headline KPI is the **fill rate** (Type 2). With EOQ-sized batches, α = 95% usually gives a fill rate well above 95%, so it holds too much stock. The project therefore:
1. computes the classic Type-1 policy (as the brief asks),
2. adds the Type-2 variant (FR-22), and
3. reports the headline only after **calibrating every policy to exactly 95% fill rate** in the simulation (§6.4), which puts all policies on equal terms.

## 6. Policies and simulation

### 6.1 Policies compared

| ID | Policy | Rule | Parameters from |
|---|---|---|---|
| **P0** | Naive weekly | Every Monday order `Q_w` = mean weekly demand in the validation period | Validation-period sales |
| **P1** | Static `(s, Q)` | Every day: while `IP ≤ s` order `Q*` | Validation-period mean and σ of daily sales |
| **P2** | Forecast-driven `(s_t, Q)` | Every day: `s_t = Σ_{k=1..L} F(t+k) + SS`; while `IP ≤ s_t` order `Q*` | Latest forecast from the best model; SS from forecast errors |

**Why P2 should win:** P1's σ_d includes predictable variation (weekday pattern, trend, season), so it over-estimates uncertainty and holds too much stock, while its fixed `s` falls behind the upward trend. P2 follows the forecast and only buffers the **unpredictable** part (forecast error).

**σ for P2:** `σ_E` = RMSE of the **L-day cumulative forecast error** measured on the validation back-test with
exactly the decision rule used in the simulation (latest origin, days *t+1 … t+L*), per series. The RMSE (not the
std) also absorbs any forecast bias, and summing the errors over L captures their autocorrelation, which
`σ_e · √L` would miss.

### 6.2 Simulation mechanics (per series, over the test period — 2017 in the demo)

Daily event order:
1. **Receive** orders with arrival date = t; serve the open backlog first (online only).
2. **Serve demand** `sales_t = min(on_hand, demand_t)`; the unmet part is split into `backordered_t = share · unmet` and `lost_t = (1 − share) · unmet` (`share = 0` for physical stores).
3. Update `on_hand`, then `IP = on_hand + on_order − backlog`.
4. **Policy decision** at end of day → order arrives at `t + L`.
5. **Record** end-of-day state (holding cost is charged on it).

- **Initial state:** `on_hand = s + Q/2` of P1 for every policy (same start); `on_order = 0`.
- **Demand** = actual test-period sales (the simulation is a replay of history).
- **Lead time** = the item's `lead_time_days` if the data provides it, otherwise the global `L`.
- **Integer units:** order quantities are rounded up.

### 6.3 Costs

| Component | Formula |
|---|---|
| Ordering | `n_orders · S` |
| Holding | `Σ_t on_hand_end(t) · c · i / 365` |
| Shortage | `Σ_t lost(t) · (p − c)` (lost margin) + `Σ_t backordered(t) · backorder_cost` (online) |
| **Total** | sum of the three |

### 6.4 Fair comparison (iso-service)

Comparing inventory between policies only makes sense at the **same fill rate**. For each policy and series, one
knob is tuned by vectorised bisection (all series at once) to the **smallest value whose simulated fill rate
reaches 95%**:
- P0 → multiplier on `Q_w` (range 0.5 – 4);
- P1, P2 → safety factor `z` (range −12 – 8).

**Negative safety stock is normal here.** With EOQ lots of two to four weeks of demand, a stock-out can only
happen in the last days of a cycle, so a 95% *fill rate* is reached with a reorder point *below* the expected
lead-time demand (calibrated `z` around −3 for P2 on the synthetic test data). It is another way of saying that a
95% fill rate is a low target when lots are large — see §6.8.

Then: **`inventory reduction = 1 − avg_inventory_P2 / avg_inventory_P0`** → the headline.

> Calibrating on the same year that is being evaluated is an *oracle* setting. It is used **only** to compare policies fairly. The non-calibrated results (z = 1.645 fixed in advance) are also reported, to show the performance you would really get.

### 6.5 Channels: physical vs. online stores

The user can mark any store as **online** (UC-11). The demand history stays the same (BR-09); only the operating rules change:

| Aspect | Physical store | Online store |
|---|---|---|
| Unmet demand | 100% lost | `backorder_share` = 80% back-ordered, 20% lost (cancellations) |
| Shortage cost | `(p − c)` per lost unit | `(p − c)` per lost unit + `backorder_cost` (€2) per back-ordered unit |
| Holding rate | 25% / year (shop floor) | 20% / year (warehouse) |
| Service KPI | Fill rate = served from stock / demand | Same definition: back-ordered units count as **not** served from stock |

**Backorder mechanics:** open backorders form a `backlog`, served first when an order arrives. The inventory position becomes `IP = on_hand + on_order − backlog`, so the policy re-orders enough to cover the waiting customers.

**Expected effect:** a lower holding rate gives a larger EOQ; backorders reduce lost margin, so the cost-optimal service level is lower. The notebook compares the same stores run as physical and as online.

**Pre-computation:** each series is simulated (and calibrated, §6.4) under **both** channels, so any mix of physical and online stores chosen in the dashboard can be put together from existing rows (see [ADR-0005](adr/0005-configurable-scenario-and-channels.md)).

**Extension — risk pooling (FR-43):** if the online stores ship from one shared warehouse, their demand is summed per item. The pooled σ is `√(Σσ_i² + 2Σρσ_iσ_j)` ≤ `Σσ_i`, so less safety stock is needed. This is computed live with a fixed z (no calibration) and shown as "pooling benefit".

### 6.6 Scenario selection

- **Stores:** first N IDs or an explicit list (1 – stores in the data).
- **Products:** top-N by revenue in the **reference period** (default — keeps the most important items), random N with a seed, or a manual list (1 – items in the data).
- The scenario only **filters and re-aggregates** precomputed per-series sums. The KPIs of a scenario are therefore exactly what you would get by simulating it from scratch (the series are independent, see §7).

### 6.7 Sensitivity analysis

Grid: lead time ∈ {3, 7, 14} × service target ∈ {90, 95, 98, 99}% (z from the target, not calibrated) → fill
rate, average inventory (€) and total cost for P1 and P2.

### 6.8 Trade-off curves and the cost-optimal service level

For each policy one knob is moved for every series at once (P0 multiplier 0.9 – 1.3, z from −3 to 4) and the
test period is simulated again: this gives the **inventory vs. fill rate** curve of each policy (the lower the
curve, the less stock for the same service) and the **total cost vs. fill rate** curve. With the assumed costs
a lost unit costs its margin (40% of the price) while holding it for a whole year costs 25% × 60% = 15% of the
price, so total cost keeps falling until the fill rate is almost 100%: the 95% of the brief is a conservative,
not a cost-optimal, target. The dashboard's what-if lets the reader change these assumptions.

### 6.9 Purchase plan on the current calendar (Orders page)

The policy comparison runs on the past; the purchase plan applies the chosen policy — **P2, forecast-driven
(s, Q)** — from today on ([`src/replenishment.py`](../src/replenishment.py)).

1. **Forward forecast.** The best model is refitted on the whole history and forecasts the 14 days after the last day
   of data; days 15–28 (and the lead time beyond) reuse the same weekday of the last forecast week. Planned
   promotions (Promotions page) multiply it by their lift, and the week after by the dip.
2. **Bringing the data to today.** When the history ends ≥ 7 days before today, every date is moved forward by
   `7·⌊gap / 7⌋` days: weekdays are kept, the last day of data becomes one of the last 7 days, and the 0–6 days left
   until today are projected with the forecast (orders the rule would have placed in those days are shown as
   "assumed already placed"). Seasons move with the dates: this is a simulation on old data; with sales up to yesterday
   no shift is applied.
3. **Starting stock.** By default the state reached at the end of the test period by the P2 simulation (stock on
   hand, back-orders, orders on the way with their arrival days). Or the user's stock file (stock of today: the
   projection then starts today). Both can be corrected by hand.
4. **Projection.** Day by day: deliveries arrive, the forecast demand is served, and the rule orders `n·Q` when the
   inventory position falls to `s_t = Σ forecast(t+1 … t+L) + SS`. `SS = max(z·σ_E, 0)` with z the calibrated knob
   ("minimum for the target") or the textbook z ("cautious"). A **negative** safety stock — common after calibration,
   because large lots already protect most demand — reaches the fill rate on average over a year, but in a day-by-day
   plan of expected demand it would schedule a stock-out at the end of every cycle, so the plan floors it at zero.
5. **Outputs.** Orders of each day (supplier, quantity, value at cost, arrival = order day + lead time, stock and
   days of cover), deliveries, and **risks**: series whose expected stock runs out (≥ 1 unit) even with the planned
   orders — typically stock too low for the supplier's lead time.

### 6.10 Supplier terms and limits in the purchase plan

The plan can respect the constraints of real purchasing ([`src/constraints.py`](../src/constraints.py)). They are
applied every day in this order:

| Constraint | Rule |
|---|---|
| **Order days / delivery days** (per supplier) | a series orders only on an order day whose delivery (day + L) falls on a delivery day. The reorder point covers the lead time **plus the days until the next chance to order**: `s_t = Σ forecast(t+1 … t+L+g−1) + SS·√((L+g−1)/L)`, g = days to the next allowed day (g = 1 when every day is allowed, which gives back the plain rule) |
| **Minimum quantity, packs** (per item) | `q = max(q, MOQ)`, then rounded up to whole packs |
| **Quantity discount** (per supplier) | a line within `break_extra_days` (default 7) days of demand of the threshold is raised to it; lines at or above it are bought at `cost · (1 − discount)`; the saving is reported |
| **Warehouse space** (per store, units) | stock + on order never above the capacity; lines served by days of cover (most urgent first), cut to whole packs or postponed |
| **Minimum order value** (per supplier and store) | a smaller order is postponed to the next order day, unless a line is **urgent** (its position is below the demand until the next chance arrives): then the order is topped up with whole packs of the supplier's items with the fewest days of cover |
| **Weekly budget** (all suppliers, Monday–Sunday) | lines accepted urgent first, then by days of cover, until the budget is used; the others are postponed |

Every order line records what changed it (minimum, packs, discount, space, top-up); postponed lines are listed with
the reason. The page compares the plan **with and without constraints** over the next 28 days (order lines,
purchases, average stock, units not served, discounts, postponed lines) — the cost of the constraints in euros. When
fixed order days are introduced, most of the extra shortfall falls in the first one or two weeks (a stock built for
daily ordering meets weekly deliveries); the page says so when it happens.

## 7. Limitations

- Sales are treated as uncensored demand. In an uploaded history with real stock-outs, days at zero because of empty shelves would lower the forecasts; a future version could take an optional `stock_on_hand` column to mask those days.
- Daily data only (v1); intermittent series (many zeros) are handled by the same models, which is not ideal (Croston/SBA = FR-58).
- Costs, prices and lead time are assumptions.
- Holt-Winters ignores yearly seasonality.
- Purchase plan: orders are proposed, not sent; one discount threshold per supplier; space in units (not volume);
  the plan uses expected demand (risks show where it does not cover it).
- Promotions: one effect per item for every store; no cannibalisation between items, no halo on other products, no
  pre-promotion dip; the price-change estimate is derived from promotions and only indicative.
- The iso-service calibration uses the evaluation year (see the note in §6.4).
- Each series is independent: no shared supplier orders, no capacity limits, no transfers between stores.
- "Online" reuses a physical store's demand history; a real online channel would have its own demand pattern (e.g. different weekday peaks). This is a what-if on the operating rules, not a model of online demand.
- Scenarios are limited to the stores and items in the active dataset.
